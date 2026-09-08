# 8 — Build: Naive vs Dynamic-Batched Server

> **You'll be able to say:** "I built one server with two endpoints over the same model code — a naive serial one and a dynamic-batched one — and load-tested both open-loop. The naive path saturated at ~2.5 req/s / 103 output tok/s; the batched path held 19.3 req/s of goodput and peaked at 786 tok/s, a **7.6× throughput and 9.9× goodput improvement on identical hardware**. Per-user TPOT got ~1.8× *worse* (8.8 → 16.3 ms) — exactly the trade lesson 5 predicts. And I can name the four bugs that bite everyone: blocking the event loop, left-padding position IDs, unbounded queues, and ignoring client cancellation."

This is the first lesson where you produce an artifact someone would hire you for. Everything before it was so that you'd know *what to measure*; everything after it (lesson 9, Phase 4, Phase 5) is a refinement of the loop you're about to write.

**Scope discipline:** you are not writing vLLM. You are writing the smallest honest server that demonstrates the throughput/latency trade, with real measurements. The one below is ~200 lines, runs on a laptop CPU or MPS with GPT-2, and every number in this lesson came out of it.

---

## What you're building

```
                              ┌──────────────────────────────────────────┐
  client ──POST /generate_naive─▶  handler: async with GLOBAL_LOCK       │
         ◀────── SSE tokens ───┤    → one request at a time, batch = 1   │
                              └──────────────────────────────────────────┘
                              ┌──────────────────────────────────────────┐
  client ─POST /generate_batched▶ handler: enqueue Req(prompt, out_queue) │
         ◀────── SSE tokens ───┤   ... and immediately await its tokens   │
                              │  ┌────────────────────────────────────┐  │
                              │  │ background worker (one, forever):  │  │
                              │  │  collect up to MAX_BATCH within    │  │
                              │  │  MAX_WAIT → run batched prefill    │  │
                              │  │  + decode in a THREAD → emit       │  │
                              │  └────────────────────────────────────┘  │
                              └──────────────────────────────────────────┘
```

Both endpoints call the *same* `Engine.step_batch()`. That is deliberate: it removes "maybe their model code is just faster" as an explanation of the difference. The only difference is **how many requests share a forward pass.**

---

## The code

Save as `projects/01-tiny-inference-server/server.py`. Dependencies: `torch`, `transformers`, `fastapi`, `uvicorn`. No GPU required.

```python
#!/usr/bin/env python3
"""Phase 3, lesson 8: one model, two endpoints, identical model code.

  /generate_naive    one request at a time (a global lock) -- batch size is always 1
  /generate_batched  dynamic batching: a background worker forms batches of up to
                     MAX_BATCH requests within a MAX_WAIT window, then decodes them
                     together to completion (a static batch per group)

Run:  MODEL=gpt2 uvicorn server:app --port 8000
Bench: python bench.py --url http://127.0.0.1:8000/generate_batched --qps 1 2 4
"""
import asyncio, json, os, time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

MODEL      = os.environ.get("MODEL", "gpt2")
DEVICE     = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu")
MAX_BATCH  = int(os.environ.get("MAX_BATCH", "16"))
MAX_WAIT   = float(os.environ.get("MAX_WAIT", "0.010"))     # dynamic-batching window, seconds
QUEUE_CAP  = int(os.environ.get("QUEUE_CAP", "64"))         # lesson 6, gate 2
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "0"))

app = FastAPI()
STATS = {"naive": 0, "batched": 0, "shed": 0, "batches": 0, "batch_tokens": 0, "steps": 0}


# ----------------------------------------------------------------------------- model
class Engine:
    """Thin wrapper: batched prefill + batched decode with left padding."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"                       # decode always reads the LAST column
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
        self.eos = self.tok.eos_token_id

    @torch.inference_mode()
    def step_batch(self, prompts: List[str], max_new: List[int], emit):
        """Generate for a whole batch, calling emit(row_index, token_text) per token."""
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        # left padding breaks default position_ids -- this is THE batched-generation bug
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        past, B = out.past_key_values, ids.shape[0]
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)      # greedy, for reproducibility
        alive = [True] * B
        produced = [0] * B
        next_pos = pos[:, -1:] + 1
        for _ in range(max(max_new)):
            for r in range(B):
                if alive[r]:
                    tid = int(nxt[r, 0])
                    emit(r, self.tok.decode([tid]))
                    produced[r] += 1
                    if tid == self.eos or produced[r] >= max_new[r]:
                        alive[r] = False                      # finished rows keep being COMPUTED:
            if not any(alive):                                # that is static batching's waste
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=DEVICE)], dim=-1)
            out = self.model(input_ids=nxt, attention_mask=mask,
                             position_ids=next_pos, past_key_values=past, use_cache=True)
            past, next_pos = out.past_key_values, next_pos + 1
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            STATS["steps"] += 1
        return produced


ENGINE: Optional[Engine] = None
NAIVE_LOCK: Optional[asyncio.Lock] = None
QUEUE: Optional[asyncio.Queue] = None


@dataclass
class Req:
    prompt: str
    max_tokens: int
    out: asyncio.Queue = field(default_factory=asyncio.Queue)


# ----------------------------------------------------------------------------- worker
async def batching_worker():
    """Dynamic batching: wait up to MAX_WAIT for up to MAX_BATCH requests, then run them."""
    loop = asyncio.get_event_loop()
    while True:
        batch = [await QUEUE.get()]
        deadline = time.perf_counter() + MAX_WAIT
        while len(batch) < MAX_BATCH:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(QUEUE.get(), timeout))
            except asyncio.TimeoutError:
                break
        STATS["batches"] += 1
        STATS["batch_tokens"] += len(batch)

        def emit(row, text):                                  # called from the worker thread
            loop.call_soon_threadsafe(batch[row].out.put_nowait, text)

        try:
            await asyncio.to_thread(ENGINE.step_batch,        # never block the event loop
                                    [r.prompt for r in batch],
                                    [r.max_tokens for r in batch], emit)
        except Exception as exc:                              # one bad row must not kill the batch
            for r in batch:
                loop.call_soon_threadsafe(r.out.put_nowait, f"[error: {type(exc).__name__}]")
        for r in batch:
            loop.call_soon_threadsafe(r.out.put_nowait, None)


@app.on_event("startup")
async def startup():
    global ENGINE, NAIVE_LOCK, QUEUE
    if TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    ENGINE = Engine()
    NAIVE_LOCK = asyncio.Lock()
    QUEUE = asyncio.Queue()
    ENGINE.step_batch(["warmup"], [4], lambda r, t: None)     # warmup: allocator, kernels, JIT
    asyncio.ensure_future(batching_worker())


def sse(text):
    return f"data: {json.dumps({'token': text})}\n\n"


@app.post("/generate_naive")
async def generate_naive(request: Request):
    body = await request.json()
    prompt, max_tokens = body.get("prompt", ""), int(body.get("max_tokens", 32))
    STATS["naive"] += 1

    async def stream():
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        async with NAIVE_LOCK:                                # one request at a time: batch = 1
            task = asyncio.ensure_future(asyncio.to_thread(
                ENGINE.step_batch, [prompt], [max_tokens],
                lambda r, t: loop.call_soon_threadsafe(q.put_nowait, t)))
            while True:
                get = asyncio.ensure_future(q.get())
                done, _ = await asyncio.wait({get, task}, return_when=asyncio.FIRST_COMPLETED)
                if get in done:
                    yield sse(get.result())
                elif task in done:
                    get.cancel()
                    while not q.empty():
                        yield sse(q.get_nowait())
                    break
            await task
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/generate_batched")
async def generate_batched(request: Request):
    body = await request.json()
    if QUEUE.qsize() >= QUEUE_CAP:                            # lesson 6: bound the queue, shed fast
        STATS["shed"] += 1
        return StreamingResponse(iter([sse("[busy]")]), status_code=429,
                                 media_type="text/event-stream")
    req = Req(body.get("prompt", ""), int(body.get("max_tokens", 32)))
    STATS["batched"] += 1
    await QUEUE.put(req)

    async def stream():
        while True:
            tok = await req.out.get()
            if tok is None:
                break
            yield sse(tok)
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/stats")
async def stats():
    s = dict(STATS)
    s["mean_batch_size"] = (STATS["batch_tokens"] / STATS["batches"]) if STATS["batches"] else 0
    s.update(model=MODEL, device=DEVICE, max_batch=MAX_BATCH, max_wait_ms=MAX_WAIT * 1e3)
    return s
```

---

## The four things that will bite you

These are not style notes. Each one is a bug that produces *plausible but wrong* numbers, which is worse than a crash.

### 1. Never block the event loop

`ENGINE.step_batch()` is synchronous, CPU/GPU-bound, and takes hundreds of milliseconds. Called directly inside an `async def` handler, it freezes the entire process — no new connections accepted, no tokens flushed to anyone, health checks timing out. Hence `await asyncio.to_thread(...)`.

Why a thread is enough (and this is [Phase 0 lesson 3](../phase-0/03-processes-threads-concurrency.md) paying off): PyTorch releases the GIL around its compute kernels, so the event loop really does get to run while the model computes. What crosses back is a **thread-safe** callback — `loop.call_soon_threadsafe(...)`, never `queue.put_nowait` directly from the thread, or you'll corrupt loop state in ways that show up as random hangs under load.

### 2. Left padding, and the position-IDs bug

A batch has prompts of different lengths, so you pad. **You must pad on the left**, because decode reads `logits[:, -1]` — with right padding, the last column of a short row is a pad token and you'd sample from garbage.

But left padding breaks positions. HuggingFace derives default `position_ids` from the *padded* sequence length, so a row padded with 40 tokens starts its real content at position 40 instead of 0 — the model sees a shifted, wrong positional encoding and produces subtly worse text. The fix is two lines and it's the same trick HF's own `generate()` uses:

```python
pos = (mask.cumsum(-1) - 1).clamp(min=0)      # per-row positions that skip pad tokens
next_pos = pos[:, -1:] + 1                    # ... and keep counting during decode
```

**Test for it:** run the same prompt alone and inside a batch with one much longer prompt. Greedy decoding must produce **identical** tokens. If it doesn't, your positions or your mask are wrong, and every quality number you report afterwards is meaningless. This single test is the difference between a demo and an engineering artifact.

### 3. Bound the queue (and make shedding visible)

`QUEUE_CAP` + HTTP 429 is [lesson 6](06-scheduling-policies-and-admission-control.md)'s gate 2, in five lines. Without it, an overloaded server accepts unbounded work, every request eventually times out, and you get congestive collapse — 100% GPU utilization, ~0 goodput. With it, you get a *decision*: some clients get a fast, honest "busy."

The measurement discipline that goes with it: **429s are data.** `bench.py` counts them in a `shed` column rather than treating them as errors, because a config that "wins" by rejecting a third of traffic must show that third.

### 4. Cancellation and error isolation

Two more lines of production reality that this server handles and most first attempts don't:

- A batch runs in `try/except` so **one malformed row cannot kill the other 15**. In a static batch, an exception mid-decode loses everyone's work.
- Client disconnects: FastAPI raises inside the generator when the client goes away, and the `Req` object is dropped. A real engine goes further — check `await request.is_disconnected()` and stop generating; a disconnected client's tokens are 100% waste ([lesson 5](05-queueing-theory.md)).

---

## Run it and measure it

Two terminals. The load generator is [lesson 7's `bench.py`](07-measuring-honestly.md) — open loop, intended-time stamping, warmup, drift check.

```bash
# terminal 1
MODEL=gpt2 MAX_BATCH=16 MAX_WAIT=0.010 TORCH_THREADS=4 uvicorn server:app --port 8000

# terminal 2 — sweep the NAIVE endpoint until it collapses
python bench.py --url http://127.0.0.1:8000/generate_naive \
                --qps 2 3 4 --duration 45 --warmup 5 --settle 15 \
                --prompt-tokens 64 --max-tokens 32 --slo 1.0
# then the BATCHED endpoint, over a much wider range
python bench.py --url http://127.0.0.1:8000/generate_batched \
                --qps 3 10 20 30 40 --duration 40 --warmup 5 --settle 15 \
                --prompt-tokens 64 --max-tokens 32 --slo 1.0
curl -s localhost:8000/stats
```

`--settle` matters more than it looks: after an overloaded point the server still has a backlog, and if the next point starts immediately it inherits somebody else's queue. Run **one server at a time** on the box, too — measurements taken while a second engine was draining were wrong by two orders of magnitude ([lesson 9](09-build-continuous-batching-engine.md) has the receipts).

**Predict before you look.** Single-request service time here is `prefill + 32 × TPOT ≈ 0.04 + 32 × 0.009 ≈ 0.33 s`, so naive capacity is `1/0.33 ≈ 3 req/s`. Batched capacity should be several times that, and per-user TPOT should get *worse* as the batch grows.

### Measured: GPT-2 124M, Apple M5 (MPS), 64-token prompts, 32 output tokens, greedy

**Naive endpoint** (`MAX_BATCH` irrelevant — one request at a time):

| Offered QPS | n | p50 TTFT | p90 TTFT | p99 TTFT | p50 TPOT | p99 ITL | e2e p99 | Output tok/s | Goodput (TTFT ≤ 1 s) | Drift |
|---|---|---|---|---|---|---|---|---|---|---|
| **2** | 77 | 44 ms | 754 ms | 1,003 ms | 8.8 ms | 31 ms | 1.28 s | 62.7 | **1.96/s** | 0.7× |
| 3 | 119 | 6,653 ms | 10,797 ms | 11,824 ms | 13.0 ms | 33 ms | 12.24 s | 73.9 | 0.14/s | 14.0× ⚠ |
| 4 | 168 | 6,126 ms | 11,406 ms | 12,024 ms | 8.9 ms | 169 ms | 12.30 s | 103.0 | 0.04/s | 7.0× ⚠ |

**Batched endpoint** (`MAX_BATCH=16`, `MAX_WAIT=10 ms`, `QUEUE_CAP=64`):

| Offered QPS | n | shed | p50 TTFT | p90 TTFT | p99 TTFT | p50 TPOT | p99 ITL | e2e p99 | Output tok/s | Goodput (TTFT ≤ 1 s) | Drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 104 | 0 | 160 ms | 490 ms | 844 ms | 11.0 ms | 67 ms | 1.22 s | 93.8 | 2.93/s | 0.3× |
| 10 | 378 | 0 | 249 ms | 441 ms | 709 ms | 10.5 ms | 41 ms | 1.15 s | 339.3 | 10.60/s | 1.1× |
| **20** | 690 | 0 | 392 ms | 634 ms | 835 ms | 14.2 ms | 62 ms | 1.32 s | 618.0 | **19.31/s** | 0.9× |
| 30 | 925 | 82 | 2,239 ms | 2,547 ms | 2,686 ms | 15.6 ms | 97 ms | 3.17 s | 785.5 | 0.11/s | 1.8× ⚠ |
| 40 | 867 | 522 | 2,502 ms | 2,681 ms | 2,760 ms | 16.3 ms | 120 ms | 3.27 s | 735.9 | 0.00/s | 1.0× |

### What the numbers say

**1. The headline, stated honestly.** Peak goodput went from **1.96 → 19.31 req/s (9.9×)** and peak output throughput from **103 → 786 tok/s (7.6×)**, on the same laptop, same model, same `step_batch` code. Batching is not a micro-optimization; it is the difference between a toy and a service.

**2. The trade is visible and it is exactly the predicted one.** Per-user `p50 TPOT` rose from 8.8-8.9 ms (naive) to 14.2 ms at 20 QPS and 16.3 ms at 40 QPS — **~1.8× worse per-token latency for 7.6× the throughput.** That's `S(B) = a + b·B` from [lesson 5](05-queueing-theory.md), measured on your own machine. Anyone who claims batching is free hasn't looked at TPOT.

**3. TTFT stays flat and then explodes — that's the knee, not a gradual slope.** Batched p99 TTFT sits between 709 ms and 844 ms across a 7× range of load (3 → 20 QPS), then triples by 30 QPS. The naive endpoint does the same thing between 2 and 3 QPS: 1.0 s → 11.8 s p99 TTFT for a 1.5× load increase. Capacity is a cliff, and the only way to know where it is, is to walk off it in a test rather than in production.

**4. Throughput keeps rising after goodput has collapsed.** 30 QPS produced the *highest* output rate in the whole table (785.5 tok/s) with goodput of **0.11 req/s** — a 175× goodput collapse from the 20 QPS row, at 27% more tokens/sec. If you optimize for tok/s you will ship the 30 QPS config. This is [lesson 7](07-measuring-honestly.md)'s central warning, reproduced on real hardware.

**5. Load shedding did its job — and shows up as data, not silence.** At 30 QPS: 82 rejections; at 40 QPS: 522. Without `QUEUE_CAP` those requests would have been admitted, queued for seconds, and delivered to clients that had given up — buying nothing and stealing capacity from requests that could still have succeeded. Note also that the served requests' p99 TTFT stops climbing (2.69 s → 2.76 s from 30 to 40 QPS): **that flat tail is the queue bound doing its job**, converting unbounded latency into bounded latency plus rejections.

**6. The drift column is not decoration.** Every collapsed row is flagged (`14.0×`, `7.0×`, `1.8×`), and the flagged rows are exactly the ones whose percentiles are meaningless — they describe a queue that was still growing. Report those points as "unstable," never as a p99.

**7. Mean batch size was 8.1 across the whole session** (from `/stats`), while `1 + λT` at 20 QPS with a 10 ms window predicts only **1.2**. The observed batch is ~7× larger because requests accumulate while the *previous* batch is still decoding (~350 ms), not during the 10 ms window. **The window is nearly irrelevant here; the batch self-forms from the service time.** That's [lesson 3](03-dynamic-batching.md)'s `B = λa/(1−λb)` result showing up in a real server, and it's the kind of observation that makes a writeup credible.

**8. What this design still cannot fix:** every batch runs `max(max_tokens)` steps, so a single long request drags a whole group. With uniform 32-token outputs (above) that waste is invisible. Change the workload to a realistic ragged mix and this endpoint collapses at ~5 QPS — measured, with numbers, in [lesson 9](09-build-continuous-batching-engine.md).

### Caveat you must state (and I'm stating it)

Client and server ran on the same laptop, so at the highest QPS points the client competes with the server for CPU — [lesson 7](07-measuring-honestly.md)'s "isolate the resource under test." The comparison stays valid because both endpoints were measured the same way, but the absolute numbers are a floor. Also: MPS, GPT-2 124M, and greedy decoding. **Re-run on a Colab T4 with a 1-7B model and keep both tables** — the shape will be the same and the ratios will be larger.

---

## Exercises that turn this into the artifact

Do these in order; each is one edit plus one bench run. Deliverable is [Project 01](../../projects/README.md), and this is the **Phase 3 exit artifact**.

1. **Find both knees** and plot p50/p99 TTFT vs QPS with one line per endpoint. Mark the goodput maximum on each. This plot *is* the project.
2. **Sweep `MAX_BATCH`** ∈ {1, 2, 4, 8, 16, 32} at the fixed QPS where goodput peaked. Plot throughput and p99 TTFT on the same x-axis. Confirm throughput saturates while latency keeps rising, and state the batch size *your SLO* would choose.
3. **Sweep `MAX_WAIT`** ∈ {0, 5, 10, 50, 200} ms. Explain why it barely matters here, using measured mean batch size vs `1 + λT`. (This is the most commonly misunderstood knob in serving; you'll have data.)
4. **Break the padding on purpose:** remove the `position_ids` fix and diff greedy outputs for a prompt run alone vs in a mixed-length batch. Record the corrupted text. Then restore it and show the outputs match. Correctness evidence beats a paragraph claiming correctness.
5. **Prove output-length waste:** make `max_tokens` heterogeneous (16 for half the requests, 256 for the rest) and report the fraction of decode slot-steps spent on already-finished rows. Predict it from lesson 2's arithmetic first.
6. **Turn admission control off** (`QUEUE_CAP=10**9`) and re-run 40 QPS. Report the goodput and the p99 TTFT of the *served* requests. This is your congestive-collapse exhibit.
7. **Honor cancellation:** add `await request.is_disconnected()` checks, then have the client abort 30% of requests halfway. Report wasted tokens before and after.
8. **Add real metrics:** a `/metrics` endpoint (queue depth, running count, mean batch size, tokens/sec, preemptions) and verify **Little's Law** — log in-flight `L` and λ, check `W ≈ L/λ` against measured e2e. When those agree, your instrumentation is trustworthy ([lesson 5](05-queueing-theory.md)).

---

## Key takeaways

- **One server, two endpoints, one model path** is the right experimental design: it makes batching the only variable, so the comparison can't be explained away.
- Measured on a laptop: **7.6× output throughput and 9.9× goodput** from dynamic batching alone — with **~1.8× worse per-user TPOT**. Both halves of that sentence belong in your writeup.
- **Never block the event loop**: run the forward pass with `asyncio.to_thread` and hand tokens back with `loop.call_soon_threadsafe`. PyTorch drops the GIL during compute, which is why this works at all.
- **Left-pad, and fix `position_ids` from the attention mask.** Verify with the batch-invariance test: greedy output alone must equal greedy output inside a mixed-length batch.
- **Bound the queue, shed with 429, and count the shedding.** Unbounded queues convert overload into congestive collapse: at 40 QPS with no cap, throughput stays high and goodput goes to zero.
- **The knee is a cliff.** p99 TTFT was flat from 3 → 20 QPS, then tripled by 30 QPS. Throughput peaked *after* goodput had already collapsed 175×.
- **The 10 ms window barely mattered**: real batch size (8.1 mean) came from requests arriving during the previous batch's ~350 ms run (`B = λa/(1−λb)`), not from the timer, which predicts 1.2.
- **The request is still the scheduling unit**, so a batch runs for `max(max_tokens)` steps. Uniform outputs hide that; ragged outputs — the realistic case — break it, which is why the next lesson exists.

**Next:** [Build: a tiny continuous-batching engine →](09-build-continuous-batching-engine.md) — make the *iteration* the scheduling unit, give every sequence its own KV-cache lifetime, and delete the head-of-line blocking that's still in this server.
