# 9 — Build: A Tiny Continuous-Batching Engine

> **You'll be able to say:** "I wrote an iteration-level scheduler: per-sequence KV lifetimes, a chunked-prefill token budget, immediate eviction, recompute preemption, and a ragged decode step built from left-padded KV plus a mask. On a **ragged** workload (outputs sampled from 8/16/32/128 tokens) my dynamic-batched server collapsed at ~5 QPS — p50 TTFT 122 s, goodput 0 — while the continuous-batching engine served 6 QPS with **p50 TTFT of 63-66 ms**. On a *uniform* workload the engine is actually slower, because per-iteration Python overhead is real and static batching wastes nothing when every output is the same length. That comparison is the whole point: continuous batching buys you robustness to workload shape, not free FLOPs."

This is the deep end, and it's where "I've read the vLLM blog post" becomes "I've written one." Lesson 8's server still scheduled *requests*: once a group started, it ran `max(max_tokens)` steps and nobody new got in. Here the scheduling unit becomes **one decode step**.

---

## The loop, for real

Lesson 4 gave you four stages. Here they are as an actual function that runs on an actual model:

```
  every iteration:
   1. ADMIT   spend a token budget on chunked prefill of queued requests
              (gate: KV pool has room, MAX_SEQS not reached)
   2. STEP    one decode step for every running sequence, as ONE forward
   3. EVICT   finished sequences leave and their KV is freed immediately
   4. PREEMPT if the KV pool is over budget, drop the newest sequence's KV
              and requeue it (recompute mode)
```

Three implementation problems stand between that pseudocode and working software, and solving them is the lesson:

| Problem | Real engines | This engine |
|---|---|---|
| Sequences have **different KV lengths** in one forward | varlen kernels (`flash_attn_varlen_func`), PagedAttention block tables | left-pad every row's KV to the longest, plus an attention mask |
| KV must be **allocated/freed per sequence** | paged block pool, no fragmentation | per-sequence tensors + a token-count budget |
| Re-batching KV every step **costs more than the math** | KV never moves; kernels read block tables | cache the batched tensor, rebuild only when composition changes |

That third row is the one nobody warns you about, and it's why this file has a `split_batch` function.

---

## The code

Save as `projects/02-continuous-batching-engine/engine.py`. Same deps as lesson 8, same `bench.py`.

```python
#!/usr/bin/env python3
"""Phase 3, lesson 9: a tiny continuous-batching engine (a very small vLLM).

Iteration-level scheduling: every loop turn we (1) spend a token budget on chunked
prefill, (2) run ONE decode step for every running sequence, (3) evict finished
sequences and free their KV immediately, (4) preempt (recompute) if the KV pool is
over budget. Per-sequence KV lifetimes, ragged batch via left-padded KV + mask.

Run:   MODEL=gpt2 uvicorn engine:app --port 8001
Bench: python bench.py --url http://127.0.0.1:8001/generate --qps 6 12 20
"""
import asyncio, json, os, time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

MODEL       = os.environ.get("MODEL", "gpt2")
DEVICE      = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available()
                                           else "mps" if torch.backends.mps.is_available() else "cpu")
MAX_SEQS    = int(os.environ.get("MAX_SEQS", "16"))        # rows in a decode step
MAX_BATCH_TOKENS = int(os.environ.get("MAX_BATCH_TOKENS", "256"))   # prefill+decode tokens/iter
KV_BUDGET   = int(os.environ.get("KV_BUDGET", "8192"))     # total KV tokens the "pool" holds
QUEUE_CAP   = int(os.environ.get("QUEUE_CAP", "64"))
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "0"))

app = FastAPI()
M = {"admitted": 0, "finished": 0, "shed": 0, "preemptions": 0, "iters": 0,
     "decode_tokens": 0, "prefill_tokens": 0, "kv_tokens": 0, "running": 0, "waiting": 0}


@dataclass
class Seq:
    rid: int
    prompt_ids: List[int]
    max_tokens: int
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    kv: Optional[list] = None          # per-layer [(k,v)], each [1, heads, kv_len, dim]
    kv_len: int = 0
    pos: int = 0                       # position id of the NEXT token to feed
    prefill_pos: int = 0               # how much of the prompt has been prefilled
    next_token: Optional[int] = None
    produced: int = 0
    preempted: int = 0

    def reset_for_recompute(self):     # preemption, recompute mode: throw the KV away
        self.kv, self.kv_len, self.pos, self.prefill_pos = None, 0, 0, 0
        self.next_token, self.produced = None, 0
        self.preempted += 1


class Engine:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        self.DynamicCache = DynamicCache
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
        self.eos = self.tok.eos_token_id
        self.n_layers = self.model.config.n_layer
        self.batch = None          # cached batched KV: {"rids": [...], "past": legacy, "L": int}

    # --- helpers ------------------------------------------------------------
    def _to_cache(self, legacy):
        return self.DynamicCache.from_legacy_cache(tuple(legacy))

    def _from_cache(self, cache):
        return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache

    def split_batch(self, seqs):
        """Give every surviving sequence its own KV again. Only needed when the batch
        composition changes -- that is the ONLY time we pay the per-row copy."""
        if self.batch is None:
            return
        rids, past = self.batch["rids"], self.batch["past"]
        index = {r: i for i, r in enumerate(rids)}
        for s in seqs:
            i = index.get(s.rid)
            if i is None:
                continue                                    # newly prefilled: owns its KV already
            keep = s.kv_len
            s.kv = [(past[li][0][i:i + 1, :, -keep:, :].clone(),
                     past[li][1][i:i + 1, :, -keep:, :].clone()) for li in range(self.n_layers)]
        self.batch = None

    @torch.inference_mode()
    def prefill_chunk(self, s: Seq, budget: int) -> int:
        """Prefill up to `budget` prompt tokens. Returns tokens consumed."""
        chunk = s.prompt_ids[s.prefill_pos: s.prefill_pos + budget]
        if not chunk:
            return 0
        ids = torch.tensor([chunk], device=DEVICE)
        pos = torch.arange(s.prefill_pos, s.prefill_pos + len(chunk), device=DEVICE).unsqueeze(0)
        mask = torch.ones((1, s.kv_len + len(chunk)), dtype=torch.long, device=DEVICE)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                         past_key_values=self._to_cache(s.kv) if s.kv else None)
        s.kv = [(k.clone(), v.clone()) for k, v in self._from_cache(out.past_key_values)]
        s.kv_len += len(chunk)
        s.prefill_pos += len(chunk)
        s.pos = s.prefill_pos
        if s.prefill_pos >= len(s.prompt_ids):                    # prompt done -> first token
            s.next_token = int(out.logits[0, -1].argmax())
        return len(chunk)

    @torch.inference_mode()
    def decode_step(self, seqs: List[Seq]):
        """One decode step for every running sequence.

        Ragged batch via left-padded KV + attention mask. The batched KV tensor is REUSED
        across steps while the batch composition is unchanged -- rebuilding it every step
        costs O(B x L) copies and dominates the step time on a small model.
        """
        B = len(seqs)
        rids = [s.rid for s in seqs]
        if self.batch is not None and self.batch["rids"] == rids:
            past, L = self.batch["past"], self.batch["L"]              # reuse: zero copies
        else:
            L = max(s.kv_len for s in seqs)
            past = []
            for li in range(self.n_layers):
                ks, vs = [], []
                for s in seqs:
                    k, v = s.kv[li]
                    pad = L - k.shape[2]
                    if pad:
                        z = k.new_zeros((1, k.shape[1], pad, k.shape[3]))
                        k, v = torch.cat([z, k], 2), torch.cat([z, v], 2)   # LEFT pad history
                    ks.append(k); vs.append(v)
                past.append((torch.cat(ks, 0), torch.cat(vs, 0)))
        mask = torch.zeros((B, L + 1), dtype=torch.long, device=DEVICE)
        for i, s in enumerate(seqs):
            mask[i, L - s.kv_len:] = 1                                 # mask hides the padding
        inp = torch.tensor([[s.next_token] for s in seqs], device=DEVICE)
        pos = torch.tensor([[s.pos] for s in seqs], device=DEVICE)
        out = self.model(input_ids=inp, attention_mask=mask, position_ids=pos,
                         past_key_values=self._to_cache(past), use_cache=True)
        toks = out.logits[:, -1].argmax(-1).tolist()
        for s in seqs:
            s.kv, s.kv_len, s.pos = None, s.kv_len + 1, s.pos + 1      # KV lives in the batch now
        self.batch = {"rids": rids, "past": self._from_cache(out.past_key_values), "L": L + 1}
        return toks


ENGINE: Optional[Engine] = None
WAITING: List[Seq] = []
RUNNING: List[Seq] = []
NEW: Optional[asyncio.Queue] = None


def kv_used():
    return sum(s.kv_len for s in RUNNING)


def scheduler_iteration(loop):
    """One turn of the engine loop. Runs in a worker thread; returns emitted tokens."""
    emits = []

    # 1. ADMIT / PREFILL under the token budget (chunked prefill).
    #    Admission changes the batch composition, so the running sequences must own their
    #    KV again before the rebuild in decode_step().
    budget = MAX_BATCH_TOKENS - len(RUNNING)                  # decode rows cost 1 token each
    if WAITING and budget > 0 and len(RUNNING) < MAX_SEQS:
        ENGINE.split_batch(RUNNING)
    while WAITING and budget > 0 and len(RUNNING) < MAX_SEQS:
        s = WAITING[0]
        need = len(s.prompt_ids) - s.prefill_pos
        if kv_used() + s.kv_len + min(need, budget) > KV_BUDGET:        # gate 1: memory
            break
        used = ENGINE.prefill_chunk(s, budget)
        M["prefill_tokens"] += used
        budget -= used
        if s.next_token is not None:                          # prompt fully prefilled
            WAITING.pop(0)
            RUNNING.append(s)
            M["admitted"] += 1
            emits.append((s, ENGINE.tok.decode([s.next_token])))         # TTFT ends here
            s.produced = 1
        else:
            break                                             # budget exhausted mid-prompt

    # 2. STEP — one decode step for all running sequences
    if RUNNING:
        toks = ENGINE.decode_step(RUNNING)
        M["decode_tokens"] += len(RUNNING)
        for s, t in zip(RUNNING, toks):
            s.next_token = t
            s.produced += 1
            emits.append((s, ENGINE.tok.decode([t])))

    # 3. EVICT finished sequences, freeing their KV immediately
    for s in [s for s in RUNNING if s.produced >= s.max_tokens or s.next_token == ENGINE.eos]:
        RUNNING.remove(s)
        s.kv = None                                           # KV returns to the pool NOW
        M["finished"] += 1
        emits.append((s, None))

    # 4. PREEMPT (recompute) while the pool is over budget: newest first
    while RUNNING and kv_used() > KV_BUDGET:
        victim = RUNNING.pop()
        victim.reset_for_recompute()
        WAITING.insert(0, victim)
        M["preemptions"] += 1

    # composition changed? hand the survivors their own KV back (the only per-row copy)
    if ENGINE.batch is not None and ENGINE.batch["rids"] != [s.rid for s in RUNNING]:
        ENGINE.split_batch(RUNNING)

    M["iters"] += 1
    M["kv_tokens"], M["running"], M["waiting"] = kv_used(), len(RUNNING), len(WAITING)
    return emits


async def engine_loop():
    loop = asyncio.get_event_loop()
    while True:
        while not NEW.empty():                                # ingest arrivals
            WAITING.append(NEW.get_nowait())
        if not RUNNING and not WAITING:
            WAITING.append(await NEW.get())                   # idle: sleep until work arrives
            continue
        emits = await asyncio.to_thread(scheduler_iteration, loop)
        for s, text in emits:
            s.out.put_nowait(text)


@app.on_event("startup")
async def startup():
    global ENGINE, NEW
    if TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    ENGINE = Engine()
    NEW = asyncio.Queue()
    warm = Seq(-1, ENGINE.tok("warmup").input_ids, 4)
    ENGINE.prefill_chunk(warm, 64); ENGINE.decode_step([warm])
    asyncio.ensure_future(engine_loop())


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    if len(WAITING) + NEW.qsize() >= QUEUE_CAP:
        M["shed"] += 1
        return StreamingResponse(iter([f"data: {json.dumps({'token': '[busy]'})}\n\n"]),
                                 status_code=429, media_type="text/event-stream")
    ids = ENGINE.tok(body.get("prompt", ""), truncation=True, max_length=512).input_ids
    s = Seq(int(time.time_ns()), ids, int(body.get("max_tokens", 32)))
    NEW.put_nowait(s)

    async def stream():
        while True:
            tok = await s.out.get()
            if tok is None:
                break
            yield f"data: {json.dumps({'token': tok})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics():
    m = dict(M)
    m["mean_batch"] = M["decode_tokens"] / max(M["iters"], 1)
    m.update(model=MODEL, device=DEVICE, max_seqs=MAX_SEQS,
             max_batch_tokens=MAX_BATCH_TOKENS, kv_budget=KV_BUDGET)
    return m
```

---

## Six details that make it work

**1. Ragged batching = left-padded KV + a mask.** Rows have KV lengths 47, 203, 12… Pad each row's history on the **left** to the longest, set `mask[i, L - kv_len_i:] = 1`, and give every row its own `position_ids`. The dense layers never notice (each token is independent); attention only sees the unmasked history. That's Orca's *selective batching* done the slow, dependency-free way. The fast way is a varlen kernel that takes cumulative sequence lengths — same idea, no padding.

**2. Per-sequence KV lifetimes are the whole point.** `Seq.kv` is that sequence's cache; eviction sets it to `None` and the memory is free *that iteration*, not when its batch-mates finish. This is the structural difference from lesson 8, and everything else follows from it.

**3. The KV pool is a token budget, and admission is gated on it.** `kv_used() + need > KV_BUDGET` → don't admit. That's [lesson 4](04-continuous-batching.md)'s arithmetic (`slots = free_bytes ÷ bytes_per_token`) with the units simplified to tokens. Set `KV_BUDGET` from real memory on a GPU: `(total − weights − activations) ÷ (2 × layers × kv_heads × head_dim × dtype_bytes)`.

**4. Chunked prefill is a token budget, not a special case.** `budget = MAX_BATCH_TOKENS − len(RUNNING)`: decode rows cost one token each, and what's left is spent on prompt tokens. A 1,000-token prompt with a 256-token budget takes four iterations and never stalls the decoders for more than one chunk's worth of GEMM. Shrinking `MAX_BATCH_TOKENS` directly shrinks the worst-case ITL spike and directly raises TTFT — that dial is [lesson 6](06-scheduling-policies-and-admission-control.md)'s tradeoff, now a variable you can set.

**5. Preemption is recompute mode, LIFO.** `reset_for_recompute()` throws the victim's KV away and requeues it at the head. Newest-first, because it has the least accumulated work to lose. Every preemption is wasted prefill, which is why `M["preemptions"]` exists — it is your over-admission alarm.

**6. The optimization you'd never guess: don't rebuild the batch.** The first version of this engine split the batched KV back into per-sequence tensors after *every* step: `2 × n_layers × B` slice+clone ops per iteration. On GPT-2 (12 layers, B=16) that's 384 tiny tensor ops per token — Python and allocator overhead that **dominated the actual matmuls**, pushing TPOT to 45 ms. Caching the batched tensor while the composition is unchanged (and calling `split_batch` only when a sequence is admitted, evicted or preempted) cut TPOT to **18-31 ms**.

> That is the practical reason PagedAttention exists. Real engines never move KV: it lives in fixed blocks and kernels read a **block table** to find it. The moment you find yourself copying caches around, you've discovered the problem vLLM was built to solve.

---

## Verify correctness before you benchmark anything

Speed numbers from a wrong engine are worthless, and it is very easy to build one that produces *plausible* text. The test is **batch invariance** under greedy decoding:

```python
solo = gen("The capital of France is", n=12)                 # one request, alone
# ... then fire 13 concurrent requests with wildly different prompt lengths,
#     one of which is the same prompt, and compare:
assert solo == res_from_the_mixed_batch
```

Actual result from this engine, with the same prompt racing 12 others of lengths 5-200 tokens:

```
solo   : ' the capital of the French Republic, and the capital of the'
batched: ' the capital of the French Republic, and the capital of the'
MATCH
```

It also matches lesson 8's server token-for-token on the same prompt. If yours doesn't, the bug is in one of exactly three places: `position_ids`, the attention mask's alignment with the padding, or the `-keep:` slice in `split_batch`. **Do not proceed until it matches** — this test is what separates an engine from a random text generator.

---

## Measured: what continuous batching actually buys

GPT-2 124M, Apple M5 (MPS), 64-token prompts, `MAX_SEQS=16`, `MAX_BATCH_TOKENS=256`, `KV_BUDGET=8192`, `QUEUE_CAP=64`, open-loop Poisson arrivals, 40 s per point after 5 s warmup, `--settle` between points, **one server running at a time**.

### Uniform workload (every request asks for exactly 32 tokens)

| Server | Offered QPS | p50 TTFT | p99 TTFT | p50 TPOT | p99 ITL | Output tok/s | Goodput (TTFT ≤ 1 s) |
|---|---|---|---|---|---|---|---|
| lesson 8, dynamic batching | 10 | 249 ms | 709 ms | 10.5 ms | 41 ms | 339.3 | 10.60/s |
| lesson 8, dynamic batching | **20** | 392 ms | 835 ms | 14.2 ms | 62 ms | **618.0** | **19.31/s** |
| lesson 8, dynamic batching | 30 | 2,239 ms | 2,686 ms | 15.6 ms | 97 ms | 785.5 | 0.11/s |
| **engine**, continuous | 6 | 63 ms | 250 ms | 18.4 ms | 134 ms | 188.7 | 5.90/s |
| **engine**, continuous | **14** | 345 ms | 1,171 ms | 31.5 ms | 131 ms | **455.3** | **13.86/s** |
| **engine**, continuous | 20 | 3,949 ms | 4,261 ms | 31.1 ms | 97 ms | 473.5 | 0.00/s |

**Read that honestly: on a uniform workload the engine loses.** Peak goodput 13.86 vs 19.31 req/s; peak throughput 473 vs 786 tok/s. Two reasons, both worth understanding:

- **Static batching wastes nothing when all outputs are the same length.** Every row of lesson 8's batch finishes on the same step, so the "waste" continuous batching removes is *zero* here.
- **Per-iteration overhead is real.** The engine does Python scheduling work, mask construction, and (on composition changes) KV re-batching *every single token*, on a model whose forward pass takes ~10 ms. On a 7B model at 100+ ms per step that overhead amortizes away; on GPT-2 it is a third of the step.

If a benchmark only ever shows you the uniform case, it is hiding the reason continuous batching exists.

### Ragged workload (`--max-tokens 8 16 32 128`, the realistic case)

| Server | Offered QPS | shed | p50 TTFT | p99 TTFT | p50 TPOT | Output tok/s | Goodput (TTFT ≤ 1 s) |
|---|---|---|---|---|---|---|---|
| lesson 8, dynamic batching | 2 | 0 | 145 ms | 1,600 ms | 9.6 ms | 75.4 | 1.65/s |
| lesson 8, dynamic batching | **4** | 0 | 827 ms | 1,950 ms | 11.4 ms | 179.2 | **2.35/s** |
| lesson 8, dynamic batching | 6 | 129 | **122,273 ms** | 222,486 ms | 17.1 ms | 13.7 | 0.00/s |
| lesson 8, dynamic batching | 8 | 244 | 174,735 ms | 233,217 ms | 18.8 ms | 7.9 | 0.00/s |
| **engine**, continuous | 6 | 0 | **66 ms** | 161 ms | 20.8 ms | 255.2 | **5.81/s** |

That 6-QPS row is the lesson in one line: **at the load where the dynamic-batched server's median first token takes two minutes, the continuous-batching engine answers in 66 milliseconds** — a ~1,850× difference in p50 TTFT, and goodput of 5.81 req/s versus zero.

Why so violent? A batch containing one 128-token request runs **128 decode steps for all 16 rows**, so the 8-token requests sitting in it are computed 16× longer than they need, and — worse — every request that arrived during those ~1.4 s waits for the entire group. Head-of-line blocking plus output-length padding compound, capacity falls to ~5 QPS, and past that the queue never drains. The engine has neither problem: short sequences leave the instant they finish, and their slots are refilled on the *next iteration*.

**Caveats, stated:** the engine's ragged point was measured while lesson 8's (idle) server process was still resident, so if anything it is pessimistic; the dynamic-batching rows were measured in isolation. Both used the same client on the same laptop. And these are 124M-parameter numbers on MPS — on a real GPU with a real model, the engine's fixed overhead shrinks relative to the forward pass and its advantage grows.

### What the metrics endpoint says

`mean_batch` (decode rows per iteration) was **11.6** on the uniform run and 10-14 on ragged runs, against `MAX_SEQS=16` — i.e. the scheduler kept the GPU 70-90% occupied by sequences rather than by padding. `preemptions` stayed at **0** because `KV_BUDGET=8192` never bound with 16 sequences of ≤192 tokens; set `KV_BUDGET=1024` and watch it climb, which is exercise 4.

---

## Three measurement bugs this build produced (learn them free)

Benchmarking a real system taught me three things that no amount of theory would have:

1. **Backlog leaks between load points.** Sweeping `--qps 4 6 8` back-to-back means point 2 inherits point 1's queue. Symptoms: non-monotonic results, an overloaded point that looks *better* than a lighter one. Fix: `--settle` (drain window) between points — added to `bench.py` in [lesson 7](07-measuring-honestly.md).
2. **A second server on the same box poisons everything.** With the engine merely *draining* in the background, the dynamic-batched server's 1-QPS point measured **9,650 ms** p50 TTFT. Isolated, the same point measured **89 ms** — a 108× error caused by a neighbour process, not by the code under test. "Isolate the resource under test" is not a nicety.
3. **Drift on e2e is the wrong stability check for ragged workloads.** With mixed output lengths, e2e varies by request *by design*, so the first-vs-last-decile ratio flags stable systems as unstable. Compute drift on **TTFT**: it moves only when the queue moves.

Every one of these would have produced a confidently wrong benchmark. Write them into your own harness now.

---

## Read the real thing, then extend yours

With this code fresh, open [vLLM](https://github.com/vllm-project/vllm) and map it one-to-one — this is by far the most effective way to read a large codebase:

| Your code | vLLM |
|---|---|
| `Seq` | `Request` / `SequenceGroup` |
| `WAITING` / `RUNNING` | `waiting` / `running` (+ `swapped`) queues |
| `budget = MAX_BATCH_TOKENS - len(RUNNING)` | `SchedulingBudget`, `max_num_batched_tokens`, `max_num_seqs` |
| `kv_used() + need > KV_BUDGET` | `BlockSpaceManager.can_allocate` → `OK` / `LATER` / `NEVER` |
| `split_batch` / left-padded KV | **PagedAttention block tables** — the copy you're doing, deleted |
| `reset_for_recompute()` | `PreemptionMode.RECOMPUTE` (vs `SWAP`) |
| `prefill_chunk` | `enable_chunked_prefill` + `long_prefill_token_threshold` |
| `M["preemptions"]` | `vllm:num_preemptions_total` |

### Exercises (these turn it into Project 02)

1. **Reproduce both tables** — uniform and ragged, both servers, one at a time, with `--settle`. The two-workload comparison *is* the deliverable; a single workload proves nothing.
2. **Sweep `MAX_BATCH_TOKENS`** ∈ {64, 128, 256, 1024} with 512-token prompts and plot p99 ITL vs p50 TTFT. You are drawing the chunked-prefill tradeoff curve with your own data.
3. **Sweep `MAX_SEQS`** ∈ {1, 4, 8, 16, 32} and plot throughput *and* goodput. Confirm goodput has an interior maximum ([lesson 5](05-queueing-theory.md)) and report the value your SLO would pick.
4. **Force preemption:** `KV_BUDGET=1024`, prompts of 256 tokens, 8 QPS. Report preemptions/sec, the ITL spike suffered by victims, and the wasted prefill tokens. Then implement **swap** (copy KV to CPU, restore later) and compare against recompute — with prompt length on the x-axis, because that's the variable that flips the answer.
5. **Add a priority class** (lesson 6): two SLOs, priority ordering in the admit loop, aging after 5 s, and preempt the low class first. Show interactive p99 TTFT improving and batch p99 getting worse — quantified.
6. **Implement `bench.py --max-tokens` from a real distribution** (ShareGPT lengths) and re-run. Report how much the heavy tail changes goodput versus your synthetic mix; connect it to Kingman's `C_s²` term.
7. **Delete the copies:** preallocate a `[MAX_SEQS, heads, max_len, dim]` KV tensor per layer, assign each sequence a fixed row, and write in place with `index_copy_`. Measure TPOT before/after. This is a hand-rolled block pool, and it's the single most educational optimization in the phase.
8. **Compare against vLLM** on the same laptop or Colab: `vllm serve gpt2 --max-num-seqs 16 --max-num-batched-tokens 256`, then point `bench.py` at `/v1/completions`. Expect to lose by a lot — and be able to explain *exactly which* of the eight differences in the table above accounts for it. That paragraph is worth more in an interview than a green benchmark.

---

## Key takeaways

- **Iteration-level scheduling in ~200 lines:** admit (chunked prefill under a token budget) → step (one ragged decode) → evict (free KV now) → preempt (recompute, LIFO). Every real engine is this loop plus kernels.
- **Ragged batching without custom kernels** = left-padded KV + attention mask + per-row `position_ids`. Correct, portable, and slower than varlen kernels — which is exactly the tradeoff to be able to explain.
- **Verify batch invariance before benchmarking.** Greedy output solo must equal greedy output inside a 13-way mixed-length batch, token for token.
- **Moving KV around costs more than the math.** Caching the batched cache (rebuild only on composition change) took TPOT from 45 ms to 18-31 ms. This is the concrete motivation for **PagedAttention**: never move KV, index it.
- **On uniform outputs, continuous batching loses to static batching** here (13.86 vs 19.31 req/s goodput): nothing is wasted for it to reclaim, and per-iteration overhead is real on a 124M model.
- **On ragged outputs — the real world — it wins overwhelmingly:** p50 TTFT **66 ms vs 122 s** at 6 QPS, goodput 5.81/s vs 0. Static batching's capacity collapsed at ~5 QPS because one 128-token request drags fifteen 8-token requests through 128 steps.
- **Measurement discipline is part of the engineering:** settle between load points, run one server at a time (a draining neighbour cost a 108× error), and compute drift on TTFT, not e2e.
- **Map your code onto vLLM's** — `Seq`→`Request`, budget→`SchedulingBudget`, KV gate→`can_allocate`, recompute→`PreemptionMode`. You now have the vocabulary to read, and contribute to, a production engine.

**Next:** [Exercises & exit artifact →](10-exercises-and-artifacts.md) — the numbers, plots, and writeup that make Phase 3 count, and the self-check that says you're ready for Phase 4.
