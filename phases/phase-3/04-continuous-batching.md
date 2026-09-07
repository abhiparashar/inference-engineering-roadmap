# 4 — Continuous Batching: Scheduling at the Iteration, Not the Request

> **You'll be able to say:** "Continuous batching moves the scheduling decision from once-per-request to once-per-decode-step. Every step the engine admits newly queued sequences into freed slots and evicts finished ones, so no sequence waits for an unrelated one to end. It's not a tuning knob — it improves throughput *and* p50 latency at once, and the binding constraint becomes KV-cache memory, not compute."

This is the single most important idea in modern LLM serving, and the reason vLLM, TGI, TensorRT-LLM and SGLang all exist in their current form. It came out of the **Orca** paper (OSDI '22), which reported up to ~37× the throughput of FasterTransformer at the same latency level; the Anyscale write-up that popularized it measured up to 23× over naive HF `generate()` batching, *while also lowering p50 latency*. Every other technique in this track trades something. This one mostly doesn't.

The mechanism is a five-line change to the loop. Understanding it well enough to write one — including what breaks — is the rest of the lesson.

---

## The idea in one diagram

```
STATIC / DYNAMIC BATCHING — the batch is the scheduling unit
step:  1    5   10   15   20   25   30   35   40
  A   ████████                                        done at 8, SLOT HELD IDLE ──────┐
  B   ██████████████████                              done at 18, SLOT HELD IDLE ─────┤
  C   ████████████████████████████████████████        done at 40                      │
  D   ████████████                                    done at 12, SLOT HELD IDLE ─────┤
  E   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ queued ─────────────── starts at 41 ─────┘
       └─ 4 rows computed for 40 steps = 160 slot-steps; 78 useful (49%) ─┘

CONTINUOUS BATCHING — one decode step is the scheduling unit
step:  1    5   10   15   20   25   30   35   40
  A   ████████ done at 8
  B   ██████████████████ done at 18
  C   ████████████████████████████████████████ done at 40
  D   ████████████ done at 12
  E           ████████████████ admitted at step 9, the instant A's slot freed
  F               ██████████ admitted at 13 (D's slot)
  G                    ████████████████████ admitted at 19 (B's slot)
       └─ same 160 slot-steps of budget; ~140 useful (88%), 7 requests served, not 4 ─┘
```

Nothing about the *math* changed. The GPU runs the same kind of batched decode step. What changed is that the slot is re-allocated **every iteration**, so freed capacity is instantly reused and a newly arrived request starts within one step (~10-30 ms) instead of waiting out the longest generation in a closed batch (seconds).

That's why the name "iteration-level scheduling" (Orca's term) is better than "continuous batching": the batch composition is a per-step decision. NVIDIA calls the same thing **in-flight batching**.

---

## The loop

```python
running = []                                  # sequences currently decoding
waiting = deque()                             # admitted-but-not-started requests

while True:
    # 1. ADMIT — fill free slots from the queue, subject to memory + token budget
    while waiting and can_allocate(waiting[0]):
        seq = waiting.popleft()
        seq.kv = kv_pool.allocate(seq.prompt_len)
        prefill(seq)                          # one forward over the whole prompt
        running.append(seq)                   # ← first token emitted here (TTFT ends)

    # 2. STEP — one decode step for every running sequence, as one batched forward
    logits = model_step(running)              # ragged batch: different lengths per row
    for seq, logit in zip(running, logits):
        seq.append(sample(logit, seq.params))
        seq.kv_len += 1

    # 3. EVICT — release finished sequences immediately
    for seq in [s for s in running if s.is_finished()]:
        kv_pool.free(seq.kv)                  # ← memory returns to the pool NOW
        seq.stream.close()
        running.remove(seq)

    # 4. PREEMPT — if memory ran out for the sequences that remain, make room
    while kv_pool.exhausted():
        victim = running.pop()                # newest first; recompute or swap its KV
        preempt(victim)
```

Four steps, and each one maps to a real production concern:

| Step | Frees you from | The hard part |
|---|---|---|
| **Admit** | head-of-line blocking | deciding *how many* and *when* — prefill steals time from decode |
| **Step** | nothing (this is the work) | the batch is **ragged**: every row has a different KV length |
| **Evict** | output-length padding waste | must be immediate, or you've reinvented static batching |
| **Preempt** | OOM crashes under load | choosing a victim and whether to recompute or swap its KV |

---

## What has to be true for this to work

Continuous batching is not a scheduler bolt-on; it needs three things from the model layer, and this is why you couldn't get it from `model.generate()`.

**1. Per-sequence KV-cache with independent lifetimes.** Each sequence owns its own cache, allocated and freed on its own schedule ([Phase 1 lesson 5](../phase-1/05-kv-cache.md)). A single contiguous `[B, heads, max_len, dim]` tensor — the naive layout — forces one allocation for the whole batch and can't release row 3 while rows 1, 2, 4 continue. Fragmentation from variable, unknown-in-advance lengths is what **PagedAttention** solves in [Phase 4](../../ROADMAP.md#phase-4--inference-optimization-techniques); a block/page pool is the natural allocator for this loop.

**2. Ragged (variable-length) batching support.** Rows in one step have KV lengths 47, 1,203, 12, 891… The dense parts of the transformer don't care — a linear layer sees `[total_tokens, hidden]` and multiplies, since each token is independent. **Attention does care**: each row must attend over exactly its own history. So either you pad (waste, back to lesson 2) or you use a variable-length kernel that takes a *cumulative-lengths* array — `flash_attn_varlen_func`, PagedAttention's block tables, xformers' `BlockDiagonalMask`. Orca's name for "batch the linear layers, run attention per sequence" is **selective batching**, and it's the second contribution of the paper.

**3. A scheduler with a memory model.** Compute is elastic — a bigger batch is a slower step. Memory is a cliff: allocate one block too many and the process dies. So the scheduler must know, before it admits anything, how many KV blocks are free and how many the candidate needs.

---

## Memory, not compute, is the real constraint

Do the arithmetic for a 7B FP16 model on an 80 GB A100/H100:

```
  KV bytes per token = 2 (K,V) × layers × kv_heads × head_dim × dtype_bytes
  Llama-2-7B (MHA):    2 × 32 × 32 × 128 × 2  =  524,288 B  ≈ 0.5 MB/token
  Llama-3-8B (GQA, 8 kv heads): 2 × 32 × 8 × 128 × 2 = 131,072 B ≈ 0.125 MB/token

  budget = 80 GB − 14 GB weights − ~4 GB (activations, CUDA ctx, fragmentation) ≈ 62 GB
  slots  = 62 GB ÷ 0.5 MB/token  ≈ 124,000 tokens of KV in total
         → at 2k tokens/sequence: ~62 concurrent sequences
         → at 8k tokens/sequence: ~15 concurrent sequences
```

Sit with that. [Phase 2](../phase-2/05-roofline-model.md) said decode becomes compute-bound at B ≈ 150-300. Memory lets you run **62** — and only 15 if contexts are long. **You will almost never reach the compute roof; you run out of KV-cache first.** Three consequences that define the rest of the track:

- The scheduler's job is really **KV-cache admission control**, not compute scheduling.
- Anything that shrinks KV bytes directly buys concurrency and throughput: **GQA/MQA** (4× here), **KV-cache quantization** (2×), paged allocation (removes fragmentation waste), prefix/prompt sharing across requests. That's Phase 4's shopping list, and now you know why each item is on it.
- **Long context is a throughput problem, not just a memory problem.** 8k contexts cut concurrency ~4×, which cuts batch size, which pushes decode back down the roofline. This is why per-token pricing for long-context requests is higher.

### When memory runs out: preemption

The queue is admitted optimistically, and sequences grow one token per step, so the engine *will* hit the wall mid-flight. Two recovery strategies, both in vLLM:

| Strategy | What happens | Cost | When |
|---|---|---|---|
| **Recompute** | drop the victim's KV, put it back in `waiting`; re-prefill from scratch later | wasted prefill FLOPs, `O(prompt)` | short prompts; default in vLLM |
| **Swap** | copy KV to host RAM over PCIe, restore later | PCIe bandwidth (~25 GB/s), 0.5 MB/token adds up fast | long prompts where re-prefill costs more than the copy |

Both are strictly better than the alternatives (crashing, or refusing to admit anything you can't guarantee to completion — which would leave the GPU half-idle). And both are why you want a **preemption counter in your metrics**: rising preemptions mean you're over-admitting, and each one is wasted work that shows up as an ITL spike for the victim.

---

## The prefill/decode collision (and the fix)

Here's the problem that continuous batching *creates*, and the question that separates people who've read a blog post from people who've operated an engine.

Prefill is compute-bound and its cost scales with prompt length; decode is memory-bound and cheap per step. Put them in the same loop and:

```
  60 sequences decoding happily, step time ≈ 25 ms  → everyone's ITL is 25 ms
  a request with a 4,000-token prompt is admitted
  → that step must also do 4,000 tokens of prefill ≈ 250-400 ms of GEMM
  → ALL 60 decoding users see one 300+ ms gap in their token stream
```

One admission, sixty stutters. This is **prefill interference** (a.k.a. generation stall), and it is the dominant source of bad p99 ITL in production LLM serving. It's also a genuine dilemma: refuse the prefill and that user's TTFT suffers instead.

Three fixes, in increasing order of ambition:

1. **Chunked prefill** (vLLM `enable_chunked_prefill`, also called split-fuse / piggybacking, from Sarathi-Serve): break the 4,000-token prefill into chunks and mix each chunk with the decode step under a single **token budget** (`max_num_batched_tokens`, e.g. 2,048 tokens per step counting both prefill and decode tokens). Prefill takes a few more steps; ITL stays bounded. This is the standard answer, and it's why the budget is expressed in *tokens*, not requests.
2. **Prioritize decode over prefill** (TGI's `waiting_served_ratio`, `max_waiting_tokens`): admit new work only when the queue justifies the interruption. Simple, effective, and it makes the tradeoff a config value.
3. **Disaggregation**: run prefill and decode on *different GPUs* — separate pools, KV transferred over the network. Each pool then batches a homogeneous workload with its own scaling. This is the frontier design (DistServe, Splitwise, vLLM/Dynamo P/D disaggregation) and it belongs to [Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale).

Notice the pattern: once scheduling is per-iteration, the *composition* of each iteration becomes a design space. Chunked prefill, priority policies, and disaggregation are all answers to "what should this step contain?" — which is [lesson 6](README.md)'s subject.

---

## Read the real code

Do this with the files open; it's the highest-value hour in the phase.

### vLLM — `vllm/core/scheduler.py` (V0) / `vllm/v1/core/sched/scheduler.py` (V1)

The V0 scheduler is the canonical readable implementation; the V1 rewrite keeps the same concepts with a flatter, faster loop. Paths drift between releases — grep for `class Scheduler` and `def schedule`. What to look for:

- **Three queues:** `waiting` (never started), `running` (decoding), `swapped` (preempted to host). Your loop above has two; the third is preemption.
- **`SchedulingBudget`** — the token budget (`max_num_batched_tokens`) *and* sequence budget (`max_num_seqs`). Every candidate must fit both. This is the single most important config pair in vLLM.
- **`_schedule_running` / `_schedule_prefills` / `_schedule_swapped`** — the priority order between continuing, starting, and resuming work. Read them in that order and you've read the policy.
- **`_preempt`** with `PreemptionMode.RECOMPUTE` vs `SWAP`, and the log line `Sequence group ... is preempted by ...` you'll eventually see in production.
- **`can_allocate` / `can_append_slots`** on the block manager — the memory model that gates everything (`AllocStatus.OK / LATER / NEVER`).

**Lab exercise (from [`labs/README.md`](../../labs/README.md)):** trace one request from `add_request` through `schedule()` and answer precisely — *what happens to a request that doesn't fit the current step's token budget?* Then find where finished sequences release their blocks.

### TGI — `router/src/queue.rs` and `router/src/infer.rs`

Rust, but the flow is readable without knowing Rust. The router owns a queue of `Entry` objects and a `batching_task` that repeatedly calls `prefill`/`decode` on the shards, then **filters** finished sequences out of the batch and **concatenates** newly admitted ones. The knobs — `max_batch_total_tokens`, `waiting_served_ratio`, `max_waiting_tokens` — are policy for exactly the prefill/decode collision above.

Reading both is the point: **same algorithm, two languages, two policy dialects.** That's how you learn which parts are essential (iteration-level scheduling, token budget, memory gate) and which are taste.

---

## Try it (laptop, no GPU)

Extend the fake engine so the difference is undeniable. Cost model: a step costs `a + b·B` and prefill costs `c·prompt_tokens`.

```python
import random, statistics

A, B_COST, C_PREFILL = 0.010, 0.0005, 0.00005      # 10 ms, 0.5 ms/seq, 50 µs/prompt token
MAX_SEQS = 32

def workload(n=400, qps=8.0):
    t, reqs = 0.0, []
    for i in range(n):
        t += random.expovariate(qps)
        reqs.append(dict(id=i, arrive=t, prompt=random.choice([64, 256, 1024]),
                         out=random.choice([16, 32, 64, 256, 512])))
    return reqs

def simulate(reqs, continuous):
    now, q, running, done = 0.0, list(reqs), [], []
    while q or running:
        if not running:                                        # idle: jump to next arrival
            now = max(now, q[0]["arrive"])
        # admit: continuous = every step; static = only when the batch has fully drained
        if continuous or not running:
            while q and q[0]["arrive"] <= now and len(running) < MAX_SEQS:
                r = q.pop(0)
                now += C_PREFILL * r["prompt"]                 # prefill blocks the loop
                r["ttft"], r["produced"] = now - r["arrive"], 1
                running.append(r)
        # one decode step for everyone
        now += A + B_COST * len(running)
        for r in running:
            r["produced"] += 1
        finished = [r for r in running if r["produced"] >= r["out"]]
        for r in finished:
            r["e2e"] = now - r["arrive"]
            done.append(r); running.remove(r)
    assert not q and not running
    p = lambda k, key: statistics.quantiles([r[key] for r in done], n=100)[k - 1]
    span = max(r["e2e"] + r["arrive"] for r in done)
    print(f"{'continuous' if continuous else 'static    '}  "
          f"ttft p50={p(50,'ttft'):6.2f}s p99={p(99,'ttft'):7.2f}s   "
          f"e2e p99={p(99,'e2e'):7.2f}s   throughput={sum(r['out'] for r in done)/span:7.1f} tok/s")

random.seed(0)
w = workload()
simulate([dict(r) for r in w], continuous=False)
simulate([dict(r) for r in w], continuous=True)
```

**Predict first:** which metric improves *most*? (Answer: p99 TTFT, by a lot — head-of-line blocking is what you deleted. Throughput improves too, from the recovered slot-steps.) Then change one thing at a time and watch: raise `MAX_SEQS` (better throughput, worse ITL — [lesson 5](05-queueing-theory.md) explains the shape), make prompts longer (watch the prefill term poison everyone's step time — that's prefill interference, and chunking `C_PREFILL` work across steps is your fix), skew `out` more heavily (static gets worse, continuous barely notices).

One run (seed 0, 400 requests at 8 QPS, outputs 16-512 tokens) gives:

```
static      ttft p50= 26.83s p99=  59.55s   e2e p99=  62.44s   throughput=  570.0 tok/s
continuous  ttft p50=  2.69s p99=  12.89s   e2e p99=  22.76s   throughput=  938.5 tok/s
```

p99 TTFT fell **4.6×** and throughput rose **1.65×** — from the *same* cost model and the *same* arrival trace. Nothing was optimized; only the scheduling granularity changed.

This simulator, made real with an actual model and a real KV pool, is **Project 02** — the large project for this phase.

---

## Key takeaways

- **Continuous batching = iteration-level scheduling**: the batch is recomposed every decode step, so freed slots are reused immediately and new requests start within one step.
- Reported gains: ~23× over naive HF batching (Anyscale), up to ~37× over FasterTransformer at equal latency (Orca) — and **p50 latency improves too**, because the waste is eliminated rather than redistributed.
- Four loop stages: **admit → step → evict → preempt.** Eviction must be immediate; admission must be memory-gated.
- It requires **per-sequence KV lifetimes**, **ragged/varlen attention kernels** (Orca's *selective batching*: batch the linear layers, per-sequence attention), and a scheduler that models memory.
- **KV-cache memory is the binding constraint, not compute**: ~0.5 MB/token for 7B MHA → ~62 concurrent 2k-token sequences on 80 GB, well under the compute-bound B ≈ 150-300. Hence GQA, KV quantization, PagedAttention.
- Over-admission is handled by **preemption** (recompute vs swap). Track a preemption counter; it's your over-admission alarm.
- **Prefill interference** is the new failure mode: one long prompt stalls every decoding user's token stream. Fixes: **chunked prefill under a token budget**, decode-priority policies, and prefill/decode **disaggregation**.
- Read `vllm/core/scheduler.py` (budget, three queues, preemption) and TGI's `router/src/queue.rs` (filter/concatenate). Same algorithm, different policy dialects.

**Next:** [Queueing theory for inference →](05-queueing-theory.md) — why the batch size that maximizes throughput is *never* the one that protects p99, in closed form.
