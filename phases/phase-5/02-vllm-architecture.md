# 2 — vLLM Architecture: Read the Code

> **You'll be able to say:** "A request enters the FastAPI server, is validated and templated in `serving.py`, crosses a ZMQ socket into a separate **engine-core process**, lands in the scheduler's waiting queue, and from then on it is just a row in `num_scheduled_tokens`. Each step, `Scheduler.schedule()` spends a **token budget** — running requests first, then waiting ones — asks `KVCacheManager.allocate_slots()` for blocks, and hands a `SchedulerOutput` to the model runner, which builds flattened input tensors plus block tables and replays a CUDA graph. There is no prefill phase and no decode phase in the scheduler: every request just tries to make `num_computed_tokens` catch up to `num_tokens`."

This is the highest-value code read in the roadmap after vLLM's block manager (which you already met in [Phase 4 lesson 5](../phase-4/05-paged-attention.md)). Do it with your own `paged_kv.py` open in the next window.

> **Version note.** Paths below are vLLM's V1 engine (`vllm/v1/…`) as of the v0.2x line. vLLM refactors aggressively; **file paths rot, structure doesn't.** Every section gives the grep that relocates the code if a path has moved — use it, and treat "find it anyway" as part of the exercise ([lesson 9](09-reading-engine-source.md)).

---

## Set up the read

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
git log -1 --format='%h %ad' --date=short     # write this hash in your notes; cite it later
tokei vllm/ 2>/dev/null | head -5              # or: find vllm -name '*.py' | xargs wc -l | tail -1
```

Then get oriented by size, not by curiosity:

```bash
find vllm/v1 -name '*.py' | xargs wc -l | sort -rn | head -20
```

The biggest files in `vllm/v1/core/sched/` and `vllm/v1/worker/gpu/` are the scheduler and the model runner. That's where the engine lives; everything else is plumbing, model definitions and hardware backends.

---

## Process architecture

vLLM V1 does **not** run the HTTP server and the engine in one Python process. That's the single most important structural fact about it.

```
  ┌─────────────────────────────── PROCESS 1: API server ───────────────────────────────┐
  │  uvicorn/FastAPI      vllm/entrypoints/openai/api_server.py                          │
  │    └─ route handlers  vllm/entrypoints/openai/{chat_completion,completion}/           │
  │         └─ serving    …/serving.py   ← chat template, sampling params, validation     │
  │              └─ tokenizer, request id, output streaming (SSE), detokenization         │
  │  AsyncLLM             vllm/v1/engine/async_llm.py                                     │
  │    └─ per-request asyncio queue + output_processor                                    │
  └───────────────────────────────┬──────────────────────────────────────────────────────┘
                                  │  ZMQ (msgpack-encoded EngineCoreRequest / Outputs)
                                  │  vllm/v1/engine/core_client.py
  ┌───────────────────────────────▼── PROCESS 2: EngineCore (the busy loop) ─────────────┐
  │  EngineCore            vllm/v1/engine/core.py     while True: schedule → execute →    │
  │    ├─ Scheduler        vllm/v1/core/sched/scheduler.py            update_from_output  │
  │    ├─ KVCacheManager   vllm/v1/core/kv_cache_manager.py                               │
  │    │     └─ BlockPool  vllm/v1/core/block_pool.py                                     │
  │    └─ Executor         vllm/v1/executor/…  (uni | multiproc | ray)                    │
  └───────────────────────────────┬──────────────────────────────────────────────────────┘
                                  │ (TP/PP: one worker process per rank)
  ┌───────────────────────────────▼── PROCESS 3..N: Workers ─────────────────────────────┐
  │  Worker + ModelRunner  vllm/v1/worker/gpu/model_runner.py, vllm/v1/worker/gpu_worker.py│
  │    ├─ InputBatch (persistent)   vllm/v1/worker/gpu/input_batch.py                     │
  │    ├─ block tables               vllm/v1/worker/gpu/block_table.py                    │
  │    ├─ attention backend          vllm/v1/attention/backends/{flash_attn,flashinfer}.py │
  │    └─ CUDA graph capture/replay  vllm/v1/worker/gpu/cudagraph_utils.py                │
  └──────────────────────────────────────────────────────────────────────────────────────┘
```

**Why the split exists.** In V0, the HTTP server and the scheduling loop shared a process and the GIL: every SSE chunk serialized, every tokenizer call, every asyncio callback stole time from the loop that had to issue the next forward pass. At 40 tokens/sec × 100 streams that is thousands of Python-level events per second contending with the thing that keeps the GPU fed. Moving the engine into its own process makes the GPU-side loop *isolated* from API-side work — the same motivation that made TGI write its router in Rust ([lesson 4](04-tgi-and-the-router-split.md)), solved with a process boundary instead of a language boundary.

Consequence you will feel in production: **there are two places to look at when latency is bad.** API-process CPU saturation (tokenization, JSON, SSE) looks nothing like engine-core saturation (queue depth, KV usage), and `/metrics` distinguishes them ([lesson 3](03-vllm-in-production.md)).

---

## The request's journey, hop by hop

| # | Hop | File (V1) | What happens |
|---|---|---|---|
| 1 | HTTP arrives | `vllm/entrypoints/openai/api_server.py` | Route, auth, request-id |
| 2 | Protocol → internal | `.../openai/chat_completion/serving.py` | Chat template applied, `SamplingParams` built, prompt tokenized, `max_tokens` validated against `max_model_len` |
| 3 | Enter the engine | `vllm/v1/engine/async_llm.py` → `core_client.py` | Request serialized, pushed over ZMQ; an asyncio output queue is created for it |
| 4 | Admission | `vllm/v1/engine/core.py::add_request` → `Scheduler.add_request` | Appended to the **waiting** queue (FCFS or priority) |
| 5 | Scheduling | `vllm/v1/core/sched/scheduler.py::schedule` | Token budget spent across running + waiting; produces `SchedulerOutput` |
| 6 | Memory | `vllm/v1/core/kv_cache_manager.py::get_computed_blocks` / `allocate_slots` | Prefix-cache lookup, then block allocation (may fail → preemption) |
| 7 | Execute | `vllm/v1/worker/gpu/model_runner.py::execute_model` | Input tensors + block tables built, forward pass (graph replay if eligible), sampling |
| 8 | Bookkeeping | `Scheduler.update_from_output` | New token appended, stop criteria checked, finished requests freed |
| 9 | Back out | `vllm/v1/engine/output_processor.py`, `detokenizer.py` | Incremental detokenization, stop-string handling, per-request queue push |
| 10 | Stream | route handler | SSE chunk to the client |

Do this trace yourself with real line numbers; it is exercise 1 of [lesson 11](11-exercises-and-artifacts.md).

---

## The scheduler, which is the whole ballgame

Open `vllm/v1/core/sched/scheduler.py` and read the comment at the top of `schedule()` before anything else. Paraphrased, it says:

> There is no "decoding phase" nor "prefill phase". Each request has `num_computed_tokens` and a target number of tokens. Each step the scheduler assigns tokens to requests so that `num_computed_tokens` catches up. This is general enough to cover chunked prefill, prefix caching, and speculative decoding.

This is the design insight worth stealing. Your Phase-3 engine almost certainly had a `if request.is_prefill: … else: …` branch. vLLM deleted that branch, and by doing so got **chunked prefill** (assign a request 512 of its 4,000 prompt tokens this step), **speculative decoding** (a decode step that assigns k+1 tokens) and **prefix-cache hits** (a request that starts life with `num_computed_tokens = 3,000` and no compute needed) as the *same* mechanism instead of three special cases.

### The loop's shape

```
schedule():
  token_budget = max_num_batched_tokens          # the step's total token allowance

  # 1) RUNNING requests first — they already hold KV blocks; keeping them
  #    moving is what bounds tail latency and avoids thrash.
  for request in running:
      tokens = min(request.remaining_tokens, token_budget)
      new_blocks = kv_cache_manager.allocate_slots(request, tokens)
      while new_blocks is None:                  # pool exhausted
          victim = running.pop()                 # preempt the LAST (newest) request
          kv_cache_manager.free(victim)          # its KV is dropped; it will recompute
          victim.status = PREEMPTED → back to the waiting queue
          new_blocks = kv_cache_manager.allocate_slots(request, tokens)
      token_budget -= tokens

  # 2) WAITING requests, while budget and blocks remain
  for request in waiting:
      computed_blocks, num_computed = kv_cache_manager.get_computed_blocks(request)  # prefix cache!
      tokens = min(request.num_tokens - num_computed, token_budget)
      if not kv_cache_manager.allocate_slots(request, tokens): break
      running.append(request); token_budget -= tokens

  return SchedulerOutput(new_reqs, cached_reqs, num_scheduled_tokens, ...)
```

Five things to notice, each of which is a Phase-3 lesson made concrete:

1. **Running-first ordering** is admission control by another name: an accepted request keeps its resources until it finishes or is preempted. That's why vLLM's TTFT degrades under load but its TPOT stays comparatively flat ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)).
2. **Preemption picks the newest running request** (LIFO victim selection) precisely so that preemption doesn't cascade into starvation of old requests. The preempted request's blocks are freed and its tokens are **recomputed** later — recompute beats swap because prefill is cheap per token and PCIe is slow ([Phase 4 lesson 5](../phase-4/05-paged-attention.md)).
3. **The token budget is the only knob that couples prefill and decode.** One 4,000-token prefill and 30 decodes cost 4,030 tokens of budget; if `max_num_batched_tokens` is 8,192, both fit and your decoders stutter by exactly the prefill's duration. This is the mechanism behind every "TPOT spikes when a long prompt arrives" report.
4. **Scheduling is pure bookkeeping — no tensors are touched.** That's why it can afford to be Python, and why V1's "zero-overhead scheduler" work was about *overlapping* it with the forward pass, not rewriting it in C++.
5. **`SchedulerOutput` is the contract** between scheduler and executor: which requests are new, which are cached, how many tokens each gets, their block tables. Read `vllm/v1/core/sched/output.py` — it's short, and it's the cleanest statement of what an engine step *is* that exists in any of these codebases.

---

## The KV-cache manager, next to your own

`vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/block_pool.py`. Map it against what you built in [Phase 4 lesson 9](../phase-4/09-build-paged-kv-and-quant-bench.md):

| Your `paged_kv.py` | vLLM V1 | Notes |
|---|---|---|
| `BlockPool.allocate()` | `BlockPool.get_new_blocks(n)` | Pops from a free-block queue; also evicts cached-but-unreferenced blocks (LRU) on demand |
| `ref[]` refcounts | `KVCacheBlock.ref_cnt` | Same idea; a block with `ref_cnt == 0` stays *cached* and reusable until reclaimed — free ≠ evicted |
| `hash_to_block` dict | `BlockHashToBlockMap` + `cache_full_blocks()` | Content hash includes parent-block hash → a chain, so a hit means the *whole* prefix matched, not one block |
| `Sequence.append_token()` | `KVCacheManager.allocate_slots(request, num_tokens)` | Allocates for a *token count* the scheduler chose, not one token |
| prefix lookup | `KVCacheManager.get_computed_blocks(request)` | Returns blocks + `num_computed_tokens`; the scheduler then skips that much prefill |
| `copy_on_write()` | (mostly unnecessary) | Full blocks are immutable once hashed; only the partial tail block ever diverges |
| `decref()` / free list | `free_blocks()`, `_maybe_evict_cached_block()` | Freed blocks go to the tail of an LRU queue; eviction happens lazily at allocation time |
| — | `usage()`, `make_prefix_cache_stats()`, `take_events()` | Observability is a first-class API here; yours had `stats` |

**Eight things theirs does that yours doesn't** (this is exactly the list [lesson 11](11-exercises-and-artifacts.md) asks you to produce yourself, so write your own version — this one is a spoiler you should check against, not copy):

1. Lazy LRU eviction of cached-but-unreferenced blocks, so a "free" block keeps serving prefix hits until the memory is actually needed.
2. Hash chaining (parent hash folded into each block hash), which makes a single map lookup prove a full-prefix match.
3. Multiple KV cache *groups* (`kv_cache_coordinator.py`) for models that mix attention types — full attention + sliding window + Mamba states in one model.
4. Sliding-window and hybrid-model managers (`single_type_kv_cache_manager.py`) that free blocks that have fallen out of the window.
5. KV connectors (`kv_connector/`) for offloading and prefill/decode disaggregation — blocks that live on another machine ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)).
6. Cache-event streams for external cache-aware routers.
7. `get_num_common_prefix_blocks()`, which lets the attention kernel run a *cascade* over the shared prefix once instead of per-sequence.
8. Careful handling of the partial tail block, hash salting for privacy/isolation, and block-size alignment for every backend.

---

## The model runner: where scheduling becomes tensors

`vllm/v1/worker/gpu/model_runner.py` is the biggest file in the engine, and 80% of it is input preparation. The important object is the **persistent batch** (`input_batch.py`): the runner keeps GPU-resident tensors for the whole batch across steps and *mutates rows* rather than rebuilding them, because rebuilding meant `H2D` copies and Python overhead every 10 ms ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)).

Per step it produces, for a batch of mixed prefill-chunks and decodes:

```
  input_ids        [total_tokens]          flattened, ragged — NOT [B, T] padded
  positions        [total_tokens]
  query_start_loc  [num_reqs + 1]          cumulative offsets (varlen attention)
  seq_lens         [num_reqs]              full context length per request
  block_table      [num_reqs, max_blocks]  the Phase-4 block tables, on GPU
  slot_mapping     [total_tokens]          where each new K/V goes: block_id*block_size + offset
```

`slot_mapping` is the line where PagedAttention becomes real: the attention kernel writes each token's K/V directly into its physical slot, and reads history through `block_table`. If you understood [Phase 4 lesson 5](../phase-4/05-paged-attention.md), you can read this file.

Also here: CUDA-graph capture for a **set of batch sizes** (padded up to the nearest captured size), the reason `--enforce-eager` exists, and the reason startup takes 30-90 seconds. And `vllm/v1/spec_decode/` (`ngram_proposer.py`, `eagle.py`, `medusa.py`) — the proposers from [Phase 4 lesson 7](../phase-4/07-speculative-decoding.md), each producing draft tokens that the scheduler then budgets for like any other tokens.

---

## What V1 changed, and why you should care

vLLM rewrote its core in 2024-2025 (V0 → V1, now the only engine). The changes are a list of production lessons:

| Change | Motivation |
|---|---|
| Engine core in a **separate process**, ZMQ IPC | API-side Python work no longer steals cycles from the scheduling loop |
| **Unified scheduling** (no prefill/decode phases; `num_computed_tokens` catch-up) | Chunked prefill, prefix caching and spec decode become one mechanism |
| **Prefix caching on by default**, near-zero overhead when it misses | It's free throughput on real chat/agent traffic ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)) |
| **Persistent batch** in the runner | Kill per-step tensor rebuild and H2D copies |
| **`torch.compile` + piecewise CUDA graphs** by default | Launch overhead dominates at small batch ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)) |
| Async scheduling / overlapped step (`async_scheduler.py`) | Schedule step *n+1* on the CPU while step *n* runs on the GPU |

If you read a 2023 vLLM blog post or a V0-era diagram (`llm_engine.py`, `block_manager_v1.py`, "swap out to CPU"), you're reading history. Check `vllm/v1/` first; `vllm/engine/` still exists mostly as configuration and compatibility surface (`vllm/engine/arg_utils.py` is where the CLI flags are defined and is genuinely useful — it's the authoritative flag list for [lesson 3](03-vllm-in-production.md)).

---

## Guided read (2-3 hours, no GPU)

Timebox each step; the goal is fluency, not completeness.

1. **(20 min)** `vllm/v1/core/sched/output.py` — the whole engine's data contract. Write down every field and what it's for.
2. **(40 min)** `Scheduler.schedule()`. Answer: where is the token budget decremented? What exactly happens when `allocate_slots` returns `None`? Which queue does a preempted request go back to, and at which end?
3. **(30 min)** `KVCacheManager.get_computed_blocks()` and `BlockPool.cache_full_blocks()`. Answer: what is hashed into a block hash, and why does the parent hash matter? What happens to the partial (non-full) tail block?
4. **(30 min)** `model_runner.py`: find where `slot_mapping` and `block_table` are built, and where the CUDA graph is replayed vs where eager runs.
5. **(20 min)** `vllm/v1/metrics/loggers.py` + `stats.py`: list the Prometheus metrics and match each to a mechanism you now know. You'll use this list in [lesson 3](03-vllm-in-production.md).
6. **(10 min)** `git log --oneline -30 vllm/v1/core/sched/scheduler.py` — read the last 30 commit messages to the scheduler. This tells you what's actually hard about this code in practice.

Deliverable for your notes: a one-page trace (10 hops, file + function + line, at your pinned commit) and the list of five things vLLM's KV manager does that yours doesn't.

---

**Next:** [Operating and tuning vLLM →](03-vllm-in-production.md) — every flag mapped to the mechanism you just read, sized with arithmetic, plus the five failure modes and what `/metrics` says about each.
