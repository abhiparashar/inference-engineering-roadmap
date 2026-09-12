# 3 — Capstone Brief: nano-vLLM

> **Proof obligation:** that you understand the serving loop well enough to rebuild its core — continuous batching, block-based KV management, prefix reuse, streaming API — and that you can explain, with profiler evidence, exactly why a production engine is still faster than yours.

This is a **brief, not a build guide**. The mechanics live in [Phase 3 lessons 4-6](../phase-3/04-continuous-batching.md), [Phase 4 lessons 4-6](../phase-4/04-kv-cache-optimization.md), and [Phase 5 lessons 2, 9](../phase-5/02-vllm-architecture.md); the discipline lives in [lesson 2](02-engineering-standards.md). What follows is the scope, the bar, and the specific ways this project goes wrong.

---

## The claim to prove

> "A ~1-2k-line engine with continuous batching, block-based paged KV cache, prefix sharing, and an OpenAI-compatible streaming API reaches *X*% of vLLM's throughput at equal p99 TTFT on the same GPU and model, and the remaining gap decomposes into *these* measured components."

The percentage is not the result. **The decomposition is the result.**

## Scope: what is in and what is out

| In (the mechanism) | Out (say so explicitly) |
|---|---|
| Continuous/in-flight batching scheduler with admission control | tensor/pipeline parallelism |
| Block-based KV allocator, block table per sequence, free-list | quantization beyond fp16/bf16 |
| Prefix sharing via block reuse with reference counting | speculative decoding (unless it *is* your paper repro) |
| Streaming `/v1/chat/completions` with correct SSE semantics and cancellation | LoRA/multi-adapter serving |
| Preemption/eviction policy when blocks run out | custom attention kernels (use FlashAttention/SDPA and say so) |
| Metrics: queue depth, running batch, block occupancy, TTFT/TPOT | multi-node anything |
| One model (1-8B), one GPU, one precision | exhaustive model coverage |

**The single most common scoping error** is attempting tensor parallelism or custom kernels. Neither demonstrates the scheduler, which is the thing this project exists to demonstrate.

---

## The acceptance bar

- [ ] **Correctness first**: greedy outputs match a HuggingFace reference for ≥ 50 prompts, token for token. Without this, every performance number is meaningless. This is the gate most solo engines quietly skip.
- [ ] **The three-way table**: naive sequential → static batching → your continuous batching, same model/GPU/workload, throughput at a fixed p99 TTFT. The deltas are the value you added ([Phase 3 lessons 2-4](../phase-3/02-static-batching.md)).
- [ ] **vLLM baseline, tuned and published**: its flags in the README, same harness, same workload, same output length.
- [ ] **The gap decomposition**: profiler evidence attributing the remaining difference to specific missing optimizations — kernel choice, CUDA graphs, sampler fusion, scheduling policy ([`playbooks/profiling.md`](../../playbooks/profiling.md), [Phase 2 lesson 8](../phase-2/08-profiling-in-practice.md)).
- [ ] **Prefix-cache evidence**: TTFT with and without prefix sharing on a workload with a shared system prompt, plus the measured hit rate ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)).
- [ ] **Memory accounting**: how `gpu_memory_utilization` translates into block count, and the KV-budget arithmetic ([Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md), [Phase 4 lesson 5](../phase-4/05-paged-attention.md)).
- [ ] **Three failure tests**: overload (bounded queue and shedding, not unbounded latency), client disconnect mid-stream (blocks freed, generation cancelled), and block exhaustion (preemption works and the preempted request completes correctly).
- [ ] **Cost**: $/1M tokens for your engine and for vLLM on the same instance.
- [ ] **Limitations**: written, specific, prioritized.

---

## Pitfalls specific to this project

| Pitfall | Symptom | Avoidance |
|---|---|---|
| Skipping the correctness gate | fast engine, subtly wrong outputs; discovered by a reviewer | token-match against HF reference before any benchmarking |
| Python-loop scheduler overhead dominating | GPU idle 40% of the time at small batch | measure the scheduler's own time per step; batch the bookkeeping; it is the classic finding of this project and worth reporting |
| Closed-loop load generation | latency looks great, queueing is invisible | fixed-arrival-rate open-loop client ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) |
| Comparing against untuned vLLM | "I beat vLLM" | publish its config; expect and explain a loss |
| Variable output lengths across runs | noisy, incomparable throughput | fix `max_tokens`, ignore EOS for benchmarks |
| Reference-counting bugs in prefix sharing | rare corruption or leaked blocks under churn | a stress test that interleaves shared-prefix and unique requests, asserting block-count conservation |
| Streaming semantics wrong | clients hang, or the last token is dropped on cancel | test SSE explicitly, including client abort |
| Unbounded admission | OOM under load instead of queueing | admission control on KV budget, not request count ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)) |

---

## What "excellent" looks like versus "adequate"

| Adequate | Excellent |
|---|---|
| It batches continuously and serves tokens | the scheduler's policy is a named, parameterized choice, and you measured two alternatives |
| A throughput number vs vLLM | a decomposition of the gap with profiler traces per component |
| Prefix caching implemented | hit-rate and TTFT curves versus shared-prefix fraction |
| "Handles many concurrent requests" | a throughput-vs-p99 frontier curve, with the knee identified |
| Code on GitHub | a write-up a reader can learn continuous batching *from* |

The excellent column is roughly one extra week of measurement on top of the same code.

---

## Where it leads

- The natural follow-on is [lesson 6](06-capstone-paper-reproduction.md): your engine is now a substrate for reproducing PagedAttention, speculative decoding, or chunked prefill against a real serving loop.
- It is the best possible preparation for the OSS-contribution track in [`resources/README.md`](../../resources/README.md) — after this, vLLM's scheduler source reads like code you have already written.
- Project index entry: **18 — nano-vLLM** in [`projects/README.md`](../../projects/README.md).

---

**Next:** [Capstone brief: multi-modal pipeline →](04-capstone-multimodal-pipeline.md) — the shipping counterpart, where the engineering is composition, observability, and degradation rather than the engine itself.
