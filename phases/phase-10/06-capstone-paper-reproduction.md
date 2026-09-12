# 6 — Capstone Brief: Reproduce a Paper's System

> **Proof obligation:** that you can read a systems paper, identify the mechanism that produces its claimed effect, implement a simplified version of that mechanism against a real serving loop, and demonstrate the *direction* of the effect with your own honest measurements — including an explanation of why your magnitude differs.

A **brief, not a build guide**. The paper's phase owns the mechanics ([Phase 4 lessons 5-7](../phase-4/05-paged-attention.md), [Phase 6 lesson 6](../phase-6/06-disaggregated-prefill-decode.md), [Phase 5 lesson 5](../phase-5/05-sglang-and-radixattention.md)); the measurement standard is [lesson 2](02-engineering-standards.md); source-reading method is [Phase 5 lesson 9](../phase-5/09-reading-engine-source.md).

**Reproducing the numbers is not the goal and is usually impossible** — you lack the cluster, the traces, and the engineering years. Reproducing the *mechanism* and showing that it moves the metric in the claimed direction, for the claimed reason, is the goal.

---

## The claim to prove

> "Implementing *mechanism M* from *paper P* in a serving loop I control changes *metric X* by *Y*% in the direction the paper predicts, under the conditions the paper says are necessary — and it does *not* help (or hurts) under *these* conditions, which is consistent with the paper's model."

The second half is the interesting half. **A reproduction that also finds the boundary where the technique stops working is a stronger result than one that only confirms the happy path.**

## Candidate papers, and what each demands

| Paper | Mechanism to implement | Predicted effect | Real difficulty |
|---|---|---|---|
| **PagedAttention** (vLLM) | block-based KV allocation + block tables + sharing | higher throughput via less fragmentation; memory utilization ↑ | needs an attention kernel that reads blocks (or a gather/copy shim — state which) |
| **Speculative decoding** | draft model + parallel verify + accept/reject | lower per-token latency at low batch; *no* gain at high batch | acceptance-rate measurement and correct output equivalence |
| **RadixAttention / prefix caching** | radix-tree prefix cache with eviction | TTFT ↓ as shared-prefix fraction ↑ | eviction policy and hit-rate accounting |
| **DistServe / prefill-decode disaggregation** | separate prefill and decode workers + KV transfer | both SLOs met at lower cost; interference removed | the KV transfer cost is the whole story; needs 2 GPUs to be honest |
| **Chunked prefill / Sarathi** | split long prefills into chunks interleaved with decode | decode ITL jitter ↓ under mixed load | scheduler bookkeeping |
| **FlashAttention** | tiled, recompute-based attention | memory ↓, speed ↑ | a real kernel project; only pick it if [Phase 2](../phase-2/README.md) was your favorite phase |

**Best substrate:** the engine from [lesson 3](03-capstone-nano-vllm.md), if you built it. Second best: a fork of a small real engine. Third: a minimal decode loop you write for the purpose — acceptable, but say so, because a mechanism's benefit depends on the loop it lives in.

---

## The acceptance bar

- [ ] **Paper summary in your own words**: the claim, the mechanism, the conditions under which it helps, and the authors' stated limitations. One page, no equations copied without understanding.
- [ ] **Output equivalence** where the paper claims it (speculative decoding is lossless; paging is lossless): proven token-for-token against the unmodified path. A "speedup" that changes outputs is a bug report, not a result.
- [ ] **Ablation, not just A/B**: the mechanism on versus off in the same binary, same workload, same harness — plus a sweep of the parameter the paper says matters (acceptance rate, block size, shared-prefix fraction, chunk size).
- [ ] **The boundary condition**: the regime where the effect disappears or inverts, measured. E.g. speculative decoding's gain vanishing as batch size grows; prefix caching doing nothing on unique prompts.
- [ ] **Directional agreement stated honestly**, with your magnitude and the paper's side by side and the differences explained (hardware, model size, missing kernels, workload).
- [ ] **What you simplified**, itemized: every simplification versus the paper, and your expectation of its effect.
- [ ] **Cost/overhead of the mechanism itself**: extra memory, extra compute, scheduler time. Papers report the win; implementations pay the overhead.
- [ ] **Limitations**, written.

---

## Pitfalls specific to this project

| Pitfall | Symptom | Avoidance |
|---|---|---|
| Chasing the paper's absolute numbers | months of frustration, no write-up | commit to direction + explanation from day one |
| Measuring the mechanism in the wrong regime | "speculative decoding didn't help" with batch 64 | read the paper's conditions; sweep the regime axis |
| No equivalence check | speedup from silently dropping tokens | token-match against the baseline path |
| Two changes at once | can't attribute the effect | one flag, one ablation |
| Implementing the paper's *system* rather than its mechanism | scope explodes | extract the single mechanism; simplify the rest deliberately |
| Ignoring the mechanism's overhead | net effect worse than reported and unexplained | measure draft-model cost, block-table cost, transfer cost |
| Treating the paper as gospel | confusion when results disagree | papers report best cases on their hardware; disagreement is data, not failure |

---

## What "excellent" looks like versus "adequate"

| Adequate | Excellent |
|---|---|
| The mechanism is implemented and it's faster | a parameter sweep showing the predicted curve shape, not just two points |
| "I got 1.4×, the paper got 2.3×" | the gap attributed to specific missing components, each measured or bounded |
| Happy-path result | the boundary where the technique stops helping, measured and explained |
| Code | a write-up that teaches the mechanism better than the paper does for practitioners |

---

## Where it leads

- This is the strongest preparation for research-engineer and MLSys roles, and for the staying-current track in [`resources/README.md`](../../resources/README.md): after one reproduction, reading new systems papers becomes fast.
- It is also the most natural bridge to an OSS contribution — you now hold a measured opinion about a mechanism inside a real project.
- No fixed project number; add it to [`projects/README.md`](../../projects/README.md) yourself, next to the phase whose mechanism you reproduced.

---

**Next:** [Capstone brief: production RAG service →](07-capstone-production-rag.md) — the shape of most real AI backends, with security and CI gates included.
