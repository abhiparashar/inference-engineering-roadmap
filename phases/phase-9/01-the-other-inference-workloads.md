# 1 — The Other Inference Workloads

> **You'll be able to say:** "I classify a serving problem by four numbers — per-request FLOPs, latency budget, per-request state, and arrival rate — and those four numbers determine the architecture before I know anything about the model. An LLM at 10 QPS with a 2-second budget and a gigabyte of KV state, and a ranking model at 1M QPS with a 10 ms budget and terabytes of *shared* state, are opposite corners of that space: almost every default flips between them. I can say for any workload which Phase 2-8 techniques transfer, which are irrelevant, and which are actively wrong."

Phases 1-8 taught one workload extremely well. This lesson is the map that tells you when that knowledge applies — and, more usefully, when reaching for it makes you the engineer who spent three weeks optimizing 12% of the latency budget.

The honest framing: **the LLM-serving skill set is deep but narrow.** Continuous batching, PagedAttention, prefix caching and speculative decoding exist because of one specific property — autoregressive generation over a shared, growing per-request cache. Remove that property and the entire toolkit collapses down to "batch requests, keep the accelerator busy, don't copy things twice."

---

## The four numbers

Ask these before you look at a single line of model code:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. FLOPs PER REQUEST    how much arithmetic does one answer cost?   │
  │                         6·P·tokens for an LLM; ~0 for an embedding  │
  │                         lookup; 50 GFLOP for one ResNet-50 image    │
  │                                                                     │
  │ 2. LATENCY BUDGET       and measured against WHAT clock?            │
  │                         wall-clock deadline (10 ms ranking),        │
  │                         perceptual (200 ms first token / audio),    │
  │                         input duration (speech: real-time factor),  │
  │                         none at all (offline batch)                 │
  │                                                                     │
  │ 3. STATE PER REQUEST    does request N+1 depend on request N's      │
  │                         memory? KV cache = GBs, per request, alive  │
  │                         for the whole generation. Vision = zero.    │
  │                         Ranking = terabytes, but SHARED and mostly  │
  │                         read-only.                                  │
  │                                                                     │
  │ 4. ARRIVAL RATE & SHAPE QPS, burstiness, and how many model calls   │
  │                         one user action triggers (fan-out)          │
  └─────────────────────────────────────────────────────────────────────┘
```

Two derived quantities do most of the work:

- **Arithmetic intensity** — FLOPs per byte moved — tells you which side of the roofline you're on ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)). LLM decode is famously terrible at ~1-2 FLOP/byte. A vision batch is 100×. An embedding lookup is *below* 1: it is pure memory/network movement with a multiply-add stapled on.
- **Budget ÷ per-request compute time** tells you how much of the budget is available for *everything else* — queueing, network, preprocessing, serialization. For an LLM that ratio is ~1.2 (compute dominates). For ranking it's ~20 (compute is noise; the budget is spent on hops).

**This is the whole lesson in one sentence:** when compute is 5% of the budget, every technique from Phases 2-4 is irrelevant, and the engineering moves to the network, the cache hierarchy, and the tail.

---

## Five archetypes

| | LLM generation | Ranking / recsys | Vision (batch) | Streaming speech | Embedding / retrieval |
|---|---|---|---|---|---|
| **FLOPs/request** | 10¹²-10¹⁴ | 10⁶-10⁸ | 10¹⁰-10¹¹ | 10⁹-10¹⁰ per chunk | 10⁸ + index search |
| **Budget** | 200 ms TTFT, 2-60 s total | **5-30 ms hard** | 0.1-10 s, or offline | RTF < 0.3, < 300 ms partials | 5-50 ms |
| **State** | GBs KV, per request, growing | TB, shared, read-mostly | none | small, per stream, per session | GB-TB index, shared |
| **Model size** | 7B-700B | < 1 GB dense + TB embeddings | 5-500 MB | 50 MB-2 GB | 100 MB-1 GB |
| **QPS/replica** | 1-50 | 10³-10⁵ | 10-10³ images/s | 10-100 streams | 10³-10⁴ |
| **Bottleneck** | HBM bandwidth | **network round trips, embedding lookup** | **preprocessing/decode** | jitter, chunk boundaries | memory bandwidth + recall tuning |
| **Batching** | continuous, in-flight | trivially free (one request *is* a batch of candidates) | static/dynamic, large | limited by stream timing | dynamic |
| **Typical stack** | vLLM/TGI/TRT-LLM | Triton, custom C++, feature store | Triton + DALI/TensorRT | custom streaming server | FAISS/ScaNN/Qdrant |
| **Cost driver** | GPU-hours | fleet CPU + memory + network | GPU-hours + egress | GPU-hours, poorly utilized | RAM (!) |

Read that table by row, not column. **Every row has a different winner**, and the "bottleneck" row is the one that decides what you actually work on.

---

## What transfers from Phases 2-8, and what doesn't

| Technique | LLM | Ranking | Vision | Speech | Notes |
|---|---|---|---|---|---|
| Roofline thinking ([P2 L5](../phase-2/05-roofline-model.md)) | ● | ● | ● | ● | universal; the *answer* differs, the method never does |
| Honest measurement ([P3 L7](../phase-3/07-measuring-honestly.md)) | ● | ● | ● | ● | universal, and most violated in recsys benchmarks |
| Static / dynamic batching ([P3 L2-3](../phase-3/02-static-batching.md)) | ✗ (wrong) | ● | ● **primary** | ◐ | the "obsolete" technique is correct for everything non-autoregressive |
| Continuous batching ([P3 L4](../phase-3/04-continuous-batching.md)) | ● **core** | ✗ | ✗ | ◐ | needs variable-length autoregressive decode; meaningless otherwise |
| PagedAttention ([P4 L5](../phase-4/05-paged-attention.md)) | ● **core** | ✗ | ✗ | ✗ | solves KV fragmentation; no KV, no problem |
| Prefix caching ([P4 L6](../phase-4/06-prefix-caching-and-radix-attention.md)) | ● | ✗ | ✗ | ◐ | analogue in speech: cached encoder states across chunks |
| Speculative decoding ([P4 L7](../phase-4/07-speculative-decoding.md)) | ● | ✗ | ✗ | ◐ | only helps when decode is memory-bound and sequential |
| Quantization ([P4 L2-3](../phase-4/02-quantization-fundamentals.md)) | ● | ● (embeddings!) | ● | ● | on CPU/NPU it's mandatory, not optional ([L5](05-hardware-diversity.md), [L7](07-edge-and-on-device.md)) |
| CUDA graphs / compile ([P2 L6](../phase-2/06-overhead-bound-and-cuda-graphs.md)) | ● | ● **huge** | ● | ● | tiny-model workloads are launch-overhead-dominated; graphs are the single biggest win |
| Tensor parallelism ([P6 L3](../phase-6/03-tensor-parallelism.md)) | ● | ✗ | ✗ | ✗ | model fits on one device in every other archetype |
| Embedding-table sharding | ✗ | ● **core** | ✗ | ✗ | the recsys analogue of TP: shard *data*, not compute ([L2](02-recommendation-and-ranking.md)) |
| Prefix-aware routing ([P6 L7](../phase-6/07-prefix-aware-routing.md)) | ● | ◐ (shard affinity) | ✗ | ● (session affinity) | "route to the replica that already has the state" generalizes |
| SLOs, burn rate ([P7 L5](../phase-7/05-slos-and-error-budgets.md)) | ● | ● | ● | ● | universal; the SLI differs |
| Cost per unit work ([P7 L9](../phase-7/09-cost-per-million-tokens.md)) | per 1M tokens | per 1M requests | per 1M images | per 1000 audio-hours | same discipline, different denominator |
| Containers, k8s, gates (P8) | ● | ● | ● | ● | universal, and the reason Phase 8 came before this one |

● applies · ◐ partially/analogue · ✗ irrelevant or wrong

The `✗` column for continuous batching and PagedAttention is worth sitting with. **Those are the two most celebrated inference techniques of the last three years, and they apply to exactly one of five archetypes.** If your mental model of "inference engineering" is those two ideas, you can be an expert on a third of the field and unemployable for the rest of it.

---

## Why the 10 ms / 1M QPS regime is a different discipline

Take a ranking service: 1M QPS, 10 ms p99 budget, 1 MFLOP of dense compute per candidate, 500 candidates per request.

```
  total dense FLOPs   = 1e6 QPS × 500 × 1e6 FLOP  = 5e14 FLOP/s  = 500 TFLOP/s
  → about 1-3 modern GPUs' worth of arithmetic for the ENTIRE FLEET
  and yet the fleet is thousands of machines. Where does it all go?

  request timeline (10 ms budget, p99):
  ├─ 0.4 ms  gateway + deserialize
  ├─ 1.2 ms  candidate generation (ANN / inverted index)     ← lesson 6
  ├─ 3.5 ms  FEATURE FETCH: 300 keys × several stores, parallel fan-out
  │          p99 dominated by the SLOWEST of 300 lookups     ← the real problem
  ├─ 0.9 ms  embedding-table gather (TB-scale, sharded over hosts)
  ├─ 1.1 ms  dense forward pass  ← the "model". 11% of the budget.
  ├─ 0.6 ms  scoring/blending/business rules
  └─ 2.3 ms  slack for GC pauses, retries, serialization
```

Three consequences that have no analogue in LLM serving:

1. **Tail amplification.** If one feature lookup has p99 = 5 ms and you issue 300 in parallel, the probability that *at least one* is slow approaches 1. The expected max of 300 draws, not the p99 of one, is your latency. Fixes are statistical, not algorithmic: hedged requests, tied requests, request cancellation, and returning a partial feature vector rather than waiting ([Phase 7 lesson 7](../phase-7/07-reliability-and-degradation.md)'s degradation ladder, with a 3 ms rung).
2. **Batching is free.** One user request already contains 500 candidates to score — a natural batch. There is no batch-size-versus-latency tradeoff to agonize over, because the batch exists before the request is queued. Almost everything in Phase 3 lessons 2-6 becomes a non-problem.
3. **Memory is the cost line, not compute.** Terabyte embedding tables live in DRAM across many hosts, mostly to be *looked up*, rarely to be multiplied. You are running a distributed hash table with a neural network attached, and you should staff and design it accordingly.

The mirror-image statement for LLMs: **a 2-second budget with 500 sequential decode steps means you have 4 ms per step**, all of which is spent streaming weights from HBM, so your entire toolkit is about reading weights fewer times (batching) or reading fewer weights (quantization, sparsity, speculation).

---

## Latency budgets are measured against different clocks

A subtlety that trips people moving between workloads: "latency" means four different things.

| Clock | Meaning | Workloads | The metric |
|---|---|---|---|
| **Wall-clock deadline** | exceed it and the result is worthless — the page already rendered | ranking, ads, fraud | hard p99/p999 with timeout-and-default |
| **Perceptual** | humans notice above a threshold | chat TTFT, voice first-audio | p95 TTFT, first-audio latency |
| **Input-relative** | must keep up with a stream arriving in real time | ASR, TTS, live video | **real-time factor** = compute time ÷ audio duration ([lesson 4](04-speech-and-streaming.md)) |
| **None (throughput)** | only $/unit and completion-by-morning matter | offline embedding, batch scoring, video indexing | items/sec/$; queue for hours, batch enormously |

The offline row is the most under-appreciated. A huge fraction of real inference compute is offline batch — nightly embedding refreshes, content moderation backfills, catalog enrichment. **It has no latency SLO at all**, which means the correct configuration is the one Phase 3 called pathological: maximum batch size, maximum queue delay, spot instances, restart-on-preemption ([Phase 8 lesson 5](../phase-8/05-scheduling-capacity-and-autoscaling.md)). Engineers who only know online serving routinely leave 5-10× of cost on the table here by serving batch jobs through an online endpoint.

---

## Fan-out: the number nobody writes on the design doc

One user action rarely equals one model call.

```
  one product-page view
    ├── 1× ranking request      → 500 candidate scores  (1 model call, batched)
    ├── 3× vision models        → thumbnail moderation, tagging, aesthetics
    ├── 1× embedding call       → "similar items" retrieval  → 1× ANN search
    └── 1× LLM call             → generated summary (streams, 2 s)
  = 6 model invocations, 4 different archetypes, 4 different SLOs,
    3 different teams, and ONE user-visible latency number.
```

Consequences:

- **The user-visible p99 is worse than any single stage's p99** unless calls are parallel *and* independently timed out ([lesson 10](10-rag-and-agentic-serving.md) formalizes this for RAG).
- **A shared model serving many callers has no single SLO.** The moderation model has 50 ms for the interactive path and no budget at all for the backfill path. Same model, same replicas, two SLOs — which is precisely why priority classes and per-tenant quotas exist ([lesson 9](09-security-and-multi-tenancy.md)).
- **Cost attribution needs the fan-out factor.** "Our vision model costs $0.0001/call" is meaningless until you multiply by 3 calls per pageview and 40M pageviews.

---

## How to recognize which archetype you've been handed

A five-question triage that works in an interview or on day one of a new team:

1. **Is the output generated token by token (or frame by frame) with each step depending on the last?** Yes → autoregressive; Phases 3-4 apply in full. No → you are in a single-shot regime and the toolkit is batching + kernel efficiency + preprocessing.
2. **Is per-request compute more than half the latency budget?** No → stop profiling the model; profile the pipeline.
3. **Does the request carry state that must survive across calls?** KV cache, encoder state, session context → you need affinity ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)) and careful draining ([Phase 8 lesson 4](../phase-8/04-kubernetes-for-gpu-serving.md)).
4. **How many stores/services does one request touch?** > 5 → tail amplification is your primary engineering problem, not FLOPs.
5. **Is there a hard deadline after which the answer is worthless?** Yes → design the *default/fallback* answer first, then the model path. Ranking systems serve a non-personalized fallback rather than being late; a late answer is strictly worse than a mediocre on-time one.

Question 5 is the cultural difference that surprises LLM engineers most. In chat serving, slow-but-correct beats fast-but-degraded. In ranking, fraud, and ads, **the timeout and its default are the product**, and the model is a best-effort improvement over them.

---

## Do this now (30 minutes)

1. **Fill in the four numbers for a workload you don't work on.** Pick one: your company's search ranker, a content-moderation vision model, or a voice assistant. Estimate FLOPs/request, budget, per-request state, QPS. Then write which of continuous batching, PagedAttention, prefix caching, and tensor parallelism apply. Most people's answer is "none of them," and that realization is the point of the phase.
2. **Compute the budget ratio for your own Phase 3/7 server**: per-request compute time ÷ end-to-end latency. If it is below 0.5, your next optimization is not in the model.
3. **Write the tail-amplification number** for a 300-way fan-out where each call has p99 = 5 ms and p50 = 1 ms. Estimate the expected maximum. Then write what you would do about it in 10 ms. (Answers in [lesson 2](02-recommendation-and-ranking.md); try first.)

---

**Next:** [Recommendation and ranking inference →](02-recommendation-and-ranking.md) — the opposite corner of the workload space: a million QPS, ten milliseconds, terabytes of embeddings, and a model that is 11% of the latency budget.
