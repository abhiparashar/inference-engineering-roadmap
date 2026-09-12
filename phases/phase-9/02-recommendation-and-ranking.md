# 2 — Recommendation and Ranking Inference

> **You'll be able to say:** "Ranking is a funnel — retrieve thousands, score hundreds, rank tens — inside a single-digit-millisecond budget where the dense forward pass is barely 10% of the time. The state is terabytes of embedding tables, sharded across hosts and mostly read; the bottleneck is the fan-out feature fetch, whose p99 is the expected *maximum* of hundreds of lookups, not the p99 of one. So the engineering is cache hierarchy, hedged and cancellable requests, kernel-launch elimination for tiny models, and a non-personalized default that serves when the budget runs out."

This is the largest inference workload on earth by request count, and it predates LLM serving by a decade. Ads, feeds, search ranking, fraud scoring, "customers also bought" — all the same shape. If you understand only LLM serving, this lesson is the largest single gap in your knowledge, and it is the one a recsys/ads interviewer will find in four minutes.

---

## The funnel: four stages, four budgets

Nobody scores a catalog of 10⁸ items per request. Ranking is a cascade of progressively more expensive models over progressively fewer items:

```
  CANDIDATE GENERATION (retrieval)      10^8 items → 10^3-10^4
  ├─ inverted index, ANN over embeddings (lesson 6), heuristics, several
  │  sources unioned; each source has its own timeout
  └─ budget ≈ 1-3 ms       cost/item: nanoseconds

  FILTERING                              10^4 → 10^3
  ├─ eligibility, blocklists, dedupe, already-seen, geo/legal
  └─ budget ≈ 0.5 ms       cost/item: a hash lookup

  SCORING (the "ranking model")          10^3 → 10^3 scores
  ├─ DLRM-style: embedding gather + MLP; ONE batched forward pass
  └─ budget ≈ 1-4 ms       cost/item: ~1 MFLOP

  RE-RANKING / BLENDING                  10^3 → 10-50 shown
  ├─ diversity, business rules, ad auction, freshness, pacing
  └─ budget ≈ 0.5-1 ms     cost/item: business logic, not math
```

Two engineering facts follow immediately:

- **Cost per item must fall by orders of magnitude as you go down the funnel.** A model that is 10× better but 100× more expensive belongs one stage later, over 100× fewer items. "Where in the funnel does this model go?" is *the* recsys design question.
- **Each stage needs its own timeout and its own degraded output.** Candidate generation that times out returns the sources that answered. Scoring that times out returns items in retrieval order. The request always answers.

---

## The model: dense compute is trivial, tables are terabytes

DLRM (Meta's *Deep Learning Recommendation Model*) is the canonical public shape, and essentially every industrial ranker is a variation:

```
   dense features            sparse/categorical features
   (counts, ratios,          (user_id, item_id, page_id, ad_id,
    age, price, CTRs)         last-50-items-clicked, ...)
        │                              │
        ▼                              ▼
   ┌─────────┐              ┌──────────────────────────────┐
   │ bottom  │              │ EMBEDDING TABLES              │
   │  MLP    │              │ one per sparse feature        │
   │ ~1 MFLOP│              │ rows: 10^6 - 10^9 each        │
   └────┬────┘              │ dim: 16-128                   │
        │                   │ TOTAL: 100 GB - 10 TB         │
        │                   │ op: GATHER (+ pooling)        │
        │                   └──────────────┬───────────────┘
        └────────────┬──────────────────────┘
                     ▼
              feature interaction  (dot products / cross layers)
                     ▼
                  top MLP  →  score
```

The parameter counts are wild in a way LLM intuition mishandles: **99.9% of parameters, and ~0% of the FLOPs.** A 1 TB model whose forward pass is 1 MFLOP. Arithmetic intensity below 1 — it is a *lookup* workload.

| Aspect | LLM (70B) | Ranker (1 TB table, 5 MFLOP dense) |
|---|---|---|
| Parameters | 70B, all read every token | 250B+, ~0.001% read per request |
| Per-request FLOPs | 10¹³+ | 10⁶-10⁸ |
| Dominant hardware resource | HBM bandwidth | **DRAM capacity + network** |
| Parallelism strategy | shard the *compute* (TP/PP) | shard the *data* (embedding tables) |
| Accelerator utilization | 40-70% achievable | often < 10%; launch-overhead-bound |
| What a 2× speedup requires | fewer weight reads | fewer/faster lookups, fewer kernel launches |

### Sharding embedding tables

You cannot fit terabytes in GPU HBM, so tables are partitioned. The vocabulary you need:

| Strategy | How | When |
|---|---|---|
| **Table-wise** | whole tables on different hosts/GPUs | many medium tables; simplest |
| **Row-wise** | one huge table split by row hash | a single table too big for one host |
| **Column-wise** | split the embedding dimension | wide embeddings, balances skewed tables |
| **Data-parallel dense + model-parallel sparse** | the standard hybrid: MLPs replicated, tables sharded | essentially every real system |
| **Hierarchical cache** | hot rows on GPU HBM, warm in host DRAM, cold on SSD/remote | Zipfian access — usually true |

The hybrid layout is why a ranking forward pass contains an **all-to-all** collective ([Phase 6 lesson 2](../phase-6/02-collectives-and-interconnects.md)): each host gathers the rows it owns for every request in the batch, then exchanges them so each request's full feature vector lands where its dense compute runs. Recsys is a distributed-systems problem wearing an ML hat; `facebookresearch/dlrm` and `pytorch/torchrec` are where to read the real code.

**Embedding quantization is the highest-leverage optimization here** ([Phase 4 lesson 2](../phase-4/02-quantization-fundamentals.md)): INT8 or INT4 rows cut table size 4-8×, which moves a table from "remote host" to "local DRAM" or from DRAM to HBM. The accuracy cost is usually tiny because embeddings are trained to be robust and the downstream MLP re-normalizes. Cutting a lookup's *network hop* is worth far more than cutting its arithmetic.

---

## The real bottleneck: fan-out and tail amplification

A scoring request needs hundreds of features from multiple stores: a user-feature store, item features, counters, real-time session state, a graph service. That is a parallel fan-out, and this is the arithmetic that defines the workload.

```
  ONE lookup:  p50 = 0.8 ms, p99 = 5 ms, p999 = 30 ms
  Issue N of them in parallel and wait for ALL:
      P(all fast) = (1 - 0.01)^N
      N = 10   → 90% of requests have no slow lookup
      N = 100  → 37%
      N = 300  → 5%     ⇒ 95% of your requests contain a p99 lookup
      N = 1000 → 0.004%
  Your request p50 ≈ the lookup p99.  Your request p99 ≈ the lookup p999+.
```

**The p99 of a 300-way fan-out is governed by the tail of the individual call, not its median.** Answer to lesson 1's exercise: with p99 = 5 ms and 300 calls, the expected maximum sits near or above 5 ms, so a naive implementation blows a 10 ms budget on a store whose "average latency is under a millisecond" — an observation that is the subject of Dean & Barroso's *The Tail at Scale*, required reading for this lesson.

The countermeasures are statistical, not algorithmic:

| Technique | Mechanic | Cost |
|---|---|---|
| **Hedged requests** | after p95 elapses, send a duplicate to another replica; take the first answer | ~5% extra load for a large tail cut |
| **Tied requests** | send to two replicas, each cancels the other's queued copy on start | more complex, less waste |
| **Request cancellation** | hard-cancel in-flight work at the deadline, everywhere, including transitively | must be plumbed end to end or you burn capacity on abandoned work |
| **Batched/multi-get RPCs** | 300 keys in 3 requests, not 300 | fewer draws from the tail; the single biggest win |
| **Deadline propagation** | every hop receives the *remaining* budget, not a fixed timeout | prevents downstream work that can never be used |
| **Partial results** | return the feature vector you have at T−1 ms, with missing features imputed | needs a model trained to tolerate missing features |
| **Co-location** | put hot features in-process or on the same host | removes the hop entirely |

That last row generalizes: **the fastest lookup is the one that doesn't cross a network boundary.** Hence the cache hierarchy.

```
  L0  in-process LRU (hot items, last-N-seconds)       ~100 ns   hit 40-70%
  L1  host-local cache / shared memory                 ~1 µs     hit 10-20%
  L2  same-rack memory store (Redis/memcached-like)    ~200 µs   hit 10-30%
  L3  sharded feature store / remote DRAM              ~1-3 ms   the rest
  L4  cold storage (SSD, blob)                         10+ ms    never on the online path
```

Item features are extremely Zipfian (a few thousand items get most impressions), so L0/L1 hit rates are high; **user** features are not, which is why per-user state usually gets fetched once per request and cached for the session rather than per candidate.

---

## Why batching is free, and what replaces it as a worry

One request brings 1000 candidates. That *is* the batch — it exists before the request is enqueued, so there is no latency-versus-throughput dial to agonize over ([Phase 3 lessons 2-5](../phase-3/02-static-batching.md) become a non-problem). Dynamic batching across *requests* is still used at low QPS to fill the accelerator, with a 1-2 ms max queue delay — exactly Triton's dynamic batcher ([Phase 5 lesson 7](../phase-5/07-triton-inference-server.md)), which exists for this workload, not for LLMs.

What replaces batching as the compute-side worry is **per-op overhead**:

| Problem | Why it dominates here | Fix |
|---|---|---|
| Kernel-launch overhead | the model is dozens of tiny ops, each < 10 µs of work | **CUDA graphs** ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)) — often 2-5× on a ranker |
| Framework/Python overhead | 50-200 µs of interpreter per call vs 1 ms total budget | compile/export the graph; serve from C++ (TorchScript, ONNX Runtime, TensorRT) |
| H2D transfer per request | PCIe latency (~10-20 µs) plus sync | pinned memory, batch the transfer, or stay on CPU |
| Variable shapes | candidate counts differ per request → recompile/replan | bucket shapes (pad to 256/512/1024) — the same trick XLA forces in [lesson 5](05-hardware-diversity.md) |
| Many small embedding gathers | one kernel per table | fused/batched embedding kernels (TorchRec, HugeCTR) |

**CPU vs GPU is a genuinely open question in ranking**, unlike LLM serving. A dense MLP of a few MFLOPs at batch 1000 runs comfortably on a handful of CPU cores, avoids PCIe and the launch overheads entirely, and lets the embedding tables live in the same DRAM you were already paying for. Many production rankers are CPU-only for precisely this reason; GPUs win once the dense part grows (transformer-based rankers, multi-task towers, large sequence features) or when the hot tables fit in HBM. Decide it by measuring cost per million requests at your p99, not by defaulting either way ([lesson 5](05-hardware-diversity.md)).

---

## The two-tower / retrieval pattern

Candidate generation usually uses a **two-tower** model: a user tower and an item tower trained so that dot product approximates relevance.

```
  OFFLINE (hours):  item tower over the whole catalog → 10^8 vectors → ANN index
  ONLINE (1 ms):    user tower on the request → 1 vector → top-k ANN search
```

This is the structural trick that makes retrieval affordable: the expensive side is precomputed and indexed; online work is one small forward pass plus an ANN query ([lesson 6](06-vector-search-and-ann.md) is that query in full detail). It also creates the recsys-specific operational hazards:

- **Index freshness.** New items are invisible until indexed. "Why is my just-uploaded video getting no traffic" is an *index pipeline* bug, not a model bug.
- **Tower skew.** Retrain the item tower, rebuild the index, and forget to ship the matching user tower → the dot products are meaningless. Both towers must be one versioned artifact ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)). This is the recsys version of the tokenizer/weights mismatch.
- **Training/serving feature skew.** The single most common quality bug in the field: a feature computed one way in the offline training pipeline and another way online (different windows, different null handling, different clock). It produces a model that is excellent offline and mediocre in production, with no error anywhere. The only real defenses are a shared feature-definition layer (the reason feature stores exist) and logging the *served* feature vectors for training.

---

## Metrics and SLIs that differ from LLM serving

| SLI | Definition | Typical target |
|---|---|---|
| p99 / p999 end-to-end | hard deadline; p999 matters because fan-out means every user hits it | 10-30 ms / 50-100 ms |
| **Timeout-default rate** | fraction of requests served by the fallback path | < 0.1%, alerted |
| **Feature coverage** | fraction of requested features actually present in the scored vector | > 99.5%; drops = silent quality loss |
| Candidate recall | fraction of the ideal top-k retrieval actually returned | tracked per source |
| Cost per 1M requests | the denominator that replaces "per 1M tokens" | your unit economics |
| Staleness | age of embeddings/counters/index at serve time | seconds to hours, per feature |

**Feature coverage is the recsys analogue of the chat-template regression** from [Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md): a store degrades, features go missing, the model happily scores a partially-zeroed vector, latency looks great, errors are zero, and revenue drops 3%. Nothing in a standard SRE dashboard catches it. You must instrument coverage per feature group and alert on it.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| p99 far above p50, model time flat | fan-out tail amplification | multi-get, hedging, deadline propagation, partial results |
| Latency fine, engagement down | feature coverage drop or training/serving skew | per-feature coverage metric; log served vectors |
| Accelerator at 8% utilization | launch-overhead-bound tiny model | CUDA graphs, op fusion, larger batch, or move to CPU |
| Periodic multi-ms latency spikes | GC pause / cache refresh / index swap | off-heap or tuned GC, incremental index swap, warm the new index before cutover |
| One shard hot, others idle | key skew (a celebrity item, a bot user) | row-wise re-sharding, hot-row replication, in-process cache for the head |
| New items get no impressions | index staleness / cold start | incremental indexing, exploration budget, content-based fallback embedding |
| Scores shift after a deploy | tower/index version skew | ship towers + index as one digest-pinned artifact |
| Fleet cost grows superlinearly with QPS | cache hit rate falling as catalog grows | re-size L0/L1, shard by access pattern, quantize embeddings |
| Retry storms during a store blip | no deadline propagation, unbounded retries | retry budgets + circuit breakers ([Phase 7 lesson 7](../phase-7/07-reliability-and-degradation.md)) |

---

## What to read and where to look in real code

- **Paper:** *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (Naumov et al., Meta) — the reference architecture; read the memory/compute split discussion.
- **Paper:** *The Tail at Scale* (Dean & Barroso, CACM 2013) — hedged/tied requests; this lesson's core latency math.
- **Paper:** *Deep Neural Networks for YouTube Recommendations* (Covington et al.) — the canonical candidate-generation-plus-ranking funnel.
- **Blog:** [Uber's Michelangelo](https://www.uber.com/blog/michelangelo-machine-learning-platform/) — the platform view: feature store, online/offline consistency, model serving as a fleet service.
- **Code:** `pytorch/torchrec` — `EmbeddingBagCollection`, sharding plans, and the all-to-all in `torchrec/distributed/`. The clearest public code for "shard the data, replicate the compute."
- **Code:** `facebookresearch/dlrm` — smaller and readable; see how tables are split across devices.
- **Code:** NVIDIA `HugeCTR` / Merlin — GPU embedding cache hierarchy (HBM → DRAM → SSD), if you want the GPU-centric end of the design space.

---

## Do this now (60 minutes)

1. **Build the funnel budget on paper** for a 20 ms service: assign milliseconds to retrieval, filtering, scoring, re-ranking, and slack; then write the degraded output for each stage when its timeout fires. This table is a complete answer to the most common recsys system-design interview question.
2. **Measure tail amplification for real.** Write a script that issues N parallel lookups against a local Redis/dict service with artificial 1% × 20 ms delay injected, for N = 1, 10, 100, 300. Plot request p50/p99 versus N. Then batch them into multi-gets of 50 and re-plot. The two curves are the lesson.
3. **Make a tiny model overhead-bound, then fix it.** Take an MLP of ~5 MFLOP, run it at batch 1000 in eager PyTorch on any device, and time it. Then apply `torch.compile` and (on CUDA) CUDA graph capture. Record the speedup and the fraction of original time that was pure launch overhead — typically 60-80%.
4. **Price CPU vs GPU** for that model at your measured batch size: cost per million requests on a CPU instance versus the cheapest GPU instance, including the utilization you can realistically reach. Keep the table for [lesson 5](05-hardware-diversity.md).

---

**Next:** [Vision and video serving →](03-vision-serving.md) — the workload where the model is genuinely compute-bound, and where you will still find that JPEG decode is 60% of the latency.
