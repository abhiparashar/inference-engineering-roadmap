# 6 — Vector Search and ANN

> **You'll be able to say:** "Recall@k is a tunable, not a property of the index — every ANN structure exposes a knob that trades recall for latency, and the job is to hit a recall target at a latency target within a memory budget. I can size HNSW (`M`, `efConstruction`, `efSearch`) or IVF-PQ (`nlist`, `nprobe`, code size) from first principles, compute index RAM before provisioning, measure recall against exact search rather than trusting a benchmark, and name what filtering, deletes and multi-tenancy do to all of it. And I know the number that surprises teams: at RAG scale, retrieval can cost as much as the LLM."

Retrieval sits in front of an enormous fraction of production LLM services and *all* of recsys candidate generation ([lesson 2](02-recommendation-and-ranking.md)'s two-tower pattern). It is also the stage most often treated as a black box, which is why it is the stage that blows the latency budget in [lesson 10](10-rag-and-agentic-serving.md).

---

## The problem, stated honestly

Given a query vector `q` and `N` stored vectors of dimension `d`, return the `k` nearest by cosine/inner-product/L2.

```
  EXACT (brute force):  N × d multiply-adds per query
     N = 1e6, d = 768   →  0.77 GFLOP/query, ~1.5 GB scanned (fp16)
                        →  a few ms on one CPU core-set; GPU: sub-ms
     N = 1e9, d = 768   →  770 GFLOP/query, 1.5 TB scanned  →  impossible online
```

Two facts follow, and they are the whole field:

1. **Below a few hundred thousand vectors, exact search is fine.** `numpy` matmul or FAISS `IndexFlatIP` answers in single-digit milliseconds and has perfect recall, zero tuning, and instant updates. **Most RAG projects do not need an ANN index and adopt one anyway**, importing tuning and freshness problems to solve a performance problem they don't have. Measure exact search first.
2. **Above that, you approximate**, and approximation means you *will* miss some true neighbors. The only question is how many, and whether it matters for your downstream task.

**Recall@k** = (true top-k ∩ returned top-k) / k, measured against exact search on the *same* data. It is not an accuracy metric of your product; it is a property of your index configuration, and it is monotone in the search-effort knob.

---

## The index families

| Family | Structure | Search knob | Memory | Build | Best at |
|---|---|---|---|---|---|
| **Flat (exact)** | none; scan all | — (always 100%) | `N·d·bytes` | none | N < ~10⁵-10⁶, or a recall baseline |
| **HNSW** | multi-layer navigable small-world graph | `efSearch` | vectors + `~N·M·2·4 B` links | slow, CPU-heavy | **default for in-memory, high recall, low latency** |
| **IVF (inverted file)** | k-means partitions; scan `nprobe` of them | `nprobe` | vectors + centroids | fast | large N, tunable, disk-friendly |
| **IVF-PQ / OPQ** | IVF + product-quantized codes | `nprobe`, code size | **`N·m` bytes** (e.g. 64 B/vector) | medium | billion-scale within RAM budget |
| **ScaNN / anisotropic quantization** | learned quantization + partitioning | leaves searched | compact | medium | Google-scale; strong recall/byte |
| **DiskANN / SPANN** | graph on SSD with memory-resident index | beam width | RAM ≈ 5-10% of data | slow | N ≫ RAM, cost-sensitive |
| **LSH** | hash buckets | #tables/probes | large | fast | mostly historical; rarely competitive now |

**In 2025 the practical choice is nearly always: Flat if small, HNSW if it fits in RAM, IVF-PQ or DiskANN if it doesn't.** Everything else is a specialization.

---

## HNSW, concretely

```
  layer 2   ●───────────────●              sparse long-range links
             \             /
  layer 1   ●──●───────●──●                
            │  │       │  │
  layer 0   ●──●──●──●──●──●──●──●         every vector, dense local links
            ▲
            search: enter at top, greedily descend, then beam-search
                    layer 0 with a candidate list of size efSearch
```

| Parameter | Meaning | Typical | Raise it → |
|---|---|---|---|
| `M` | links per node per layer | 16-48 | better recall ceiling; **+memory** (`≈ M·2·4 B/vector`); slower build |
| `efConstruction` | candidate breadth while building | 100-400 | better graph quality; build time up (no query cost) |
| `efSearch` | candidate breadth at query time | 32-512 | **higher recall, higher latency** — the runtime dial |

Properties that matter operationally:

- **`efSearch` must be ≥ k**, and recall grows roughly logarithmically in it: going 64 → 128 might buy 0.5% recall for 2× latency. Find the knee empirically.
- **HNSW is the fastest at high recall (0.95-0.99)** among in-memory structures, which is why it is the default in Qdrant, Weaviate, Milvus, pgvector (`hnsw`), Elasticsearch/OpenSearch and Lucene.
- **Memory = full vectors + graph.** 1M × 768-dim fp32 = 2.95 GB of vectors plus ~0.13 GB of links at `M=16`. Store fp16 or int8 vectors and you halve/quarter the dominant term (see quantization below).
- **Deletes are soft.** HNSW cannot cheaply remove a node from the graph; databases tombstone and periodically rebuild/compact. A workload with heavy churn either accepts degraded recall and growing memory or schedules rebuilds — this is the #1 HNSW operational surprise.
- **Builds are expensive and single-index.** Building a 100M-vector HNSW index is hours of CPU; plan it as a batch pipeline artifact, not something you do in a request path.

## IVF and product quantization, concretely

```
  build:  k-means → nlist centroids (rule of thumb: nlist ≈ 4·√N … 16·√N)
  query:  compare q to all nlist centroids  (cheap)
          scan the nprobe nearest lists     (the cost)
  PQ:     split d into m subvectors, each quantized to 256 codes (1 byte)
          storage per vector = m bytes  (e.g. d=768, m=96 → 96 B, 32× smaller)
          distances computed on codes via lookup tables (SIMD-friendly)
```

| Parameter | Effect |
|---|---|
| `nlist` ↑ | finer partitions: less scanned per probe, but more centroid comparisons and more risk that the true neighbor is in an unprobed cell |
| `nprobe` ↑ | **the recall dial**: linear-ish cost, diminishing recall gains |
| `m` (subquantizers) ↑ | more bytes/vector, better recall; `m` must divide `d` |
| OPQ pre-rotation | a learned rotation before PQ; usually +1-3% recall for free at query time |
| Re-ranking with exact vectors | fetch the top-`R` PQ candidates and rescore exactly → recovers most PQ loss; needs the full vectors somewhere (SSD is fine) |

**The PQ re-rank pattern is the standard billion-scale design:** compressed codes in RAM for the coarse pass, exact vectors on disk for rescoring the top few hundred. It gets you 0.95+ recall at 50-100× compression.

---

## Sizing: do this arithmetic before you provision

```
  raw vectors:      N × d × bytes_per_component
     1e6 × 768 × 4 (fp32)  = 2.95 GB
     1e6 × 768 × 2 (fp16)  = 1.47 GB       ← almost always fine
     1e6 × 768 × 1 (int8)  = 0.74 GB       ← usually ~1% recall loss, verify
     1e6 × 96 B  (PQ, m=96)= 0.10 GB       ← 30× smaller, needs re-rank

  HNSW links:       N × M × 2 × 4 B   (1e6, M=16 → 0.13 GB)
  payload/metadata: often LARGER than the vectors (text chunks, ids, filters)
  replicas:         × (1 + replicas)  for HA and read throughput
  headroom:         × 1.5-2 for compaction/rebuild, or you cannot rebuild at all
```

That last line is where teams get stranded: an index that exactly fills its node cannot be compacted or rebuilt in place. **Budget rebuild headroom from day one.**

Cost framing worth internalizing ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md) listed retrieval as a line item that "can rival the LLM"): a 100M × 768 fp16 HNSW index is ~150 GB of vectors plus links plus payload plus replicas — call it 400-500 GB of *RAM*, which is a five-figure annual bill before a single LLM token is generated. PQ or DiskANN turns that into a four-figure bill with a recall and complexity cost. That tradeoff is the single biggest cost decision in a RAG system.

---

## Measuring recall — the part people skip

```
  1. sample 1,000-10,000 real queries (real, not synthetic: distribution matters)
  2. compute EXACT top-k with a flat index (slow, offline, once)
  3. for each candidate config: run it, compute recall@k, record p50/p99 latency
  4. plot recall vs p99 latency; each config is a point; you want the frontier
  5. pick the cheapest config on the frontier that meets your recall target
```

Non-negotiables:

- **Your own data and your own queries.** Public ANN benchmarks (ann-benchmarks, VectorDBBench) are useful for shortlisting and useless for configuration: recall behavior depends on the intrinsic dimensionality and clustering of *your* embeddings.
- **Report recall with latency, always.** A vendor claim of "sub-millisecond search" without a recall number is meaningless — I can give you 0.1 ms at 30% recall on any index.
- **Choose the recall target from downstream impact, not from 0.99 by reflex.** For RAG with k=20 and a reranker behind it, recall@20 of 0.90 is often indistinguishable in end answer quality from 0.99 while costing half the latency. *Measure the end task*, then set the retrieval target. This experiment is the single most valuable thing in this lesson and almost nobody runs it.
- **Watch p99, not p50.** Graph searches have long tails (unlucky entry points, cold pages). A p50 of 3 ms with a p99 of 60 ms wrecks a pipeline budget.

---

## Filtering: the feature that breaks everything

Real queries are "nearest neighbors **where** `tenant_id = 42 AND lang = 'de' AND published > X`".

| Strategy | How | Failure mode |
|---|---|---|
| **Post-filter** | ANN for top-`k'`, then filter | with a selective filter you may return *nothing*; requires `k'` ≫ k and unbounded retries |
| **Pre-filter** | build the allowed-id set, then exact-search it | great when the set is small; degenerates to brute force when large |
| **Filtered ANN** | filter-aware graph traversal (Qdrant, Milvus, pgvector+HNSW with conditions) | recall degrades as selectivity rises; implementation-specific |
| **Partitioned indexes** | one index per tenant/language/shard | clean and fast; poor when partitions are many and tiny (per-tenant overhead, memory waste) |

**The selectivity crossover is the thing to know:** with a filter passing < ~1% of the corpus, pre-filtering plus exact search usually *wins*. With > ~10%, filtered ANN wins. In the middle, measure. For multi-tenant systems the safest design is **partition by tenant** — it also makes deletes, quotas and isolation tractable ([lesson 9](09-security-and-multi-tenancy.md)).

---

## Freshness, updates, and the rest of the pipeline

The index is one component of a retrieval system; the pipeline around it causes more incidents than the index does.

```
  ingest → chunk → embed → upsert → (index build/merge) → search → rerank → LLM
           ▲        ▲                  ▲                      ▲
           │        │                  │                      └ cross-encoder,
           │        │                  └ segment merge/compaction pauses          
           │        └ EMBEDDING MODEL VERSION = part of the index identity
           └ chunk size/overlap decides what "a neighbor" even means
```

- **The embedding model is part of the index artifact.** Change the model and every stored vector is meaningless against new queries — the same skew as [lesson 2](02-recommendation-and-ranking.md)'s two-tower mismatch. Re-embedding 100M documents is a real project. Version them together and pin by digest ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)); never let a query embedder auto-upgrade.
- **Normalization must match.** Cosine similarity requires L2-normalized vectors; inner-product search on unnormalized vectors ranks by magnitude. A mismatch between ingest and query is a silent quality bug with no error.
- **Chunking is a retrieval hyperparameter**, and usually a bigger quality lever than the index: 200-800 token chunks with 10-20% overlap is the common range, and it should be tuned against the end task.
- **Hybrid retrieval (BM25 + vectors) beats either alone** on most real corpora, fused by Reciprocal Rank Fusion or a learned reranker. Keyword search still wins on exact identifiers, names, and rare tokens — which is precisely what users type.
- **A cross-encoder reranker over the top 50-100** is the highest-quality-per-millisecond addition to a retrieval stack: it lets you lower ANN recall (cheaper search) and recover quality. It is also a GPU or CPU inference workload of its own with its own budget — commonly 10-30 ms, which must be in the pipeline arithmetic.
- **Compaction/merge pauses** cause p99 spikes and RAM spikes; schedule them, monitor them, and never let them coincide with peak traffic.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Good p50, terrible p99 | graph tail / cold pages / merge running | pin memory, schedule compaction, cap `efSearch`, add replicas |
| Recall collapsed after adding a filter | post-filtering on a selective predicate | partitioned indexes or pre-filter + exact |
| Quality dropped, no code change | embedder version drift, or normalization mismatch | pin embedder with the index; assert vector norms at ingest |
| Memory grows without new data | tombstoned deletes never compacted | scheduled rebuild/compaction with headroom |
| Index build OOMs | no rebuild headroom, or `efConstruction`/`M` too high | build on a bigger box offline; ship the built index as an artifact |
| "Nearest" results are nonsense | wrong metric (L2 vs IP vs cosine) for the model | use the metric the embedding model was trained with |
| Retrieval cost rivals the LLM bill | full-precision in-RAM index at scale | int8/PQ vectors, DiskANN, or fewer replicas with a cache |
| Recall fine offline, bad in production | offline eval used synthetic queries | re-measure with sampled production queries |
| New documents unfindable for hours | batch-only index builds | incremental upserts plus a small "fresh" index searched in parallel |
| Multi-tenant noisy neighbor | one huge tenant in a shared index | per-tenant partitions and quotas ([lesson 9](09-security-and-multi-tenancy.md)) |

---

## What to read and run

- **Papers:** *Efficient and robust approximate nearest neighbor search using HNSW* (Malkov & Yashunin); *Product Quantization for Nearest Neighbor Search* (Jégou et al.); *ScaNN* (Guo et al.); *DiskANN* (Subramanya et al.). Read HNSW and PQ — they are short and they explain every knob in this lesson.
- **Code:** `facebookresearch/faiss` — `IndexFlatIP`, `IndexHNSWFlat`, `IndexIVFPQ`; the `index_factory` strings are a compact language for everything above. `qdrant/qdrant` for a production Rust implementation with filtered search and per-collection quantization.
- **Docs:** engineering blogs from Pinecone/Qdrant/Weaviate on recall-vs-latency tradeoffs, read with "what recall did they hold fixed?" in mind.

---

## Do this now (60 minutes — this feeds project 17)

1. **Build the frontier.** Embed 100k-1M real text chunks (any small embedding model). Build: `Flat`, `HNSW(M=16, efSearch ∈ {32,64,128,256})`, `IVF(nlist=4√N, nprobe ∈ {1,4,16,64})`, `IVFPQ(m=d/8)`. Measure recall@10 against Flat plus p50/p99 latency for each. Plot the frontier and mark the config you'd ship.
2. **Do the RAM arithmetic** for your configs at N = 1M, 10M, 100M, with fp32/fp16/int8/PQ vectors, plus payload and 2 replicas and rebuild headroom. Convert to $/month at your cloud's memory price. This table is why people use PQ.
3. **Run the experiment nobody runs:** hold the LLM fixed and measure *end-task* answer quality at recall 0.99, 0.95, 0.90, 0.80. Find where quality actually degrades. Then set your retrieval target from that, not from reflex.
4. **Break filtering.** Add a metadata filter passing 50%, 5%, and 0.5% of the corpus. Measure recall and latency for post-filter vs pre-filter-and-exact. Record the crossover.

---

**Next:** [Edge and on-device inference →](07-edge-and-on-device.md) — the same engineering discipline with the budget in megabytes and milliwatts, on hardware you do not control and cannot log into.
