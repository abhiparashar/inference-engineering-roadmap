# 11 — Build: CPU/GPU Serving Shootout + RAG Service with Guardrails

> **You'll be able to say:** "I built both. Project 16: one model on three runtimes with honest latency, throughput, accuracy and cost-per-unit-work numbers, and a crossover QPS where GPU becomes cheaper than CPU. Project 17: a RAG service with a per-stage budget table, a timeout and fallback on every stage, a multi-dimension token-bucket limiter, and an injection guardrail I attacked myself and documented the bypasses of. Both end in a numbers table, and both survived me deliberately breaking a dependency."

Two projects, two afternoons each if you are efficient. They are deliberately complementary: project 16 is the *hardware and format* half of Phase 9 ([lessons 5](05-hardware-diversity.md), [8](08-model-formats-and-runtimes.md)); project 17 is the *pipeline, retrieval and security* half ([lessons 6](06-vector-search-and-ann.md), [9](09-security-and-multi-tenancy.md), [10](10-rag-and-agentic-serving.md)). Project 17 is also the direct precursor to [capstone 21](../phase-10/07-capstone-production-rag.md).

Everything here runs on a laptop. A single rented GPU-hour makes project 16's comparison complete; without it, compare CPU runtimes against your recorded Phase 3/4 GPU numbers.

---

# Project 16 — CPU-only serving shootout

**Goal:** one model, three runtimes, one honest table, one decision.

## Step 1 — Pick the model and fix the benchmark harness first

Pick **one** track:

| Track | Model | Runtimes to compare |
|---|---|---|
| **LLM track** | a 1-8B instruct model | PyTorch GPU FP16 (or MPS) · `llama.cpp` GGUF Q4_K_M CPU · `llama.cpp` GGUF Q8 CPU |
| **Vision/tabular track** | ResNet-18/50 or a small ViT | PyTorch GPU FP16 · ONNX Runtime CPU FP32 · ONNX Runtime CPU INT8 (static) |

Write the harness **before** collecting any number, reusing the [benchmarking playbook](../../playbooks/benchmarking.md) and [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md):

- Fixed input set (same prompts/images for every runtime), fixed output length for LLMs (`max_tokens`, `ignore_eos` if available — variable output length invalidates comparisons).
- Warmup runs discarded; ≥ 10 measurement runs; report p50/p99 **and σ**.
- Record: runtime version, quantization, thread count, batch size, host CPU model, CPU frequency governor, GPU model/driver.
- One command reproduces everything; output is a CSV you commit.

**Acceptance:** `bench.py --runtime X --batch B` produces a row; ten identical invocations give σ < 5% of the mean. If σ is larger, fix the environment (thermal throttling, other processes, frequency scaling) before proceeding — this is the same discipline that makes a CI performance gate possible ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)).

## Step 2 — Convert and verify (not just convert)

Follow [lesson 8](08-model-formats-and-runtimes.md)'s build-step discipline:

```
  convert.py  →  artifact + metadata.json
    metadata.json: source digest, tool versions, opset/quant type,
                   dynamic axes/buckets, calibration-set digest,
                   measured tolerances
  verify.py   →  MUST PASS BEFORE BENCHMARKING
    · loads on the target runtime
    · partition/placement report: zero unexpected CPU fallbacks (if using an accelerator EP)
    · numeric: max|Δ| and mean cosine vs the reference on 200 fixed inputs
    · task: top-1 agreement (vision) or greedy-token agreement + perplexity delta (LLM)
```

For INT8: calibrate on 200-500 samples drawn from the same distribution you benchmark on. Record the accuracy delta; a quantized runtime that is 3× faster and 8% less accurate is not "faster," it is a different product.

**Acceptance:** every artifact you benchmark has a `verify.py` pass recorded, with numbers. An unverified artifact does not enter the table.

## Step 3 — Sweep and measure

For each runtime: batch sizes 1, 2, 4, 8, 16, 32 (LLM: concurrency 1, 2, 4, 8, 16 instead), and thread counts 1, 4, physical-core-count, 2× cores for CPU runtimes.

```
  runtime        quant  threads  batch  p50(ms)  p99(ms)  thr(u/s)  mem(GB)  acc
  torch-cuda     fp16      -       1      ...      ...      ...       ...    ref
  torch-cuda     fp16      -      32      ...
  llama.cpp      Q4_K_M    8       1      ...
  llama.cpp      Q4_K_M    8       4      ...
  onnxrt-cpu     int8      8      32      ...
```

Things to record while you are there, because they are the interesting findings:

- **The thread-count knee.** CPU throughput usually peaks at physical cores and *degrades* beyond it; oversubscription with hyperthreads often loses. Note whether NUMA pinning (one process per socket) helps.
- **Prefill vs decode split for LLMs.** Report TTFT for a 1000-token prompt separately from tokens/sec. CPU decode is often tolerable while CPU prefill is disqualifying — that asymmetry is the real finding of this project ([lesson 5](05-hardware-diversity.md)).
- **Memory high-water mark**, because on CPU the model may be huge and that is the point.

## Step 4 — Convert to money, and find the crossover

```
  cost per 1M units = ($/hour ÷ (sustained_units_per_sec × 3600)) × 1e6
  Do it twice: at 80% utilization (busy service) and 30% (typical reality).

  Then, for a given QPS, total hourly cost:
    CPU:  ceil(QPS / cpu_units_per_sec_per_instance) × $cpu_per_hour
    GPU:  ceil(QPS / gpu_units_per_sec_per_instance) × $gpu_per_hour
  Plot both vs QPS. The intersection is the crossover QPS.
```

**Acceptance for project 16:**

- [ ] A table with ≥ 3 runtime configurations, each with p50/p99/throughput/memory/accuracy and σ.
- [ ] Every artifact has a recorded verification (numeric tolerance + task metric).
- [ ] Prefill/TTFT reported separately from steady-state throughput (LLM track).
- [ ] A cost-per-1M-units table at two utilization levels, with real instance prices cited.
- [ ] A crossover-QPS plot or number.
- [ ] One paragraph: which runtime you would ship for a named workload, and the single measurement that would change your mind.
- [ ] `projects/16-cpu-vs-gpu-serving/README.md` containing all of the above, and the commit that reproduces it.

---

# Project 17 — Mini RAG service with budgets, limits, and guardrails

**Goal:** a service whose *engineering* is the deliverable, not its answer quality. Corpus size is irrelevant (1k-100k chunks is plenty); the per-stage budget table, the fallback on every stage, the limiter and the attacked guardrail are the point.

## Step 1 — Budget first, code second

Write the nine-row table from [lesson 10](10-rag-and-agentic-serving.md) for your own SLO before writing a line of pipeline code: stage, budget, timeout, fallback. Commit it as `BUDGET.md`. Every later step implements one row.

## Step 2 — Corpus, chunking, index

```
  ingest.py:  documents → chunks (size/overlap recorded) → embeddings → index
              write index_manifest.json:
                embedder model + revision, chunk size/overlap, metric,
                normalization, index type + params, corpus digest, built_at
```

Requirements:

- **Exact search baseline.** Build a flat index too; you need it to measure recall ([lesson 6](06-vector-search-and-ann.md)).
- **The four-artifact bundle**: embedder, index, chunking config, prompt template versioned together with one digest. Assert at startup that the query embedder matches `index_manifest.json` and **refuse to start** on mismatch — the fail-closed pattern from [Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md).
- **Per-tenant partitioning** if you include a tenant concept (recommended: two tenants, so you can test isolation).

## Step 3 — The pipeline, with a timeout and fallback per stage

```python
# shape only — the requirements are the structure, not this code
async def handle(req, deadline):
    with stage("guardrail.input", budget=50, deadline=deadline) as s:
        verdict = await guard_input(req.query)          # fallback: allow + log
    with stage("embed", budget=120, deadline=deadline) as s:
        try:    qv = await embed(req.query)
        except (TimeoutError, DependencyError):
            s.fallback("bm25_only"); qv = None         # rung 3
    with stage("retrieve", budget=200, deadline=deadline) as s:
        docs = await (ann_search(qv, k=20) if qv is not None
                      else bm25_search(req.query, k=20))
    with stage("rerank", budget=250, deadline=deadline) as s:
        try:    docs = await rerank(req.query, docs)[:5]
        except (TimeoutError, DependencyError):
            s.fallback("ann_order"); docs = docs[:5]    # rung 1
    ctx = assemble(docs, budget_tokens=2000)            # record context_tokens
    return stream_llm(ctx, req.query, deadline=deadline)  # rung 5/6 on failure
```

Non-negotiables in the implementation:

1. **Deadline propagation**: the request deadline is computed once at the edge and every stage receives the *remaining* budget; a stage with no remaining budget fails fast instead of starting work.
2. **Stage spans and metrics**: per-stage latency histograms, fallback counters by rung, `context_tokens` histogram, cache-hit counters, and `truncated` rate ([Phase 7 lessons 2, 4](../phase-7/02-instrumenting-with-prometheus.md)).
3. **Prompt ordering for prefix reuse**: static system prompt first, variable content last; no request id or timestamp at the top ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)).
4. **Streaming response** with the guardrail applied to the stream, not only to the finished text.
5. **Retrieval cache** keyed by `(tenant, query_embedding_hash, filters)` — the cheapest, safest cache in the stack.

## Step 4 — The limiter

Implement the multi-dimension token bucket from [lesson 9](09-security-and-multi-tenancy.md):

- Dimensions: requests/min, input tokens/min, **output tokens/min with reserve-and-refund**, and concurrent requests.
- Two tiers (free, paid) with different capacities and refill rates, plus per-request caps (max input tokens, max `max_tokens`, max `n`, max `k`).
- `429` with `Retry-After` and a machine-readable reason naming the exceeded dimension.
- Prove it: a load test that exceeds each dimension in turn and gets a clean 429 for the *right* reason, while a well-behaved client in the other tier keeps being served.

**That last clause is the real test** — isolation, not just rejection. If the abusive tenant's traffic degrades the polite tenant's p99, add per-tenant queues with weighted service ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)).

## Step 5 — The guardrail, and attacking it

1. **Ingest-time scanning**: flag documents containing instruction-like text; tag every chunk with `source` and `trust` level.
2. **Query-time input guardrail**: heuristics (instruction phrases, zero-width/tag characters, base64 blobs, excessive imperative density) with a logged verdict.
3. **Context-time check**: refuse to place a chunk flagged as instruction-bearing into the prompt, or place it with an explicit data frame; log every such decision.
4. **Output handling**: escape model output at the sink, strip/deny external image and link targets (the exfiltration vector), validate that every citation id was actually retrieved.
5. **Attack it.** Write ≥ 10 adversarial inputs and ≥ 5 poisoned documents. Try at minimum: another language; base64/ROT13; zero-width or homoglyph characters; instructions split across two chunks so neither alone looks malicious; instructions inside a code block; instructions in PDF/HTML metadata; a "system:" role-play frame; a markdown image with a query-string exfiltration payload.

**Record a bypass table** — attack, whether it bypassed, what you changed (or why you accepted it). A guardrail with a documented 40% bypass rate plus a least-privilege tool policy is honest engineering; a guardrail claimed to be complete is not. Keep the attacks as a regression suite.

## Step 6 — Break it on purpose

Run a fixed-QPS load test through each of these, and record client-visible behavior:

| Chaos | Expected behavior |
|---|---|
| Kill the vector DB | rung 3 (BM25 or cached) — a degraded answer, not a 500 |
| Add 2 s latency to the reranker | rung 1 (skip rerank), circuit breaker opens, p99 stays inside SLO |
| Kill the LLM backend | rung 6 (return passages with citations) or a clean 503 + `Retry-After` |
| Flood from one tenant | 429s for that tenant only; the other tenant's p99 unchanged |
| Submit a 100k-token input | rejected at the edge before any GPU work |
| Poisoned document retrieved | logged, quarantined or framed; no tool call and no exfiltration link in the output |
| Embedder version mismatch | service refuses to start (fail closed) |

**Acceptance for project 17:**

- [ ] `BUDGET.md`: nine rows, each with budget, timeout, and fallback.
- [ ] Per-stage p50/p99 over ≥ 200 queries, with the stage that owns your p99 named.
- [ ] Recall@k of your shipped index config measured against exact search, plus the end-task quality at 2-3 recall levels.
- [ ] Deadline propagation demonstrated: a request that arrives with insufficient remaining budget fails fast without starting LLM work.
- [ ] Chaos table above, all seven rows, with observed behavior.
- [ ] Limiter evidence: clean 429s per dimension, and tenant isolation under flood.
- [ ] Guardrail bypass table with ≥ 10 attacks, and the regression suite in the repo.
- [ ] Cost per request broken down by stage, showing retrieval versus LLM share.
- [ ] `projects/17-rag-pipeline-guardrails/README.md` with all of the above.

---

## How the two projects come back

| Built here | Reused in |
|---|---|
| Verified conversion + benchmark harness | any future hardware/runtime migration decision ([lesson 5](05-hardware-diversity.md)) |
| Cost-per-unit-work and crossover analysis | the [cost-optimization capstone](../phase-10/05-capstone-cost-optimization.md) |
| Per-stage budget + fallback ladder | the [production RAG capstone](../phase-10/07-capstone-production-rag.md), and every multi-stage pipeline you ever own |
| Limiter + guardrails + bypass suite | the same capstone, plus any public-facing model API |
| Per-stage instrumentation | [Phase 7](../phase-7/README.md)'s dashboards, now with stage attribution |
| Index manifest + fail-closed startup | [Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)'s artifact discipline, applied to retrieval |

Before declaring either project done, run the [production-readiness checklist](../../playbooks/production-readiness-checklist.md) against project 17 and update the status column in [`projects/README.md`](../../projects/README.md).

---

**Next:** [Exercises & exit artifact →](12-exercises-and-artifacts.md) — the drills that make this phase permanent, and the self-check before the capstones.
