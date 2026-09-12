# 4 — Capstone Brief: Multi-Modal Production Pipeline

> **Proof obligation:** that you can compose several heterogeneous models into one service that holds an end-to-end latency budget, degrades instead of failing when a stage breaks, scales its stages independently, and is observable enough to diagnose a regression from dashboards alone.

A **brief, not a build guide**. Composition mechanics: [Phase 5 lessons 7-8](../phase-5/07-triton-inference-server.md). Stage budgets and fallbacks: [Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md). Per-workload engineering: [Phase 9 lessons 3-4](../phase-9/03-vision-serving.md). Observability: [Phase 7](../phase-7/README.md). Delivery: [Phase 8](../phase-8/README.md). The bar: [lesson 2](02-engineering-standards.md).

---

## The claim to prove

> "A three-stage pipeline (e.g. speech → LLM → speech, or image → detection → LLM captioning/moderation) holds a *p99 = N ms* end-to-end budget, with every stage independently budgeted, timed out, scaled and instrumented — and it degrades gracefully through *these* rungs when any single stage fails or slows."

Pick **one** shape and do it properly:

| Shape | Stages | Hard part |
|---|---|---|
| Voice assistant | streaming ASR → LLM → streaming TTS | the turn-taking budget; partials; barge-in cancellation ([Phase 9 lesson 4](../phase-9/04-speech-and-streaming.md)) |
| Visual moderation/captioning | decode/preprocess → detector → LLM | preprocessing is the bottleneck, not the model ([Phase 9 lesson 3](../phase-9/03-vision-serving.md)) |
| Document understanding | OCR/layout → embedding → LLM | throughput mode plus a per-page cost model |

## Scope

| In | Out |
|---|---|
| Three real stages with different hardware needs and different natural batch sizes | training or fine-tuning any of the models |
| A composition layer that owns budgets, timeouts, retries, cancellation (Ray Serve graph, Triton ensemble, or your own async orchestrator) | a polished UI |
| Per-stage autoscaling or at least per-stage replica sizing from measured capacity | multi-region, multi-cluster |
| Full observability: per-stage spans, per-stage histograms, one diagnosis dashboard | training-pipeline orchestration |
| A degradation ladder with a tested rung per stage | exhaustive model/quality tuning |
| Streaming end-to-end where the shape demands it | more than one pipeline shape |

**The most common scoping error** is treating this as "three models behind one endpoint." The deliverable is the *engineering between* the models: budget arithmetic, timeouts, cancellation, backpressure, and scaling ratios.

---

## The acceptance bar

- [ ] **Budget table**: every stage with budget, measured p50/p99, timeout, and fallback — the nine-row artifact from [Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md), adapted.
- [ ] **End-to-end p99 at a stated offered load**, measured open-loop from the client, with the stage that owns the tail named.
- [ ] **Stage capacity ratios derived from measurement**: e.g. "1 ASR replica per 3 LLM replicas per 2 TTS replicas at 50 concurrent sessions," with the measurement that produced it ([Phase 8 lesson 5](../phase-8/05-scheduling-capacity-and-autoscaling.md)).
- [ ] **Backpressure proven**: overload one stage and show the pipeline sheds or queues in a bounded way instead of collapsing; the mismatched-capacity failure (fast producer, slow consumer) must be visible in a metric.
- [ ] **Cancellation proven**: client disconnect or barge-in cancels *all* downstream in-flight work; show the resource release in metrics, not just in code.
- [ ] **Degradation tested per stage**: kill or slow each stage in turn; record client-visible behavior against the ladder.
- [ ] **One diagnosis dashboard** that answers "which stage is slow right now?" in a single screen ([Phase 7 lesson 6](../phase-7/06-dashboards-and-diagnosis.md)).
- [ ] **Streaming correctness** for streaming shapes: first-audio/first-token latency reported separately from total, no gaps/underruns under sustained load.
- [ ] **Cost per unit** (per conversation minute, per image, per page) with the per-stage breakdown showing which stage dominates — it is frequently *not* the LLM.
- [ ] **Limitations**, written.

---

## Pitfalls specific to this project

| Pitfall | Symptom | Avoidance |
|---|---|---|
| Only end-to-end instrumentation | you can't attribute a regression | one span and one histogram per stage from commit one |
| Serial execution of independent stages | latency 2× worse than necessary | parallelize what doesn't depend; prove it in the trace |
| Uniform replica counts | one stage saturated, others idle | size from per-stage measured capacity |
| No cancellation plumbing | abandoned work keeps burning GPU | propagate cancellation across every stage boundary |
| Stage-local timeouts summing past the deadline | client times out while stages still work | deadline propagation, remaining-budget arithmetic |
| Preprocessing ignored | GPU at 20% while CPUs are pegged | five-stage attribution ([Phase 9 lesson 3](../phase-9/03-vision-serving.md)) |
| Payloads as JSON floats | serialization dominates latency | binary/compressed payloads, gRPC or Triton binary tensors |
| Orchestrating with a glue framework and calling it done | no budgets, no backpressure, no scaling | own the composition layer's contracts explicitly |

---

## What "excellent" looks like versus "adequate"

| Adequate | Excellent |
|---|---|
| Three models chained, it works | a budget table where every row has a tested fallback |
| End-to-end latency reported | per-stage frontier curves and the named tail owner |
| Replicas set by trial and error | capacity ratios derived from measured per-stage throughput |
| "It has Prometheus metrics" | a dashboard that diagnoses a stage regression in one screen, demonstrated on an injected regression |
| A demo video | a chaos table: each stage killed, each observed behavior recorded |

---

## Where it leads

- Directly reusable in the production RAG capstone ([lesson 7](07-capstone-production-rag.md)) — same pipeline discipline, different stages.
- The strongest single artifact for ML-platform and applied-AI roles, because it is the closest thing in this roadmap to what those teams actually operate.
- Project index entry: **19 — multi-modal production pipeline** in [`projects/README.md`](../../projects/README.md).

---

**Next:** [Capstone brief: cost optimization →](05-capstone-cost-optimization.md) — the capstone that is literally an open job description.
