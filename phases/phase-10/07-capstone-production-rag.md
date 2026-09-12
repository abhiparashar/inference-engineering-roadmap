# 7 — Capstone Brief: Production-Grade RAG Service

> **Proof obligation:** that you can run a retrieval-plus-generation pipeline as a *service* — per-stage latency budgets with tested fallbacks, measured retrieval quality, token-denominated rate limits with tenant isolation, injection guardrails you attacked yourself, observability that attributes a regression to a stage, and a CI gate that blocks a merge on latency or quality regression.

A **brief, not a build guide**. It is the full-scale version of [Phase 9 lesson 11](../phase-9/11-build-rag-and-cpu-serving.md)'s project 17, with Phase 7's observability and Phase 8's delivery machinery added. Retrieval mechanics: [Phase 9 lesson 6](../phase-9/06-vector-search-and-ann.md). Pipeline discipline: [Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md). Security: [Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md). The bar: [lesson 2](02-engineering-standards.md).

**If you already built project 17, this capstone is that project plus four additions**: real observability, a CI gate, a retrieval-quality evaluation harness, and deployment with a timed rollback. Do not rebuild it from scratch.

---

## The claim to prove

> "A retrieval + generation service holds p95 TTFT ≤ *N* ms at *Q* QPS over a *corpus size* corpus at recall@k = *R*, degrades through a tested ladder when any dependency fails, isolates tenants under abuse, resists *these* injection attacks (and not *those*), and cannot merge a regression past its CI gates. Cost per request is $*C*, of which retrieval is *P*%."

## Scope

| In | Out |
|---|---|
| Ingest → chunk → embed → index (with a manifest) | building a vector database |
| Query pipeline: guardrail → embed → ANN (+ optional rerank) → prompt → LLM stream → output guardrail | agent loops with many tools (a bounded 2-3 step variant is fine, and interesting) |
| Per-stage budgets, timeouts, deadline propagation, fallback per stage | multi-region replication |
| Multi-dimension token-bucket limiter, ≥ 2 tenants/tiers, per-tenant queues | a billing system |
| Retrieval-quality harness: recall@k vs exact, plus end-task eval | a novel retrieval method |
| Prometheus + traces + one diagnosis dashboard | a front end beyond `curl`/a minimal page |
| CI: quality gate + latency gate + recall gate, thresholds from measured σ | full Kubernetes/production platform (nice, not required) |
| Deployment with a timed rollback | fine-tuning the LLM |

---

## The acceptance bar

- [ ] **`BUDGET.md`**: every stage with budget, measured p50/p99, timeout, and fallback rung.
- [ ] **Per-stage latency and cost attribution** over ≥ 500 queries at a stated QPS, with the tail-owning stage named and the retrieval-vs-LLM cost split reported.
- [ ] **Retrieval quality measured**: recall@k against exact search for the shipped index config, plus end-task quality at two or three recall levels — and the resulting choice of recall target justified ([Phase 9 lesson 6](../phase-9/06-vector-search-and-ann.md)).
- [ ] **Four artifacts versioned together**: embedder, index build, chunking config, prompt template, one digest; the service fails closed on mismatch.
- [ ] **Degradation tested**: vector DB down, reranker slow, LLM down, embedder down — each with recorded client-visible behavior, including the retrieval-only rung.
- [ ] **Abuse and isolation**: per-dimension 429s (requests, input tokens, output tokens with reserve-and-refund, concurrency), and proof that a flooding tenant does not move the other tenant's p99.
- [ ] **Guardrail bypass table**: ≥ 10 attacks including indirect injection via a poisoned document, with results and the honest bypass rate; plus output-sink handling (escaping, link/image stripping, citation validation).
- [ ] **Groundedness/citation metric** on a sampled set — the RAG-specific quality SLI, not just latency.
- [ ] **Observability**: per-stage spans and histograms, `context_tokens` and fallback-rung counters, and a dashboard demonstrated diagnosing an injected regression ([Phase 7 lessons 4, 6](../phase-7/04-tracing-and-logging.md)).
- [ ] **CI gates that fire**: a latency regression caught by the perf gate, a prompt/chunking change caught by the quality-or-recall gate, thresholds derived from measured noise ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)).
- [ ] **Timed rollback** of a bad prompt/index/model version, with no rebuild ([Phase 8 lesson 6](../phase-8/06-deploying-and-rollouts.md)).
- [ ] **Cost per 1k requests**, itemized, and **Limitations**, written.

---

## Pitfalls specific to this project

| Pitfall | Symptom | Avoidance |
|---|---|---|
| Optimizing answer quality instead of engineering | a nice demo, no budgets, no gates | the deliverable is the service properties; quality is one measured SLI among several |
| No exact-search baseline | recall unmeasurable, index tuning is guesswork | build a flat index alongside |
| Metadata filter used as a tenant boundary | cross-tenant leak | partition indexes per tenant where data is sensitive |
| Shared semantic cache | wrong answers *and* cross-tenant leakage | tenant + auth scope in every cache key; conservative threshold; log hits |
| Variable content at the top of the prompt | prefix cache never hits, TTFT inflated | static-first prompt ordering, assert hit rate in CI |
| `context_tokens` unmonitored | cost and TTFT silently triple after a chunking change | histogram + alert on p95 |
| Fixed per-hop timeouts | LLM starts a 2 s generation with 200 ms left | deadline propagation end to end |
| Guardrail claimed complete | one demo bypass destroys credibility | publish the bypass rate and pair guardrails with least-privilege |
| Evaluating retrieval and generation together only | can't tell which regressed | separate retrieval eval from end-task eval |

---

## What "excellent" looks like versus "adequate"

| Adequate | Excellent |
|---|---|
| RAG works and answers questions | a budget table where every row's fallback has been tested by breaking the dependency |
| "Uses HNSW" | a recall/latency frontier, the shipped config marked, and the end-task experiment that set the recall target |
| A prompt-injection filter exists | a bypass table with an honest rate, plus blast-radius controls that hold *when* it fails |
| Metrics exposed | a dashboard shown diagnosing an injected per-stage regression |
| Tests pass | two CI gates each catching a regression the other misses |

---

## Where it leads

- This is the most transferable capstone for the current market: nearly every "AI product" backend is this shape, and most are missing four of the acceptance rows above.
- Combine with [lesson 5](05-capstone-cost-optimization.md) for a cost analysis of the same service, and the pair covers both the reliability and the economics conversation.
- Project index entry: **21 — production-grade RAG service** in [`projects/README.md`](../../projects/README.md). Run [`playbooks/production-readiness-checklist.md`](../../playbooks/production-readiness-checklist.md) against it before calling it done.

---

**Next:** [Write-up and portfolio →](08-writeup-and-portfolio.md) — turning a finished system into evidence someone can evaluate in ten minutes.
