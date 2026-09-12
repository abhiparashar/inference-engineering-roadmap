# 10 — RAG and Agentic Serving

> **You'll be able to say:** "A RAG request is a pipeline, and a pipeline's p99 is not the sum of its stages' p50s — it's dominated by whichever stage has the worst tail. So every stage gets a budget, a timeout derived from that budget, a fallback that produces a usable answer without it, and a cache. An agent loop multiplies the whole pipeline by the number of round trips, which is why unbounded agents cannot hold an SLO and need step limits, deadline propagation, and parallel tool execution. The most common production RAG failure isn't retrieval quality — it's the absence of a degradation path when one dependency is slow."

This is where the phase converges: retrieval ([lesson 6](06-vector-search-and-ann.md)), untrusted content ([lesson 9](09-security-and-multi-tenancy.md)), LLM serving (Phases 3-4), observability and degradation ([Phase 7 lessons 4, 7](../phase-7/04-tracing-and-logging.md)) in one request path. It is also the shape of most "AI product" backends being built right now, which makes it the most immediately employable lesson in Phase 9.

---

## The pipeline and its budget

```
  REQUEST BUDGET: 3000 ms to first token   (a real, stated SLO)

  stage                       budget   p50    p99    timeout  fallback
  ─────────────────────────────────────────────────────────────────────────
  1 auth + rate limit          20 ms     3     15      50 ms  reject (429/401)
  2 input guardrail            50 ms    12     40     100 ms  fail open + log
  3 query embed               120 ms    35     90     200 ms  BM25-only retrieval
  4 ANN search                200 ms    45    180     300 ms  cached/popular docs
  5 metadata filter + fetch   150 ms    40    140     250 ms  drop enrichment
  6 rerank (cross-encoder)    250 ms    80    220     300 ms  skip → use ANN order
  7 prompt assembly            20 ms     5     15      30 ms  —
  8 LLM prefill → TTFT       1500 ms   600   1400    2500 ms  smaller model / cache
  9 output guardrail (stream)  50 ms    10     40     100 ms  fail closed on stream
  ─────────────────────────────────────────────────────────────────────────
  sum of p50  = 830 ms      ← what your demo shows
  sum of p99  = 2140 ms     ← the pessimistic bound, if stages were
                              perfectly correlated (they aren't)
  measured p99 ≈ 1500-1900 ms  ← reality: dominated by stage 8, then 6, 4
  headroom to SLO = ~1100 ms   ← this is what you have to spend on retries
```

Three rules that fall out of that table:

1. **Budget the stages, then derive timeouts.** A stage timeout is `budget × 1.5-2` (room for its own tail), never "30 s because that's the client default." The sum of timeouts must be less than the request deadline, or your timeouts are decorative.
2. **`p99(total) ≤ Σ p99(stage)`, and the bound is loose** unless stages share a common cause (they often do: the same node is slow, the same GC pauses, the same network). Measure the total; use the sum of p99s only as a design bound.
3. **Every stage needs a fallback that yields a usable answer.** "Retrieval failed" is not an answer; "here is an answer without retrieved context, flagged as such" is. Write the fallback column *before* you write the stage.

### Deadline propagation, not per-hop timeouts

```
  client deadline 3000 ms, arrives at t=0
  gateway:   remaining = 3000 → passes 2950 downstream
  retrieval: remaining = 2900 → ANN gets min(300, remaining)
  LLM:       remaining = 2100 → max_tokens capped so generation can finish
  any hop with remaining ≤ 0 fails FAST rather than starting work
```

Fixed per-hop timeouts produce the pathology where stage 8 starts a 2.5-second generation with 200 ms of budget left: the work completes, the client is gone, and you paid for it. Deadline propagation ([lesson 2](02-recommendation-and-ranking.md)'s recsys discipline) is the fix, and it must be plumbed through every client library.

---

## Caching: the biggest lever, four layers deep

| Layer | Key | Hit rate (typical) | Saves | Risk |
|---|---|---|---|---|
| **Exact response cache** | hash(normalized query + filters + model + prompt version + tenant) | 5-30% | everything | staleness; must invalidate on corpus/model/prompt change |
| **Semantic cache** | nearest-neighbor over query embeddings above a threshold | 10-40% claimed | everything | **wrong answers**: "cheapest plan" vs "most expensive plan" are near-neighbors. Tune the threshold conservatively and log every hit |
| **Retrieval cache** | hash(query embedding, filters) → doc ids | 20-50% | stages 3-6 | cheap and safe — the best value in the table |
| **Embedding cache** | hash(text) | high for repeated docs/queries | stage 3 | trivial; just do it |
| **LLM prefix cache** | shared system prompt + retrieved context | engine-level | most of prefill | [Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md) |

Two structural notes:

- **Order your prompt for prefix reuse.** Static system prompt → tenant/persona → retrieved context → user query. Put the *most* stable content first and the most variable last, or prefix caching gets you nothing. A common mistake is injecting a timestamp or request id at the top of the system prompt, which destroys every prefix hit in the fleet.
- **Cache keys must include the tenant and authorization scope** ([lesson 9](09-security-and-multi-tenancy.md)). A shared semantic cache is a cross-tenant data-leak vector, and it is an easy mistake to ship.

---

## Failure isolation and the degradation ladder

RAG adds dependencies (vector DB, embedding service, reranker, tool APIs) to a request path that previously had one. Each is a new way to be down, and the pipeline must degrade rather than fail ([Phase 7 lesson 7](../phase-7/07-reliability-and-degradation.md)'s ladder, specialized):

```
  RUNG 0  full pipeline: embed → ANN → filter → rerank → LLM
  RUNG 1  drop the reranker (use ANN order)                    −250 ms, small quality cost
  RUNG 2  drop enrichment/metadata fetch                       −150 ms
  RUNG 3  degrade retrieval: BM25 keyword search only, or a cached popular set
  RUNG 4  fewer chunks (k: 20 → 5) → shorter prefill           big TTFT win
  RUNG 5  no retrieval: answer from parametric knowledge, clearly flagged
  RUNG 6  retrieval-only: return the passages with citations and NO generation
          (surprisingly acceptable to users, and very cheap)
  RUNG 7  cached/static response, 503 + Retry-After, free tier shed first
```

Rung 6 is the one teams forget and users like: when the LLM is saturated, a list of relevant sources is a real product, not an error page.

**Circuit breakers per dependency.** If the reranker is failing, stop calling it for 30 seconds rather than paying its timeout on every request — a dependency that times out on 100% of calls costs you its full timeout × QPS in latency and capacity. Combine with retry *budgets* (a global cap on retry traffic, e.g. 10% of requests) so a slow dependency cannot cause a self-inflicted retry storm.

---

## Observability: stage spans or nothing

One span per stage, with the attributes that let you answer "why was this request slow?" without re-running it:

```
  trace: rag.request  (3 attrs: tenant, tier, prompt_version)
   ├─ guardrail.input        verdict, rule_hits
   ├─ embed.query            model, tokens, cache_hit
   ├─ retrieve.ann           index_version, k, ef/nprobe, candidates, cache_hit
   ├─ retrieve.fetch         docs, bytes, store
   ├─ rerank                 model, pairs_scored, skipped(reason)
   ├─ prompt.assemble        context_tokens, chunks_used, truncated(bool)
   ├─ llm.generate           model, prefix_cache_hit, prompt_tokens,
   │                         completion_tokens, ttft_ms, tpot_ms, finish_reason
   └─ guardrail.output       verdict
```

| Metric | Why it earns its place |
|---|---|
| Per-stage latency histograms | the only way to attribute a p99 regression to a stage |
| Stage timeout/fallback counters (by rung) | your actual degradation rate — alert on it, it is invisible otherwise |
| `context_tokens` histogram | the dominant driver of TTFT *and* cost; regressions here are silent |
| `truncated` rate | retrieved context silently dropped = quality loss with no error |
| Cache hit rates per layer | directly proportional to your bill |
| Retrieval recall proxy (offline) | quality drift from index/embedder changes ([lesson 6](06-vector-search-and-ann.md)) |
| Citation/groundedness rate | the RAG-specific quality SLI |
| Cost per request, by stage | retrieval can rival the LLM ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md)) |

**`context_tokens` is the sleeper metric.** A chunking change that raises average context from 1500 to 4000 tokens triples prefill cost and doubles TTFT while every functional test passes. Alert on its p95 like you would on latency.

---

## Agentic serving: multiplying the whole pipeline

An agent loop is "LLM decides → tool runs → result appended → repeat." Each iteration is a full prefill over a *growing* context plus a tool round trip.

```
  per step:  prefill(context) + decode(decision tokens) + tool latency
  context grows every step ⇒ prefill cost grows superlinearly in step count

  5-step agent, 2 s per LLM turn, 500 ms per tool:
      5 × (2000 + 500) = 12.5 s   — and that is the GOOD case
  p99 with one slow tool and one retry: 25-40 s
```

What that implies:

| Control | Why |
|---|---|
| **Hard step limit** (5-10) | unbounded loops are unbounded cost; the model will not stop on its own |
| **Global deadline for the whole loop**, propagated | per-step timeouts alone let 8 steps eat 40 s legitimately |
| **Parallel tool execution** | independent tool calls must be concurrent; serial fan-out is the single biggest agent latency bug |
| **Prefix caching across steps** | the conversation prefix is stable and grows by suffix — the ideal case for [Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md); without it you re-prefill everything each step |
| **Context compaction** | summarize or drop old tool outputs; otherwise prefill grows until it dominates |
| **Cheap model for routing, strong model for synthesis** | most steps are classification, not reasoning |
| **Idempotency keys on tool calls** | retries and reruns must not double-charge a card or double-send an email |
| **Cost/step accounting per request** | one agent request can cost 100× a chat request; unmetered agents produce astonishing bills |
| **Stream progress to the user** | 12 s with visible steps is tolerable; 12 s of blank screen is not |
| **Cancellation propagation** | user abandons → cancel LLM generation and in-flight tools, or you keep paying |

**Speculative parallelism** is the advanced move: launch the likely-needed retrieval concurrently with the LLM's planning turn, and discard it if unused. Same tradeoff as speculative decoding ([Phase 4 lesson 7](../phase-4/07-speculative-decoding.md)) — pay extra compute for latency — and only worth it when the prediction is right most of the time.

---

## Quality evaluation, because latency isn't the whole SLO

A RAG service can be fast, cheap, and confidently wrong. Minimum viable evaluation:

| Layer | Measure | Method |
|---|---|---|
| Retrieval | recall@k, MRR/nDCG | fixed query set with labeled relevant docs; run on every index/embedder change ([lesson 6](06-vector-search-and-ann.md)) |
| Grounding | is every claim supported by a retrieved passage? | NLI model or LLM-as-judge over a sampled set; report a rate, not a vibe |
| Answer quality | task-specific correctness | golden set of 100-300 Q/A pairs, scored offline |
| Citations | correct and resolvable? | mechanical check: every cited id exists and was actually retrieved |
| Refusal/abstention | does it say "I don't know" when retrieval was empty? | inject empty-retrieval cases into the golden set |
| Regression gating | CI on prompt/index/model changes | distributional comparison versus baseline ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)) |

**The four artifacts that must be versioned together**, because a change in any one invalidates your evals: embedding model, index build, chunking config, and prompt template. Treat them as one deployable unit with one digest ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)); a prompt edited in a dashboard is an unrecorded deploy.

---

## Security inside the pipeline

Everything in [lesson 9](09-security-and-multi-tenancy.md) applies, but three items are specific to this shape and easy to get wrong:

1. **Retrieved documents are untrusted instructions.** This is the indirect-injection path: whoever can add a document to your corpus can attempt to steer every answer that retrieves it. Scan documents at *ingest* (not just at query time), tag them with provenance, and never let retrieved content authorize a tool call.
2. **Retrieval must be tenant-scoped at the index level.** A metadata filter is a query feature, not a security boundary; a bug in filter construction is a cross-tenant data leak. Partition where the data is sensitive.
3. **Fan-out amplification is a DoS vector.** One request that triggers 10 searches, 10 reranks, and 5 LLM calls is a 100× cost amplifier. Bound fan-out per request and count the *amplified* cost against the tenant's quota, not the request count.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| p99 >> sum of stage p50s | one stage's tail dominates | per-stage p99 dashboards; fix or bound the worst |
| Requests fail entirely when the vector DB is slow | no timeout or no fallback on retrieval | stage timeout + BM25/cached fallback (rung 3) |
| Cost doubled after a "chunking improvement" | `context_tokens` grew | alert on context-token p95; cap chunks and total context |
| TTFT regressed after a prompt change | prefix cache invalidated (variable content moved to the front) | static-first prompt ordering; assert cache hit rate in CI |
| Wrong-but-plausible cached answers | semantic cache threshold too loose | raise the threshold, log hits, add a negative-example test set |
| Answers cite documents that weren't retrieved | prompt/citation plumbing bug or hallucinated ids | mechanical citation validation before responding |
| Agent runs for minutes | no step limit or global deadline | hard caps; propagate the deadline; stream progress |
| Agent costs 100× a chat request | serial tool calls, no compaction, no prefix cache | parallelize, compact, cache; meter per request |
| Cross-tenant content in answers | shared index/cache without tenant scoping | partition indexes; tenant in every cache key |
| Retrieval quality decayed silently | embedder/index/chunking changed independently | version the four artifacts together; recall gate in CI |
| Retry storm during a dependency blip | unbounded retries, no circuit breaker | retry budgets + breakers + deadline checks before retrying |

---

## What to read

- **Papers:** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al.) for the original framing; *ReAct* (Yao et al.) for the tool-loop pattern every agent framework implements.
- **Read with a critical eye:** any "production RAG" blog post — check whether it reports per-stage latency, a fallback per stage, and a recall number. Most report none of the three, which tells you what is missing in the field.
- **Code:** a serving-graph framework rather than an orchestration DSL — Ray Serve deployment graphs ([Phase 5 lesson 8](../phase-5/08-ray-serve-and-composition.md)) or Triton ensembles ([Phase 5 lesson 7](../phase-5/07-triton-inference-server.md)) — because per-stage autoscaling, concurrency and timeouts are the actual production requirements, and they are what LangChain-style glue does not give you.

---

## Do this now (60 minutes — this is project 17's design)

1. **Write the budget table** for a 3-second-TTFT RAG SLO: stage, budget, timeout, fallback. Nine rows, no gaps. This table is the design document and the interview answer.
2. **Instrument a real pipeline per stage.** Even with a 1000-document corpus, produce p50/p99 per stage over 200 queries, and identify which stage owns your p99.
3. **Kill a dependency.** Stop your vector DB mid-load-test and record what your client sees. If it's a 500 rather than a degraded answer, implement rung 3 and re-run. That before/after pair is the proof this lesson asks for.
4. **Measure the prompt-ordering effect.** Run the same pipeline with (a) static system prompt first and (b) a per-request id prepended to the system prompt. Compare TTFT and your engine's prefix-cache hit rate. The delta is free performance most services throw away.

---

**Next:** [Build: CPU/GPU shootout + RAG with guardrails →](11-build-rag-and-cpu-serving.md) — the two projects that turn this phase into artifacts, with explicit acceptance criteria.
