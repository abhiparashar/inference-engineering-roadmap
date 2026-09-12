# 8 — Canary, Shadow Traffic, and Quality Gates

> **You'll be able to say:** "A new engine version, a new quantization, or a new prompt template are all *model changes*, and none of them can be validated by a latency graph. I run shadow traffic to get zero-risk latency/throughput evidence, then a percentage canary with automated analysis on latency, error rate **and** output quality, using a statistically honest comparison: same prompts, fixed seeds where possible, paired sampling, and a difference metric appropriate to sampling nondeterminism. I know why exact-match diffing fails on a temperature>0 model, what to use instead, and how to build a promotion/rollback decision that is arithmetic rather than opinion."

[Lesson 7](07-reliability-and-degradation.md) kept the service up under failure. This lesson keeps it *correct* under change — and change is the dominant cause of incidents in every production system ever measured. Inference adds a failure class ordinary services don't have: the deploy succeeds, every metric is green, and the answers got worse.

---

## What you are actually shipping

| Change | Latency risk | Quality risk | Typical surprise |
|---|---|---|---|
| Engine version bump (vLLM 0.x → 0.y) | medium | **medium** | default sampling params, chat-template handling, or tokenizer behaviour changed |
| New quantization (FP16 → INT8/INT4/FP8) | improves | **high** | fine on benchmarks, degrades on *your* long-tail prompts ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)) |
| Model weights update (v1 → v2 fine-tune) | low | **high** | different verbosity, refusal rate, format adherence |
| Chat template / system prompt change | low | **very high** | a whitespace or role-tag difference silently degrades everything |
| Scheduler/config change (`max_num_seqs`, chunked prefill) | **high** | low | throughput up, ITL tail up more |
| Speculative decoding enablement | improves TPOT | low-medium (should be output-neutral) | acceptance rate too low → slower *and* more expensive |
| Hardware change (A100 → H100, TP 2 → 4) | improves | low | numerics differ slightly; kernel selection changes |
| Routing change (round-robin → prefix-aware) | improves TTFT | none | hotspots ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)) |

Note how many rows have quality risk that no infrastructure metric detects. **That is the gap this lesson closes.** Also note: every row belongs in the `model_version` label from [lesson 4](04-tracing-and-logging.md) — quantization, engine version, template hash, and weights all together. If you can't attribute a request to the exact configuration that produced it, you cannot run any of what follows.

---

## The rollout ladder

```
  0. OFFLINE EVAL          fixed prompt set, no traffic, no risk
       │                   catches: gross quality regressions, format breaks
  1. SHADOW / MIRROR       copy of real traffic, response DISCARDED
       │                   catches: real-workload latency, OOM, crashes, throughput
       │                   risk: extra GPU cost; side effects if the model calls tools!
  2. CANARY 1%             real traffic, real users, automated analysis
       │                   catches: quality on real prompts, tail behaviour
  3. CANARY 5% → 25% → 50% with a bake time at each step
       │                   catches: capacity effects, slow burns, cache interactions
  4. 100% + keep the old version deployable for one full rollback window
```

Choose the entry rung by risk: a config tweak can start at step 2; a new quantization should always do step 1 first (latency evidence for free) and then step 2 with quality analysis. A chat-template change should do step 0 aggressively — it is the cheapest place to catch the most damaging class.

### Shadow traffic mechanics

```
  client ──► router ──┬──► PRIMARY (fp16)    ──► response to client
                      └──► SHADOW  (int4)    ──► response DISCARDED, metrics + sample stored
  Properties:
    + zero user risk (nothing returned), real prompt distribution, real concurrency shape
    − 2× GPU cost for the mirrored fraction (mirror 5-20%, not 100%)
    − shadow load must not steal capacity from primary: separate replicas, lower priority,
      hard concurrency cap, and shed shadow first under pressure
    ! DANGER: if the model performs SIDE EFFECTS (tool calls, writes, emails, payments),
      shadowing executes them twice. Tool execution must be stubbed in shadow mode.
    ! Privacy: shadow responses stored for comparison are user content — see lesson 4's
      content-capture policy; store hashes/scores by default, raw text only in the
      access-controlled eval pipeline.
```

Shadowing is the highest-value, lowest-risk step and the one most teams skip. It gives you *your own* workload's latency/throughput/OOM evidence before a single user is exposed — far better evidence than any public benchmark, because prompt-length distribution and concurrency shape dominate the result.

---

## Comparing outputs when the model is nondeterministic

This is the technical heart of the lesson. Exact-match diffing between two model versions does not work, for four independent reasons:

1. **Sampling.** `temperature > 0` means the same prompt yields different text on every call, even from the same weights.
2. **Batch-dependent numerics.** Even at `temperature=0`, floating-point reduction order depends on batch composition and kernel selection, so the same prompt can produce different logits — and one flipped argmax early in a sequence diverges the whole continuation. (This is why "deterministic inference" needs batch-invariant kernels, not just a fixed seed.)
3. **Hardware/kernel differences.** Different GPU, different TP degree, different attention backend → different numerics.
4. **Quantization.** The point of the change is that the numbers differ.

So "0% diff" is unachievable and "any diff = fail" is a broken gate. What works:

| Method | What it measures | Cost | When |
|---|---|---|---|
| **Greedy + fixed seed, same hardware, same batch config** | maximal comparability; diffs are then meaningful | cheap | pre-production A/B of weights or quantization |
| **Token-level agreement of top-1 under teacher forcing** | per-position divergence rate on a fixed prompt set — the most sensitive numeric check available | cheap | quantization validation ([Phase 4 lesson 2](../phase-4/02-quantization-fundamentals.md)) |
| **Perplexity / KL divergence vs baseline** on held-out text | distributional shift, one number | cheap | quantization and weight changes |
| **Embedding similarity** (cosine) between baseline and candidate answers | semantic equivalence, tolerant of paraphrase | medium (an embedding model per response) | shadow/canary on real traffic |
| **Task metrics**: exact-match/F1 for extraction, JSON-schema validity, unit tests for code, tool-call correctness | *the* thing users care about, when your task has a checkable answer | medium | any change, if your product has structure |
| **LLM-as-judge pairwise preference** | subjective quality, human-like | expensive, noisy, biased toward verbosity | last resort; use with position-swapping and a fixed rubric |
| **Cheap behavioural proxies**: output-length distribution, refusal-phrase rate, repetition n-gram rate, language-ID match, format-adherence rate, finish-reason mix | regressions, with near-zero cost, computable on 100% of traffic | very cheap | **always on**; this is your production quality SLI ([lesson 5](05-slos-and-error-budgets.md)) |

**Start with the last row.** Output-length distribution plus refusal rate plus JSON-validity plus finish-reason mix catches a remarkable share of real regressions for essentially no cost, on all traffic, without storing any content. A quantization that broke instruction-following usually shows up as "mean output length fell 40%" or "`length` finishes tripled" long before anyone files a ticket.

### Statistics: don't ship on a 200-request canary

```
  You want to detect a change in a rate (e.g. JSON-validity 99% → 97%).
  Required sample size per arm, roughly, for a two-proportion test at 80% power, α=0.05:

     baseline p    detect Δ      n per arm
     99%           2 pts         ~2,000
     99%           0.5 pts       ~30,000
     50% (judge)   5 pts         ~1,600
     latency p99   need ~10x more samples than for a mean; percentiles are noisy

  ⇒ a 1% canary at 100 req/s gives 1 req/s → 2,000 samples takes ~33 minutes.
    That is your minimum bake time, and it is arithmetic, not a guess.
```

Additional statistical hygiene that matters in practice:

- **Pair the comparison.** Same prompts to both arms (shadow makes this exact) removes prompt-difficulty variance and cuts the required sample size dramatically.
- **Watch for traffic skew.** A 1% canary chosen by hash of `tenant_id` can land on one big customer with atypical prompts. Randomize per request for quality comparisons; use sticky assignment only when session consistency matters — and then check the arms' prompt-length distributions match before believing the result.
- **Correct for multiple comparisons.** Ten metrics at α=0.05 gives a ~40% chance of one false alarm; either raise the bar per metric or define a small set of primary gates and treat the rest as diagnostic.
- **Beware Simpson's paradox by tenant/route/length bucket.** Always break the primary metric down by `tenant_class` and prompt-length bucket before promoting.
- **Cold-start contamination.** A fresh canary replica has an empty prefix cache and un-captured CUDA graphs, so its early TTFT is *structurally* worse ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md)). Warm it, then start the measurement window. Otherwise every canary looks like a latency regression for its first minutes.

---

## Automated canary analysis

The decision must be a computed verdict against pre-declared thresholds, not a human squinting at Grafana.

```
  CANARY GATE (evaluated every 60 s over the accumulated window)
  ─────────────────────────────────────────────────────────────────────────────
  HARD FAIL (abort immediately, no waiting):
    error rate (class ∈ {capacity, infra}) > baseline × 2, OR any crash/OOM
    p99 TTFT > SLO threshold sustained 2 min
    JSON-schema validity < baseline − 1 pt         (if applicable)
    empty-output rate > baseline × 2
  SOFT FAIL (needs n ≥ required sample size, then abort):
    p95 TTFT ratio canary/baseline > 1.15
    p95 TPOT ratio > 1.15
    mean output length ratio outside [0.85, 1.15]
    refusal-phrase rate > baseline + 1 pt
    embedding-similarity mean < 0.92 (paired, shadow-derived)
    finish_reason mix χ² p < 0.01
  PASS CRITERIA to advance a step:
    all hard gates clear, all soft gates clear, n ≥ required, bake time elapsed
  ROLLBACK: revert traffic weight to 0 within one router reconfiguration (<30 s),
            keep the canary replicas alive for post-hoc analysis, page nobody unless
            the baseline is also affected
```

Argo Rollouts / Flagger implement exactly this control loop with Prometheus queries as the analysis provider; the work you own is **the metric list, the thresholds, and the sample-size requirement**. Write them into the rollout manifest so the gate is version-controlled with the code ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)).

### Cost of the canary itself

Canarying GPU workloads is expensive in a way canarying web services is not: a 1% canary still needs **at least one whole replica** (you cannot have 1% of a GPU), so a 10-replica fleet running a canary costs +10% capacity minimum, and for a TP8 model that's 8 GPUs. Consequences:

- Prefer **shadowing a small fraction to a single replica** for latency/throughput evidence, since it doesn't need to scale with the traffic share.
- Use **surge-based rollouts with a plan**: `maxSurge: 1` on GPU nodes means one extra replica's GPUs must exist — reserve that headroom or the rollout will simply be `Pending` forever ([Phase 8 lesson 6](../phase-8/06-deploying-and-rollouts.md)).
- For very large models, canary in a **separate cell/region** and shift traffic at the router, rather than surging inside the same cluster.

---

## Rollback: the property that makes everything else safe

| Requirement | Why |
|---|---|
| Old version's **image, weights and config** remain fetchable for the full rollback window | "roll back" fails if the artifact was garbage-collected; pin by digest ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)) |
| Rollback is a **traffic-weight change**, not a rebuild | seconds, not the `T_cold` minutes of a fresh deploy |
| Rollback is **tested** on a schedule | an untested rollback path is a hypothesis; exercise it quarterly |
| State compatibility | KV caches, prefix caches, and any persisted session state must not break when versions swap; prefix caches are per-replica so this is usually free — verify for your stack |
| One-way changes are flagged | template changes that alter stored conversation formats, tokenizer changes that invalidate cached prompts, schema migrations: identify them explicitly and require a separate plan |

**Time-to-rollback is the reliability metric of your deploy system.** Track it. Anything above a couple of minutes means incidents get resolved by debugging under pressure instead of by reverting, which is strictly worse.

---

## Worked example: shipping INT4 quantization

```
  WEEK 0  offline: token-agreement + perplexity vs FP16 on 2k held-out prompts
                   + task metrics on your product's eval set (JSON validity, extraction F1)
          gate: perplexity delta < 2%, top-1 agreement > 95%, task metrics within 1 pt
  WEEK 0  shadow 10% of live traffic to ONE int4 replica for 24 h
          measures: p50/p95/p99 TTFT & TPOT, tokens/s per GPU, KV occupancy (int4 weights
          free up KV room → bigger batches), OOM/crash count, output-length distribution,
          paired embedding similarity vs primary responses
          gate: no crashes, TTFT ≤ baseline, tokens/s per GPU ≥ 1.3× baseline,
                embedding similarity ≥ 0.94, length ratio in [0.9, 1.1]
  WEEK 1  canary 1% for ≥ 60 min (n ≈ 3,600) → 5% for 2 h → 25% for 4 h → 50% overnight
          automated gate as above, broken down by tenant_class and prompt-length bucket
  WEEK 1  100%; FP16 deployment kept warm for 48 h, rollback = one weight change
  AFTER   cost review: cost per 1M tokens before/after, at the SLO (lesson 9).
          If cost didn't improve, the quantization was pointless — that is the actual
          decision criterion, and it is an arithmetic one.
```

The last line is the discipline that separates engineering from cargo cult: a quantization that doesn't reduce cost per 1M tokens *at your SLO* has traded quality for nothing.

---

## Do this now (75 minutes)

1. **Build a shadow proxy.** A small async service that forwards each request to primary and (for a sampled fraction) to a canary, returns only primary's response, and records both arms' TTFT/TPOT/E2E/output-length/finish-reason plus a paired similarity score. Verify a kill of the canary backend does not affect primary responses at all — that isolation is the feature.
2. **Run one real comparison.** Same model, two configs you actually have: FP16 vs 8-bit, or `enforce_eager` on/off, or two `max_num_seqs` values. Produce the paired table (latency percentiles, tokens/s, length distribution, similarity) over ≥1,000 paired prompts, and state a promote/reject verdict against thresholds you wrote *before* looking.
3. **Compute your required sample size and bake time** for your highest-traffic route and a 1 pt regression in your most checkable quality metric. Then write the canary gate YAML/config with those numbers in it. If the bake time is longer than your team's patience, that tension is the real finding — resolve it by shadowing more traffic rather than by shipping on less evidence.

---

**Next:** [Cost per 1M tokens →](09-cost-per-million-tokens.md) — the unit-economics formula, how every lever from Phases 3-6 lands in it, and how to attribute cost per tenant without a cardinality explosion.
