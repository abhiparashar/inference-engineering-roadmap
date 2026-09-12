# 9 — Cost per Million Tokens: Unit Economics of Inference

> **You'll be able to say:** "The unit metric is `$ per 1M output tokens at the SLO`, and it equals `(GPU-$/hr × GPUs per replica) / (output tokens/sec × 3600) × 10⁶` — with the vital qualifier that the throughput number must be measured at the latency you actually ship, not at peak. I can decompose that number into utilization, batch efficiency, headroom, idle trough capacity and purchasing mode; attribute it per tenant and per feature without exploding metric cardinality; price prefill and decode separately because they have different costs and different cache economics; and say which Phase 3-6 lever moves it most for a given workload."

[Lesson 8](08-canary-and-shadow-traffic.md) ended with "if cost per 1M tokens didn't improve, the change was pointless." This lesson makes that sentence computable. It is also the lesson that most directly maps to a job title — "inference cost optimization" is a real role at every AI company right now, and it is 80% this arithmetic plus the measurement discipline from Phase 3.

---

## The formula, and the qualifier that makes it honest

```
                          $/GPU-hour × GPUs per replica
  cost per 1M tokens  =  ───────────────────────────────── × 10⁶
                         output tokens/sec/replica × 3600

  EXAMPLE: H100 at $3.00/hr, TP=1, measured 2,400 output tok/s at p99 TTFT ≤ 500 ms
           = 3.00 / (2400 × 3600) × 1e6 = $0.35 per 1M output tokens

  SAME HARDWARE, measured at PEAK throughput (4,100 tok/s, p99 TTFT 4 s — unshippable)
           = $0.20 per 1M tokens   ← the number in the vendor chart
           ⇒ 43% "savings" that consists entirely of violating your SLO
```

**Always report the SLO with the cost.** `$0.35/1M @ p99 TTFT 500 ms, p99 TPOT 50 ms` is a fact; `$0.20/1M` alone is marketing. This is the same honesty rule as [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md), applied to dollars.

### Where the real bill diverges from the formula

The formula prices *a busy replica*. Your invoice prices *every allocated GPU-hour*, busy or not:

```
  effective cost per 1M tokens
      =  cost_formula  ×  1/utilization_factor  ×  1/goodput_factor

  utilization_factor = tokens actually served ÷ tokens the fleet could have served
                       (losses: diurnal trough, headroom for T_cold, spare for failover,
                        canary replicas, autoscaler lag, fragmented/unschedulable GPUs)
  goodput_factor     = tokens delivered to satisfied users ÷ tokens generated
                       (losses: aborted streams, timed-out requests, retried requests,
                        shadow traffic, speculative-decoding rejected drafts)

  TYPICAL REALITY CHECK
    formula (busy replica)                       $0.35 / 1M
    ÷ 0.55 average fleet utilization (diurnal)   $0.64
    ÷ 0.93 goodput (aborts, retries, rejects)    $0.69   ← the number your CFO sees
```

That factor-of-two gap between "benchmark cost" and "invoice cost" is where most of the available savings live, and it is invisible to anyone measuring only single-replica throughput. **Instrument the two factors, not just the formula.**

---

## Prefill and decode cost differently

A single blended number hides the most important structural fact about your traffic.

```
  PREFILL: compute-bound, parallel over prompt tokens
    cost ≈ (2 × params × prompt_tokens FLOPs) / (achieved FLOP/s × $/s)
    → cost per prompt token is roughly FLAT in batch size (already efficient)
    → and it goes to ~0 for cache-hit tokens (prefix caching, Phase 4 lesson 6)

  DECODE: memory-bandwidth-bound, sequential
    cost ≈ (weights bytes read per step) / (HBM bandwidth) × steps / batch
    → cost per output token falls ~linearly with batch size until KV runs out
    → this is why batching is THE cost lever (Phase 3 lesson 4)

  CONSEQUENCE FOR PRICING AND OPTIMIZATION
    input tokens are cheap and cacheable      → cached-input discounts exist in the market
                                                 for a real engineering reason
    output tokens are expensive and serial    → ~3-5x the price of input tokens, typically
    long prompts + short outputs = prefill-heavy: optimize cache hit rate, chunked prefill
    short prompts + long outputs = decode-heavy: optimize batch size, quantization, spec-dec
```

So: report **cost per 1M input tokens and cost per 1M output tokens separately**, plus your **prefix-cache hit rate**, because a hit converts prefill cost to nearly zero. A 60% hit rate on a prefill-heavy workload is a direct ~60% reduction of the prefill half of your bill ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md) is therefore a cost lever, not just a latency lever).

---

## The full cost model of a deployment

| Term | Magnitude | Notes |
|---|---|---|
| GPU compute | 70-90% of total | the only term most people track |
| Headroom / spare capacity | +20-50% of GPU | required by `T_cold × dλ/dt` ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md)); a *deliberate* SLO purchase |
| Diurnal trough waste | often the largest single loss | peak:trough of 3-5× is normal for consumer traffic |
| Redundancy (N+1, multi-AZ) | +1 replica minimum, or +100% for 2-replica services | reliability purchase |
| Canary/shadow capacity | +1 replica per rollout | [lesson 8](08-canary-and-shadow-traffic.md) |
| Idle standby pool | pre-warmed replicas | burst insurance, priced in GPU-hours |
| CPU/host, memory | 5-15% | tokenization, router, gateway |
| Networking / egress | small for text, **large for audio/video/images** | streaming text is cheap; egress pricing bites multimodal ([Phase 9 lesson 3](../phase-9/03-vision-serving.md)) |
| Storage | small but annoying | weights per version × regions; registry retention |
| Observability | 2-10% (!) | traces and logs at high QPS are not free ([lesson 4](04-tracing-and-logging.md)); this is why sampling exists |
| Vector DB / retrieval (RAG) | can rival the LLM | [Phase 9 lesson 6](../phase-9/06-vector-search-and-ann.md) |
| Engineering time | dominates for small fleets | below ~10 GPUs, an engineer-week costs more than the GPUs; optimize accordingly |

**The last row is a real decision rule.** At 4 GPUs, spending two weeks to save 20% of $9k/yr is negative value; at 400 GPUs the same work saves $180k/yr. Know which regime you are in before starting an optimization project — and say so in the proposal.

---

## Per-tenant and per-feature attribution

You need "which customer/feature costs what" for pricing, quotas, and prioritization. [Lesson 2](02-instrumenting-with-prometheus.md) forbade `tenant_id` as a Prometheus label, so attribution runs through the structured event stream from [lesson 4](04-tracing-and-logging.md):

```
  PER-REQUEST COST ESTIMATE (computed at request completion, logged, not a metric)

  gpu_seconds_attributed ≈  prefill_tokens_uncached × c_prefill
                          + output_tokens × c_decode(batch_size_avg)

  where c_prefill  [GPU-s per prompt token]  and
        c_decode(b) [GPU-s per output token at batch b]
  are CALIBRATED CONSTANTS from your own benchmark sweep (Phase 3 harness):
     c_decode(b) ≈ step_time(b) / b          ← falls with batch size
     c_prefill   ≈ prefill_time / prompt_tokens

  cost_usd ≈ gpu_seconds_attributed × ($/GPU-hour ÷ 3600) × GPUs_per_replica
             × overhead_multiplier        ← the utilization/goodput factors above
```

Then aggregate in the warehouse: `SUM(cost_usd) GROUP BY tenant, feature, model_version, day`. Properties worth knowing:

- **It's an estimate, and that's fine.** The sum across requests won't exactly match the invoice (shared batch effects, idle time). Reconcile by scaling the attributed total to the actual GPU-hours: `factor = actual_gpu_hours / Σ attributed_gpu_hours`, and publish the factor. A 1.3-1.8× factor is normal and is itself a KPI — it *is* your utilization inefficiency.
- **Batch-size dependence is the subtle part.** The same request costs less when it rode in a big batch. Attributing with the *observed* average batch size is fair and gives tenants an incentive-compatible signal: bursty, cache-hostile traffic genuinely costs more.
- **Cached prompt tokens must be discounted**, or your attribution will mis-rank tenants and hide the benefit of the routing work.
- **Partial/aborted generations are billed at what they consumed** — the GPU time was real ([lesson 2](02-instrumenting-with-prometheus.md) counted those tokens deliberately).

---

## The levers, ranked by typical impact

| Lever | Typical effect on $/1M tokens | Cost/risk | Phase |
|---|---|---|---|
| **Continuous batching** (vs static/none) | 3-20× improvement | none — table stakes | [3 L4](../phase-3/04-continuous-batching.md) |
| **Raise batch size until the SLO binds** | 1.5-4× | ITL/TPOT rises; bounded by KV | [3 L4](../phase-3/04-continuous-batching.md), [3 L5](../phase-3/05-queueing-theory.md) |
| **Weight quantization** (FP8/INT8/INT4) | 1.5-3× (also frees KV room → bigger batches) | quality risk; validate per [lesson 8](08-canary-and-shadow-traffic.md) | [4 L2-3](../phase-4/02-quantization-fundamentals.md) |
| **Prefix caching + prefix-aware routing** | up to ~the prefill share of your bill | routing hotspots | [4 L6](../phase-4/06-prefix-caching-and-radix-attention.md), [6 L7](../phase-6/07-prefix-aware-routing.md) |
| **KV-cache quantization / GQA-aware sizing** | more concurrency at the same memory | small quality risk | [4 L4](../phase-4/04-kv-cache-optimization.md) |
| **Fix the diurnal trough** (autoscale + predictive) | 1.3-2× on the *fleet* number | SLO risk during ramps | [6 L8](../phase-6/08-autoscaling-gpu-fleets.md) |
| **Spot/preemptible mix** | 0.3-0.6× on the numerator | eviction handling required | [6 L8](../phase-6/08-autoscaling-gpu-fleets.md) |
| **Reserved/committed pricing for the baseline** | 0.4-0.7× on the baseline portion | commitment risk | this lesson |
| **Right-size the model** (8B vs 70B; cascade/distill) | 3-10× | the biggest quality decision | [4 L1](../phase-4/01-what-to-optimize.md) |
| **Speculative decoding** | 1.2-2× on latency; cost-neutral-to-worse at high batch | acceptance-rate dependent | [4 L7](../phase-4/07-speculative-decoding.md) |
| **Right-size the parallel layout** (TP 8 → 2 + replicas) | 1.2-1.6× | latency/size constraints | [6 L1](../phase-6/01-when-one-gpu-isnt-enough.md), [6 L3](../phase-6/03-tensor-parallelism.md) |
| **Prefill/decode disaggregation** | goodput gains at scale | operational complexity | [6 L6](../phase-6/06-disaggregated-prefill-decode.md) |
| **Newer GPU generation** | often better $/token despite higher $/hr | availability | [2 L1](../phase-2/01-why-gpus.md) |
| Cheaper GPU per hour (e.g. A10G vs H100) | frequently **worse** $/token | — | compute $/token, never $/hour |

The last row is the most common procurement mistake: **a cheaper GPU is not a cheaper token.** An H100 at 3× the hourly price of an A10G can be 5-8× the tokens/sec for a model that fits comfortably, making it substantially cheaper per token *and* faster. Always convert to $/1M tokens at your SLO before choosing hardware.

---

## Making it a live metric

```promql
# output tokens/sec, fleet
sum(rate(inf_tokens_total{direction="output"}[10m]))

# allocated GPUs (from kube-state-metrics or a static recording rule per deployment)
sum(inference_replica_gpus)

# $/hr per GPU is a config constant → a recording rule
# cost per 1M output tokens, live, fleet-wide (INCLUDES idle: this is the honest one)
  (sum(inference_replica_gpus) * 3.00)
/ (sum(rate(inf_tokens_total{direction="output"}[10m])) * 3600)
* 1e6

# efficiency per replica: tokens/sec/GPU  → the regression detector
sum by (pod) (rate(inf_tokens_total{direction="output"}[10m])) / on(pod) inference_replica_gpus

# utilization factor: delivered vs capacity (capacity = replicas × measured μ at SLO)
  sum(rate(inf_tokens_total{direction="output"}[10m]))
/ (sum(inference_replica_gpus) / <gpus_per_replica> * <measured_tokens_per_s_at_slo>)

# energy per 1M tokens — the physical efficiency metric, useful and hard to game
  sum(rate(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION[10m])) / 1000
/ (sum(rate(inf_tokens_total{direction="output"}[10m])) * 3600) * 1e6
```

Alert on **cost per 1M tokens rising >20% vs the 7-day baseline at the same hour** ([lesson 5](05-slos-and-error-budgets.md)) — the same-hour comparison is essential because the metric is strongly diurnal. A silent efficiency regression (a config change that halved batch size, a deploy that disabled CUDA graphs, a router change that tanked cache hit rate) is otherwise invisible until the monthly invoice.

---

## Worked example: a cost review that changes a decision

```
  SETUP: Llama-3.1-8B chat, p99 TTFT ≤ 500 ms, p99 TPOT ≤ 50 ms
         peak 25 QPS, trough 5 QPS, mean 12 QPS; 600 prompt / 250 output tokens mean
         H100 80GB on-demand $3.00/hr, 1 GPU per replica

  MEASURED AT SLO (Phase 3 harness): 2,400 output tok/s per replica
  demand at peak: 25 QPS × 250 tok = 6,250 output tok/s → 2.6 replicas → 3 at 100% util
  with 70% target utilization (headroom rule): 4 replicas at peak
  with reactive scale-in to trough (5 QPS → 1,250 tok/s): 1-2 replicas at night
  average fleet over 24 h ≈ 2.6 replicas

  BILL: 2.6 × $3.00 × 24 × 30 = $5,616/month
  TOKENS: 12 QPS × 250 output tok = 3,000 output tok/s
          × 2.592e6 s/month = 7.78e9 tokens = 7,776 million output tokens
  EFFECTIVE COST: 5616 / 7776 = $0.72 per 1M output tokens
  FORMULA COST (busy replica):        $0.35 per 1M   ← 2.06× gap = utilization loss

  OPTIONS PRICED:
  A. FP8 quantization: measured 3,900 tok/s at SLO → 1.7 replicas avg → $0.44/1M  (−39%)
     risk: quality validation required (lesson 8). Cost of the work: ~1 week.
  B. Predictive autoscaling + 1 warm standby: avg 2.0 replicas → $0.55/1M       (−24%)
     risk: SLO during ramps; no quality risk at all.
  C. 60% spot mix with drain handling: numerator ×0.70 → $0.50/1M               (−30%)
     risk: eviction storms; needs lesson 7's drain path.
  D. A+B+C combined (multiplicative on different terms) → ≈ $0.24/1M            (−67%)
  E. Move to A10G at $1.00/hr: measured 620 tok/s at SLO → 10 replicas → $1.72/1M (+139%)
     ⇒ the "cheaper" GPU is 2.4x MORE expensive per token. Rejected by arithmetic.

  DECISION: B first (no quality risk, one week), then A behind a full canary, then C.
  Report line: "$0.72 → $0.24 per 1M output tokens at unchanged SLO; 3 changes, 4 weeks."
```

That final line is what an inference-cost-optimization role delivers. Note that every input was *measured* — the formula alone would have made options A and E look very different.

---

## Do this now (60 minutes)

1. **Compute your two numbers.** For any model you can run: measure output tokens/sec at your SLO (not at peak) with the Phase-3 harness, then compute formula cost per 1M tokens and — using replica-hours actually allocated over a day — the effective cost. Report the ratio. Understanding that ratio is the entire lesson.
2. **Calibrate `c_prefill` and `c_decode(b)`** for your stack: sweep batch size, record step time, fit per-token GPU-seconds. Then add `cost_usd_estimate` to your structured completion event ([lesson 4](04-tracing-and-logging.md)) and aggregate a day's logs by `tenant_class`. Check the reconciliation factor against real GPU-hours.
3. **Price three options** for your deployment (e.g. quantization, better scale-in, spot mix) with measured or clearly-labelled estimated throughput, and write the one-paragraph recommendation including risk and engineering cost. Compare against renting the same model from a hosted API per 1M tokens — sometimes that comparison is the honest answer, and knowing *at what scale* it flips is a genuinely senior judgment.

---

**Next:** [Incident response, runbooks, and chaos testing →](10-incident-response-and-chaos.md) — the on-call reality: an inference-specific incident catalogue, runbooks that fit on one screen, blameless postmortems that produce alerts instead of adjectives, and deliberately breaking your own GPUs before they break themselves.
