# 5 — SLOs, Error Budgets, and Burn-Rate Alerting

> **You'll be able to say:** "An SLI is a ratio of good events to valid events; an SLO is a target on that ratio over a window; the error budget is `1 − SLO` and it is a *spending account*, not a failure. For inference I write separate SLOs for availability, TTFT and TPOT, and I exclude intentionally shed and client-aborted requests from the denominator on purpose. I alert on multi-window multi-burn-rate conditions — `14.4×` over 1 h confirmed by 5 min for a page, `6×` over 6 h for a ticket — instead of alerting on p99 crossing a line, because the latter pages on noise and misses slow burns."

[Lessons 1-4](01-what-to-measure.md) produced signals. This lesson turns a handful of them into a *promise* and a small set of alerts, and deletes everything else from the paging path. This is the lesson that changes how you're perceived on a team: most engineers add dashboards; the ones who get trusted with production delete alerts.

---

## SLI → SLO → error budget, precisely

```
  SLI  = good events / valid events            (a ratio in [0,1], per window)
  SLO  = a target on the SLI                   e.g. 99.5% over a rolling 28 days
  ERROR BUDGET = (1 − SLO) × valid events      the number of bad events you may spend

  99.5% over 28 days  ⇒ 0.5% of events may be bad
                      ⇒ at 100 req/s that's 0.005 × 100 × 86400 × 28 ≈ 1.2M bad requests
                      ⇒ or, as time-equivalent at full outage: ~3 h 22 m
```

Two consequences people resist and then find liberating:

1. **The budget is meant to be spent.** A team at 100% budget remaining all quarter is over-invested in reliability and under-invested in shipping. An error budget is permission to take risk — to roll out the INT4 quantization, to try the new scheduler.
2. **The budget defines the policy, not a feeling.** "Budget exhausted → feature freeze, reliability work only, no risky rollouts" is the standard rule, and it converts an argument about vibes into arithmetic.

### Event-based, not time-based

Time-based availability ("minutes of downtime") is a poor fit for inference because degradation is partial and load-dependent: at 3 a.m. a broken replica affects 40 requests; at peak it affects 40,000. **Count events.** `good/valid` over a rolling window is the metric; time-to-equivalent is just a communication aid for executives.

---

## Choosing the denominator: "valid events" is where the design is

This is the most commonly botched part of an inference SLO, and it's worth being pedantic:

| Request outcome | In denominator? | In numerator (good)? | Rationale |
|---|---|---|---|
| 200, first token within threshold | yes | yes | the happy path |
| 200, first token too slow | yes | **no** | latency is the SLI |
| 500 / engine crash / OOM | yes | no | our fault, squarely |
| 429 from admission control | **no** (excluded) or separate SLO | — | shedding is *designed* behaviour ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)); counting it as failure means load-shedding — the correct overload response — destroys your SLO and incentivizes the wrong engineering. Give shedding its *own* SLO: "≤0.1% of requests shed." |
| 400 / invalid prompt / over context limit | no | — | client error; not a reliability event |
| 401/403 | no | — | ditto |
| client disconnected before first token | **no** | — | you cannot deliver to an absent client; but *track* it — a rising rate means users are giving up, which is a real signal ([lesson 1](01-what-to-measure.md)) |
| client disconnected mid-stream after a good TTFT | yes | yes | you delivered the SLI |
| 200 but empty/garbage output | yes | **no** | the quality-failure class; requires output-side instrumentation, and most teams silently exclude it. Don't. |
| requests during a declared maintenance window | no | — | if and only if the window is announced and bounded |

**Write the exclusion list down and put it in the SLO document.** Every exclusion is a place where your SLO can be gamed, deliberately or by accident, and an undocumented exclusion is indistinguishable from a bug.

---

## The inference SLO set

One SLO is never enough here because the user-visible failure modes are independent.

| # | SLO | SLI definition | Typical target | Window |
|---|---|---|---|---|
| 1 | **Availability** | non-error responses / valid requests | 99.9% | 28 d rolling |
| 2 | **TTFT latency** | requests with TTFT ≤ 500 ms / valid requests | 99% | 28 d rolling |
| 3 | **Streaming pace (TPOT)** | requests with TPOT ≤ 50 ms/token / valid streamed requests | 99% | 28 d rolling |
| 4 | **Stall-freedom (optional, strong)** | requests with max ITL ≤ 2 s / valid streamed requests | 99.5% | 28 d rolling |
| 5 | **Shedding** | shed requests / offered requests | ≤ 0.1% | 7 d rolling |
| 6 | **Output quality (if you can measure it)** | responses passing automated checks / sampled responses | 99.9% | 7 d rolling |

Guidance on picking the numbers, which matters more than the structure:

- **Derive thresholds from the product, then check against measurement.** For chat: TTFT under ~500 ms reads as instant; TPOT of 50 ms/token = 20 tok/s ≈ comfortably faster than reading speed. For a code-completion product the TTFT budget may be 150 ms and long outputs irrelevant. For a batch summarization API, TTFT is nearly irrelevant and E2E-per-1k-tokens is the real SLI.
- **Never set the target above what you have measured at your SLO-load.** Your Phase-3 harness already told you the achievable p99 at your saturation point ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)). An SLO you can't meet at your current capacity is a budget-exhaustion machine.
- **Differentiate by tier, not by tenant.** `tenant_class` labels give you "pro: p99 TTFT 400 ms / free: 1.5 s" without cardinality explosion ([lesson 2](02-instrumenting-with-prometheus.md)).
- **Long-output requests need a normalized SLI.** If a single SLO must cover 20-token and 2000-token responses, use TTFT + TPOT (both length-independent) and leave E2E out of the SLO entirely. This is why lesson 1 insisted on splitting them.

### Writing it down

```
  SLO-2: TTFT
  ─────────────────────────────────────────────────────────────────────────────
  Service        : chat-inference (model family llama-3.1-8b-instruct, route /v1/chat)
  SLI            : fraction of valid requests whose first output token is emitted
                   within 500 ms of admission, measured at the ROUTER
  Good           : inf_ttft_seconds_bucket{le="0.5"}
  Valid          : inf_ttft_seconds_count  minus  {error_class="shed"} and
                   {outcome="client_abort_before_first_token"}
  Target         : 99.0% over a rolling 28-day window
  Budget         : 1.0% of valid requests
  Burn policy    : page at 14.4x(1h/5m) and 6x(6h/30m); ticket at 1x(3d/6h)
  Exclusions     : announced maintenance windows (max 2 h/month, tracked in CHANGELOG)
  Owner          : inference-serving team;  reviewed quarterly
  Dependencies   : router, engine, GPU fleet, model registry
```

The `le="0.5"` bucket boundary is not a coincidence — [lesson 2](02-instrumenting-with-prometheus.md) put it there so this SLI is exact rather than interpolated. **The SLO document and the bucket configuration are the same decision.**

---

## Burn-rate alerting: the only latency alerting that works

The naive alert — `p99_ttft > 0.5 for 5m` — fails in both directions. At low traffic, ten slow requests move p99 and page you at 4 a.m. for nothing. Meanwhile a steady 0.9% failure rate never crosses any threshold while quietly eating your entire 28-day budget in a week.

**Burn rate** fixes both. Define:

```
  burn rate = (observed bad-event ratio) / (budgeted bad-event ratio)

  burn rate 1   = you will consume exactly 100% of the budget by the end of the window
  burn rate 14.4 = you will consume the whole 28-day budget in 2 days
                   (equivalently: 2% of the budget in 1 hour)
  burn rate 6   = whole budget in ~4.7 days  (5% of budget in 6 hours)
```

The standard SRE workbook configuration, and why each row exists:

| Severity | Long window | Burn rate | Short window (confirm) | Budget consumed before it fires | Catches |
|---|---|---|---|---|---|
| **Page** | 1 h | 14.4× | 5 min | 2% | fast, severe breakage |
| **Page** | 6 h | 6× | 30 min | 5% | serious partial degradation |
| **Ticket** | 24 h | 3× | 2 h | 10% | meaningful slow burn |
| **Ticket** | 3 d | 1× | 6 h | 10% | the quiet drip that exhausts the budget |

Two properties make this the right design:

1. **The short window kills flapping.** The alert requires *both* the long window (significance: enough events to be real) and the short window (currency: it's still happening now) to exceed the rate. Recovery is therefore fast — when the incident ends, the short window drops below threshold within minutes instead of waiting out the long window.
2. **Severity is proportional to budget impact**, so alert urgency automatically tracks user harm, and low-traffic periods stop generating pages: at 2 req/s, a 14.4× burn over an hour is a real, substantial number of affected requests, not three unlucky ones.

```yaml
groups:
  - name: inference-slo-ttft
    rules:
      # good-ratio recording rules at several windows (cheap, shared with dashboards)
      - record: inf:ttft:good_ratio_5m
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5",error_class!="shed"}[5m]))
            / sum(rate(inf_ttft_seconds_count{error_class!="shed"}[5m]))
      - record: inf:ttft:good_ratio_1h
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5",error_class!="shed"}[1h]))
            / sum(rate(inf_ttft_seconds_count{error_class!="shed"}[1h]))
      - record: inf:ttft:good_ratio_30m
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5",error_class!="shed"}[30m]))
            / sum(rate(inf_ttft_seconds_count{error_class!="shed"}[30m]))
      - record: inf:ttft:good_ratio_6h
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5",error_class!="shed"}[6h]))
            / sum(rate(inf_ttft_seconds_count{error_class!="shed"}[6h]))

      - alert: TTFTSLOBurnRateFast          # 2% of the 28d budget in 1 hour
        expr: (1 - inf:ttft:good_ratio_1h)  > (14.4 * 0.01)
          and (1 - inf:ttft:good_ratio_5m)  > (14.4 * 0.01)
        for: 2m
        labels: { severity: page, slo: ttft }
        annotations:
          summary: "TTFT SLO burning at >14.4x (2% of 28d budget/hour)"
          runbook: "https://runbooks/inference/ttft-slo-burn"
          dashboard: "https://grafana/d/inference-overview"

      - alert: TTFTSLOBurnRateSlow          # 5% of budget in 6 hours
        expr: (1 - inf:ttft:good_ratio_6h)  > (6 * 0.01)
          and (1 - inf:ttft:good_ratio_30m) > (6 * 0.01)
        for: 15m
        labels: { severity: page, slo: ttft }
```

Note `0.01` is the budget (1 − 0.99 target) appearing literally in the expression. Keep the SLO target in exactly one place — ideally generated from the SLO document (Sloth, OpenSLO, Pyrra all do this) so the alert, the dashboard and the document cannot drift apart.

---

## The alerts that should exist besides SLO burn

Burn-rate alerts are symptom-based, which is correct for paging. You still want a short list of **cause-based** alerts — as tickets, not pages — that catch conditions users haven't felt yet:

| Alert | Condition | Severity | Why |
|---|---|---|---|
| KV occupancy saturated | `avg_over_time(vllm:gpu_cache_usage_perc[10m]) > 0.9` | ticket | preemption and ITL tails are imminent ([Phase 4 lesson 5](../phase-4/05-paged-attention.md)) |
| Preemption storm | `rate(vllm:num_preemptions_total[5m]) > threshold` | ticket→page | the engine is thrashing; capacity or config is wrong |
| Queue persistently non-empty | `avg_over_time(num_requests_waiting[10m]) > N` | ticket | ρ→1; scale before the SLO burns ([Phase 3 lesson 5](../phase-3/05-queueing-theory.md)) |
| Replica count below minimum | `count(up{job="inference"}) < N` | page | redundancy lost; one more failure = outage |
| Hardware fault | XID in {48,79,94,95} or ECC DBE > 0 | page (node-level) | drain now ([lesson 3](03-gpu-and-host-telemetry.md)) |
| Throttling sustained | throttle reason != 0 for 15 m | ticket | you are paying for clocks you don't get |
| Prefix cache hit-rate collapse | hit rate drops >30% vs 7-day baseline | ticket | routing regression ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)); TTFT will follow |
| Finish-reason shift | `length`-finishes or empty outputs up >3σ | ticket | quality regression or a prompt-template break ([lesson 8](08-canary-and-shadow-traffic.md)) |
| Cost per 1M tokens up | >20% vs 7-day baseline | ticket | a silent efficiency regression ([lesson 9](09-cost-per-million-tokens.md)) |
| Metrics pipeline down | `absent(up{job="inference"})` or scrape failures | page | you are blind; treat as an incident |

**Everything not on this list or the burn-rate list should not page.** The test for every proposed page: *"is a human required to do something right now that automation can't?"* If the answer is no, it's a ticket or a dashboard panel.

---

## Practical failure modes of this machinery

| Failure | Symptom | Fix |
|---|---|---|
| SLO target above measured capability | permanent budget exhaustion, alert fatigue, people mute the page | re-measure at SLO-load, set an achievable target, then invest to raise it |
| Shed requests counted as failures | load shedding, which *protects* the SLO, appears to violate it | exclude shed from the denominator; give shedding its own SLO |
| No quality SLI | green dashboards during a real user-visible regression | add finish-reason/empty-output metrics at minimum ([lesson 1](01-what-to-measure.md)) |
| Alert on p99 instead of burn rate | 4 a.m. pages at 3 req/s; slow burns missed | burn-rate alerts with short-window confirmation |
| Budget policy unwritten | budget exhausted, nothing changes, SLO becomes decoration | write the freeze/rollback policy before you need it |
| One SLO per model version | churn on every deploy; no continuity | SLO is on the *service*; model version is a label used for attribution |
| Window too short (e.g. 1 h SLO) | noisy, meaningless | 28 d rolling for user-facing SLOs; short windows only inside burn-rate alerts |
| Prober-only measurement | synthetic looks fine, real traffic is broken | SLIs from real traffic at the router; probers for black-box availability only |

---

## Do this now (45 minutes)

1. **Write two real SLO documents** (availability and TTFT) for your Phase-3 server using the template above, including the exclusion list and the burn policy. Verify the bucket boundary required by your TTFT SLI actually exists in your histogram; if not, add it and redeploy — that mismatch is the most common reason an SLO number is subtly wrong.
2. **Implement the four-row burn-rate alert set** in Prometheus for that TTFT SLO. Then test it: inject latency (add a `sleep` in the prefill path or throttle the GPU) so ~15% of requests exceed the threshold, and confirm the fast-burn alert fires within minutes *and* clears within minutes of the fix. An untested alert is a hypothesis.
3. **Compute your current budget position.** Over your longest available window, calculate `1 − good/valid` against your chosen target and express it as "X% of budget consumed, Y days of headroom at the current burn rate." That one sentence is the entire content of a weekly reliability review, and producing it monthly is what makes an SLO real rather than a document.

---

**Next:** [Dashboards and diagnosis →](06-dashboards-and-diagnosis.md) — the four-dashboard hierarchy, and a symptom→cause decision tree that turns "TTFT is up" into a specific Phase-3-to-6 lever in under five minutes.
