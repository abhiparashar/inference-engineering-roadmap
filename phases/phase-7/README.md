# Phase 7 — Observability, Reliability, and Cost (Deep Dive)

> **Goal:** stop judging an inference system by benchmark numbers and start judging it the way a business does — against an SLO, an error budget, a cost per million tokens, and an incident record. By the end of this phase you can instrument a streaming engine correctly, define SLOs that survive contact with real traffic, alert on burn rate instead of noise, diagnose a latency regression to a specific layer in minutes, ship a new model version behind automated quality gates, price the service per 1M tokens, and break your own fleet on purpose before it breaks itself.

This folder is the long-form version of [Phase 7 in the ROADMAP](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference). Phases 3-6 built and scaled the thing. This phase is everything that happens *after* it works once — which is where the job actually is.

The intellectual core:

> **You cannot operate what you cannot measure, you cannot promise what you cannot measure, and in inference the two failure modes that matter most are invisible to the metrics people copy from web services: a replica that is alive and producing nothing, and a response that is 200 OK and useless. Everything else — batching, quantization, parallelism — is an optimization inside a contract that this phase defines: an SLI, an SLO, an error budget, and a cost per million tokens at that SLO.**

---

## Prerequisites

- **[Phase 3 lessons 1, 5, 7](../phase-3/01-what-a-serving-system-is.md)** — TTFT/TPOT/E2E definitions, queueing theory (ρ→1 is why latency explodes), and honest open-loop load generation with percentiles. Every SLI here is one of those metrics made permanent, and every capacity claim is a load-test claim.
- **[Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)** — admission control and load shedding. Shedding is the correct overload response, which is why lesson 5 excludes it from the SLO denominator.
- **[Phase 4 lessons 4-6](../phase-4/04-kv-cache-optimization.md)** — KV-cache economics, paging, prefix caching. KV occupancy is *the* saturation metric, preemption is *the* tail-latency source, and prefix-cache hit rate is a cost lever.
- **[Phase 5 lesson 3](../phase-5/03-vllm-in-production.md)** — vLLM's `/metrics` and flag-to-mechanism tuning. vLLM's metric set is the answer key for "what should I even emit."
- **[Phase 6 lessons 7-9](../phase-6/07-prefix-aware-routing.md)** — routing, autoscaling with minute-scale cold starts, multi-node failure modes. Lessons 6-10 here assume a fleet with replicas, routing skew, and gang-scheduled blast radius.

### About hardware

- **Everything in this phase runs on a laptop.** The instrumentation, Prometheus/Grafana/Jaeger stack, SLO math, burn-rate alerts, shadow proxy, quality analysis and chaos experiments all work against a CPU model or the stub server in [lesson 11](11-build-observability-and-canary.md). This is the cheapest high-value phase in the roadmap.
- **A GPU adds one lesson's worth of measurement:** [lesson 3](03-gpu-and-host-telemetry.md)'s DCGM work (the utilization decoy, the live roofline, throttling). One rented GPU-hour is enough; ~2-4 hours if you also want real vLLM metrics and a real quantization comparison in [lesson 8](08-canary-and-shadow-traffic.md).
- **No GPU at all?** Do lesson 3 by reading the DCGM field reference and reasoning from [Phase 2 lesson 5](../phase-2/05-roofline-model.md), and mark those numbers as inferred in your artifact. Everything else is unaffected.

---

## The map of this phase

```
                        WHAT DOES THIS SERVICE PROMISE?
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
    MEASURE                      PROMISE                       PAY FOR
  lessons 1-4                  lessons 5-6                   lesson 9
  SLIs, Prometheus,            SLO, error budget,            $ per 1M tokens
  DCGM, traces/logs            burn-rate alerts,             at the SLO, and
        │                      diagnosis tree                 the utilization gap
        │                            │                            │
        └──────────────┬─────────────┴──────────────┬─────────────┘
                       ▼                            ▼
                 KEEP IT UP                    CHANGE IT SAFELY
                 lessons 7, 10                 lesson 8
                 failure catalogue,            shadow → canary → 100%,
                 timeouts, retries,            quality gates that catch
                 shedding, degradation         a "green dashboard" regression
                 ladder, runbooks, chaos
  ───────────────────────────────────────────────────────────────────────────────
  THE TWO INFERENCE-SPECIFIC BLIND SPOTS THIS PHASE EXISTS TO CLOSE
    1. GPU_UTIL reads ~100% at every load  ⇒ saturation must come from KV occupancy
                                             and queue depth, never utilization
    2. 200 OK can carry garbage            ⇒ availability metrics can be perfectly
                                             green during a total quality outage
```

Two things fall out of this picture that most engineers get backwards:

1. **Alerting on p99 crossing a threshold is worse than not alerting.** It pages on statistical noise at low traffic and stays silent through a slow burn that eats a month's budget in a week. Burn-rate alerting fixes both, and it costs four recording rules ([lesson 5](05-slos-and-error-budgets.md)).
2. **Cost per 1M tokens is not the number in the benchmark.** The formula prices a busy replica; the invoice prices every allocated GPU-hour, and the gap — trough waste, headroom, aborts, retries, canary capacity — is routinely a factor of two. That gap is where the savings are, and it is invisible to anyone measuring single-replica throughput ([lesson 9](09-cost-per-million-tokens.md)).

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [What to measure](01-what-to-measure.md) | "The golden signals for inference are TTFT/TPOT/ITL, requests *and* tokens per second, five distinct error classes, and KV occupancy plus queue depth as saturation. I know why per-step tails get amplified by output length." |
| 2 | [Instrumenting with Prometheus](02-instrumenting-with-prometheus.md) | "Histogram not summary; a bucket boundary at every SLO threshold; bounded labels with tenant identity in logs instead of metrics; and streaming-aware timing that records TTFT at first token and still counts aborted requests." |
| 3 | [GPU and host telemetry](03-gpu-and-host-telemetry.md) | "`GPU_UTIL` is a busy-flag. `SM_ACTIVE`/`PIPE_TENSOR_ACTIVE`/`DRAM_ACTIVE` are the live roofline. XID 48/79/94/95 means drain the node, and clock throttling explains the slowdown that looks like a code regression." |
| 4 | [Tracing and logging](04-tracing-and-logging.md) | "One trace splits a slow request into queue wait, prefill, decode and stream. `gen_ai.*` attributes, tail-based sampling that keeps the slow ones, exemplars to jump from p99 to a real trace — and a content policy that keeps prompts out of the pipeline." |
| 5 | [SLOs and error budgets](05-slos-and-error-budgets.md) | "SLI = good/valid, SLO = a target over a window, error budget = permission to take risk. I alert at 14.4× over 1 h confirmed by 5 min, and I exclude shed traffic from the denominator on purpose." |
| 6 | [Dashboards and diagnosis](06-dashboards-and-diagnosis.md) | "Four dashboards, not forty. Given 'TTFT p99 doubled' I split queue wait from prefill time and walk a written decision tree to the responsible layer in under five minutes." |
| 7 | [Reliability and graceful degradation](07-reliability-and-degradation.md) | "Retries cost a full prefill and arrive during overload, so they need budgets and jitter and must never follow a streamed token. Bounded queues, aligned timeouts, circuit breakers, a progress watchdog for the hang, and a degradation ladder that spends quality to keep availability." |
| 8 | [Canary and shadow traffic](08-canary-and-shadow-traffic.md) | "Shadow first for free latency evidence, then a percentage canary with automated analysis on latency, errors *and* quality. Exact-match diffing can't work; here's what does, and here's the sample size the gate requires." |
| 9 | [Cost per 1M tokens](09-cost-per-million-tokens.md) | "`(GPU-$/hr × GPUs) / (output tok/s × 3600) × 10⁶`, measured at the SLO — then divided by utilization and goodput to get the invoice. I can attribute cost per tenant and rank every Phase 3-6 lever by its effect on it." |
| 10 | [Incident response and chaos](10-incident-response-and-chaos.md) | "Mitigate before diagnose. I have a one-screen runbook per alert with copy-pasteable mitigations and a `DO NOT` section, blameless postmortems whose action items are alerts and limits, and a chaos suite that includes freezing a rank and blackholing my own metrics." |
| 11 | [Build: observability stack + canary harness](11-build-observability-and-canary.md) | "I stood up the whole stack, proved the burn-rate alert fires and clears under injected faults, and built a shadow harness that rejects a truncating canary which every latency metric called healthy." |
| 12 | [Exercises & exit artifact](12-exercises-and-artifacts.md) | "Here is the dashboard set with its fault-injection table, the canary report with a computed verdict, and the SLO/runbook/cost document trio." |

---

## How to work through this phase

1. **Instrument before you theorize.** Lessons 1-2 against your own Phase-3 server on day one. Every later lesson is a query over metrics you emitted yourself, and the exercises assume they exist.
2. **Write the SLO document before the alerts.** The bucket boundary in your histogram and the threshold in your SLO are the same decision ([lesson 2](02-instrumenting-with-prometheus.md), [lesson 5](05-slos-and-error-budgets.md)); discovering that in the wrong order means re-instrumenting.
3. **Test every alert by causing it.** The fault-injection endpoint in [lesson 11](11-build-observability-and-canary.md) exists for this. An untested alert is a hypothesis, and half of them are wrong in a way you'd rather find on a Tuesday afternoon.
4. **Delete panels.** The [lesson 6](06-dashboards-and-diagnosis.md) exercise is as much about removing dashboards as building them. A dashboard nobody can read during an incident is a liability.
5. **Do the cost arithmetic on real measurements.** [Lesson 9](09-cost-per-million-tokens.md) is the phase's most transferable skill and it takes a calculator plus your Phase-3 harness. Report cost *with* the SLO attached, always.
6. **Break something on purpose, early.** [Lesson 10](10-incident-response-and-chaos.md)'s `SIGSTOP` experiment finds a missing progress watchdog in most stacks, including production ones. Better you than a user.

**Time budget:** 2-3 weeks part-time. Lessons 1, 2, 5 and 9 are load-bearing; lesson 11 produces the artifact.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. **Define an SLO for a hypothetical chat product** — SLI, target, window, exclusions — and write the specific Prometheus alert rules that page someone when it's at risk. ([lessons 5](05-slos-and-error-budgets.md), [2](02-instrumenting-with-prometheus.md))
2. Explain **why GPU utilization is the wrong saturation signal** and name the two metrics that are right. ([lessons 1](01-what-to-measure.md), [3](03-gpu-and-host-telemetry.md))
3. Given "p99 TTFT doubled," **name the first split you make** and the two branches it produces. ([lesson 6](06-dashboards-and-diagnosis.md))
4. Explain **why a retry in an inference service is dangerous** and state the rules that make retries safe. ([lesson 7](07-reliability-and-degradation.md))
5. Describe **how you would ship an INT4 quantization to production** without betting the SLO on it, including what evidence gates each step. ([lesson 8](08-canary-and-shadow-traffic.md))
6. Write the **cost per 1M tokens formula**, state the qualifier that makes it honest, and explain the gap between it and the invoice. ([lesson 9](09-cost-per-million-tokens.md))

## Projects that belong to this phase

- **[13 — Prometheus + Grafana observability stack](../../projects/README.md)** (small): instrumented server, metrics/traces stack, dashboards, burn-rate alerts, and a fault-injection table proving each alert fires and clears. **Required Phase 7 artifact.**
- **[14 — Canary/shadow deployment harness with automated quality diff](../../projects/README.md)** (large): traffic mirroring with hard isolation, paired latency and quality comparison, and a promote/reject verdict computed against pre-declared thresholds. **This is the Phase 7 exit artifact.**

Also do the [Phase 7 labs](../../labs/README.md#phase-7-lab--observabilitysre) — the SRE-book SLO chapter and the DCGM metric-selection exercise pair directly with lessons 5 and 3.

---

Next after this: **[Phase 8 — MLOps Glue: Containers, Orchestration, CI/CD, IaC](../phase-8/README.md)**. Phase 7 tells you whether the service is healthy and what it costs; Phase 8 is how it gets built, scheduled and provisioned reproducibly — and whether your rollback takes 30 seconds or 30 minutes.
