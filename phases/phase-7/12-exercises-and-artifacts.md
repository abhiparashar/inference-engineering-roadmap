# 12 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 6 made a fleet exist. Phase 7 proves you can **run it as a service**: that you know what it promises, whether it is keeping that promise, what it costs per million tokens, who gets paged when it breaks, and how a new model version ships without betting the SLO on it.

Everything here is doable on a laptop with a stub or a tiny CPU model. This is the phase where **hardware is not the bottleneck and discipline is** — and it is also the phase that most directly changes how a hiring manager or a tech lead evaluates you, because it is the material that separates "can build a server" from "can be trusted with production."

---

## Warm-up exercises

**Measurement and instrumentation** — do these first; the rest depend on them

1. **Write the SLI contract** for one service: every metric, its type, labels, buckets, vantage point, and the decision it drives. Delete any metric that drives no decision. ([1](01-what-to-measure.md))
2. **Tail amplification.** From per-token timestamps over ≥50 requests of ≥200 tokens, compute the ITL distribution, then the fraction of *requests* containing at least one gap above the p99-of-gaps. Explain, in one sentence, why per-step SLIs must be tighter than request-level intuition suggests. ([1](01-what-to-measure.md))
3. **Bucket error.** Deliberately configure TTFT buckets with a gap around your p99, compare `histogram_quantile` against a client-side exact p99, then add a boundary at your SLO threshold and show the exact-ratio query matching truth. Report both errors as percentages. ([2](02-instrumenting-with-prometheus.md))
4. **Cardinality budget.** Count your series (`count({__name__=~"inf_.*"})`), then compute the count at 50 replicas, and again with a `tenant_id` label at 5,000 tenants. Put the three numbers next to your Prometheus memory limit. ([2](02-instrumenting-with-prometheus.md))
5. **Abort accounting.** Disconnect 20 clients mid-stream. Prove (a) the abort counter increments, (b) partial output tokens are counted, (c) any KV/slot resource is released promptly. Whichever fails is a real bug you just found. ([2](02-instrumenting-with-prometheus.md), [7](07-reliability-and-degradation.md))

**Hardware and traces**

6. **The utilization decoy, measured.** Record `GPU_UTIL`, `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`, power and tokens/sec at batch 1 and at your maximum batch. Two rows, one conclusion. (No GPU? Do it on a rented hour, or reason it through from a published DCGM field reference and mark it as inferred.) ([3](03-gpu-and-host-telemetry.md))
7. **The live roofline.** Prefill-heavy vs decode-heavy load; show `PIPE_TENSOR_ACTIVE` and `DRAM_ACTIVE` swapping dominance. State which lever each regime implies. ([3](03-gpu-and-host-telemetry.md), [Phase 2 lesson 5](../phase-2/05-roofline-model.md))
8. **XID triage table.** For XID 13, 31, 48, 63, 74, 79, 94, 95: write cause, blast radius, and the exact action (kill process / drain replica / cordon node / RMA). One line each, from memory afterwards. ([3](03-gpu-and-host-telemetry.md))
9. **A trace that diagnoses.** Produce one trace whose dominant span is queue wait, one whose dominant span is prefill, and one containing a decode stall event. For each, name the fix and the phase it comes from. ([4](04-tracing-and-logging.md), [6](06-dashboards-and-diagnosis.md))
10. **Tail-sampling window.** Configure tail-based sampling; then set `decision_wait` below your longest request duration and show the slow traces disappear. Explain why this failure is silent. ([4](04-tracing-and-logging.md))
11. **Privacy audit.** Grep your own code and config for anything that could put prompt text into logs, traces or metrics. Write the one-paragraph content-capture policy you would defend in a review. ([4](04-tracing-and-logging.md))

**SLOs, alerts, cost**

12. **Two SLO documents** (availability + TTFT) with explicit exclusion lists and burn policy. Justify each exclusion in one clause. ([5](05-slos-and-error-budgets.md))
13. **Burn-rate arithmetic.** For a 99% SLO over 28 days at 100 req/s: how many bad requests is the budget? How long until exhaustion at burn rates 1, 6, and 14.4? What fraction of the budget does a 20-minute total outage consume? ([5](05-slos-and-error-budgets.md))
14. **Alert test.** Fire your fast-burn alert with injected latency and measure detection time and clear time. Then compute what a naive `p99 > threshold for 5m` alert would have done at 1/10 of the traffic. ([5](05-slos-and-error-budgets.md))
15. **Cold diagnosis drill.** Have someone inject one of {CPU limit throttling, `max_num_seqs`=4, lowered power limit, 10× longer prompts, a hot replica} without telling you which. Time yourself to the correct diagnosis using dashboards only. ([6](06-dashboards-and-diagnosis.md))
16. **Timeout audit.** List client → ingress → router → engine → generation-cap timeouts with numbers and verify strict inner-< outer ordering. Fix any inversion and describe the failure it was causing. ([7](07-reliability-and-degradation.md))
17. **Retry-storm simulation.** Load a saturated service with (a) no retries, (b) 3 retries no backoff, (c) 3 retries with full jitter, (d) a 10% retry budget. Report success rate, p99, and total GPU work wasted for each. ([7](07-reliability-and-degradation.md))
18. **Degradation ladder.** Write your service's rungs 0-6 with the metric and label for each, then implement one rung and prove your SLO query and logs can see when it was active. ([7](07-reliability-and-degradation.md))
19. **Sample size.** For your highest-traffic route, compute the canary sample size needed to detect a 1-point regression in your most checkable quality metric, and convert it to a bake time at 1% and 5% traffic. ([8](08-canary-and-shadow-traffic.md))
20. **Cost decomposition.** Compute formula cost per 1M output tokens at your SLO, then effective cost from allocated replica-hours over a day; report the ratio and attribute the gap to trough waste, headroom, goodput loss. ([9](09-cost-per-million-tokens.md))
21. **The GPU-choice trap.** Price the same model on two GPU types you can get numbers for (measured or clearly-labelled vendor figures) in $/1M tokens at a fixed SLO. Show that the cheaper $/hour option is not necessarily cheaper per token. ([9](09-cost-per-million-tokens.md))
22. **Three chaos experiments**, hypothesis written first: SIGKILL a replica; `SIGSTOP` the engine (watchdog test); blackhole the metrics pipeline. Record predicted vs measured. ([10](10-incident-response-and-chaos.md))

---

## Conceptual self-check (no notes)

1. Why is `DCGM_FI_DEV_GPU_UTIL` useless as a scaling or capacity signal, and what are the two things it *is* good for? ([3](03-gpu-and-host-telemetry.md))
2. Why can a Prometheus **summary** not be aggregated across replicas while a **histogram** can? ([2](02-instrumenting-with-prometheus.md))
3. Where must a bucket boundary sit for an SLO query to be exact rather than interpolated, and why does that make the SLO document and the histogram config the same decision? ([2](02-instrumenting-with-prometheus.md), [5](05-slos-and-error-budgets.md))
4. Why do shed (429) requests belong outside the SLO denominator, and what goes wrong if you include them? ([5](05-slos-and-error-budgets.md))
5. State the multi-window multi-burn-rate configuration and explain what each of the two windows is for. ([5](05-slos-and-error-budgets.md))
6. A p99 TTFT doubling with flat traffic and no deploy: give the first split you'd make and the two branches it produces. ([6](06-dashboards-and-diagnosis.md))
7. Why is an inference retry categorically more dangerous than a web-service retry, and what are the three rules that make retries safe? ([7](07-reliability-and-degradation.md))
8. Why can't exact-match diffing validate a new model version, and what do you use instead at each cost tier? ([8](08-canary-and-shadow-traffic.md))
9. Write the cost-per-1M-tokens formula and name the qualifier that makes it honest. Then explain the utilization and goodput factors that separate it from the invoice. ([9](09-cost-per-million-tokens.md))
10. Why does "mitigate before diagnose" hold even when you don't know the cause, and what engineering investments make it cheap? ([10](10-incident-response-and-chaos.md))
11. What probe catches a replica that is alive, healthy, and producing no tokens — and why do liveness/readiness probes miss it? ([7](07-reliability-and-degradation.md), [Phase 6 lesson 9](../phase-6/09-multi-node-operations.md))
12. Which failure class shows 200 OK on every dashboard, and what is the cheapest instrumentation that detects it? ([1](01-what-to-measure.md), [8](08-canary-and-shadow-traffic.md))

If any answer takes more than ~60 seconds, re-read the linked lesson — these are the questions that get asked in a production-engineering interview, almost verbatim.

---

## Exit artifact

Produce **Option A**. A + B is the strongest pair; C is cheap and disproportionately useful in design reviews and interviews.

### Option A — Project 13: the observability stack (required)

`projects/13-observability-stack/README.md` containing:

- The **instrumented server** (streaming-aware TTFT/ITL, abort handling, structured completion events with no prompt text) and the **`docker-compose`** stack: Prometheus with exemplars, Grafana, Jaeger, Alertmanager (+ DCGM if you have a GPU).
- **`rules.yml`**: recording rules and the four-row burn-rate alert set against a written SLO.
- The **L1 + L2 dashboards** (panels from [lesson 6](06-dashboards-and-diagnosis.md)) with deploy annotations and 7-day overlays — exported JSON committed, plus screenshots.
- The **fault-injection table**: for each injected fault (prefill slowdown, decode stalls, overload, killed replica, frozen engine, blackholed metrics) — which alert fired, detection time, and which panel identified the cause.
- One **trace** each for a queue-dominated and a prefill-dominated slow request.

### Option B — Project 14: the canary/shadow harness (large)

`projects/14-canary-shadow-harness/README.md` containing:

- The **shadow proxy** with proven isolation (killing the canary backend does not change client latency — show the measurement).
- The **analysis script** with thresholds declared *before* the run, and two results: **REJECT** for a deliberately degraded canary (e.g. truncated outputs — green on every latency metric, caught by length ratio and similarity) and **PROMOTE** for an identical-config control.
- The **required sample size** computed for your primary quality gate, and the bake time it implies at 1% / 5% / 25% traffic.
- Breakdowns of the primary metric by `tenant_class` and prompt-length bucket, with a sentence on Simpson's-paradox risk.
- Ideally: the same harness run against two *real* engine configs (FP16 vs INT8/AWQ), making this simultaneously your Phase-4 quantization quality answer.

### Option C — The SRE document set (cheap, high signal)

Three pages for one real service:

1. **SLO document** — SLIs, targets, windows, exclusions, burn policy, owner (from [lesson 5](05-slos-and-error-budgets.md)).
2. **One runbook** for your TTFT burn alert, in the [lesson 10](10-incident-response-and-chaos.md) structure, with working mitigation scripts (`drain-replica`, `rollback`, `set-admission`, `degrade`) and a `DO NOT` section.
3. **Cost review** — formula vs effective cost per 1M tokens, the decomposition, three priced options with risk and engineering cost, and a one-paragraph recommendation ([lesson 9](09-cost-per-million-tokens.md)).

Plus, if you ran the chaos experiments: a **postmortem** for one of them using the [lesson 10](10-incident-response-and-chaos.md) skeleton, with typed action items.

Update [`projects/README.md`](../../projects/README.md) status for 13 and 14 when done, and run the [production-readiness checklist](../../playbooks/production-readiness-checklist.md) against the result — this is the phase where that checklist stops being aspirational.

---

## How you know you're ready for Phase 8

- [ ] You can state your service's SLIs, SLOs, exclusions, and current error-budget position from memory.
- [ ] Your histogram buckets have a boundary at every SLO threshold, and you know why that matters.
- [ ] You can name the five error classes and point at the metric that detects each — including silent quality failures.
- [ ] You can look at one dashboard and split TTFT into queue wait vs prefill, and you know which branch each implies.
- [ ] You have fired and cleared a burn-rate alert on purpose, and know its detection latency.
- [ ] You can say what your service costs per 1M output tokens at the SLO, and what the invoice-vs-formula ratio is.
- [ ] You have a written mitigation path — rollback, shed, drain, degrade — that takes seconds, not a rebuild.
- [ ] You have deliberately broken something (killed a replica, frozen an engine, blackholed metrics) and measured recovery.
- [ ] You would refuse to ship a quantization without paired quality evidence, and you know what evidence to ask for.

---

## Where these ideas come back

| Phase 7 idea | Comes back as |
|---|---|
| Prometheus histograms and burn-rate alerts | the CI benchmark gate's thresholds ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)) |
| Readiness/liveness/progress watchdog | k8s probe configuration and `terminationGracePeriodSeconds` ([Phase 8 lesson 4](../phase-8/04-kubernetes-for-gpu-serving.md)) |
| Canary gates and rollback windows | progressive-delivery pipelines and model-registry digest pinning ([Phase 8 lessons 6-7](../phase-8/06-deploying-and-rollouts.md)) |
| Cost per 1M tokens | the capstone cost-optimization case study ([Phase 10 lesson 5](../phase-10/05-capstone-cost-optimization.md)) |
| Per-stage latency budgets and timeouts | RAG/agentic pipelines where retrieval is a separate stage ([Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md)) |
| Quality metrics and content-capture policy | security, PII handling, and multi-tenancy ([Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md)) |
| Error taxonomy and degradation ladder | the production-readiness checklist you apply to every capstone ([Phase 10 lesson 2](../phase-10/02-engineering-standards.md)) |

---

**Next:** [Phase 8 — MLOps Glue: Containers, Orchestration, CI/CD, IaC](../phase-8/README.md). Phase 7 told you whether the service is healthy and what it costs; Phase 8 is how it gets built, shipped, scheduled and provisioned reproducibly — the layer that decides whether your rollback takes 30 seconds or 30 minutes.
