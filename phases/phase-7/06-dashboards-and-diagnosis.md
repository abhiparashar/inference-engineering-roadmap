# 6 — Dashboards and Diagnosis: From "TTFT Is Up" to a Specific Lever

> **You'll be able to say:** "I build four dashboards, not forty: a service overview (the SLOs and nothing else), a capacity/saturation view, a per-replica view, and a cost view. Each panel exists to advance a decision in a written decision tree. Given 'p99 TTFT doubled at 14:05', I can split queue wait from prefill time, check whether prompt lengths shifted, check KV occupancy and preemptions, check prefix-cache hit rate, check clocks and stragglers, and land on the responsible layer — engine config, routing, capacity, hardware, or client behaviour — in under five minutes, because the panels are ordered the way the tree branches."

[Lesson 5](05-slos-and-error-budgets.md) decided what pages you. This lesson decides what you look at once paged. The skill being built is not Grafana proficiency; it is **owning a diagnosis procedure** that a tired person can follow at 3 a.m. and that a new teammate can learn in a day.

---

## Why most inference dashboards are useless

```
  THE ANTI-PATTERN                            WHY IT FAILS
  ─────────────────────────────────────────────────────────────────────────────
  60 panels, one per available metric         nobody can hold it in working memory
  every panel shows an average                averages hide the tail = hide the incident
  GPU utilization as the headline number      always ~100%; conveys nothing (lesson 3)
  latency without a token-rate panel          can't tell "slow" from "bigger prompts"
  no baseline/comparison                      "is 1.2 s bad?" is unanswerable
  per-replica lines on a fleet panel          40 lines of spaghetti; use heatmaps/quantiles
  no link to runbook/trace                    dashboard ends where the work begins
```

The fix is a **hierarchy**, where each level answers one question and hands off to the next.

```
  L1  SERVICE OVERVIEW      "are we meeting the promise?"        → SLOs, budget, traffic
        │  (if no: which dimension is broken?)
  L2  CAPACITY / SATURATION "are we out of room, and of what?"   → queue, KV, batch, tokens
        │  (if room exists: is it one replica or all of them?)
  L3  REPLICA DETAIL        "which replica, and is it hardware?" → per-pod, DCGM, clocks
        │  (if a single request: what did its time look like?)
  L4  TRACE                 "where did THIS request's time go?"  → exemplar → span tree
  ────────────────────────────────────────────────────────────────────────────────────
  L5  COST                  "what do we pay per 1M tokens, and why?"  (weekly, not on-call)
```

---

## L1 — Service overview (the only dashboard on the wall)

Eight panels, in this order. If a panel does not map to an SLO or to the first branch of the decision tree, it doesn't belong here.

| Panel | Query shape | Reads as |
|---|---|---|
| SLO compliance, per SLO, current 28 d | `inf:ttft:good_ratio_28d` vs target | the promise |
| Error budget remaining + burn rate | `(good_ratio − target)/(1 − target)` | how much risk is left |
| TTFT p50/p95/p99 | `histogram_quantile` over the fleet | the headline latency |
| TPOT p50/p95/p99 + ITL p99 | ratio-of-sums + quantile | streaming pace and stalls |
| Traffic: req/s and **prompt + output tokens/s** | `rate()` counters | load, in both units ([lesson 1](01-what-to-measure.md)) |
| Errors by class (stacked) | `rate(errors_total)` by `class` | which failure class ([lesson 1](01-what-to-measure.md) taxonomy) |
| Finish reasons (stacked %) | `rate(finish_total)` by `reason` | the silent-quality canary |
| Replica count + model version mix | `count(up)`, `count by (model_version)` | "did a deploy just happen?" |

Two design rules that do a lot of work:

- **Annotate deploys.** Grafana annotations from your CD pipeline turn "latency changed at 14:05" into "latency changed 90 seconds after the 14:03 rollout." This single feature resolves a large fraction of incidents without any further investigation.
- **Put the window comparison on the panel.** Every latency panel should have a "same time last week" overlay or a `offset 7d` series. Diurnal traffic means absolute numbers are meaningless without it ([lesson 9](09-cost-per-million-tokens.md) uses the same trick for cost).

---

## L2 — Capacity and saturation

| Panel | Metric | Threshold that matters |
|---|---|---|
| Queue depth | `num_requests_waiting`, fleet sum + p95 per replica | sustained > 0 means ρ→1 |
| Queue wait time | `queue_wait_seconds` p95 | the part of TTFT you own |
| Running batch size vs limit | `num_requests_running` / `max_num_seqs` | at the limit = concurrency-capped |
| **KV-cache occupancy** | `gpu_cache_usage_perc`, max and mean over replicas | > 0.9 sustained = preemption zone |
| Preemptions | `rate(num_preemptions_total)` | any sustained rate is a config/capacity bug |
| Token-budget utilization | tokens per step / `max_num_batched_tokens` | prefill/decode mix pressure |
| Prefix-cache hit rate | `rate(prefix_cache_hits)/rate(prefix_cache_queries)` | drops predict TTFT rises |
| Prompt-length + output-length heatmaps | histograms over time | detects client-side workload shifts |
| Admission rejects | `rate(errors{class="shed"})` | shedding is working / overloaded |
| Replica-level TTFT heatmap | p95 per pod, as a heatmap | one-bad-replica detection at a glance |

The token-length heatmaps are the panel teams most often lack and most often need: **half of "we got slower" incidents are "our traffic changed"** — a customer started sending 32k-token prompts, or a prompt template grew a 4k-token preamble. Without the distribution over time, that is invisible and you will spend the incident looking at your own code.

---

## L3 — Replica detail (the hardware-vs-software split)

Per-pod, filterable by replica, six rows matching the health model from [lesson 3](03-gpu-and-host-telemetry.md):

```
  ALIVE      up, restarts, /health latency, engine version, model version
  FAST       step time, tokens/s, SM_CLOCK, throttle reasons, temp/power
  FULL       KV occupancy, batch size, queue depth, preemptions
  CORRECT    finish reasons, error rate, XID/ECC counters
  HOST       container CPU throttling, RSS, /dev/shm, NIC, disk
  COLLECTIVE (TP/PP only) NVLink TX/RX, collective wait time, per-rank step-time spread
```

The **per-rank step-time spread** panel is the straggler detector for TP replicas: a single throttled or degraded GPU sets the pace for the whole group and appears on every other rank as "collective wait" ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)). Without it you conclude the whole replica is slow; with it you point at one GPU.

---

## The decision tree

Print this. It is the deliverable of the lesson.

```
 SYMPTOM: TTFT p99 UP
 ├─ 1. Did traffic change?   (req/s, tokens/s, prompt-length heatmap)
 │     ├─ prompt lengths grew            → client-side change. Verify per-tenant.
 │     │                                   Lever: per-tenant limits, chunked prefill tuning
 │     ├─ QPS grew beyond capacity       → capacity. Lever: scale out (Phase 6 L8),
 │     │                                   admission control (Phase 3 L6)
 │     └─ traffic flat                   → go to 2
 ├─ 2. Split TTFT: queue wait vs prefill time  (the single most informative split)
 │     ├─ QUEUE dominates
 │     │    ├─ batch size at max_num_seqs        → concurrency-capped: raise limit if KV allows
 │     │    ├─ KV occupancy > 0.9 / preemptions  → memory-capped: quantize KV, more GPUs,
 │     │    │                                      lower max_num_seqs, shorter max_model_len
 │     │    └─ neither, GPU idle-ish            → HOST-side: cgroup CPU throttling, tokenizer,
 │     │                                          event-loop blocking (lesson 3)
 │     └─ PREFILL dominates
 │          ├─ prefix-cache hit rate dropped    → routing regression (Phase 6 L7),
 │          │                                     cache eviction pressure, template change
 │          ├─ prompts longer                   → see 1
 │          └─ compute slower per token         → go to 3
 ├─ 3. Is it hardware?      (SM_CLOCK, throttle reasons, temp, per-rank spread, XID)
 │     ├─ clocks down / throttle bits set       → power or thermal cap (lesson 3)
 │     ├─ one rank slow                         → straggler GPU: drain the replica
 │     └─ XID/ECC events                        → drain node, RMA path (lesson 10)
 ├─ 4. Is it one replica or all?   (per-pod TTFT heatmap)
 │     ├─ one pod hot                           → bad replica (cold cache, bad GPU, or the
 │     │                                           router is overweighting it) → drain/restart
 │     └─ uniform                                → fleet-wide: config, deploy, or capacity
 └─ 5. Did we deploy?       (deploy annotations, model_version mix)
       ├─ new engine/model version              → roll back first, diagnose after (lesson 8)
       └─ no deploy                              → config drift? check ConfigMap/flags diff

 SYMPTOM: TPOT / ITL p99 UP (streaming feels choppy)
 ├─ batch size grew a lot        → expected: TPOT rises with batch size. Check it's within SLO;
 │                                 this is the throughput/latency trade (Phase 3 L4)
 ├─ preemptions > 0              → KV thrashing: reduce max_num_seqs or add KV capacity
 ├─ long prefills interleaving   → enable/tune chunked prefill so prefill doesn't stall decode
 ├─ ITL spikes periodic (~secs)  → host-side pauses: GC, logging, metrics scrape blocking loop
 └─ clocks/throttling            → hardware (lesson 3)

 SYMPTOM: ERROR RATE UP
 ├─ class=capacity (OOM)         → gpu_memory_utilization too high, max_model_len too long,
 │                                 or a long-prompt spike. Fix config, don't retry harder
 ├─ class=timeout                → upstream timeout < your p99 E2E. Align timeouts (lesson 7)
 ├─ class=shed                   → overload: is this correct behaviour? check offered load
 ├─ class=infra                  → XID/eviction/node loss (lesson 3, 10)
 └─ class=quality                → model/prompt/template regression (lesson 8)

 SYMPTOM: COST PER 1M TOKENS UP
 ├─ output tokens/s per replica down      → efficiency regression: batch size, preemption,
 │                                          throttling, or a deploy changed the config
 ├─ replica count up at flat traffic      → autoscaler thrash or headroom too generous
 ├─ traffic mix shifted to long prompts   → prefill-heavy: cache/prefix strategy, pricing
 └─ idle replicas / low batch at trough   → scale-in policy, spot mix (lesson 9)
```

**The "split queue wait from prefill" step at node 2 is the highest-value move in the whole tree.** It takes one panel and it separates "we are over capacity" from "compute got slower," which are different teams, different fixes, and different hours of the night.

---

## Panel construction details that matter

| Practice | Reason |
|---|---|
| Heatmaps for per-replica and per-length distributions | 40 pods × 3 quantiles = unreadable lines; a heatmap shows the outlier instantly |
| `histogram_quantile` over `sum by (le)` — never `avg` of quantiles | mathematically wrong otherwise ([lesson 2](02-instrumenting-with-prometheus.md)) |
| Fixed y-axis ranges on SLO panels, with a threshold line | makes "bad" visible pre-attentively; auto-scaling hides magnitude |
| Every panel has a description: what decision it drives | the panel that drives nothing gets deleted at the next review |
| Links: panel → runbook, panel → trace (exemplars), panel → logs (same `request_id`) | on-call time is dominated by navigation, not thinking |
| Template variables for `model`, `model_version`, `tenant_class`, `pod` | one dashboard instead of one per model |
| Recording rules behind expensive panels | dashboards that time out during incidents are worse than none ([lesson 2](02-instrumenting-with-prometheus.md)) |
| Deploy + incident annotations | correlation for free |

### One number to keep visible at all times

If you may only have a single panel: **`output tokens/sec` and `p95 TTFT`, overlaid with `offset 7d`.** Throughput delivered plus the latency it was delivered at, compared to normal, is the smallest complete statement about an inference service's health. Everything else is elaboration.

---

## Diagnosis walkthrough: a realistic incident

```
  14:05  page: TTFTSLOBurnRateFast (14.4x, 1h/5m)
  14:06  L1: TTFT p99 1.9 s (was 0.35). Traffic flat. Errors flat. No deploy annotation.
         Finish reasons normal. Replica count 12/12. → not a deploy, not a crash
  14:07  L2: queue wait p95 = 1.55 s (was 0.02) ; prefill p95 = 0.21 s (unchanged)
         → QUEUE dominates. Compute is fine. Node 2 of the tree, left branch
  14:08  L2: batch size 64/64 pinned at max_num_seqs; KV occupancy 0.71 (not full);
         preemptions 0
         → concurrency-capped, NOT memory-capped
  14:09  L2: prefix-cache hit rate dropped 0.62 → 0.11 at 14:03
         → routing changed? L1 shows model_version mix unchanged...
  14:10  L3: one pod (vllm-7d9-abc) receives 4x the traffic of its peers
         → router is overweighting one replica: a sticky-routing hotspot
           (Phase 6 lesson 7: a popular shared prefix pinned to one replica)
  14:12  MITIGATE: raise the router's per-replica concurrency cap → spill over to peers;
         TTFT p99 back to 0.5 s by 14:15
  14:40  FIX: bound the sticky-hash load per replica (consistent hashing with bounded loads),
         alert on per-replica traffic skew > 2x
```

Total: six minutes to mitigation, because each step was a panel that existed in the order the tree branches. The postmortem action item is not "add a dashboard" — it is "add the skew alert," which is exactly what [lesson 10](10-incident-response-and-chaos.md) means by a useful postmortem.

---

## Do this now (60 minutes)

1. **Build the L1 dashboard** (eight panels, that order) against your instrumented Phase-3 server, with deploy annotations wired from your run script and an `offset 7d` overlay on the latency panels. Then delete every panel you had before that isn't on the L1/L2/L3 lists — the deletion is the exercise.
2. **Build the queue-vs-prefill split panel** and prove it discriminates: run a load test above saturation (queue dominates) and then a single-request test with a 32k-token prompt (prefill dominates). Two workloads, two distinct panel signatures, one decision tree node validated.
3. **Run the tree cold.** Have someone (or a script) induce one of: a CPU limit that throttles the container, `max_num_seqs` set to 4, a lowered GPU power limit, or a traffic generator with 10× longer prompts — without telling you which. Time yourself to the correct diagnosis using only the dashboards. Any step where you had to leave the dashboard and read code is a missing panel; add it and re-run.

---

**Next:** [Reliability and graceful degradation →](07-reliability-and-degradation.md) — the failure catalogue of a real inference fleet, why retries are more dangerous here than in ordinary services, and how to shed, hedge, and degrade instead of falling over.
