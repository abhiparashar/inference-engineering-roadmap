# 10 — Incident Response, Runbooks, and Chaos Testing

> **You'll be able to say:** "An incident has a structure — detect, declare, mitigate, diagnose, resolve, review — and mitigation comes before diagnosis. I can write a one-screen runbook whose first section is 'how to make it stop' (roll back, shed, drain, scale, fail over) and whose second is 'how to find out why'. I know the inference-specific incident catalogue and the mitigation for each entry. I run blameless postmortems whose action items are alerts, limits and automation — never 'be more careful'. And I inject failures deliberately: kill a rank of a TP group, throttle a GPU, fill the KV cache, revoke a spot instance, and blackhole the metrics pipeline, because a failure mode you have never seen is a failure mode you cannot debug at 3 a.m."

[Lessons 1-9](01-what-to-measure.md) built the instruments and the guardrails. This lesson is the human protocol that uses them, and the deliberate practice that keeps it sharp. This is the "boring" material that makes the difference between an engineer who can build a serving system and one a company trusts to run it.

---

## The shape of an incident

```
  DETECT     an alert fires (lesson 5) or a human reports it
     │       ▸ MTTD matters: an SLO burn alert should beat the first customer ticket
  DECLARE    name it, assign roles, open a channel. Cheap to declare, expensive to delay.
     │       ▸ roles: Incident Commander (decides, communicates), Ops (hands on keyboard),
     │         Comms (status page, stakeholders). One person may hold all three at 3 a.m.,
     │         but the IC role must be explicit or two people will "fix" it simultaneously.
  MITIGATE   STOP THE BLEEDING BEFORE UNDERSTANDING IT            ◀── the core discipline
     │       ▸ roll back, shed load, drain the bad replica, scale out, fail over, degrade
     │       ▸ "we don't know why yet" is not a reason to delay a rollback
  DIAGNOSE   the decision tree (lesson 6), traces (lesson 4), hardware signals (lesson 3)
     │
  RESOLVE    the real fix, deployed through the normal canary path (lesson 8) — an
     │       emergency change bypassing the gate is how you get a second incident
  REVIEW     blameless postmortem within days, action items with owners and dates
```

**Mitigate before diagnose** is the rule most engineers get backwards, because debugging is intellectually satisfying and rollback feels like giving up. The user-facing cost of ten extra minutes of diagnosis at full impact is almost always higher than the cost of a rollback you later discover was unnecessary. Corollary: *the ability to mitigate fast is an engineering investment* — one-click rollback, a traffic-weight knob, a global degradation switch, a "drain this replica" command ([lesson 7](07-reliability-and-degradation.md)).

---

## The inference incident catalogue

| Incident | First signal | Mitigate (minutes) | Resolve (days) |
|---|---|---|---|
| **Traffic spike beyond capacity** | queue depth, TTFT burn | shed at the edge, clamp `max_tokens`, scale out, degrade to smaller model | predictive scaling, standby pool ([Phase 6 L8](../phase-6/08-autoscaling-gpu-fleets.md)) |
| **Bad deploy (latency)** | burn alert right after a deploy annotation | roll back by traffic weight | canary gate that would have caught it ([lesson 8](08-canary-and-shadow-traffic.md)) |
| **Bad deploy (quality)** | finish-reason/length shift; user reports | roll back | quality gates on the rollout, template hash pinning |
| **One degraded GPU / straggler** | per-pod latency heatmap, per-rank spread | drain the replica; cordon the node | automated drain on XID/clock anomalies ([lesson 3](03-gpu-and-host-telemetry.md)) |
| **KV thrashing / preemption storm** | preemption rate, ITL p99 | lower `max_num_seqs`, stop admitting, restart with a sane config | KV budget sizing ([Phase 4 L4](../phase-4/04-kv-cache-optimization.md)), occupancy alert |
| **NCCL hang (no tokens, process alive)** | tokens/s = 0, running > 0 | kill the replica (progress watchdog should already have) | watchdog + NCCL timeouts ([Phase 6 L9](../phase-6/09-multi-node-operations.md)) |
| **Poison request killing replicas serially** | replica restarts correlated with one prompt pattern | block the pattern/tenant at the edge; quarantine | input validation, hard limits, crash-loop quarantine |
| **Retry storm** | offered load ≫ client-visible load | disable client retries at the gateway, shed | retry budgets, `Retry-After` honoring ([lesson 7](07-reliability-and-degradation.md)) |
| **Model registry / weight fetch failure** | new replicas never become ready | stop the rollout; pin to the last-good digest | local NVMe weight cache, digest pinning ([Phase 8 L7](../phase-8/07-model-registry-and-artifacts.md)) |
| **Spot eviction wave** | replica count dropping, capacity errors | fail over to on-demand; raise the on-demand floor | spot fraction policy, drain path ([Phase 6 L8](../phase-6/08-autoscaling-gpu-fleets.md)) |
| **Region/zone capacity exhausted** | scale-out requests failing | spill to another region/provider | multi-region capacity plan |
| **Cache hit-rate collapse (routing bug)** | prefix hit rate, TTFT up | revert routing config; raise per-replica caps | bounded-load hashing, skew alert ([Phase 6 L7](../phase-6/07-prefix-aware-routing.md)) |
| **Cost blowout** | cost per 1M tokens alert | revert the config change; cap concurrency | efficiency alerting ([lesson 9](09-cost-per-million-tokens.md)) |
| **Observability outage** | scrape failures / `absent()` | treat as an incident: you are flying blind; freeze risky changes | independent monitoring of the monitoring |
| **Upstream dependency slow** (vector DB, auth, tool APIs) | per-stage spans | timeout + fallback per stage | stage budgets ([Phase 9 L10](../phase-9/10-rag-and-agentic-serving.md)) |

Keep this table in the team wiki. Half of incident response is recognizing "oh, this is the poison-request one" in the first two minutes.

---

## Runbooks that work

A runbook is a **one-screen document per alert**, linked from the alert annotation ([lesson 5](05-slos-and-error-budgets.md)). Structure:

```
  RUNBOOK: TTFTSLOBurnRateFast
  ─────────────────────────────────────────────────────────────────────────────
  WHAT IT MEANS   TTFT SLO budget burning at >14.4x. Users see slow first tokens now.
  IMPACT          All chat traffic on <model>. Free tier degrades first if rung 3 active.
  DASHBOARDS      L1 overview → L2 capacity  (links)
  FIRST 3 CHECKS  1. deploy in the last 30 min?      → if yes: ROLL BACK (cmd below)
                  2. queue wait vs prefill split?     → queue ⇒ capacity path
                                                        prefill ⇒ cache/compute path
                  3. one pod hot or all pods?          → one ⇒ drain it (cmd below)
  MITIGATIONS     rollback:  ./deploy rollback --service chat --to last-good
                  drain:     ./ops drain-replica <pod>         (safe, bounded)
                  scale:     ./ops scale --service chat --replicas +3
                  shed:      ./ops set-admission --max-queue 32   (fast 429s)
                  degrade:   ./ops degrade --rung 2               (clamps max_tokens)
  DO NOT          restart all replicas at once (T_cold is minutes; you will make it worse)
                  raise max_num_seqs while KV occupancy >0.85 (causes preemption storms)
  ESCALATE IF     mitigations don't move p99 in 10 min → page <team>; if hardware XID
                  present → page infra/hardware on-call
  AFTER           file postmortem if >5% of budget consumed; attach the trace links
```

Rules for runbooks:

- **Copy-pasteable commands, not prose.** At 3 a.m., "consider reducing concurrency" is useless; `./ops set-admission --max-queue 32` is not.
- **A `DO NOT` section.** Inference has specific foot-guns (mass restart with minute-scale cold starts; raising limits while memory-bound) that an experienced engineer will otherwise reach for out of web-service habit.
- **Stale runbooks are worse than none.** Review them when the alert fires: the last responder updates it, every time. Untouched-in-a-year is a smell.
- **One runbook per alert**, not one per service. The alert tells you the symptom; the runbook is the response to *that* symptom.

---

## Postmortems: the only artifact that compounds

Blameless means the analysis targets systems, not people: if a human action caused the outage, the finding is "the system permitted that action without a check."

```
  POSTMORTEM SKELETON
  ────────────────────────────────────────────────────────────
  Summary        2 sentences: what users experienced, for how long
  Impact         quantified: requests affected, % of error budget consumed, $ if relevant,
                 tenants affected. (Numbers from lessons 5 and 9 — this is why they exist.)
  Timeline       detection → declaration → each mitigation → resolution, with timestamps
                 and MTTD/MTTM/MTTR computed
  Root causes    plural. There is never one. Include the contributing conditions
                 (e.g. "KV occupancy alert existed but was routed to a muted channel")
  What went well explicitly — the rollback path worked, the watchdog fired
  Action items   each with OWNER, DATE, and TYPE:
                   detect   (new alert / lower threshold / new metric)
                   mitigate (new automation / faster rollback / a degradation rung)
                   prevent  (limit, validation, canary gate, config guardrail)
                   process  (runbook update, training, ownership)
  Lucky breaks   what could have made this much worse (this section finds the next incident)
```

The quality test for a postmortem: **could a new team member, reading only this, prevent or handle a recurrence?** Action items like "be more careful with config changes" fail it; "add a validating admission webhook that rejects `max_num_seqs` > KV-derived ceiling" passes.

Two metrics worth tracking across incidents, because they tell you whether the investment is working: **MTTD** (are the alerts finding things before customers?) and **MTTM** (time to mitigation — the number the mitigate-first discipline and the rollback investment move).

---

## Chaos engineering for GPU inference

You cannot debug a failure mode you have never seen. And inference has failure modes that are hard to imagine and easy to inject.

| Experiment | How | What you're validating |
|---|---|---|
| **Kill one replica under load** | `kubectl delete pod` / SIGKILL | router ejects it fast; in-flight requests fail cleanly; no retry storm |
| **Graceful drain under load** | send SIGTERM | preStop drain works, streams finish, `terminationGracePeriodSeconds` is sufficient ([lesson 7](07-reliability-and-degradation.md)) |
| **Kill one rank of a TP group** | SIGKILL one process | the whole replica dies fast and restarts, instead of hanging forever ([Phase 6 L9](../phase-6/09-multi-node-operations.md)) |
| **Freeze a rank** | `SIGSTOP` one rank | the progress watchdog trips — this is the hang test, and most stacks fail it initially |
| **Throttle a GPU** | `nvidia-smi -pl <low watts>` or `-lgc <low clock>` | throttle detection works; the straggler shows in per-rank spread; SLO impact is visible |
| **Fill the KV cache** | long prompts + high `max_tokens` at high concurrency | preemption path, admission control, occupancy alert |
| **Inject a CUDA OOM** | shrink `gpu_memory_utilization`, or allocate a large tensor in a sidecar context | the replica fails cleanly, doesn't corrupt other requests, restarts |
| **Slow the network / degrade NVLink** | `tc netem` delay; `NCCL_P2P_DISABLE=1` | collective cost visible, TTFT/TPOT impact quantified |
| **Spot eviction** | cloud eviction simulation or SIGTERM with a 30 s deadline | drain-within-notice path works |
| **Blackhole the metrics/trace pipeline** | drop the collector | serving is unaffected (observability must never be on the request path!), and you notice |
| **Slow the model registry / object store** | proxy with injected latency | startup timeouts, readiness probes, weight-cache fallback |
| **Poison request** | a pathological prompt (max context, huge `n`, adversarial grammar) | edge validation rejects it; if it kills a replica, quarantine works |
| **Dependency failure (RAG)** | kill the vector DB | per-stage timeout + fallback ([Phase 9 L10](../phase-9/10-rag-and-agentic-serving.md)) |
| **Clock/tokenizer mismatch** | deploy a mismatched tokenizer to one canary replica | your quality metrics detect it — the test of [lesson 8](08-canary-and-shadow-traffic.md) |

Discipline for running these:

1. **Hypothesis first, in writing.** "Killing one of six replicas will cause <1% request failures and recovery within 30 s." Then measure. A confirmed hypothesis is worth little; a *refuted* one is the whole value.
2. **Start in staging, small, during business hours**, with an abort condition and a person watching. Graduate to production game days once each experiment has passed in staging.
3. **GameDay format**: one person injects, the rest respond using only dashboards and runbooks. Track time-to-diagnosis. Every gap becomes a postmortem-style action item.
4. **Automate the passing ones** into a periodic job, so regressions in the *recovery* path get caught. Drain behaviour in particular rots silently with every config change.
5. **Never inject on hardware you can't restore** — power-limit and clock experiments must be reverted, and `nvidia-smi` persistence-mode settings survive reboots on some configurations.

---

## On-call that is survivable

| Practice | Inference-specific reason |
|---|---|
| Paging alerts only from the burn-rate + hardware + capacity-loss list ([lesson 5](05-slos-and-error-budgets.md)) | GPU fleets generate a lot of noisy signals; utilization-based alerts are the classic noise source |
| A weekly on-call review: every page → keep / tune / delete | the only mechanism that keeps the paging set small |
| Rotations with real handoffs and a known "what's risky this week" note | rollouts and capacity changes dominate risk |
| Access + permissions pre-granted | an incident is not the time to discover you can't drain a node |
| The mitigation commands exist as scripts, tested, in the runbook | see MTTM above |
| Load-shedding and degradation are *switches*, not code changes | a code change during an incident takes a build + `T_cold` minutes |
| Capacity headroom known and published | "can we absorb this?" must be answerable in seconds ([lesson 9](09-cost-per-million-tokens.md)) |
| Status page + customer comms templates | quality degradation especially needs explaining; users notice "the model got worse" and file it as a bug |

---

## Do this now (60 minutes)

1. **Write one real runbook** for your TTFT burn alert, in the exact structure above, including a `DO NOT` section and working commands (`./ops drain-replica`, a rollback, a shed switch) — write the scripts if they don't exist. The scripts are the deliverable; the document is the index.
2. **Run three chaos experiments** against your Phase-3/6 setup, hypothesis written first: (a) SIGKILL one replica under load — measure failed requests and recovery time; (b) `SIGSTOP` the engine process — confirm the progress watchdog trips and the replica restarts; (c) blackhole your Prometheus/collector — confirm serving is unaffected and that you detect the blindness. Record measured vs predicted for each.
3. **Write a postmortem for one of them** using the skeleton, with quantified impact from your own SLO/budget math and at least three action items typed `detect` / `mitigate` / `prevent`. If the exercise produces no action items, your hypothesis was too easy — pick a harder experiment.

---

**Next:** [Build: observability stack + canary/shadow harness →](11-build-observability-and-canary.md) — the two projects that make this phase real: a Prometheus/Grafana/traces stack around your server, and a shadow-traffic harness that produces an automated quality-and-latency diff report.
