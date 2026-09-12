# 5 — Capstone Brief: Cost-Optimization Case Study

> **Proof obligation:** that you can take a model and a realistic traffic pattern, drive down dollars per million tokens through a sequence of levers applied in a defensible order, quantify each lever's individual contribution, and state the quality cost of every one — on rented hardware, with real invoices.

A **brief, not a build guide**. The levers are Phases 3-8; the accounting is [Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md); the purchase-mode and autoscaling mechanics are [Phase 8 lesson 5](../phase-8/05-scheduling-capacity-and-autoscaling.md); the measurement standard is [lesson 2](02-engineering-standards.md).

This is the capstone that maps most directly onto a posted job title. If you do only one, do this one — and it is also the cheapest to run well, because the deliverable is a report rather than a system.

---

## The claim to prove

> "For *this* model under *this* diurnal traffic pattern at *this* SLO, cost per 1M output tokens falls from $A to $B — a *N*× reduction — and each lever's contribution and quality cost is separately measured and attributed."

Two constraints make it credible: **the SLO is held constant** across all configurations, and **quality is measured**, not assumed. Cost reductions that quietly relax p99 or degrade output are not reductions; they are trades, and you must price both sides.

## Scope

| In | Out |
|---|---|
| One model family, 7-14B, on rented cloud GPUs | training or distillation of a new model |
| A synthetic-but-realistic traffic profile: diurnal QPS curve, prompt/output length distributions from a stated source | a real production traffic trace you don't have |
| Levers: batching/scheduler config, quantization, KV/prefix caching, hardware choice, autoscaling policy, purchase mode (on-demand/spot/committed), request routing | multi-region cost engineering |
| Quality evaluation per configuration | a novel quantization method |
| An itemized cost model including non-GPU lines | building a billing system |

**Define the traffic profile first and freeze it.** Every number in the report is relative to it, and changing it mid-study invalidates the comparison.

---

## The lever ladder, in the order to apply it

```
  0  BASELINE            unoptimized fp16, default server config, fixed replicas
                         → measure $/1M tokens at the SLO. This is your denominator.
  1  BATCHING/SCHEDULER  the free win: continuous batching + admission tuning
                         typical 2-10× vs naive; zero quality cost        [P3]
  2  CACHING             prefix/prompt caching for shared system prompts
                         large on chat-shaped traffic; zero quality cost  [P4 L6]
  3  QUANTIZATION        INT8/FP8/INT4 weights; measure quality every time
                         1.5-3× throughput; NONZERO quality cost         [P4 L2-3]
  4  HARDWARE            right-size the GPU to the model; consider CPU for
                         small models; TPU/Inferentia if the stack allows [P9 L5]
  5  AUTOSCALING         track the diurnal curve; warm pool sized by cold start
                         saves the idle hours, which are most of them     [P8 L5]
  6  PURCHASE MODE       committed baseline + on-demand peak + spot batch
                         40-60% on the committed portion; no engineering  [P8 L5]
  7  ROUTING/CASCADES    small model first, escalate on difficulty
                         biggest remaining lever; quality risk to measure
```

**Report the ladder as a waterfall**: each rung's marginal contribution, cumulative cost, and quality delta. That waterfall chart is the single most persuasive artifact in this entire roadmap, because it is exactly the deliverable the job produces.

Note the ordering logic, which you should state: free-and-safe levers first (1-2), then levers with a quality cost (3), then capital/planning levers that require no model change (5-6), then architectural changes (7). Applying quantization before fixing batching is the classic backwards move — it optimizes the wrong term and muddles the attribution.

---

## The acceptance bar

- [ ] **Baseline and final $/1M tokens**, both measured at the same held p99, on named instance types with cited prices.
- [ ] **Waterfall table**: lever, config change, throughput delta, $/1M delta, cumulative, quality delta.
- [ ] **Quality measured per configuration**: a fixed eval (task accuracy, or perplexity plus a distributional output comparison), not vibes ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)).
- [ ] **Utilization honesty**: costs computed against the *diurnal* curve, including idle hours — not only at saturation. Report the fleet's average utilization before and after.
- [ ] **Full cost model**: GPU, host/CPU, storage, egress, observability, vector DB if present, engineering amortization ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md)'s table).
- [ ] **Spot risk quantified**: preemption rate you observed or assumed, and what the interruption costs in-flight requests.
- [ ] **The regime statement**: at what fleet size each lever is worth the engineering time (the "4 GPUs vs 400 GPUs" rule) — this is the judgment the role is hired for.
- [ ] **Actual project spend** disclosed, and **Limitations**.

---

## Pitfalls specific to this project

| Pitfall | Symptom | Avoidance |
|---|---|---|
| SLO drifts between configurations | apparent savings are just higher latency | pin p99 and report it in every row |
| Quality never measured | a 4× "saving" that shipped a worse product | eval per configuration, no exceptions |
| Costs at 100% utilization only | the headline number is unreachable in production | compute against the traffic curve |
| Peak-only benchmarking | autoscaling and purchase-mode levers look worthless | simulate the full diurnal cycle |
| Stacking levers without attribution | you can't say what worked | one lever per measurement, then combined |
| Ignoring non-GPU lines | observability and egress are 5-15% in reality | itemize everything |
| Cherry-picked instance prices | comparison unfair | same region, same date, cited source, on-demand unless stated |
| Uncontrolled variance | "improvements" inside the noise | n ≥ 10, σ reported ([lesson 2](02-engineering-standards.md)) |
| Runaway spend on your own study | a surprise invoice | budget alerts and hard `max_size` limits ([Phase 8 lesson 9](../phase-8/09-infrastructure-as-code.md)), and pick small models |

---

## What "excellent" looks like versus "adequate"

| Adequate | Excellent |
|---|---|
| "We got 3× cheaper with quantization and batching" | a seven-rung waterfall with marginal contributions and quality deltas |
| Cost at saturation | cost over a diurnal curve with utilization before/after |
| Quality "seemed fine" | a per-configuration eval table with the one configuration you rejected on quality |
| A number | a recommendation: what to do at 4 GPUs, at 40, at 400, with reasons |
| A blog post | a report someone could act on in a real company this week |

The "one configuration you rejected on quality grounds" is worth calling out: it proves the study had a standard rather than a conclusion.

---

## Where it leads

- The report is the most directly interviewable artifact in the roadmap; expect the whole conversation to be about it.
- Pairs naturally with [lesson 3](03-capstone-nano-vllm.md) (depth) or [lesson 7](07-capstone-production-rag.md) (shipping) as your second capstone.
- Project index entry: **20 — cost-optimization case study** in [`projects/README.md`](../../projects/README.md). Cost discipline while running it: [`GETTING-STARTED.md`](../../GETTING-STARTED.md).

---

**Next:** [Capstone brief: reproduce a paper's system →](06-capstone-paper-reproduction.md) — research literacy demonstrated against a real serving loop.
