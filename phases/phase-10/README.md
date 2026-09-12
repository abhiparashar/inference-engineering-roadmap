# Phase 10 — Capstones (Deep Dive)

> **Goal:** convert nine phases of mechanics into two or more pieces of work that a stranger can evaluate in ten minutes and conclude "this person can do the job." Not tutorials followed, not a course certificate — a system you built, measured honestly, and wrote up with numbers, tradeoffs, and the parts that failed.

This folder is the long-form version of [Phase 10 in the ROADMAP](../../ROADMAP.md#phase-10--capstones-this-is-where-top-1-gets-proven). It is deliberately **not** a build guide. Phases 0-9 contain the build guides; the capstones are where nobody hands you steps, because *choosing* the steps is the skill being demonstrated.

The intellectual core:

> **A capstone is an argument, and its evidence is measurement.** The artifact that gets you hired is not the code — it is a repository where the code is reproducible, the numbers are honest, the comparison is fair, and the write-up states what you tried that didn't work. Two such projects beat ten half-finished ones, and either beats any number of completed tutorials.

---

## What these lessons are for

| Lesson | Role |
|---|---|
| 1 | how to choose a capstone that actually proves something, and how to scope it so it finishes |
| 2 | the engineering bar every capstone must clear — this is the load-bearing lesson |
| 3-7 | one brief per ROADMAP capstone: what it proves, its acceptance bar, and the specific ways each one goes wrong |
| 8 | the write-up, the repository layout, and how to talk about it in an interview |

Lessons 3-7 are **briefs, not build guides**: scope, proof obligation, acceptance criteria, pitfalls. The implementation knowledge is in the phase that taught it, and each brief points there. If a brief told you every step, the project would no longer be evidence of anything.

---

## Prerequisites

You do not need every phase finished, but each capstone has a hard dependency:

| Capstone | Requires, genuinely |
|---|---|
| 1 — nano-vLLM | [Phase 1](../phase-1/README.md) (KV cache, decode loop), [Phase 3 lessons 4-6](../phase-3/04-continuous-batching.md), [Phase 4 lessons 4-6](../phase-4/04-kv-cache-optimization.md) |
| 2 — multi-modal pipeline | [Phase 5 lessons 7-8](../phase-5/07-triton-inference-server.md), [Phase 7](../phase-7/README.md), [Phase 9 lessons 3-4](../phase-9/03-vision-serving.md) |
| 3 — cost optimization | [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md), [Phase 4](../phase-4/README.md), [Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md), [Phase 8 lesson 5](../phase-8/05-scheduling-capacity-and-autoscaling.md) |
| 4 — paper reproduction | the phase that owns the paper, plus [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md) |
| 5 — production RAG service | [Phase 9 lessons 6, 9-11](../phase-9/06-vector-search-and-ann.md), [Phase 7](../phase-7/README.md), [Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md) |

Across all five: [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md), [`playbooks/profiling.md`](../../playbooks/profiling.md), and [`playbooks/production-readiness-checklist.md`](../../playbooks/production-readiness-checklist.md) are not optional reading — they are the standard lesson 2 formalizes.

---

## The map of this phase

```
                     PICK TWO (lesson 1)
                            │
   ┌──────────────┬─────────┴────────┬──────────────┬──────────────┐
   ▼              ▼                  ▼              ▼              ▼
 nano-vLLM   multi-modal        cost study     paper repro    prod RAG
 lesson 3    lesson 4           lesson 5       lesson 6       lesson 7
 depth in    composition +      $ and unit     research       product shape
 the engine  observability      economics      literacy       + security
   │              │                  │              │              │
   └──────────────┴─────────┬────────┴──────────────┴──────────────┘
                            ▼
              THE BAR THEY ALL MUST CLEAR (lesson 2)
        reproducible · measured honestly · fair comparison ·
        failure modes tested · cost stated · scope limits admitted
                            ▼
                  THE WRITE-UP (lesson 8)
        numbers table first, method second, code third
  ───────────────────────────────────────────────────────────────────────
  PICK FOR CONTRAST, NOT FOR COMFORT
    one project that proves DEPTH  (3 or 6: you understand the machine)
    one project that proves SHIP   (2, 5, or 3-as-cost-study: you deliver)
```

Two conclusions that decide whether this phase pays off:

1. **An unmeasured capstone is a tutorial.** The differentiating content is the numbers table and the explanation of the gap between your system and the reference one. "I built a continuous-batching server" is a claim anyone can make; "mine reaches 62% of vLLM's throughput at the same p99, and here are the four optimizations that account for the gap" is evidence.
2. **Unfinished-but-honest beats finished-but-vague.** A project that hit a wall, documented the wall with profiler output, and stated what it would take to break through reads as senior. A project whose README says "achieves high performance" reads as junior regardless of the code.

---

## The lessons

| # | File | What it gives you |
|---|---|---|
| 1 | [Choosing and scoping a capstone](01-choosing-a-capstone.md) | a selection rule based on the job you want, a scoping method that makes it finishable, and the four capstone anti-patterns |
| 2 | [Engineering standards](02-engineering-standards.md) | **the bar**: reproducibility, honest measurement, fair comparison, tested failure modes, stated cost, admitted limits |
| 3 | [Capstone brief: nano-vLLM](03-capstone-nano-vllm.md) | the from-scratch continuous-batching engine — proof obligation, acceptance bar, pitfalls |
| 4 | [Capstone brief: multi-modal pipeline](04-capstone-multimodal-pipeline.md) | composed multi-stage serving with real observability and autoscaling |
| 5 | [Capstone brief: cost optimization](05-capstone-cost-optimization.md) | the $/1M-token case study — the most directly employable of the five |
| 6 | [Capstone brief: reproduce a paper's system](06-capstone-paper-reproduction.md) | mechanism reproduction with a directional result, not a number match |
| 7 | [Capstone brief: production RAG service](07-capstone-production-rag.md) | the shape of most real AI backends: pipeline, budgets, guardrails, CI gate |
| 8 | [Write-up and portfolio](08-writeup-and-portfolio.md) | repo layout, README structure, the numbers-first template, and how to present it under questioning |

---

## How to work through this phase

1. **Read lessons 1-2 before choosing.** Lesson 2 is the definition of "done"; picking a project without it produces a demo.
2. **Pick two, of different kinds.** One depth project (3 or 6) and one shipping project (2, 5, or 7). Companies screen for both and most candidates show only one.
3. **Timebox, then cut scope, never quality.** Four to six weeks part-time per capstone. When you run out of time, reduce *scope* (fewer models, one hardware type, smaller corpus) and keep the measurement bar intact. A narrow, rigorous result is publishable; a broad, sloppy one is not.
4. **Write the README skeleton on day one**, with empty numbers tables. It forces you to decide what you are going to prove, and it stops the classic failure where the code works and nobody can tell.
5. **Measure continuously, not at the end.** Keep the benchmark in the repo from the first commit and run it on every significant change; you want the history, because "here is where the 3× came from" is the story.
6. **Ship it publicly.** A public repo with a real README, plus a short write-up (blog, gist, or the repo itself), is what makes the work findable. Then take one OSS contribution from [`resources/README.md`](../../resources/README.md)'s track — a merged PR in a serving framework is the single most verifiable credential in this field.

**Time budget:** 8-12 weeks part-time for two capstones, including write-ups. This is the phase where slowing down is correct.

## Projects that belong to this phase

- **[18 — nano-vLLM](../../projects/README.md)** ([brief](03-capstone-nano-vllm.md))
- **[19 — Multi-modal production pipeline](../../projects/README.md)** ([brief](04-capstone-multimodal-pipeline.md))
- **[20 — Cost-optimization case study](../../projects/README.md)** ([brief](05-capstone-cost-optimization.md))
- **[21 — Production-grade RAG service](../../projects/README.md)** ([brief](07-capstone-production-rag.md))
- Paper reproduction ([brief](06-capstone-paper-reproduction.md)) has no fixed project number — it takes the number of whichever phase's mechanism you reproduce, or add it to the index yourself.

There are no labs in this phase ([`labs/README.md`](../../labs/README.md#phase-10--capstone-labs) says so explicitly): the projects *are* the work.

---

After this phase, the roadmap stops having phases. What continues is in [`resources/README.md`](../../resources/README.md): the OSS-contribution ladder and the staying-current track, both of which are indefinite and both of which compound.
