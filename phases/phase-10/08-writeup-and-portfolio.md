# 8 — Write-Up and Portfolio

> **You'll be able to say:** "My repository is structured so that a reviewer sees the claim, the numbers table, and the method within thirty seconds of opening it, and can reproduce the headline number in one command. I lead with results, admit limits, explain gaps with measurements, and I can defend every number under questioning because I know how each was produced and what its variance was."

The code is done. This lesson is about the fact that **an unread project is equivalent to an unbuilt one**, and that the write-up is itself an engineering artifact judged by the same standard as the code.

---

## The thirty-second test

A reviewer opens your repo. In thirty seconds they must be able to answer: *what does this do, what did it achieve, against what baseline, on what hardware.* That dictates README ordering, which is the inverse of how projects are usually written:

```
  ┌─ README.md ──────────────────────────────────────────────────┐
  │ 1  ONE-LINE CLAIM, with a number                             │
  │ 2  THE NUMBERS TABLE  ← baseline vs yours, at a stated SLO    │
  │ 3  What this is / isn't (3-5 bullets)                        │
  │ 4  Reproduce it: one command, stated hardware, ~runtime      │
  │ 5  How it works: one diagram, then the 3-5 key design         │
  │    decisions with the alternative you rejected and why       │
  │ 6  Method: workload, warmup, run count, variance             │
  │ 7  What I found: the surprises, the gap analysis with data   │
  │ 8  Limitations & what I'd do next, prioritized               │
  │ 9  Code map: where to look, in reading order                 │
  └──────────────────────────────────────────────────────────────┘
```

Sections 2 and 7 are the ones that differentiate. **Section 7 — "what I found" — is the section that makes a hiring manager want to talk to you**, because it is the only part that cannot be produced by following a tutorial. Examples of what belongs there: "the scheduler's own Python overhead was 22% of step time at batch 8"; "JPEG decode was 46% of the pipeline before NVDEC"; "INT4 cost 1.8 points of task accuracy, which is why configuration 6 was rejected"; "recall below 0.90 was indistinguishable in end-task quality, so I ship `efSearch=64`."

### The numbers table, concretely

| | Baseline (naive) | Reference (vLLM 0.6.3, tuned) | Mine |
|---|---|---|---|
| Throughput @ p99 TTFT ≤ 500 ms | 180 tok/s | 5,240 tok/s | 2,110 tok/s |
| p50 / p99 TTFT | 210 / 1,850 ms | 140 / 480 ms | 165 / 495 ms |
| Peak KV utilization | 22% | 91% | 84% |
| $/1M output tokens (A10G, on-demand) | $41.10 | $1.41 | $3.51 |
| n runs, σ (throughput) | 10, 2.8% | 10, 1.9% | 10, 3.1% |

Four properties make this table trustworthy: a *naive* baseline, a *tuned reference*, a **held SLO in the metric name**, and *variance*. Publish the reference's flags immediately below it.

---

## Repository layout

```
  README.md            the nine sections above
  RESULTS.md           full tables, sweeps, plots; raw CSV/JSON committed
  ENVIRONMENT.md       hardware, driver, versions, instance type, $/hour
  BUDGET.md            (pipeline projects) per-stage budget/timeout/fallback
  RELIABILITY.md       the failure tests and observed behavior
  LIMITATIONS.md       or a README section — but it must exist
  bench/               the harness; one command; committed raw output
  scripts/             reproduce.sh, load generator, eval
  src/                 the system, readable, commented where non-obvious
  docs/                one architecture diagram; design decisions
```

Three rules that reviewers notice:

1. **Committed raw results.** Tables can be retyped; CSVs cannot be faked casually, and their presence signals the measurement really happened.
2. **A code map in reading order.** "Start at `src/scheduler.py:step()`" saves a reviewer five minutes and reads as consideration for the reader — the same instinct good code review requires.
3. **No dead scaffolding.** Empty directories, `TODO: implement`, commented-out experiments, and an unfinished `v2/` branch in `main` all subtract. Delete or finish.

---

## Writing style for technical evidence

| Do | Don't |
|---|---|
| "2,110 tok/s at p99 TTFT 495 ms (n=10, σ=3.1%)" | "very fast" |
| "41% of vLLM's throughput; gap decomposed below" | "comparable to vLLM" |
| "I chose FCFS with a KV-budget admission cap; priority queueing would help mixed traffic but I didn't test it" | "uses an advanced scheduler" |
| "Not tested: multi-GPU, quantization, output > 1024 tokens" | silence about scope |
| "This surprised me: X. The cause was Y, confirmed by Z" | omitting the interesting part |
| Present tense for the system, past tense for experiments | marketing voice |

**Length:** a README that takes more than ten minutes to read will not be read. Put depth in `RESULTS.md` and link to it. One page of README plus one page of results beats five pages of prose.

**Diagrams:** one, showing data flow with the stage or component names that appear in the code. ASCII is fine and often better — it renders everywhere and diffs cleanly, which is why every lesson in this repo uses it.

---

## Publishing

| Channel | What it's for | Effort |
|---|---|---|
| Public GitHub repo | the artifact itself; the only mandatory one | — |
| A write-up post (blog/gist/repo `docs/`) | reach; the narrative version with the gap analysis | half a day |
| Résumé line | one line per capstone, with the headline number in it | minutes |
| Interview talking point | a three-minute verbal version, practiced | an hour |
| OSS contribution | the strongest external validation of all | see [`resources/README.md`](../../resources/README.md) |

**Résumé line template**, which works because it is specific:

> *Built a 1.6k-line continuous-batching LLM server with paged KV management and prefix caching; reached 41% of vLLM's throughput at equal p99 TTFT on an A10G, with the gap attributed via profiling to kernel and CUDA-graph differences.*

Compare against "built an LLM inference server (Python, PyTorch, CUDA)." Same work, different evidence.

---

## Defending it in an interview

Expect these, in roughly this order, for any capstone:

1. **"Walk me through the architecture."** Two minutes, top-down, no code. Practice it out loud; the written version is in README section 5.
2. **"Why is the reference implementation faster?"** The decomposition. If you answer with speculation rather than measurements, the interview turns into a different conversation.
3. **"How did you measure that?"** Workload, warmup, run count, variance, open-loop client, held SLO. Knowing your σ from memory is disproportionately impressive.
4. **"What would you do with another month?"** Your prioritized limitations list, with a reason for the order — this question is testing judgment, not ambition.
5. **"What broke?"** Tell a real debugging story with the tool you used and the evidence that resolved it. The best answer names a wrong hypothesis you held first.
6. **"What does it cost?"** $/1M units, at what utilization, on what hardware.
7. **"What's wrong with it?"** Answer honestly and specifically. Candidates who can't criticize their own work read as unable to review anyone else's.

Two failure modes to avoid: **overclaiming** (one unfair comparison discredits the whole repo) and **underclaiming** ("it's just a toy" — you spent six weeks on it; describe what it demonstrates).

---

## The ten-minute self-review before publishing

- [ ] Claim with a number in the first two lines.
- [ ] Numbers table with naive baseline, tuned reference, held SLO, and variance.
- [ ] One-command reproduction, tested from a clean clone.
- [ ] `ENVIRONMENT.md` with hardware, versions, and price.
- [ ] Raw results committed.
- [ ] "What I found" section with at least two non-obvious findings.
- [ ] Failure tests recorded.
- [ ] Cost per unit of useful work stated.
- [ ] Limitations, prioritized, honest.
- [ ] No dead scaffolding, no "TODO: benchmarks", no marketing adjectives.
- [ ] A code map so a reviewer knows where to start.

---

## After the capstones

The phases end here; two tracks do not ([`resources/README.md`](../../resources/README.md)):

1. **OSS contribution** — docs fix → reproduce and triage a real perf bug with profiler evidence → add a benchmark → fix a bug → implement a feature. One merged PR in vLLM, SGLang, TGI or `llama.cpp` is the single most verifiable credential in this field, and after two capstones you are qualified to attempt it.
2. **Staying current** — releases, MLSys/OSDI/NSDI proceedings, issue trackers. The specific techniques in Phases 4-6 will churn; the measurement discipline in [lesson 2](02-engineering-standards.md) will not.

The through-line of the whole roadmap, if it compresses to one sentence: **know where the bottleneck is, prove it with a measurement, and be honest about what you haven't tested.** Everything else is technique, and technique is learnable on demand once that habit is in place.

---

**Done.** You have reached the end of the roadmap. The index of everything you built is in [`projects/README.md`](../../projects/README.md); the next move is a merged PR.
