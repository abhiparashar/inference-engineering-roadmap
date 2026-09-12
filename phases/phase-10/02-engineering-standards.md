# 2 — Engineering Standards

> **You'll be able to say:** "Done means six things, all of them checkable by someone else: it reproduces from a clean clone on stated hardware, every number has a method and a variance, the baseline was configured as carefully as my system, the failure modes I claim to handle were tested by breaking them, the cost is stated in dollars per unit of work, and the limits are admitted in writing. A project missing any one of those is a demo, and I can tell which one is missing in about two minutes of reading someone else's repo."

This is the load-bearing lesson of Phase 10. The capstone briefs that follow are short precisely because this is where the standard lives: the same six requirements apply to all of them.

---

## The six requirements

```
  1  REPRODUCIBLE     clean clone → one command → the numbers
  2  MEASURED         method, warmup, N runs, variance, percentiles
  3  FAIRLY COMPARED  a baseline tuned as hard as your system
  4  FAILURE-TESTED   you broke the things you claim to survive
  5  PRICED           $ per 1M tokens / requests / images, with the math
  6  BOUNDED          what it does NOT do, in writing, before anyone asks
```

Every one of them is a claim *about you* rather than about the software: that your results can be trusted, that you know what you didn't test, and that you know what it costs. That is the entire hiring signal.

---

## 1. Reproducible

The test: someone clones the repo on the stated hardware and, from the README alone, reproduces a headline number within the stated variance.

| Requirement | Concretely |
|---|---|
| Pinned dependencies | lockfile or exact versions; `torch`, CUDA, engine, driver — all recorded |
| One-command run | `make bench` / `./scripts/bench.sh`, not eleven prose steps |
| Recorded environment | GPU model, driver, CUDA, CPU, RAM, instance type, $/hour ([`ENVIRONMENT.md`](01-choosing-a-capstone.md)) |
| Digest-pinned artifacts | model revision/digest, dataset digest, index manifest ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)) |
| Deterministic-where-possible | fixed seeds, fixed prompt/input sets, greedy decoding for comparisons |
| Raw data committed | the CSV/JSON the tables were generated from, not just the tables |

**The rawest form of this test is the one people fail:** delete your virtualenv and your Docker cache, follow your own README on a fresh box, and time it. Whatever breaks is what a reviewer will hit in minute two.

## 2. Measured honestly

The full standard is [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md) plus [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md). The minimum, restated because it is violated constantly:

- **Warm up, then discard the warmup.** First-request numbers measure compilation and cache state, not steady state.
- **N ≥ 10 runs, report σ** (or p50/p95 with a run count). A single number with no variance cannot support a comparison — and your CI-gate threshold, if you have one, must be derived from that σ ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)).
- **Percentiles, never means, for latency.** p50/p95/p99, plus the metric decomposition that matters for the workload: TTFT and TPOT separately for LLMs ([Phase 3 lesson 1](../phase-3/01-what-a-serving-system-is.md)), per-stage for pipelines ([Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md)).
- **Measure at a fixed SLO, not at maximum batch.** Throughput without a latency constraint is a number you can inflate arbitrarily; report the throughput achievable at your stated p99.
- **Include client-side queueing.** Latency measured from "the server started the request" hides the queue, which is where the tail lives. Measure from request submission, with an open-loop or fixed-arrival-rate load generator, not a closed loop of N threads ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)).
- **Fix output length** when comparing generation systems, or you are comparing sampling luck.
- **State the workload distribution** — prompt and output length distributions, arrival pattern. "Realistic" means you wrote down what you assumed.

**One table row you should always include: the trivial baseline.** Naive/unoptimized numbers alongside your optimized ones. The delta is the value you added, and its absence is the most common reason a good project reads as unconvincing.

## 3. Fairly compared

| Rule | Why |
|---|---|
| Same hardware, same model, same precision, same workload | any difference is a confound you must name |
| Baseline tuned, not defaulted | vLLM with untouched defaults is not a baseline, it is a strawman |
| Baseline's config published | reviewers will check the flags; publish them first |
| Same measurement harness for both | never compare your harness's numbers to a number from a blog post |
| Equal batch/concurrency policy | batch 512 vs batch 1 is not a comparison |
| Report where you lose | expected, and the explanation is the interesting content |

**Expect to lose to production frameworks and say so.** A single engineer's engine reaching 30-60% of vLLM's throughput is a good result; claiming to beat it invites scrutiny that will find the flaw. The sentence that earns trust: *"vLLM is 2.4× faster at equal p99; the gap is attributable to FlashAttention kernels (measured 1.6×), CUDA graph capture (1.3×), and a fused sampler — none of which I implemented."* That is a senior answer, and it requires having profiled rather than guessed ([`playbooks/profiling.md`](../../playbooks/profiling.md)).

## 4. Failure-tested

Any reliability claim you make must have a test that breaks it. Pick at least three relevant to your project and record observed behavior:

| Break | Expected | Where it's taught |
|---|---|---|
| Kill a dependency mid-load (vector DB, reranker, model backend) | degraded answer or clean 503, never a hang | [Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md) |
| Overload beyond capacity | bounded queue, load shed, 429 — not unbounded latency | [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md) |
| Poison request (max context, huge `n`, adversarial input) | rejected at the edge; no replica death | [Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md) |
| `SIGSTOP`/hang a worker | liveness detects progress failure, not just process liveness | [Phase 8 lesson 4](../phase-8/04-kubernetes-for-gpu-serving.md) |
| Rolling restart under load | zero dropped/truncated streams | [Phase 8 lessons 4, 6](../phase-8/06-deploying-and-rollouts.md) |
| Client disconnect mid-stream | resources released, generation cancelled | [Phase 7 lesson 7](../phase-7/07-reliability-and-degradation.md) |
| Corrupt an artifact byte | fail closed with a readable error | [Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md) |

Two lines of README per test — what you broke, what happened — and this requirement is satisfied. It is also the cheapest credibility in the entire phase: almost nobody does it, so doing it is differentiating out of proportion to the effort.

## 5. Priced

Every capstone states cost per unit of useful work, with the arithmetic visible ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md), [Phase 9 lesson 5](../phase-9/05-hardware-diversity.md)):

```
  cost per 1M units = ($/hour ÷ (sustained_units_per_sec × 3600)) × 1e6

  report it at your SLO, and at a realistic utilization (30-60%),
  not only at saturation — and name the instance type and price source.
```

Also state the project's own spend: "this study cost $140 in rented GPU time" is a detail that signals you operate under real constraints. Cost discipline for the whole roadmap is in [`GETTING-STARTED.md`](../../GETTING-STARTED.md).

## 6. Bounded

A `## Limitations` section, written before anyone asks, containing the things you know are true:

- what is not implemented (and what it would cost to implement),
- which configurations were never tested,
- where the measurement is weak (single hardware type, synthetic traffic, small corpus),
- known correctness gaps,
- what you would do next, in priority order, with a reason for the order.

**This section is read as competence, not as weakness**, and it defuses the interview move of finding a gap you hadn't mentioned. A candidate who lists their own project's four biggest weaknesses is demonstrating the exact judgment the job requires.

---

## The self-review pass

Before declaring a capstone done, read your own repo as a stranger and answer:

1. Can I run the headline benchmark within 15 minutes of cloning?
2. Does every number in the README have a method, a run count, and a variance?
3. Is the baseline's configuration published, and would its maintainers call it fair?
4. Which three failures did I test, and is the observed behavior written down?
5. What does one million units cost, on what hardware, at what utilization?
6. Do the limitations name the things I'm least proud of?
7. Is the gap between my system and the reference *explained with measurements* rather than speculation?

Then run [`playbooks/production-readiness-checklist.md`](../../playbooks/production-readiness-checklist.md) against any capstone that serves traffic (4, 5, 7). Rows you cannot check are legitimate `## Limitations` entries — that is the correct use of the checklist, not a reason to hide it.

---

## Anti-standards: phrases that cost you credibility

| Phrase | Why it hurts | Replacement |
|---|---|---|
| "blazing fast" / "highly optimized" | unfalsifiable | "2,140 tok/s at p99 TTFT 480 ms (n=10, σ=3%)" |
| "production-ready" | nobody believes it about a solo project | "handles the seven failure cases in `RELIABILITY.md`" |
| "beats vLLM" (without config) | invites and fails scrutiny | "reaches 41% of vLLM's throughput; gap attributed below" |
| "scales to millions of users" | untested extrapolation | "measured to 400 QPS on 1 GPU; extrapolation and its assumptions below" |
| "state-of-the-art" | means nothing in a README | cite the specific paper and number you compared to |
| "TODO: benchmarks" | signals the project is unfinished | delete the feature, keep the measurement |

---

## Do this now (60 minutes, on whatever you have already built)

1. **Audit an existing project of yours** — a Phase 3/7/8/9 build is ideal — against the six requirements. Score each 0/1. The zeros are your work list.
2. **Fix requirement 1 first**: pin versions, write the one-command script, add `ENVIRONMENT.md`, commit the raw results file. It is the cheapest and it gates everything else.
3. **Add the trivial baseline row** to your main table if it is missing.
4. **Run one failure test** from the table above and write the two-line result. Notice how much more credible the README instantly reads.

---

**Next:** [Capstone brief: nano-vLLM →](03-capstone-nano-vllm.md) — the depth project, stated as a proof obligation rather than a tutorial.
