# Phase 3 — Serving Fundamentals: Batching, Queueing & Scheduling (Deep Dive)

> **Goal:** stop thinking about *a* request and start thinking about *traffic*. By the end of this phase you can take a model that runs fine in a notebook, put it behind a server that stays fast for 200 concurrent users, and defend every number you report with percentiles at a fixed offered load.

This folder is the long-form version of [Phase 3 in the ROADMAP](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling). Phases 0-2 were about one machine running one model: what a computer does, what a transformer does, why the GPU is starved at batch 1. **Phase 3 is the systems answer to that starvation** — keep the GPU fed with many requests at once, without letting the queue eat your tail latency.

This is the highest-leverage phase in the whole track for getting hired. Almost everyone applying to an inference role can say "vLLM is fast because of continuous batching." Very few can draw the scheduler loop, explain with Little's Law why p99 explodes at 90% utilization, or say what happens to a 4,000-token prefill that arrives while 60 sequences are decoding. That gap is this phase.

---

## Prerequisites

- **Phase 0 lesson 3** ([processes, threads, concurrency](../phase-0/03-processes-threads-concurrency.md)) — you'll be writing an async server with a background worker; you need to know why the GIL doesn't block you here and where a blocking call kills throughput.
- **Phase 0 lesson 6** ([build a server from scratch](../phase-0/06-build-a-server-from-scratch.md)) — sockets, request lifecycle, HTTP.
- **Phase 1 lessons 5-6** ([KV-cache](../phase-1/05-kv-cache.md), [prefill vs decode](../phase-1/06-prefill-vs-decode.md)) — every batching decision in this phase is really a decision about KV-cache memory and about two workloads with opposite bottlenecks.
- **Phase 2 lesson 5** ([the roofline model](../phase-2/05-roofline-model.md)) — the single result `decode arithmetic intensity ≈ batch size` is the *reason* this phase exists.

### About hardware

Good news: this is the first phase you can do almost entirely on your laptop, **including the Apple Silicon path**. The scheduling logic, the queueing math, and the benchmark harness don't care what's behind them. Use a tiny model (GPT-2 124M, Qwen2.5-0.5B) on CPU/MPS, or even a `sleep()`-based fake model with a realistic cost function — the *shape* of the latency-vs-load curve is the lesson, not the tokens/sec.

Then re-run the same harness once on a Colab T4 to see real numbers. Keep both tables. "Here's the same benchmark on CPU and on a T4, and here's why the knee moved" is a better artifact than either alone.

---

## The big idea of this phase

Phase 2 ended with an uncomfortable fact: at batch 1, decode uses under 1% of the GPU's math. The fix is to run many sequences at once. But requests **don't arrive together and don't finish together**, and that single sentence generates every idea in this phase:

```
        arrival misalignment          lifetime misalignment
        (they start at different      (they finish at different
         times)                        times — 20 tokens vs 800)
                │                              │
                ▼                              ▼
        DYNAMIC BATCHING              CONTINUOUS BATCHING
        wait T ms, then run           schedule per ITERATION, not per request
        (lesson 3)                    (lesson 4)
                └──────────────┬───────────────┘
                               ▼
                  and both are bounded by
              KV-CACHE MEMORY + QUEUEING THEORY
                     (lessons 4, 5)
```

And one discipline sits above all of it: **a serving number without a stated offered load and a percentile is not a number.** "We do 3,000 tokens/sec" and "p99 TTFT is 400 ms" only mean something together.

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [What a serving system actually is](01-what-a-serving-system-is.md) | "I can draw a request's full timeline — arrival, queue wait, prefill, N decode steps, stream close — and name the metric that owns each segment: TTFT, TPOT, e2e, throughput, goodput." |
| 2 | [Static batching (and why it hurts)](02-static-batching.md) | "Static batching pads to the longest prompt and waits for the longest *output*, so a batch of 8 can waste 60%+ of its decode steps and block every request behind it." |
| 3 | [Dynamic batching](03-dynamic-batching.md) | "Waiting T ms to form a batch only helps if `λ·T` is comparable to the batch size — and it fixes arrival misalignment only. For fixed-shape models it's the right answer; for LLMs it isn't enough." |
| 4 | [Continuous batching (iteration-level scheduling)](04-continuous-batching.md) | "Batch at the iteration, not the request: every decode step, admit newly queued sequences into freed slots and evict finished ones. I can trace vLLM's `schedule()` and say what preempts what." |
| 5 | [Queueing theory for inference](05-queueing-theory.md) | "`L = λW`. Latency scales as `1/(1−ρ)`, so at 95% utilization mean wait is 20× service time and p99 is far worse — and bigger batches raise throughput *and* p99 at the same time." |
| 6 | Scheduling policies & admission control | *(next commit)* |
| 7 | Measuring it honestly: load generation & percentiles | *(next commit)* |
| 8 | Build: naive vs dynamic-batched FastAPI server | *(next commit)* |
| 9 | Build: a tiny continuous-batching engine | *(next commit)* |
| 10 | Exercises & exit artifact | *(next commit)* |

---

## How to work through this phase

1. **Do lesson 5's arithmetic by hand.** Little's Law and the `1/(1−ρ)` curve are the two pieces of math that make you sound like a systems engineer rather than an ML enthusiast. They take twenty minutes to learn and last a career.
2. **Predict, then measure — again.** Before your first load test, write down the QPS at which you expect the knee, from `1/service_time × max_batch`. Being wrong and explaining why is the exercise.
3. **Read real scheduler code.** [vLLM's scheduler](https://github.com/vllm-project/vllm) and [TGI's router queue](https://github.com/huggingface/text-generation-inference) are both readable in an afternoon. Lesson 4 walks you through them. This is also where your first OSS contribution is most likely to come from.
4. **Never report a mean.** From this phase onward, every latency number you write down has a percentile and an offered load attached. See [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md).

**Time budget:** 2-4 weeks part-time. Lessons 4 and 5 are the load-bearing ones; the two build lessons produce your first genuinely portfolio-grade artifact.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Explain with **Little's Law** why raising max batch size increases p99 latency even while throughput keeps improving. ([lesson 5](05-queueing-theory.md))
2. Explain what **continuous batching** changes relative to dynamic batching, in terms of *when* a scheduling decision is made. ([lesson 4](04-continuous-batching.md))
3. Say what happens to a long **prefill** that arrives while many sequences are decoding, and name two fixes. ([lesson 4](04-continuous-batching.md))
4. Describe how you'd benchmark a serving system so the numbers are honest — offered load, warmup, percentiles, TTFT vs TPOT reported separately. ([lesson 1](01-what-a-serving-system-is.md), [benchmarking playbook](../../playbooks/benchmarking.md))

## Projects that belong to this phase

- **[01 — Tiny inference server](../../projects/README.md)** (small): `/generate_naive` vs `/generate_batched`, load-tested, with a committed p50/p90/p99 + throughput table. **This is the Phase 3 exit artifact.**
- **[02 — Continuous-batching engine](../../projects/README.md)** (large): an in-flight scheduler with per-sequence KV-cache, eviction, and admission — a tiny vLLM, benchmarked against the two endpoints above.

---

Next after this: **[Phase 4 — Inference Optimization Techniques](../../ROADMAP.md#phase-4--inference-optimization-techniques)**. Phase 3 keeps the GPU busy; Phase 4 makes each token cheaper — quantization, PagedAttention, speculative decoding.
