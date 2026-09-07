# 1 — What a Serving System Actually Is

> **You'll be able to say:** "A served request has five segments — queue wait, prefill, first token out, N decode steps, stream close — and each one is owned by a different metric and breaks for a different reason. TTFT includes queue wait, TPOT doesn't, and reporting either as a mean hides the only cases anyone cares about."

In Phase 1 you generated text with a loop that called your model. That is not a serving system, and the difference is not "add FastAPI." A notebook loop has **one** request, so latency and throughput are the same statement and there is nothing to schedule. A serving system has *traffic*: requests arriving at times you don't control, with lengths you don't know, competing for one GPU's memory and one GPU's clock.

This lesson builds the vocabulary. Everything after it is a scheduling decision, and you cannot judge a scheduling decision without knowing exactly which number it moves.

---

## From `generate()` to a service

```
  NOTEBOOK                              SERVICE
  ────────                              ───────
  out = model.generate(prompt)          client ──HTTP/gRPC──▶ web layer (FastAPI/uvicorn)
                                                                  │  request object + future
  one caller                                                      ▼
  one request in flight                                     ┌───────────┐
  latency == 1/throughput                                   │   QUEUE   │ ← the thing this phase is about
  no queue, no scheduler                                    └───────────┘
  nothing to tune                                                 │  scheduler picks a batch
                                                                  ▼
                                                            engine loop (owns the GPU,
                                                            one process, one CUDA context)
                                                                  │  tokens, streamed
                                                                  ▼
                                                            client (SSE / chunked HTTP)
```

Four structural facts fall out of that picture, and they trip up almost every first attempt:

1. **One GPU, one engine loop.** The GPU is a single resource with a single CUDA context worth using; you don't get to run four copies of the model in four threads on one card and hope. The engine is a **single-threaded loop** that owns the device and pulls work from a queue. All concurrency lives *in front of* it.
2. **The web layer must never block.** If your handler calls `model.generate()` directly, that handler holds the loop while the GPU works, and request #2 waits even though the GPU is 99% idle. This is the `/generate_naive` endpoint you'll build in the project — it exists to be beaten.
3. **A request is an object with state, not a function call.** It has an arrival timestamp, a prompt, sampling params, generated tokens so far, a KV-cache, a status (queued/running/finished), and a channel back to its caller. Every serving framework has this class: vLLM calls it a `Request`/`SequenceGroup`, TGI calls it an `Entry`. When you write yours in lesson 9, it looks the same.
4. **Output is a stream, not a value.** LLM responses are delivered token-by-token (Server-Sent Events or gRPC streaming), which is *why* first-token latency is a separate metric from total latency. If you only return complete responses, you've thrown away the single biggest perceived-latency win available to you.

---

## The five segments of a request's life

This timeline is the most important diagram in the phase. Memorize the segment names.

```
  t_arrive        t_start          t_first_token                          t_last_token
     │               │                   │                                     │
     ├───────────────┼───────────────────┼─────────────────────────────────────┤
     │  QUEUE WAIT   │     PREFILL       │   DECODE: N-1 steps, one token each │
     │  (scheduler   │  (whole prompt,   │   (each step reads ALL weights +     │
     │   hasn't      │   one big GEMM,   │    the KV-cache; memory-bound)       │
     │   picked me)  │   compute-bound)  │                                     │
     └───────────────┴───────────────────┴─────────────────────────────────────┘
     ├──────────── TTFT ─────────────────┤
                                         ├────────── TPOT × (N-1) ─────────────┤
     ├─────────────────────── end-to-end latency ────────────────────────────  ┤
```

| Metric | Definition | What breaks it | Typical target (chat) |
|---|---|---|---|
| **TTFT** — time to first token | `t_first_token − t_arrive` | queue depth, prompt length, prefill batching | < 200-500 ms |
| **TPOT** — time per output token | `(t_last − t_first) ÷ (N−1)` | batch size, HBM bandwidth, launch overhead | 10-50 ms (20-100 tok/s) |
| **ITL** — inter-token latency | the *distribution* of individual gaps, not the mean | one long prefill jumping into your batch | p99 matters, not mean |
| **e2e latency** | `t_last_token − t_arrive` = TTFT + TPOT×(N−1) | everything above + output length | task-dependent |
| **Normalized latency** | `e2e ÷ N` — from the Orca paper | best single number for comparing schedulers | — |
| **Throughput** | output tokens/sec (and requests/sec) across all users | batch size, i.e. this whole phase | maximize at fixed SLO |
| **Goodput** | requests/sec that met their SLO | tail latency; the honest business metric | maximize |

Three of these deserve emphasis:

**TTFT includes queue wait.** This is the mistake to avoid. It is very easy to build a server whose *prefill* is a fast 80 ms while users experience 3-second TTFT, because 40 requests were ahead of them. If you instrument only the engine, you will not see it: you need the timestamp from when the request **entered the process**, not from when the scheduler picked it up. Every real framework exposes both (vLLM's `vllm:time_to_first_token_seconds` vs its queue-time gauge).

**TPOT and TTFT have opposite bottlenecks and opposite fixes.** Prefill is compute-bound (a big GEMM over hundreds of tokens); decode is memory-bound (one token, all the weights) — this is [Phase 1 lesson 6](../phase-1/06-prefill-vs-decode.md) and [Phase 2 lesson 5](../phase-2/05-roofline-model.md) restated. So: bigger batches make TPOT *worse* per-user but throughput much better; quantization mostly helps TPOT; more compute mostly helps TTFT. **A single "latency" number averages two variables that move in opposite directions.** Report them separately, always.

**ITL is a distribution, and the tail is the product.** If 49 of 50 gaps are 25 ms and one is 900 ms because a 4,000-token prefill was scheduled into your batch, the mean ITL is 42 ms and looks fine, while the user watched the text freeze mid-sentence. Streaming makes stutter visible in a way that batch APIs never did. Track p95/p99 ITL. Lesson 4 explains exactly where that 900 ms comes from.

---

## Do the arithmetic: where does the time actually go?

A concrete request: 500-token prompt, 200-token output, 7B model in FP16 on an A100.

```
  prefill:   500 tokens × 2 × 7e9 FLOPs   = 7.0e12 FLOPs
             ÷ ~150 TFLOP/s achieved       ≈ 47 ms          ← compute-bound, scales with prompt
  decode:    200 steps × (14 GB ÷ 2 TB/s)  ≈ 200 × 7 ms = 1,400 ms
                                                            ← memory-bound, scales with output
  e2e ≈ 47 + 1,400 ≈ 1.45 s     TTFT ≈ 47 ms     TPOT ≈ 7 ms
```

Read the ratio: **prefill is 3% of the wall clock, decode is 97%.** That is the normal shape of a chat request, and it is why this whole phase is about keeping *decode* batched. Now change one thing at a time and watch which metric moves:

| Change | TTFT | TPOT | Throughput | Why |
|---|---|---|---|---|
| Prompt 500 → 4,000 tokens | 47 → ~380 ms | ~unchanged | ↓ | prefill FLOPs scale with prompt |
| Output 200 → 800 tokens | unchanged | unchanged | ↓ per-request | 4× more decode steps |
| Batch 1 → 32 concurrent decodes | ↑ (queueing) | ~7 → ~9 ms | **≈ 25×** | weights read once, serve 32 tokens ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)) |
| FP16 → INT8 weights | slightly ↓ | ~7 → ~3.5 ms | ↑ | half the bytes per decode step (Phase 4) |
| 40 requests already queued | +3,000 ms | unchanged | unchanged | **queue wait is invisible to engine-side metrics** |

That third row is the entire economic case for this phase: **throughput rose ~25× and per-user TPOT got 30% worse.** Nothing else in inference engineering has that ratio. And the last row is the entire reliability case: the biggest TTFT regression in production is usually not the model at all.

---

## Offered load: the axis every benchmark needs

A latency number means nothing without the load it was measured at. The load knob is **offered load** — the arrival rate (QPS or requests/sec) your clients *generate*, independent of how fast you serve them.

```
  p99 latency
      │                                              ╱  ← "knee": queue starts growing
      │                                          ╱      faster than you drain it
      │                                    ╱
      │                            ╱
      │        ────────────────╱
      │  flat: server keeps up
      └──────────────────────────────────────────────▶ offered QPS
        the only interesting number is WHERE the knee is
```

Two ways to generate load, and they measure different things:

- **Closed loop** (`locust` defaults, most naive scripts): C virtual users, each sends a request, *waits for the response*, then sends the next. Concurrency is fixed at C; arrival rate falls automatically when the server slows down. This measures "how does the system behave at fixed concurrency" — and it **cannot show you overload**, because a struggling server gets sent less work. That self-correction is called **coordinated omission**, and it silently deletes your worst latencies.
- **Open loop** (fixed QPS, Poisson arrivals): clients send at a rate regardless of what came back. Queues grow without bound past capacity — which is exactly what real traffic does, and exactly what you need to find the knee.

**Benchmark open-loop at several fixed QPS values, report percentiles at each.** Then, and only then, quote a throughput number: "3,100 output tok/s at p99 TTFT ≤ 500 ms." That sentence is a serving result. "23 tok/s" is not. Full procedure — warmup, workload shape, percentile reporting — is in [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md), and lesson 7 turns it into a harness.

---

## Why "average latency" is worse than useless

It's not just imprecise; it actively hides the failure. Two servers, same mean:

```
  Server A: every request 500 ms                        → mean 500 ms, p99 500 ms
  Server B: 95% at 100 ms, 5% at 8,100 ms               → mean 500 ms, p99 ≈ 8,100 ms
```

Server B has a 1-in-20 chance of a user leaving. And it gets worse with fan-out: a page that makes **10** backend calls and waits for all of them hits the p99 path with probability `1 − 0.99¹⁰ ≈ 10%`. Your p99 is somebody else's p90. This is why tail latency (not mean) is the currency of SLOs everywhere from AWS to Google — Marc Brooker's and Gil Tene's writing on this is the canonical reading, and *DDIA* Ch. 1 says it in three pages.

The practical rules, which you will follow for the rest of the track:

1. Report **p50/p90/p99**, never a mean. Compute percentiles from raw samples; **never average percentiles** across shards or time buckets (that's not a percentile of anything).
2. Report **TTFT and TPOT separately**; they have different bottlenecks.
3. State the **offered load** and the **workload shape** (prompt/output length distribution).
4. **Warm up** before measuring: CUDA context, allocator, autotune, `torch.compile`, and CUDA-graph capture all make the first requests unrepresentative ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)).
5. Prefer **goodput** — "req/s meeting a 500 ms TTFT SLO" — when comparing configurations. It's the only metric that can't be gamed by making some users very slow.

---

## Try it (laptop, no GPU needed)

You need the vocabulary in your fingers before lesson 2. Model the timeline with a fake engine so nothing is hidden:

```python
import asyncio, random, statistics, time

SERVICE_PREFILL = 0.005     # 5 ms prefill   (scaled down so the whole sweep runs in ~1 min)
SERVICE_TOKEN   = 0.002     # 2 ms per decode step

async def handle(req_id, out_len, results):
    t_arrive = time.perf_counter()
    async with SLOT:                                     # one "GPU": strictly serial
        t_start = time.perf_counter()
        await asyncio.sleep(SERVICE_PREFILL)
        t_first = time.perf_counter()
        for _ in range(out_len - 1):
            await asyncio.sleep(SERVICE_TOKEN)
        t_last = time.perf_counter()
    results.append(dict(queue=t_start - t_arrive, ttft=t_first - t_arrive,
                        tpot=(t_last - t_first) / max(out_len - 1, 1), e2e=t_last - t_arrive))

async def run(qps, n=40):
    global SLOT
    SLOT = asyncio.Semaphore(1)                          # batch size 1, on purpose
    results, tasks = [], []
    for i in range(n):
        await asyncio.sleep(random.expovariate(qps))     # open loop, Poisson arrivals
        tasks.append(asyncio.create_task(handle(i, random.choice([20, 50, 200]), results)))
    await asyncio.gather(*tasks)
    p = lambda k, q: statistics.quantiles([r[k] for r in results], n=100)[q - 1]
    print(f"qps={qps:>4}  queue p50={p('queue',50)*1e3:7.0f}ms p99={p('queue',99)*1e3:8.0f}ms  "
          f"ttft p99={p('ttft',99)*1e3:8.0f}ms  tpot p50={p('tpot',50)*1e3:5.1f}ms")

for qps in (2, 4, 5, 6):
    asyncio.run(run(qps))
```

**Predict before running:** mean service time is `0.005 + 0.002 × (avg output ≈ 90) ≈ 0.183 s`, so capacity is about **5.5 req/s**. A real run:

```
qps=   2  queue p50=      0ms p99=     367ms  ttft p99=     373ms  tpot p50=  2.3ms
qps=   4  queue p50=    448ms p99=    1112ms  ttft p99=    1118ms  tpot p50=  2.3ms
qps=   5  queue p50=    394ms p99=    1651ms  ttft p99=    1657ms  tpot p50=  2.3ms
qps=   6  queue p50=   1648ms p99=    2344ms  ttft p99=    2349ms  tpot p50=  2.3ms
```

`tpot` is flat at 2.3 ms across a 3× change in load — the per-token work never changed — while queue wait, and therefore TTFT, grows without bound. Past ~5.5 QPS the numbers don't converge at all; they grow for as long as you run the test. That's the knee, that's why TTFT and TPOT must be reported separately, and that's the curve [lesson 5](05-queueing-theory.md) derives in closed form.

---

## Key takeaways

- A serving system = **web layer (never blocks) + queue + single-threaded engine loop that owns the GPU**. All concurrency lives in front of the engine.
- A request is a **stateful object** (arrival time, tokens, KV-cache, status, response channel), not a function call — that's what makes scheduling possible.
- **Five segments:** queue wait → prefill → first token → decode steps → close. `TTFT = queue + prefill`, `TPOT = decode`, `e2e = TTFT + TPOT×(N−1)`.
- **TTFT includes queue wait.** Timestamp requests at process entry, or you'll be blind to your most common regression.
- **Prefill is compute-bound, decode is memory-bound**, so batching hurts TPOT slightly and helps throughput enormously (~25× at B=32) — the trade this phase manages.
- **Offered load is a required axis.** Measure open-loop at fixed QPS; closed-loop hides overload via coordinated omission.
- **Percentiles or nothing**: p50/p90/p99 from raw samples, TTFT and TPOT separately, warmed up, with the workload shape stated. Prefer **goodput** to compare configs.

**Next:** [Static batching (and why it hurts) →](02-static-batching.md) — the obvious way to batch, the 60% waste it creates, and the head-of-line blocking that makes it unusable for chat.
