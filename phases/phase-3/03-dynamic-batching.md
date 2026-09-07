# 3 — Dynamic Batching: Buying Throughput with a Time Window

> **You'll be able to say:** "Dynamic batching holds requests for up to T ms or until N have arrived, then runs them as one static batch. The window is only worth anything if `λ·T` is comparable to N — and under real load the batch size self-tunes to `λa/(1−λb)` and the timer stops mattering entirely. It fixes arrival misalignment; the convoy problem survives untouched."

Lesson 2's static batch had one absurd property: the batch was assembled by hand. Real requests arrive whenever they want, so the first design question of any server is *"how long do I wait for company before I start work?"* Dynamic batching is the answer, it is a genuine and large win, and it is the correct final answer for a lot of production models. It is also the design that most people mistake for "how LLM serving works."

---

## The mechanism

One queue, one worker loop, two knobs:

```
  clients ──▶ ┌──────── queue ────────┐
              │ r1  r2  r3 …          │
              └───────────────────────┘
                        │  worker loop:
                        │    batch = [ queue.get() ]              ← block for the first one
                        │    deadline = now + T
                        │    while len(batch) < N and now < deadline:
                        │        try: batch.append(get(timeout=deadline-now))
                        │        except Timeout: break
                        ▼    run_static_batch(batch); return each result to its caller
                     GPU (one static batch at a time)

  knobs:  N = max_batch_size   (memory/throughput ceiling)
          T = max_queue_delay  (latency you're willing to pay for company)
```

Three properties follow directly, and each is a design lever:

1. **The first request in a window pays the most wait; the last pays ~none.** So added latency is *unfair by construction* — and if the window rarely fills, the average added wait is about `T/2`.
2. **Under load the window never expires.** If N requests are already queued when the worker comes back, the batch fills instantly and T is irrelevant. **T only ever matters at low load** — which is precisely when you didn't need a batch. Hold on to that irony; it's the punchline of this lesson.
3. **It is still static batching inside.** The batch, once formed, runs to completion as a closed set. Everything lesson 2 said about output-length waste and the convoy effect still applies for autoregressive models.

---

## Do the arithmetic (1): when is the window worth anything?

Arrivals are Poisson at rate λ. In a window of T seconds you expect `λ·T` arrivals, so:

```
  achieved batch size  B ≈ min(N, 1 + λ·T)
  to reach batch N you need  T ≥ N / λ      ⟺    λ ≥ N / T
```

Plug in the numbers people actually configure:

| Offered load λ | T = 5 ms | T = 20 ms | T = 100 ms | To fill N=32 you need |
|---|---|---|---|---|
| 10 req/s | B ≈ 1.05 | B ≈ 1.2 | B ≈ 2 | T = 3.2 **s** — unusable |
| 100 req/s | B ≈ 1.5 | B ≈ 3 | B ≈ 11 | T = 320 ms |
| 1,000 req/s | B ≈ 6 | B ≈ 21 | B = 32 (full) | T = 32 ms |
| 5,000 req/s | B ≈ 26 | B = 32 (full) | B = 32 (full) | T = 6.4 ms |

**A 10 ms window at 100 QPS buys you a batch of two.** You paid 10 ms of TTFT for a ~2× improvement on a memory-bound step — probably worth it, but nothing like the 25× that B=32 promised in [lesson 1](01-what-a-serving-system-is.md). This table is the antidote to the most common tuning mistake in serving: **tuning T when the real problem is that λ is too low to fill a batch on any timescale a user will tolerate.** At low λ the honest options are to accept small batches, use a smaller/cheaper model or instance, or consolidate traffic from several endpoints onto one engine.

---

## Do the arithmetic (2): the batch size tunes itself

Model one batched step as **fixed cost + per-item cost**, which is exactly what a memory-bound decode step is (`a` = reading the weights, once, no matter how many rows; `b` = the marginal per-sequence work):

```
  step time     S(B) = a + b·B
  throughput    X(B) = B / (a + b·B)      →  X(∞) = 1/b     ← hard capacity ceiling
```

Now the key observation. Under sustained load the worker doesn't wait on a timer at all: it takes **everything that arrived while it was busy**. That's a fixed point:

```
  B = λ · S(B) = λ(a + bB)        ⟹        B = λa / (1 − λb)
```

With `a = 10 ms` (weight read) and `b = 0.5 ms` (per-sequence marginal cost), so a capacity ceiling of `1/b = 2,000 req/s`:

| λ (req/s) | self-tuned B | step time S(B) | queue wait ≈ S | latency ≈ 2S | utilization ρ = λ/2000 |
|---|---|---|---|---|---|
| 200 | 2.2 | 11 ms | 11 ms | ~22 ms | 10% |
| 1,000 | 20 | 20 ms | 20 ms | ~40 ms | 50% |
| 1,600 | 80 | 50 ms | 50 ms | ~100 ms | 80% |
| 1,800 | 180 | 100 ms | 100 ms | ~200 ms | 90% |
| 1,950 | 780 | 400 ms | 400 ms | ~800 ms | 97.5% |
| 2,000 | ∞ | ∞ | — | — | 100% |

This one table contains most of what a serving engineer needs to know:

- **Dynamic batching is self-regulating.** Load goes up, batches get bigger, per-request efficiency improves. You never hand-tune B for peak; you cap it (`N`) so memory doesn't explode.
- **The system finds its own operating point**, and that point drifts toward larger batches and higher latency as traffic grows — silently, without any deploy. Your p99 gets worse on Monday morning because more people showed up.
- **Latency blows up as `1/(1−ρ)`.** From 50% to 90% utilization, throughput improves 1.8× and latency gets 5× worse. That functional form is not an artifact of this cost model — [lesson 5](05-queueing-theory.md) derives it from queueing theory, and it's the most important curve in the phase.
- **`N` is the safety valve.** Capping B bounds latency and KV-cache memory at the cost of shedding or queueing the excess. Choosing that cap *is* the throughput-vs-tail-latency decision, made explicit.

---

## Implementation shape (this is Project 01)

The pattern is a queue of `(request, future)` pairs plus one background worker. Note carefully what makes it correct: the HTTP handler `await`s a future and **never touches the model**, so the event loop stays free to accept more requests while the GPU works.

```python
import asyncio, time
from fastapi import FastAPI
from pydantic import BaseModel

MAX_BATCH, MAX_DELAY_S = 32, 0.010          # N and T

app, queue = FastAPI(), asyncio.Queue()

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 64

@app.post("/generate_batched")
async def generate_batched(req: Req):
    fut = asyncio.get_running_loop().create_future()
    await queue.put((req, fut, time.perf_counter()))     # timestamp at ENTRY, for honest TTFT
    return {"text": await fut}                            # handler blocks; event loop does not

async def worker():
    while True:
        first = await queue.get()                         # block until there is any work
        batch, deadline = [first], time.perf_counter() + MAX_DELAY_S
        while len(batch) < MAX_BATCH:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout))
            except asyncio.TimeoutError:
                break                                     # window expired: run what we have
        try:
            texts = await asyncio.to_thread(              # keep the GPU call off the event loop
                run_static_batch, [r for r, _, _ in batch])
            for (_, fut, _), text in zip(batch, texts):
                if not fut.cancelled(): fut.set_result(text)
        except Exception as e:                            # one bad request must not kill the worker
            for _, fut, _ in batch:
                if not fut.cancelled(): fut.set_exception(e)

@app.on_event("startup")
async def _start(): asyncio.create_task(worker())
```

The failure modes to build in from the start, because every one of them is a real outage:

- **Blocking the event loop.** A synchronous `model.generate()` inside an `async def` freezes *all* request handling, including health checks — the server looks dead while the GPU is fine. Use `asyncio.to_thread` (or a separate process) as above.
- **An unbounded queue.** Under overload, memory grows and every queued request eventually times out client-side; you burn GPU on answers nobody is waiting for. Bound the queue and **reject fast** (HTTP 429/503) — see [lesson 5](05-queueing-theory.md) on load shedding.
- **Ignoring cancellation.** Clients disconnect. Check `fut.cancelled()` before computing and drop dead work; a busy server can waste double-digit percentages of its GPU on abandoned requests.
- **One exception killing the worker.** Wrap the batch call; fail that batch, keep the loop alive.
- **Cold-start measurements.** Warm up with a dummy batch at startup before serving or benchmarking ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)).

This mechanism is not a toy: NVIDIA **Triton Inference Server** ships exactly it as `dynamic_batching { max_queue_delay_microseconds, preferred_batch_size }`, and TensorFlow Serving, TorchServe, KServe, Ray Serve and BentoML all expose the same two knobs under different names. For a vision or embedding model, **this is the production answer** and you should stop here.

---

## Why it isn't enough for LLMs

Dynamic batching solves **arrival misalignment**: requests that show up at different times can still run together. It does nothing about **lifetime misalignment**: they finish at wildly different times.

```
  DYNAMIC BATCHING, 4 requests, outputs of 20/50/200/500 tokens
  ├─── window T ───┤
  batch #1: ████████████████████████████████████████████████  500 steps (all 4 slots held)
                   ↑A done (20)  ↑B done (50)      ↑C done (200)
                   └──────────── 3 slots idle for most of the batch ─────────┘
  r5 arrives here ─┘                                        r5 starts only HERE ──────┘
```

Everything from [lesson 2](02-static-batching.md) is still true: ~66% of decode slot-steps wasted, r5 head-of-line blocked behind an essay it has nothing to do with, freed slots unusable. The window changed *who joins* the batch; it didn't change the fact that **the batch is the scheduling unit**.

And the fix is now obvious when stated in the right units. The unit of work in autoregressive serving is not a *request*, it's **one decode step for one sequence**. Make that the thing you schedule and all three pathologies disappear at once.

---

## Try it (laptop)

Take the fake-engine harness from [lesson 1](01-what-a-serving-system-is.md) and add a batching worker with the `S(B) = a + bB` cost model. **Predict first**, and be careful with the capacity number: `1/b = 2,000 req/s` is the ceiling only for *unbounded* B. With the cap `N = 32` the step is at most `a + bN = 26 ms`, so real capacity is `N/(a+bN) = 32/0.026 ≈ 1,230 req/s`. Predict what happens at 1,600 QPS before you run it.

```python
import asyncio, random, statistics, time

A, B_COST, MAX_BATCH, T = 0.010, 0.0005, 32, 0.010

async def run(qps, n):
    q, lat = asyncio.Queue(), []
    async def client():
        for _ in range(n):
            await asyncio.sleep(random.expovariate(qps))
            await q.put(time.perf_counter())
    async def worker():
        while True:
            batch = [await q.get()]
            dl = time.perf_counter() + T
            while len(batch) < MAX_BATCH and (to := dl - time.perf_counter()) > 0:
                try: batch.append(await asyncio.wait_for(q.get(), to))
                except asyncio.TimeoutError: break
            await asyncio.sleep(A + B_COST * len(batch))          # the "GPU"
            done = time.perf_counter()
            lat.extend(done - t0 for t0 in batch)
            sizes.append(len(batch))
    sizes = []
    w = asyncio.create_task(worker())
    await client(); await asyncio.sleep(3.0); w.cancel()
    p = lambda k: statistics.quantiles(lat, n=100)[k - 1] * 1e3
    print(f"qps={qps:>5} n={n:>5} meanB={statistics.mean(sizes):5.1f} "
          f"p50={p(50):7.1f}ms p99={p(99):8.1f}ms  "
          f"first100 p50={statistics.median(lat[:100])*1e3:6.1f}ms "
          f"last100 p50={statistics.median(lat[-100:])*1e3:8.1f}ms")

for qps, n in ((200, 800), (1000, 3000), (1600, 4000), (1800, 4000), (2200, 4000)):
    asyncio.run(run(qps, n))
```

A real run (numbers move a little with machine and event-loop overhead):

```
qps=  200 n=  800 meanB=  4.4 p50=   24.9ms p99=    38.1ms  first100 p50=  23.8ms last100 p50=    24.2ms
qps= 1000 n= 3000 meanB= 27.5 p50=   40.3ms p99=    59.4ms  first100 p50=  39.6ms last100 p50=    38.3ms
qps= 1600 n= 4000 meanB= 31.7 p50=  294.1ms p99=   598.0ms  first100 p50=  41.9ms last100 p50=   593.6ms
qps= 1800 n= 4000 meanB= 31.7 p50=  506.3ms p99=   987.8ms  first100 p50=  39.9ms last100 p50=   985.0ms
qps= 2200 n= 4000 meanB= 31.7 p50=  688.2ms p99=  1352.1ms  first100 p50=  41.9ms last100 p50=  1348.3ms
```

Everything predicted is there. Mean batch size climbs (4.4 → 27.5) and then pins at the cap of 32; p99 runs ~1.5-2× p50 the whole way. And look at the last two columns for the three overloaded rows: **the first hundred requests saw ~42 ms and the last hundred saw ~600-1,350 ms.** That reported p50 of 294 ms is not a property of the system — it's an average over a queue that was still growing when the test stopped. Run it twice as long and every number above 1,230 QPS doubles.

**An unstable system doesn't have a p99; it has a slope.** Always check whether latency is drifting across the run before you quote a percentile — comparing the first and last decile is the cheapest possible test, and it catches the single most common lie in serving benchmarks.

---

## Key takeaways

- **Dynamic batching = queue + (max wait T, max size N), then run one static batch.** Universal in non-LLM serving; Triton's `dynamic_batching` is exactly this.
- **`B ≈ min(N, 1 + λT)`.** The window only helps if `λ·T` approaches N; a 10 ms window at 100 QPS gets you a batch of 2. Low load can't be fixed with timers.
- **Under load the batch self-tunes to `B = λa/(1−λb)`** and T becomes irrelevant. Capacity ceiling is `1/b`; efficiency comes from amortizing `a`.
- **Latency degrades as `1/(1−ρ)`** — 50%→90% utilization is 1.8× throughput for 5× latency. `N` is the explicit cap that trades throughput for tail latency and bounds KV memory.
- **Engineering musts:** never block the event loop, bound the queue and shed load, honor client cancellation, isolate batch failures, warm up before measuring.
- **It fixes arrival misalignment only.** Output-length waste, convoy effect, and head-of-line blocking all survive, because the *request* is still the scheduling unit.
- The right unit for autoregressive serving is **one decode step for one sequence** — which is the next lesson.

**Next:** [Continuous batching (iteration-level scheduling) →](04-continuous-batching.md) — schedule per step instead of per request, and get better throughput *and* better latency at the same time.
