# 7 — Measuring It Honestly: Load Generation & Percentiles

> **You'll be able to say:** "A closed-loop client cannot measure overload — it stops sending when the server slows down, which is coordinated omission. In my own harness the same server at the same offered load reported p99 = 328 ms to a naive 8-slot client and 121 s when latency was timestamped from intended arrival: a **370× lie**. So I generate open-loop at fixed QPS with Poisson arrivals, timestamp at intended send time, discard warmup, check for drift across the run, use ≥1,000 samples before quoting a p99, and report **goodput** — which peaks and then collapses while throughput is still going up."

Everything up to here was mechanism and policy. This lesson is the part that makes the rest *true*. It is also the single most transferable skill in the phase: benchmarking discipline is what separates "we think this helped" from "we know this helped, here's the curve."

Blunt version: **most published inference benchmarks — including ones from serious companies — are wrong in at least one of four specific ways.** All four are mechanical, all four are avoidable in an afternoon, and being the person who spots them in a design review is a career asset.

---

## The four ways a serving benchmark lies

```
  1. CLOSED LOOP        client waits for a reply before sending the next request
                        → the server's slowdown throttles your load generator
                        → overload becomes literally unmeasurable  (coordinated omission)

  2. NO WARMUP          CUDA context, allocator, autotune, torch.compile, CUDA-graph capture
                        → first requests are 10-100x slow, or (worse) a warm prefix cache
                          makes later ones fake-fast

  3. PERCENTILE OF AN   the queue was still growing when the run ended
     UNSTABLE RUN       → the "p99" is an average over a moving target, not a property
                          of the system. Run twice as long and it doubles.

  4. WRONG AGGREGATION  averaging percentiles across shards/windows; p99 from 50 samples;
                        one run; means instead of percentiles; client is the bottleneck
```

Number 1 is the deep one, so it gets its own section.

---

## Coordinated omission: the bug in almost every load test

The name is Gil Tene's. The mechanism is this: your client has a finite number of in-flight slots (threads, virtual users, connection-pool entries). When the server gets slow, requests **queue up inside the client**, and if you start the stopwatch at *send* time, that client-side queueing is invisible. Worse, the offered load silently drops — a struggling server gets sent *less* work, which is the exact opposite of what real traffic does.

The fix is one line of code: **start the stopwatch at the time the request was *supposed* to be sent**, not when you got around to sending it.

```python
# WRONG — measures the server, conditioned on the client keeping up
t0 = time.perf_counter(); resp = await send(req); record(time.perf_counter() - t0)

# RIGHT — measures what a user experiences
t_intended = schedule[i]                      # decided BEFORE the run, from the arrival process
await sleep_until(t_intended)                 # (or: note how late you are and keep the intent)
resp = await send(req); record(time.perf_counter() - t_intended)
```

### Do the experiment (30 lines, exact, no server needed)

One server with 20 ms mean service time — capacity 50 req/s. One client with `slots` in-flight limit. Every request is measured *both* ways.

```python
import random, statistics

S = 0.020            # mean service time: 20 ms  ->  capacity = 50 req/s

def measure(offered_qps, client_slots, T=120.0, seed=0):
    """One server (FCFS) + a client with a bounded number of in-flight slots."""
    rnd = random.Random(seed)
    intended, t = [], 0.0
    while t < T:
        t += rnd.expovariate(offered_qps)
        intended.append(t)
    slot_free = [0.0] * client_slots
    server_free = 0.0
    naive, honest = [], []
    for a in intended:
        j = min(range(client_slots), key=lambda k: slot_free[k])
        sent  = max(a, slot_free[j])                 # client-side wait: NOT in the naive number
        start = max(sent, server_free)               # server-side queue wait
        server_free = start + rnd.expovariate(1 / S)
        slot_free[j] = server_free
        naive.append(server_free - sent)             # timestamped at send  -> lies
        honest.append(server_free - a)               # timestamped at intended arrival -> truth
    P = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1] * 1e3
    achieved = len(intended) / max(slot_free)
    print(f"offered={offered_qps:>4}/s slots={client_slots:>3}  achieved={achieved:5.1f}/s  "
          f"naive p50={P(naive,50):6.1f}ms p99={P(naive,99):8.1f}ms   "
          f"honest p50={P(honest,50):7.1f}ms p99={P(honest,99):9.1f}ms   "
          f"lie factor={P(honest,99)/P(naive,99):5.1f}x")

print("--- capacity is 50 req/s; watch what the client's slot count does to the report ---")
for qps, slots in ((25, 8), (45, 8), (60, 8), (60, 64), (60, 100000), (100, 8), (100, 100000)):
    measure(qps, slots)
```

```
--- capacity is 50 req/s; watch what the client's slot count does to the report ---
offered=  25/s slots=  8  achieved= 24.5/s  naive p50=  28.3ms p99=   188.9ms   honest p50=   28.3ms p99=    189.6ms   lie factor=  1.0x
offered=  45/s slots=  8  achieved= 44.4/s  naive p50=  90.7ms p99=   287.5ms   honest p50=   98.6ms p99=    474.2ms   lie factor=  1.6x
offered=  60/s slots=  8  achieved= 51.1/s  naive p50= 150.5ms p99=   305.9ms   honest p50= 8202.2ms p99=  19102.3ms   lie factor= 62.4x
offered=  60/s slots= 64  achieved= 51.1/s  naive p50=1230.4ms p99=  1611.7ms   honest p50= 8202.2ms p99=  19102.3ms   lie factor= 11.9x
offered=  60/s slots=100000  achieved= 51.1/s  naive p50=8202.2ms p99= 19102.3ms   honest p50= 8202.2ms p99=  19102.3ms   lie factor=  1.0x
offered= 100/s slots=  8  achieved= 49.8/s  naive p50= 152.4ms p99=   327.5ms   honest p50=61403.6ms p99= 121244.5ms   lie factor=370.2x
offered= 100/s slots=100000  achieved= 49.8/s  naive p50=61403.6ms p99=121244.5ms   honest p50=61403.6ms p99= 121244.5ms   lie factor=  1.0x
```

Read those rows carefully, because this table is the whole lesson:

- **Under load (ρ = 0.5), everything agrees.** Coordinated omission is invisible when the system is healthy — which is exactly why it survives code review.
- **At 60 QPS against a 50 QPS server, an 8-slot client reports p99 = 306 ms. The truth is 19.1 s.** The naive number isn't merely optimistic; it is *bounded* by the client's slot count and can never show overload. It reports the server's **service time**, and calls it latency.
- **More client slots make the lie smaller but don't remove it** (64 slots → still 11.9× off). "Use more threads" is not a fix; only unbounded concurrency (a true open loop) or intended-time stamping is.
- **At 100 QPS the report is off by 370×.** Note the `achieved` column: 49.8/s. The client sent 50 req/s no matter what you asked for, because that's all the server would let it send. **A closed-loop test at any overload just measures your server's capacity and lies about the latency.**
- Both honest columns are identical regardless of client slots, which is the tell: honest measurement is a property of the *system*, not of your test rig.

> The rule: **generate load open-loop and timestamp from intended arrival.** If you use a closed-loop tool (Locust's default user model, naive `for` loops, most `ab`/`wrk` setups), you must at minimum report the concurrency, never call the result "latency under load," and never use it to find a knee.

---

## How many samples does a p99 need?

A percentile from too few samples is noise with a decimal point. Same underlying distribution (exponential, mean 20 ms, true p99 = 91.9 ms), 60 independent trials at each sample count:

```python
rnd = random.Random(1)
pop = [rnd.expovariate(1 / S) * 1e3 for _ in range(500_000)]
true_p99 = statistics.quantiles(pop, n=100)[98]
print(f"true p99 = {true_p99:.1f}ms")
for n in (50, 100, 500, 1_000, 5_000, 20_000):
    ests = []
    for trial in range(60):
        r = random.Random(1000 + trial)
        ests.append(statistics.quantiles([r.expovariate(1 / S) * 1e3 for _ in range(n)], n=100)[98])
    lo, hi = min(ests), max(ests)
    print(f"n={n:>6}  p99 estimate range over 60 trials: {lo:6.1f} - {hi:6.1f}ms  "
          f"(spread {100*(hi-lo)/true_p99:5.1f}% of true)")
```

```
true p99 = 91.9ms
n=    50  p99 estimate range over 60 trials:   49.4 -  223.0ms  (spread 188.9% of true)
n=   100  p99 estimate range over 60 trials:   67.3 -  178.5ms  (spread 121.0% of true)
n=   500  p99 estimate range over 60 trials:   70.0 -  113.5ms  (spread  47.4% of true)
n=  1000  p99 estimate range over 60 trials:   77.0 -  108.1ms  (spread  33.8% of true)
n=  5000  p99 estimate range over 60 trials:   84.0 -   97.0ms  (spread  14.2% of true)
n= 20000  p99 estimate range over 60 trials:   89.3 -   95.4ms  (spread   6.6% of true)
```

**At n = 50, the p99 estimate ranges from 49 ms to 223 ms — a 4.5× spread on identical systems.** Any A/B comparison at that sample size is a coin flip you will interpret as a result. Practical floors:

| You want to quote | Minimum samples | Why |
|---|---|---|
| p50 | ~100 | median is cheap; ±10% |
| p90 | ~500 | 50 samples in the tail |
| **p99** | **≥1,000, prefer 5,000** | only `n/100` samples define it |
| p999 | ≥50,000 | usually not worth it; report max instead |

Two corollaries you must internalize:

1. **Never average percentiles.** The mean of two shards' p99s is not the p99 of anything. Aggregate **raw samples** (or use a mergeable sketch: HDRHistogram, t-digest, DDSketch — which is what Prometheus histograms and Datadog do under the hood, and why `histogram_quantile` over a `_bucket` metric is legitimate while averaging a `p99` gauge is not).
2. **Quoting a p99 from a 60-second run at 2 QPS (n = 120) is malpractice.** Either run longer or don't quote it. This is the single most common flaw in blog-post benchmarks.

---

## The harness

Save this as `labs/phase3/bench.py`. Stdlib only (works against FastAPI/uvicorn, vLLM's OpenAI server, or your own socket server — anything speaking HTTP/1.1 with SSE), open-loop, streaming-aware, and it computes every metric this phase cares about. You will reuse it in lessons 8 and 9 and in every project from here on.

```python
#!/usr/bin/env python3
"""Open-loop, streaming-aware load generator. Stdlib only.

Usage:  python bench.py --url http://127.0.0.1:8000/generate --qps 4 --duration 30
Reports TTFT / ITL / e2e percentiles, throughput, goodput, and a drift check.
"""
import argparse, asyncio, json, random, statistics, time
from urllib.parse import urlsplit


async def one_request(url, body, out):
    """Send one POST, timestamp every streamed token. Latency starts at INTENDED time."""
    u = urlsplit(url)
    t_intended = out["t_intended"]
    payload = json.dumps(body).encode()
    head = (f"POST {u.path or '/'} HTTP/1.1\r\nHost: {u.netloc}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
            f"Accept: text/event-stream\r\nConnection: close\r\n\r\n").encode()
    reader, writer = await asyncio.open_connection(u.hostname, u.port or 80)
    try:
        writer.write(head + payload)
        await writer.drain()
        chunked, status = False, 0
        line = await reader.readline()
        status = int(line.split()[1]) if len(line.split()) > 1 else 0
        while True:                                            # headers
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            if line.lower().startswith(b"transfer-encoding") and b"chunked" in line.lower():
                chunked = True
        stamps, buf = [], b""

        def feed(data):                                        # SSE: one token per "data:" line
            nonlocal buf
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if raw.strip().startswith(b"data:"):
                    stamps.append(time.perf_counter())

        if chunked:
            while True:
                size = await reader.readline()
                if not size:
                    break
                n = int(size.strip().split(b";")[0] or b"0", 16)
                if n == 0:
                    await reader.readline()
                    break
                feed((await reader.readexactly(n + 2))[:-2])
        else:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                feed(data)
    finally:
        writer.close()
    if status != 200:
        out.update(status=status)                              # 429s are DATA, not errors
        return
    if not stamps:
        out.update(status=-1)
        return
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    out.update(status=200, tokens=len(stamps),
               ttft=stamps[0] - t_intended, e2e=stamps[-1] - t_intended,
               itl_p99=(statistics.quantiles(itls, n=100)[98] if len(itls) > 1 else 0.0),
               tpot=((stamps[-1] - stamps[0]) / len(itls)) if itls else 0.0)


async def run(url, qps, duration, warmup, prompt_tokens, max_tokens, slo):
    rnd = random.Random(0)
    results, tasks, t0 = [], [], time.perf_counter()
    while time.perf_counter() - t0 < duration:
        await asyncio.sleep(rnd.expovariate(qps))              # OPEN loop: never waits for a reply
        rec = {"t_intended": time.perf_counter(), "status": None}
        results.append(rec)
        body = {"prompt": "word " * prompt_tokens, "max_tokens": max_tokens}
        tasks.append(asyncio.ensure_future(one_request(url, body, rec)))
    await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if r["status"] == 200 and r["t_intended"] - t0 >= warmup]
    shed = [r for r in results if r["status"] not in (200, None)]
    if len(ok) < 20:
        print(f"only {len(ok)} usable samples ({len(shed)} shed/failed) — raise duration")
        return
    P = lambda k, p: statistics.quantiles([r[k] for r in ok], n=100)[p - 1]
    span = max(r["e2e"] + r["t_intended"] for r in ok) - min(r["t_intended"] for r in ok)
    first, last = ok[: len(ok) // 10], ok[-len(ok) // 10:]
    drift = statistics.median([r["e2e"] for r in last]) / statistics.median([r["e2e"] for r in first])
    print(f"qps={qps:<5} n={len(ok):<5} shed={len(shed):<4} "
          f"ttft p50={P('ttft',50)*1e3:7.0f} p90={P('ttft',90)*1e3:7.0f} p99={P('ttft',99)*1e3:8.0f}ms  "
          f"tpot p50={P('tpot',50)*1e3:6.1f}ms  itl p99={P('itl_p99',99)*1e3:7.1f}ms  "
          f"e2e p99={P('e2e',99):6.2f}s  out={sum(r['tokens'] for r in ok)/span:7.1f} tok/s  "
          f"goodput={sum(1 for r in ok if r['ttft'] <= slo)/span:5.2f}/s  drift={drift:4.1f}x"
          + ("  <-- UNSTABLE" if drift > 1.5 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/generate")
    ap.add_argument("--qps", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--slo", type=float, default=0.5)
    a = ap.parse_args()
    for q in a.qps:
        asyncio.get_event_loop().run_until_complete(
            run(a.url, q, a.duration, a.warmup, a.prompt_tokens, a.max_tokens, a.slo))
```

Seven design decisions in there, each fixing one of the four lies:

| Line | Decision | Fixes |
|---|---|---|
| `await asyncio.sleep(rnd.expovariate(qps))` then `ensure_future` | **open loop**: arrivals happen on schedule, never gated on replies | lie 1 |
| `ttft = stamps[0] - t_intended` | latency measured from **intended** arrival | lie 1 |
| `r["t_intended"] - t0 >= warmup` | warmup window discarded | lie 2 |
| `drift = median(last decile) / median(first decile)` | **stability check** before any percentile is believed | lie 3 |
| `if len(ok) < 20: ...raise duration` | refuses to print statistics it can't support | lie 4 |
| `status != 200` counted as `shed`, not dropped silently | 429s are the load-shedding *result*, not an error to hide | honesty |
| per-token `stamps`, so TTFT / TPOT / **ITL p99** are separate | the two bottlenecks never get averaged together | [lesson 1](01-what-a-serving-system-is.md) |

### Prove it works on something you fully understand

Before pointing it at a GPU, point it at a fake server whose cost model you *chose*. Save as `labs/phase3/stub_server.py`:

```python
#!/usr/bin/env python3
"""Stdlib SSE stub 'model server': one serial GPU slot, chunked streaming."""
import asyncio, json

A, B, C = 0.010, 0.0005, 0.00005      # step base, per-seq, per prefill token
SLOT = None

async def handle(reader, writer):
    await reader.readline()
    length = 0
    while True:
        h = await reader.readline()
        if h in (b"\r\n", b"\n", b""):
            break
        if h.lower().startswith(b"content-length"):
            length = int(h.split(b":")[1])
    body = json.loads(await reader.readexactly(length)) if length else {}
    n = int(body.get("max_tokens", 32))
    prompt_tokens = len(body.get("prompt", "").split())
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                 b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
    async with SLOT:                                   # serial: batch size 1, on purpose
        await asyncio.sleep(C * prompt_tokens)         # prefill
        for i in range(n):
            await asyncio.sleep(A + B)                 # one decode step
            payload = f"data: {json.dumps({'token': f't{i}'})}\n\n".encode()
            writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            await writer.drain()
    writer.write(b"0\r\n\r\n")
    await writer.drain()
    writer.close()

async def main():
    global SLOT
    SLOT = asyncio.Semaphore(1)
    server = await asyncio.start_server(handle, "127.0.0.1", 8000)
    print("stub server on :8000", flush=True)
    async with server:
        await server.serve_forever()

asyncio.get_event_loop().run_until_complete(main())
```

**Predict before you run.** Service time for a 64-token prompt and 16 output tokens is `0.00005×64 + 16×0.0105 ≈ 0.171 s`, so capacity `μ ≈ 5.9 req/s`, and TPOT should be pinned at 10.5 ms regardless of load. Then:

```
$ python stub_server.py &
$ python bench.py --qps 2 4 5 6 7 --duration 40 --warmup 5 --max-tokens 16 --slo 0.5

qps=2.0   n=63    shed=0    ttft p50=     16 p90=    149 p99=     205ms  tpot p50=  11.5ms  itl p99=   12.2ms  e2e p99=  0.38s  out=   29.1 tok/s  goodput= 1.82/s  drift= 1.0x
qps=4.0   n=127   shed=0    ttft p50=     85 p90=    393 p99=    1096ms  tpot p50=  11.5ms  itl p99=   12.0ms  e2e p99=  1.27s  out=   57.7 tok/s  goodput= 3.38/s  drift= 3.9x  <-- UNSTABLE
qps=5.0   n=172   shed=0    ttft p50=    291 p90=   1061 p99=    1625ms  tpot p50=  11.5ms  itl p99=   11.9ms  e2e p99=  1.80s  out=   74.7 tok/s  goodput= 3.01/s  drift= 6.1x  <-- UNSTABLE
qps=6.0   n=202   shed=0    ttft p50=   1299 p90=   4316 p99=    5193ms  tpot p50=  11.5ms  itl p99=   12.1ms  e2e p99=  5.37s  out=   82.7 tok/s  goodput= 1.07/s  drift=11.4x  <-- UNSTABLE
qps=7.0   n=234   shed=0    ttft p50=   4287 p90=   8845 p99=    9665ms  tpot p50=  11.5ms  itl p99=   12.4ms  e2e p99=  9.84s  out=   84.1 tok/s  goodput= 0.13/s  drift=10.9x  <-- UNSTABLE
```

Five things this run teaches, and every one of them generalizes to a real engine:

1. **TPOT is flat at 11.5 ms across a 3.5× change in load; TTFT moves 270×** (16 ms → 4.3 s). Exactly [lesson 1](01-what-a-serving-system-is.md)'s claim, now measured. If you had reported one blended "latency" number you'd have destroyed the only signal in the table.
2. **Throughput keeps rising past the knee and tells you nothing.** 57.7 → 84.1 tok/s while p99 TTFT went from 1.1 s to 9.7 s. **Throughput is monotone in offered load right up to collapse, which is why it is a terrible headline metric.**
3. **Goodput has an interior maximum: 3.38 req/s at 4 QPS, then 3.01, 1.07, 0.13.** There it is — the curve lesson 5 predicted and lesson 6 exploited. The best operating point is *below* the point of maximum throughput, and only goodput shows it.
4. **The drift flag is doing its job, including a false positive.** At 4 QPS (ρ ≈ 0.68) the system is genuinely stable, but the first/last decile is only 12 samples, so `drift = 3.9×` is mostly noise — the same small-n problem as the percentile table above. At 6-7 QPS the drift is real: `ρ > 1`, the queue never drains, and those p99 values would keep climbing with a longer run. **Diagnosis: raise `--duration` until the flag settles; if it doesn't, the system is actually unstable and it has no p99 to report.**
5. **`shed=0` everywhere**, because this stub has no admission control — so it converts overload into 9-second TTFTs instead of fast rejections. Add lesson 6's gates and this column becomes the interesting one.

---

## The rest of the checklist

**Warm up properly.** Discard the first 30-60 s or first N requests. On a GPU the first request pays CUDA context init, allocator growth, cuDNN/cuBLAS autotune, `torch.compile` compilation, and CUDA-graph capture ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)) — often 10-100× the steady-state cost. And beware the *opposite* error: with prefix caching on, repeating the same prompt makes everything after the first request fake-fast. **Randomize prompts unless you are explicitly measuring cache hits.**

**State the workload shape, always.** Prompt and output length distributions change every conclusion in this phase, because they set `C_s²` (lesson 5) and the prefill/decode mix (lesson 6). Minimum disclosure: model, dtype, hardware, prompt-length distribution, output-length distribution, `max_tokens`, streaming on/off, and whether outputs were length-capped. The standard realistic choice is **ShareGPT** (~200 in / ~250 out, heavy-tailed); a fixed 128/128 synthetic workload is fine for A/B but will overstate throughput and understate tails.

**Isolate the resource under test.** Run the generator on another machine, or at minimum another process — an asyncio client at high QPS *will* become your bottleneck and silently cap the "server" number. Sanity checks: does the client's own CPU sit below ~70%? Does `L = λW` hold (lesson 5)? Does doubling client processes change the result? If yes to the last, you measured your client.

**Repeat and report variance.** Three runs minimum; cloud GPUs have noisy neighbours. Report median-of-runs plus spread, and change **one variable at a time**.

**Count everything, including failures.** Rejections (429), timeouts, cancellations, and truncated streams are results. A config that "wins" by shedding 40% of traffic must show that 40%.

### Tools worth knowing (and their loop model)

| Tool | Model | Use it for |
|---|---|---|
| `vllm bench serve` / `benchmarks/benchmark_serving.py` | open loop, Poisson (`--request-rate`), ShareGPT-aware | the reference LLM harness; read it before writing your own |
| [`ray-project/llmperf`](https://github.com/ray-project/llmperf) | open loop | comparing hosted API providers |
| `vegeta`, `k6`, `oha`, `wrk2` | open loop / constant-rate (wrk2 explicitly corrects coordinated omission) | generic HTTP; `wrk2` exists *because* `wrk` had this bug |
| `locust`, `ab`, `wrk` | **closed loop by default** | fine for capacity probing; do **not** quote latency under overload |
| HDRHistogram / t-digest | mergeable percentile sketches | aggregating percentiles across shards correctly |

Your own `bench.py` is not redundant with these — it's how you *understand* them, and it's a portfolio artifact. But quote vLLM's harness when you compare against published numbers, so the methodology matches.

---

## The reporting template

Every benchmark writeup in this repo — starting with Project 01 in the next lesson — uses this shape (see [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md)):

| Config | Offered QPS | n | p50 TTFT | p99 TTFT | p50 TPOT | p99 ITL | Output tok/s | Goodput (TTFT ≤ 500 ms) | Shed % | Drift |
|---|---|---|---|---|---|---|---|---|---|---|
| naive (serial) | 4 | 1,200 | | | | | | | | |
| dynamic batching, T = 10 ms | 4 | 1,200 | | | | | | | | |
| continuous batching | 4 | 1,200 | | | | | | | | |

Plus **two plots**: p50/p99 TTFT vs offered QPS (find the knee) and **goodput vs offered QPS** (find the operating point). And one sentence of the form:

> *"3,100 output tok/s at p99 TTFT ≤ 500 ms, 7B FP16 on one A100-80GB, ShareGPT prompts, open-loop Poisson arrivals, 5 min per point after 60 s warmup, median of 3 runs."*

That sentence is a serving result. Anything shorter is a claim.

---

## Key takeaways

- **Closed-loop load generation cannot measure overload.** The server throttles your client, so the report is bounded by service time. Measured lie factor in this lesson: **62× at 1.2× load, 370× at 2× load.**
- **Fix: open loop (fire on schedule, unbounded in-flight) + timestamp from intended arrival.** More client threads is not a fix.
- **p99 needs ≥1,000 samples** (at n = 50 the estimate spread was 4.5×). **Never average percentiles** — aggregate raw samples or use HDRHistogram/t-digest/DDSketch.
- **Check stability before quoting any percentile:** compare the first and last decile of the run. A drifting run has a slope, not a p99.
- **Warm up** (CUDA context, autotune, compile, graph capture) and **randomize prompts** unless you're deliberately measuring prefix-cache hits.
- **Report TTFT, TPOT and ITL separately**, with the workload shape, hardware, and offered load stated. Count 429s, timeouts and cancellations as data.
- **Throughput is monotone up to collapse; goodput has a maximum.** Measured here: throughput 57.7 → 84.1 tok/s while goodput fell 3.38 → 0.13 req/s. Optimize the operating point, not the headline.
- Run the generator off the server's box, verify with `L = λW`, repeat 3×, change one variable at a time.

**Next:** [Build: naive vs dynamic-batched FastAPI server →](08-build-naive-vs-batched-server.md) — the first real server, the first real numbers, and the first artifact you'd put in front of an interviewer.
