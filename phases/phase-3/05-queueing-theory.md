# 5 — Queueing Theory for Inference (Why p99 Explodes)

> **You'll be able to say:** "`L = λW`, and latency scales as `1/(1−ρ)`. At 90% utilization, mean wait is already 10× service time and p99 is ~4.6× that again; at 95% it's 20×. Service-time variance multiplies it further, which is why LLM tails are so bad. So throughput and p99 are optimized at different operating points, and the batch size that maximizes tokens/sec is never the one that protects your SLO."

Everything so far has been mechanism: how to form a batch, how to schedule a step. This lesson is the *math of load*, and it's the part interviewers use to separate ML engineers from systems engineers. It's also short — two formulas do 90% of the work, and you can derive the rest at a whiteboard.

The reason it matters: a serving system near saturation does not degrade gracefully, it degrades **hyperbolically**. The difference between 70% and 95% utilization is not 25% more risk; it's a 6× worse mean wait and a queue that never drains after any hiccup. If you don't have this curve in your head, you will size capacity by "GPU util looks fine" and get paged.

---

## Little's Law: `L = λW`

For any stable system, over any long enough window:

```
   L  =  λ  ×  W
   │      │      └── W = average time a request spends in the system (seconds)
   │      └───────── λ = average arrival rate (requests/second)
   └──────────────── L = average number of requests IN the system (concurrency)
```

No assumptions. Not about arrival distribution, service distribution, scheduling order, or number of servers — it's essentially conservation of flow. That generality is why it's the most useful equation in this entire track. The intuition: each request contributes `W` seconds of "occupancy," and you get `λ` of them per second, so at any instant `λW` are resident.

Three uses, all of them things you'll actually do:

**1. Size the concurrency you must support.** Chat service, λ = 20 req/s, average e2e latency 4 s (typical for a 200-token answer):

```
  L = 20 × 4 = 80 sequences in flight, on average
```

Now check that against [lesson 4](04-continuous-batching.md)'s KV-cache arithmetic: a 7B model on 80 GB holds ~62 concurrent 2k-token sequences. **You cannot serve this traffic on one GPU** — not because of FLOPs, but because 80 > 62. That's a two-line capacity plan, and it's the correct answer to "how many GPUs do we need?"

**2. Derive the latency you can't measure directly.** Your metrics show `num_requests_running = 45` and 15 req/s arriving. Then `W = L/λ = 3 s`, whether or not you instrumented e2e latency. Both numbers are gauges every serving framework already exports (vLLM: `vllm:num_requests_running`, `vllm:request_success_total`).

**3. Catch dishonest benchmarks.** If someone reports 500 req/s at 50 ms mean latency, then `L = 500 × 0.05 = 25` requests must have been in flight. If their load generator had 4 workers, the numbers are impossible and something is wrong (usually: they measured a warm cache, or the client was the bottleneck). Little's Law is a free consistency check on every result you'll ever read.

**Corollary you'll use constantly:** the same law applied to the queue alone gives `L_queue = λ × W_queue`. So if your SLO says "no request waits more than 500 ms" and λ = 20 req/s, your queue must never exceed `20 × 0.5 = 10` requests. **That's how you pick a queue bound** — not by guessing a "big enough" number.

---

## Utilization: the `1/(1−ρ)` cliff

Define utilization `ρ = λ/μ = λ·S`, where `S` is mean service time and `μ = 1/S` is capacity. For the simplest queue (Poisson arrivals, exponential service, one server, FCFS — "M/M/1"):

```
  W    = S / (1 − ρ)              total time in system
  W_q  = ρ·S / (1 − ρ)            time waiting in queue
  L    = ρ / (1 − ρ)              requests in system
```

The `1/(1−ρ)` factor is the whole story. With `S = 100 ms`:

| ρ (utilization) | L (in flight) | W (mean) | p99 of W | vs. service time |
|---|---|---|---|---|
| 50% | 1.0 | 200 ms | 0.92 s | 2× |
| 70% | 2.3 | 333 ms | 1.53 s | 3.3× |
| **80%** | 4.0 | 500 ms | 2.30 s | 5× |
| **90%** | 9.0 | 1.00 s | 4.61 s | 10× |
| 95% | 19.0 | 2.00 s | 9.21 s | 20× |
| 99% | 99.0 | 10.0 s | 46.1 s | 100× |
| 100% | ∞ | ∞ | ∞ | — |

The p99 column comes from a second fact worth memorizing: **in M/M/1 the sojourn time is exponentially distributed**, so

```
  W_p = W · ln(1/(1−p))      →      p99 = W · ln(100) ≈ 4.6 × mean
```

A mean latency of 1 s implies a p99 near 4.6 s *by the mathematics of queueing alone* — no bug, no slow GPU, no bad request. This is the deepest reason "average latency" is a lie ([lesson 1](01-what-a-serving-system-is.md)): the mean and the tail are locked together by a factor of ~5, and it's the tail your users experience.

Read the table's shape, not its numbers: **from 50% to 90% utilization you gained 1.8× throughput and paid 5× latency.** From 90% to 99%, you gain 10% throughput and pay 10× latency. Which is why production systems for latency-sensitive traffic run at **60-75% utilization** and why "our GPUs are only 70% utilized" is often the correct engineering answer rather than waste. If someone insists on 95%, they are buying 25% more throughput with a 4× worse tail and zero headroom to absorb a traffic spike, a node failure, or a GC pause.

### Variance makes it worse: Kingman's formula

Real service times aren't exponential, and LLM service times are *wild* — 20 tokens vs 2,000 tokens is a 100× spread. The general single-server approximation (Kingman, for a G/G/1 queue):

```
  W_q  ≈  ( ρ / (1 − ρ) ) · ( (C_a² + C_s²) / 2 ) · S
                             └── squared coefficients of variation
                                 (arrival and service), = 1 each for M/M/1
```

Wait is the product of a **utilization term** and a **variability term**. Take eight requests with output lengths `[20, 20, 20, 50, 50, 100, 200, 2000]`: mean 307, standard deviation 642, so `C_s ≈ 2.1` and `C_s² ≈ 4.4`. The variability term becomes `(1 + 4.4)/2 = 2.7`:

**Same hardware, same utilization, 2.7× the queue wait — purely because output lengths are heavy-tailed.**

That single insight explains an enormous amount of real behavior:

- Why LLM serving has famously bad tails even at modest load.
- Why `max_tokens` caps are a *latency* control, not just a cost control: they truncate `C_s²`.
- Why separating traffic classes works so well (short/interactive vs long/batch on different pools): splitting a heavy-tailed workload into two low-variance workloads shrinks the variability term for both. This is the strongest argument for the routing policies in [lesson 6](06-scheduling-policies-and-admission-control.md).
- Why continuous batching helps the tail even at fixed throughput: it decouples each sequence's completion from its neighbours', so a long generation no longer injects its variance into everyone else's wait.

---

## The answer to the phase self-check: batch size vs p99

*"Using Little's Law, why does increasing max batch size increase p99 even as throughput keeps rising?"* Here is the full argument — practice saying it out loud.

```
  step time      S(B) = a + b·B                (a = read the weights, b = marginal per-sequence)
  throughput     X(B) = B / (a + b·B)          rising in B, saturating at 1/b
  per-token lat  ITL  = S(B) = a + b·B         rising in B, without limit
  in flight      L    ≤ B                      the cap IS the concurrency limit
  by Little      W    = L / λ                  at fixed λ, more concurrency = more time per request
```

Three effects, all pushing the same way:

1. **Direct:** every sequence's per-token latency *is* the step time, and the step time grows linearly in B. Batch 32 → 64 makes every user's ITL worse, permanently.
2. **Via Little's Law:** raising the cap admits more concurrent work at the same arrival rate. `L` rises, so `W = L/λ` rises. The extra requests aren't served faster; they're served *simultaneously and slower*.
3. **Diminishing returns:** `X(B) → 1/b`. Past the knee (`B ≈ a/b`, where the fixed cost is amortized), throughput gains are tiny while the latency cost stays linear. With `a = 10 ms, b = 0.5 ms`: B=20 gives 1,000 req/s at 20 ms/step; B=200 gives 1,818 req/s (+82%) at 110 ms/step (+450%).

And the tail specifically is worse than the mean, because p99 is owned by requests that arrived when the batch was *fullest* — exactly the moments the big cap allows. So:

> **`max_num_seqs` / `max_batch_size` is not a performance knob. It is the throughput-versus-tail-latency dial, and it must be set from your SLO, not from what fits in memory.**

The right procedure: pick your SLO (say p99 TTFT ≤ 500 ms, p99 ITL ≤ 50 ms), sweep the batch cap, and report **goodput** — requests/sec meeting the SLO — at each. Goodput has an interior maximum; throughput doesn't. That sweep is a chart, and it's a genuinely impressive artifact to have in a portfolio.

---

## Head-of-line blocking, FCFS, and why deep queues are a trap

Two more failure patterns that fall straight out of the math.

**Head-of-line blocking.** In FCFS with a heavy-tailed service distribution, a short request stuck behind a long one waits for work it has nothing to do with. From [lesson 2](02-static-batching.md), a static batch is the extreme form. Even with continuous batching, an admission decision that spends the whole token budget on one 8k-token prefill blocks everyone else's first token. Mitigations are scheduling policy — shortest-job-first-ish ordering, priority classes, preemption, fair-share — and each one trades fairness for tail latency in a way you must choose deliberately ([lesson 6](06-scheduling-policies-and-admission-control.md)).

**Deep queues don't add capacity; they add latency.** This is the most common instinctive mistake. A queue absorbs *bursts*; it cannot fix `λ > μ`. If arrivals exceed capacity, the queue grows without bound and every queued request eventually exceeds a timeout — so you spend 100% of your GPU producing answers that nobody is still waiting for. That's the classic **congestive collapse**: goodput falls to zero while utilization reads 100%.

The correct posture under overload:

- **Bound the queue** at `λ_target × W_slo` (Little's Law, above), not at "a big number."
- **Shed load** (HTTP 429/503) the instant the bound is exceeded. Rejecting 10% of traffic in 1 ms is strictly better than serving 100% of it 30 s late.
- **Deadline-aware dropping**: attach `arrival_time + slo` to each request and discard it before scheduling if it's already doomed. Cheap, and it directly protects goodput.
- **Honor client cancellation** — disconnected clients are pure waste ([lesson 3](03-dynamic-batching.md)).
- **Never retry blindly.** Retries multiply λ exactly when λ is the problem; use a retry *budget* and exponential backoff with jitter.

---

## Two servers, one queue

Scaling out interacts with all of the above, and one result is worth carrying into [Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale): **a shared queue beats independent queues.** Same total capacity, ρ = 0.9, mean service S:

| Arrangement | Mean queue wait |
|---|---|
| 2 replicas, requests randomly assigned to each replica's own queue | `ρS/(1−ρ)` = **9.0 S** |
| 2 replicas pulling from one shared queue (M/M/2) | **4.3 S** |

Half the wait, zero extra hardware — because with separate queues, one replica can idle while a request waits at the other. That's why a load balancer that dispatches *only when a worker is free* (least-outstanding-requests / "power of two choices") beats round-robin or random, and why in-engine queueing beats per-replica queueing. It also explains the tension you'll meet in Phase 6: prefix-cache-aware **sticky** routing deliberately gives up some of this pooling benefit to win KV-cache hits. Real systems measure which wins.

---

## Try it (laptop, 25 lines)

Verify the `1/(1−ρ)` curve yourself, then go find it in your own server's metrics.

```python
import random, statistics

def mm1(rho, S=0.1, n=200_000, cs2=1.0):
    """Single server, FCFS, Poisson arrivals. cs2 = squared CV of service time."""
    lam = rho / S
    p = 0.1                                    # two-point service law: mean S, variance cs2·S²
    d = S * (cs2 / (p * (1 - p))) ** 0.5
    low, high = S - p * d, S + (1 - p) * d
    t = free_at = 0.0
    sojourn = []
    for _ in range(n):
        t += random.expovariate(lam)                                        # arrival
        s = random.expovariate(1 / S) if cs2 == 1.0 else \
            (high if random.random() < p else low)                          # service
        free_at = max(t, free_at) + s                                       # FCFS, one server
        sojourn.append(free_at - t)
    W = statistics.mean(sojourn)
    p99 = statistics.quantiles(sojourn, n=100)[98]
    print(f"rho={rho:.2f} cs2={cs2:>4}  W={W*1e3:8.1f}ms  (M/M/1 theory {S/(1-rho)*1e3:8.1f}ms)  "
          f"p99={p99*1e3:8.1f}ms  p99/W={p99/W:4.1f}  L=lam*W={lam*W:6.2f}")

random.seed(0)
for rho in (0.5, 0.7, 0.8, 0.9, 0.95):
    mm1(rho)
mm1(0.8, cs2=4.0)          # same utilization, heavy-tailed service times
```

**Predict before running:** `W` should track `S/(1−ρ)`, `p99/W` should sit near `ln(100) ≈ 4.6`, `L` should match `ρ/(1−ρ)`, and the last line — same ρ = 0.8, higher service variance — should be much worse than the ρ = 0.8 row above it. A real run:

```
rho=0.50 cs2= 1.0  W=   199.9ms  (M/M/1 theory    200.0ms)  p99=   911.7ms  p99/W= 4.6  L=lam*W=  1.00
rho=0.70 cs2= 1.0  W=   331.7ms  (M/M/1 theory    333.3ms)  p99=  1531.2ms  p99/W= 4.6  L=lam*W=  2.32
rho=0.80 cs2= 1.0  W=   504.6ms  (M/M/1 theory    500.0ms)  p99=  2246.1ms  p99/W= 4.5  L=lam*W=  4.04
rho=0.90 cs2= 1.0  W=   962.7ms  (M/M/1 theory   1000.0ms)  p99=  4161.7ms  p99/W= 4.3  L=lam*W=  8.66
rho=0.95 cs2= 1.0  W=  2034.4ms  (M/M/1 theory   2000.0ms)  p99=  9387.8ms  p99/W= 4.6  L=lam*W= 19.33
rho=0.80 cs2= 4.0  W=  1071.7ms  (M/M/1 theory    500.0ms)  p99=  5185.9ms  p99/W= 4.8  L=lam*W=  8.57
```

Every prediction holds, including Kingman: at ρ = 0.8 with `C_s² = 4`, the formula says `W_q ≈ (0.8/0.2)·((1+4)/2)·100 ms = 1,000 ms`, so `W ≈ 1,100 ms` — measured 1,072 ms. **Service-time variance alone doubled the latency at unchanged utilization**, and the ρ=0.8 heavy-tailed system is now worse than the ρ=0.9 exponential one. That's the LLM tail problem in one line of output.

Then extend it: add a *bounded* queue with load shedding and plot goodput vs offered load. It rises, peaks, and — without shedding — collapses. Seeing that collapse in your own plot is the most useful ten minutes in this lesson.

Finally, do it for real: on your Project-01 server, log `L` (in-flight count) every 100 ms and λ, and check `W ≈ L/λ` against your measured latencies. When those three numbers agree, your instrumentation is trustworthy — and when they don't, you've found a bug in your metrics before it lies to you in production.

---

## Key takeaways

- **`L = λW`** holds for any stable system, no assumptions. Use it to size concurrency, to derive latency from gauges you already export, and to sanity-check every benchmark you read.
- **Queue bound = `λ_target × W_slo`.** That's the principled way to size a queue; anything larger only converts capacity shortfall into timeouts.
- **`W = S/(1−ρ)`**: 80% → 5× service time, 90% → 10×, 95% → 20×. Latency is hyperbolic in utilization, so run latency-sensitive inference at **60-75%**.
- **`p99 ≈ 4.6 × mean`** for M/M/1 — the tail is structural, not a bug. Means are worse than useless.
- **Kingman:** `W_q ≈ (ρ/(1−ρ)) · ((C_a²+C_s²)/2) · S`. LLM output lengths are heavy-tailed (`C_s² ≈ 4`+), multiplying wait ~2.7× at the same load. Hence `max_tokens` caps and separate traffic classes.
- **Batch cap is the throughput/tail dial:** step time grows linearly in B, `L ≤ B` so `W = L/λ` grows, and throughput saturates at `1/b`. Set it from the SLO and pick it by sweeping **goodput**, which has a real maximum.
- **Deep queues are latency, not capacity.** Under overload: bound, shed, drop doomed requests, honor cancellation, and never retry without a budget — or you get congestive collapse at 100% "utilization" and zero goodput.
- **One shared queue beats N private queues** (4.3 S vs 9.0 S at ρ=0.9) — the argument for least-outstanding-requests balancing, revisited under prefix-aware routing in Phase 6.

**Next:** lesson 6 — scheduling policies & admission control: FCFS vs priority vs fair-share, preemption, chunked prefill budgets, and how to encode an SLO in a scheduler.
