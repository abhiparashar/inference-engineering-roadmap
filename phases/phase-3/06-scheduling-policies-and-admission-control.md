# 6 — Scheduling Policies & Admission Control

> **You'll be able to say:** "The scheduler answers four questions every step: *who goes next*, *how much work fits in this step*, *who do I refuse*, and *who do I kick out*. FCFS is the default because output lengths are unknown, but shortest-job-first cuts mean wait ~2.5× on a convoy and priority classes isolate interactive traffic — both by starving somebody, which is why aging and fair-share exist. And under overload no ordering policy helps at all: only admission control does. Bounding the queue plus dropping already-doomed requests took goodput from 0.09 to 2.47 req/s in my own simulator."

Lesson 4 gave you the loop; lesson 5 gave you the math of load. This lesson is the **policy** that sits inside the loop — the part that a config file exposes and that you will actually tune on a real system at 2 a.m.

The framing that makes it click: a scheduler is not "the thing that runs the model." It is a **resource allocator with an SLO**, and every one of its decisions is a choice about *whose* latency gets worse. There is no policy that makes everyone faster. Once you accept that, the whole design space becomes readable.

---

## The four questions

```
                        ┌─────────────── requests arrive ───────────────┐
                        ▼                                               │
   Q3: ADMIT?   ┌──────────────┐   reject → 429 / 503                   │
   (gatekeeper) │ queue bound  │   drop   → deadline already blown       │
                └──────┬───────┘                                        │
                       ▼                                                │
   Q1: ORDER?   ┌──────────────┐   FCFS · SJF/SRPT · priority ·         │
   (who next)   │  wait queue  │   fair-share · EDF · MLFQ              │
                └──────┬───────┘                                        │
                       ▼                                                │
   Q2: HOW MUCH? ┌─────────────┐  token budget: max_num_batched_tokens  │
   (step content)│ this step   │  seq budget:   max_num_seqs            │
                 └─────┬───────┘  prefill chunks vs decode rows         │
                       ▼                                                │
   Q4: EVICT?    ┌─────────────┐  KV pool exhausted → recompute / swap  │
   (preemption)  │  running    │  victim choice: LIFO, lowest priority  │
                 └─────────────┘                                        │
```

Q1 is what textbooks call scheduling. **Q2, Q3 and Q4 are where production incidents live.** Most people tuning a serving engine only know about Q1, and Q1 is the least important of the four.

---

## Q1: ordering policies

| Policy | Rule | Optimizes | Cost | Who does this |
|---|---|---|---|---|
| **FCFS** | arrival order | fairness, predictability, zero starvation | head-of-line blocking; convoy effect | vLLM default, TGI, most engines |
| **SJF / SRPT** | shortest (remaining) job first | provably minimal *mean* flow time | needs to know the length; starves long requests | research; approximated by length prediction |
| **Priority classes** | tier first, arrival within tier | protects one traffic class | lower tiers can starve indefinitely | vLLM `--scheduling-policy priority`, most gateways |
| **Fair-share** | equalize *service received* per tenant | no tenant can hog the GPU | needs per-tenant accounting; not latency-aware | VTC (Virtual Token Counter, OSDI '24) |
| **EDF / deadline** | earliest `arrival + slo` first | goodput against an explicit SLO | thrashes when everything is late | SLO-aware routers, real-time systems |
| **MLFQ** | new jobs start high, demote as they consume | approximates SJF *without knowing lengths* | preemption cost; needs KV swapping | FastServe (skip-join MLFQ) |

Three of these deserve real attention.

**Why FCFS wins by default.** SRPT is optimal for mean flow time — that's a theorem — and yet nobody ships it. The reason is specific to LLMs: **you don't know the output length when you admit the request.** A 20-token answer and a 2,000-token answer look identical at arrival; the only signal is the prompt and `max_tokens` (an upper bound that clients set to 4,096 out of laziness). Everything clever in this row of the table is therefore an *estimation* problem, which is why "predict output length, then schedule" is an active research area rather than a config flag. FCFS also has two properties you should not undervalue: it cannot starve anyone, and it makes latency *predictable*, which is what SLOs are actually written against.

**MLFQ is the trick that dodges the estimation problem.** Give every new request the top priority level; each time it consumes its quantum without finishing, demote it. Short requests finish while still at high priority; long requests sink to the bottom and stop blocking anyone. You get SJF-like behaviour with **no length knowledge at all** — the cost is preemption, which for an LLM means evicting a partially-generated sequence's KV-cache (recompute or swap, [lesson 4](04-continuous-batching.md)). That cost is exactly why this is a paper (FastServe) and not vLLM's default.

**Fair-share is a different axis, not a better policy.** FCFS is fair *per request*; if tenant A sends 1,000 requests and tenant B sends 5, FCFS gives A 200× the GPU. VTC's answer is to count **tokens served per tenant** and always serve the tenant with the smallest counter — the LLM version of weighted fair queueing. Reach for it when you're multi-tenant, not when you're chasing p99.

### Do the arithmetic: the convoy

Five requests arrive together; service times 10, 1, 1, 1, 1 (arbitrary units).

```
  FCFS order  10,1,1,1,1  → completions 10, 11, 12, 13, 14  → mean 12.0
  SJF  order  1,1,1,1,10  → completions  1,  2,  3,  4, 14  → mean  4.8
```

**2.5× better mean latency, same hardware, same work, zero cost** — and the long request finished at 14 either way. That is the entire appeal of SJF, and the entire trap: change the workload so long requests keep arriving and that one job never runs. Which brings us to the rule you should carry:

> Any policy that reorders by size **must** be paired with an anti-starvation mechanism — **aging** (bump priority with waiting time) or a reserved share of each step for the oldest request. Otherwise you have traded a bad p99 for an infinite p99.

---

## Q2: what should this step contain? (the token budget)

This is the question that makes LLM scheduling different from web-request scheduling, and it barely appears in the classic literature.

A decode step for `B` sequences costs `a + b·B`. A prefill of `P` tokens costs roughly `c·P` and is *compute*-bound. So the composition of a step — how many prefill tokens, how many decode rows — sets both throughput and everyone's inter-token latency. Engines express this as **two budgets that every candidate must fit**:

```
  max_num_batched_tokens   ← tokens processed this step, prefill + decode COMBINED (e.g. 2048)
  max_num_seqs             ← rows in the batch, i.e. concurrent sequences (e.g. 256)
```

Each decoding sequence contributes exactly 1 token per step; a prefill contributes its whole prompt (or its chunk). So a budget of 2,048 tokens with 60 decoding sequences leaves ~1,988 tokens of prefill room — which is how **chunked prefill** turns a 4,000-token prompt into three steps instead of one 300 ms stall.

Then you must pick a bias, and it is a genuine dilemma with no free answer:

| Bias | Effect | Hurts |
|---|---|---|
| **Prefill-priority** (fill the budget with waiting prompts) | best TTFT, best raw throughput | ITL of everyone already decoding — the stutter from [lesson 4](04-continuous-batching.md) |
| **Decode-priority** (serve running sequences first, prefill with leftovers) | smooth ITL, protects streaming users | TTFT of queued requests |
| **Chunked/hybrid** (cap prefill tokens per step) | bounded ITL *and* bounded TTFT | slightly lower peak throughput; one more knob |

Chunked prefill is the right default, and the knob you're really setting is *how much ITL jitter you'll tolerate to admit new work*. Numerically: at `a = 10 ms, b = 0.5 ms/seq`, 60 decoding sequences give a 40 ms step. Adding a 2,048-token prefill chunk at ~50 µs/token adds ~100 ms → a 140 ms step, so every streaming user sees one 3.5× ITL spike. Halve the chunk and you halve the spike and double the number of steps the prefill takes. **That's the whole tradeoff, and it's arithmetic, not taste.**

---

## Q3: admission control — the part that actually saves you

Lesson 5 proved that a queue cannot add capacity. Admission control is the mechanism that acts on that fact. Three gates, in order:

**Gate 1 — the memory gate (per candidate).** Before admitting, ask the KV pool whether the request's blocks fit *with a watermark of headroom* (vLLM keeps a few percent free so that running sequences can still append). Answers are three-valued, and the third one matters: `OK` (admit), `LATER` (doesn't fit now, retry next step), `NEVER` (can't fit even on an empty GPU → reject immediately with an error, don't queue it forever).

**Gate 2 — the queue bound (per system).** Size it with Little's Law, not vibes: `queue_cap = λ_target × W_slo`. At 20 req/s with a 500 ms queue-wait SLO, the cap is 10. Past the cap, return **HTTP 429 with `Retry-After`** — fast failure is a feature; it lets the client fail over or degrade while it still has time to.

**Gate 3 — the deadline drop (per request, continuously).** Stamp every request with `deadline = arrival + slo`. Before scheduling, discard anything already past its deadline. It has *no chance* of being useful, and every token you spend on it is stolen from a request that could still succeed. This gate is cheap, it's rarely implemented, and in the experiment below it is worth **19× goodput** on its own.

```python
def admit(req, now, pool, waiting, cap, slo):
    if now - req.arrive > slo:            return "DROP"     # gate 3: already doomed
    if len(waiting) >= cap:               return "SHED"     # gate 2: 429 + Retry-After
    status = pool.can_allocate(req)                          # gate 1: KV memory + watermark
    if status == "NEVER":                 return "REJECT"   # 413-style: too big for this GPU, ever
    if status == "LATER":                 return "WAIT"
    return "ADMIT"
```

Two more things belong in this layer, and both are pure profit:

- **Honor cancellation.** A disconnected client's tokens are 100% waste. Watch the request's disconnect event and free the sequence immediately.
- **Never retry blindly.** Retries multiply λ exactly when λ is the problem. Retry budgets, backoff with jitter, and — better — a client that respects `Retry-After`.

---

## Q4: preemption and starvation

Admission is optimistic (sequences grow one token per step), so the engine *will* hit the KV wall mid-flight. Lesson 4 covered the mechanics — recompute vs swap. The **policy** questions are:

- **Who is the victim?** LIFO (newest running sequence) is the standard choice: it has the least accumulated work to throw away and re-prefilling it is cheapest. Priority schedulers instead evict the lowest-priority sequence, even if it's old — which is what makes "priority" mean something under memory pressure rather than only at admission.
- **How do you stop the victim from being re-evicted forever?** A preempted request goes back to `waiting`; if the policy keeps picking it, you've built a livelock. Aging, or a "never preempt the same request twice in a row" rule, fixes it.
- **How do you know it's happening?** A **preemption counter** in your metrics. Rising preemptions = over-admission, and each one is wasted prefill FLOPs that shows up as an ITL spike for the victim. `vllm:num_preemptions_total` exists for exactly this; alert on its rate, not its value.

---

## Encoding an SLO in a scheduler

Here's what all four questions look like as one concrete policy for a two-class service. This is the artifact to have in your head when someone asks "how would you design the scheduler?":

```
CLASSES
  interactive   SLO: p99 TTFT ≤ 500 ms, p99 ITL ≤ 50 ms      (chat UI)
  batch         SLO: none, best effort                        (offline summarization)

Q1 ORDER      priority: interactive before batch; FCFS within class;
              aging: batch waiting > 5 s is promoted to interactive tier
              (bounds batch starvation; without it, batch never runs under sustained load)

Q2 BUDGET     max_num_seqs = 64 (from the goodput sweep, not from what fits)
              max_num_batched_tokens = 2048, chunked prefill ON
              (ITL budget: 64 rows ≈ 42 ms step; a 2048-token chunk adds ~100 ms
               → cap the chunk at 1024 to keep p99 ITL near the 50 ms target)

Q3 ADMIT      interactive queue cap = λ_target × 0.5 s ; batch queue cap = large
              drop interactive past deadline; never drop batch, just delay it
              429 + Retry-After on interactive shed, so the client can degrade

Q4 PREEMPT    evict batch first, LIFO within class; recompute mode (prompts are short)
              alarm on preemptions/sec > 1
```

Notice that every number traces to either an SLO or a measured cost coefficient. **That's the difference between tuning and guessing**, and it's the answer interviewers are listening for.

---

## Read the real code

- **vLLM** — `vllm/v1/core/sched/scheduler.py` (V1) or `vllm/core/scheduler.py` (V0). Find the policy switch (`fcfs` vs `priority`, exposed as `--scheduling-policy`; requests carry a `priority` field where lower = more urgent), the `SchedulingBudget`, and the long-prefill guard that keeps one huge prompt from eating a whole step. Flag names drift between releases — grep for `scheduling_policy` and `max_num_batched_tokens`.
- **TGI** — `router/src/queue.rs`: `max_concurrent_requests` (gate 2 — this is where your 429 comes from), `max_batch_total_tokens` (gate 1, in tokens), and `waiting_served_ratio` / `max_waiting_tokens`, which are literally the Q2 prefill-vs-decode bias expressed as config.
- **SGLang** — `--schedule-policy` (including longest-prefix-match ordering, which optimizes *cache hit rate* rather than latency — a fifth axis) and `--schedule-conservativeness`, which is the admission optimism dial.
- **Triton Inference Server** — priority levels + queue policies (`timeout_action: REJECT`, `default_timeout_microseconds`, `max_queue_size`) in `model_config.proto`. It's the cleanest declarative statement of gates 2 and 3 in any production system; read it even if you'll never deploy Triton.

---

## Try it (laptop, no GPU)

This extends lesson 4's continuous-batching simulator with real policies and real gates. The metric to watch is **goodput** — requests whose TTFT met the SLO, per second — because that's the only number that can't be gamed by making somebody else very slow.

```python
import random, statistics

A, B_COST, C_PREFILL = 0.010, 0.0005, 0.00005     # step base(s), per-seq(s), per prefill token(s)
MAX_SEQS, TTFT_SLO = 16, 1.0                      # batch cap; goodput = requests meeting this TTFT

def workload(n=900, qps=2.8, seed=0):
    rnd, t, reqs = random.Random(seed), 0.0, []
    for i in range(n):
        t += rnd.expovariate(qps)
        reqs.append(dict(id=i, arrive=t, prompt=rnd.choice([64, 256, 1024]),
                         out=rnd.choice([16, 32, 64, 128, 512, 1024]),
                         cls="int" if rnd.random() < 0.7 else "batch"))
    return reqs

def key_for(policy, now):
    if policy == "fcfs":  return lambda r: r["arrive"]
    if policy == "sjf":   return lambda r: r["out"]                      # oracle: knows output length
    if policy == "prio":  return lambda r: (r["cls"] != "int", r["arrive"])
    if policy == "aging": return lambda r: (r["cls"] != "int" and now - r["arrive"] < 2.0, r["arrive"])
    raise ValueError(policy)

def simulate(reqs, policy="fcfs", queue_cap=None, drop_doomed=False, label=None):
    pend = sorted((dict(r) for r in reqs), key=lambda r: r["arrive"])
    i, now, waiting, running, done, shed = 0, 0.0, [], [], [], 0
    while i < len(pend) or waiting or running:
        while i < len(pend) and pend[i]["arrive"] <= now:                            # arrivals
            r = pend[i]; i += 1
            if queue_cap is not None and len(waiting) >= queue_cap: shed += 1        # 429 at the door
            else: waiting.append(r)
        if not waiting and not running:
            now = pend[i]["arrive"]; continue
        if drop_doomed:                                                              # deadline-aware drop
            keep = [r for r in waiting if now - r["arrive"] <= TTFT_SLO]
            shed += len(waiting) - len(keep); waiting = keep
        waiting.sort(key=key_for(policy, now))
        while waiting and len(running) < MAX_SEQS:                                   # admit
            r = waiting.pop(0)
            now += C_PREFILL * r["prompt"]                                           # prefill blocks the step
            r["ttft"], r["produced"] = now - r["arrive"], 1
            running.append(r)
        now += A + B_COST * len(running)                                             # one decode step
        for r in running: r["produced"] += 1
        for r in [r for r in running if r["produced"] >= r["out"]]:
            r["e2e"] = now - r["arrive"]; r["norm"] = r["e2e"] / r["out"]
            done.append(r); running.remove(r)
    Q = lambda k, p, rs: statistics.quantiles([r[k] for r in rs], n=100)[p - 1]
    span = max(r["arrive"] + r["e2e"] for r in done)
    good = sum(1 for r in done if r["ttft"] <= TTFT_SLO) / span
    ints = [r for r in done if r["cls"] == "int"]; bats = [r for r in done if r["cls"] == "batch"]
    print(f"{label or policy:<22} ttft p50={Q('ttft',50,done)*1e3:7.0f}ms p99={Q('ttft',99,done):6.2f}s  "
          f"norm-lat p99={Q('norm',99,done)*1e3:5.1f}ms/tok  goodput={good:5.2f} req/s  shed={shed:>3}  "
          f"[int ttft p99={Q('ttft',99,ints):6.2f}s  batch ttft p99={Q('ttft',99,bats):6.2f}s]")

w = workload()
print("--- 900 requests, 2.8 req/s offered (rho ~= 0.93), MAX_SEQS=16, TTFT SLO 1.0s ---")
for p in ("fcfs", "sjf", "prio", "aging"):
    simulate(w, policy=p)
print()
print("--- overload: 4.5 req/s (~1.5x capacity), FCFS, admission control on/off ---")
hot = workload(n=900, qps=4.5, seed=1)
simulate(hot, label="no admission control")
simulate(hot, queue_cap=20, label="queue cap 20")
simulate(hot, queue_cap=20, drop_doomed=True, label="cap 20 + drop doomed")
```

A real run:

```
--- 900 requests, 2.8 req/s offered (rho ~= 0.93), MAX_SEQS=16, TTFT SLO 1.0s ---
fcfs                   ttft p50=   1048ms p99=  9.83s  norm-lat p99=496.1ms/tok  goodput= 1.29 req/s  shed=  0  [int ttft p99= 10.11s  batch ttft p99=  9.57s]
sjf                    ttft p50=    109ms p99= 12.28s  norm-lat p99=121.9ms/tok  goodput= 1.95 req/s  shed=  0  [int ttft p99= 13.33s  batch ttft p99= 13.08s]
prio                   ttft p50=    148ms p99= 21.15s  norm-lat p99=607.4ms/tok  goodput= 1.75 req/s  shed=  0  [int ttft p99=  4.38s  batch ttft p99= 23.83s]
aging                  ttft p50=    970ms p99=  9.96s  norm-lat p99=486.9ms/tok  goodput= 1.30 req/s  shed=  0  [int ttft p99= 10.02s  batch ttft p99=  9.63s]

--- overload: 4.5 req/s (~1.5x capacity), FCFS, admission control on/off ---
no admission control   ttft p50=  65240ms p99=123.31s  norm-lat p99=7188.7ms/tok  goodput= 0.09 req/s  shed=  0  [int ttft p99=123.82s  batch ttft p99=122.68s]
queue cap 20           ttft p50=   6002ms p99=  9.59s  norm-lat p99=589.0ms/tok  goodput= 0.13 req/s  shed=325  [int ttft p99=  9.61s  batch ttft p99=  9.66s]
cap 20 + drop doomed   ttft p50=    656ms p99=  1.03s  norm-lat p99= 79.6ms/tok  goodput= 2.47 req/s  shed=380  [int ttft p99=  1.03s  batch ttft p99=  1.04s]
```

Five results worth more than the code that produced them:

1. **SJF's p50 TTFT is 10× better than FCFS (109 ms vs 1,048 ms) and its goodput is 51% higher — while its p99 TTFT is *worse* (12.3 s vs 9.8 s).** That's starvation, visible in one table: the long requests it deprioritized are now the tail. Any single-number comparison of these two policies is a lie.
2. **Priority does exactly what it promises and charges exactly what you'd expect:** interactive p99 TTFT 4.4 s (vs 9.8 s under FCFS) bought with batch p99 TTFT of 23.8 s (vs 9.6 s). No throughput was created; latency was *moved* from one class to another. This is the honest description of every priority scheme.
3. **The aging run is nearly identical to FCFS** — because at ρ ≈ 0.93 almost every batch request waits longer than the 2 s aging threshold, so it ages into the top tier and the policy degenerates. Aging thresholds must be set relative to your *measured* queue-wait distribution or they're a no-op. Change it to 8 s and watch the run move back toward `prio`.
4. **Under overload, no ordering policy helps.** The first overload row is congestive collapse: 65-second median TTFT, goodput 0.09 req/s while the "GPU" is 100% busy. Reordering a doomed queue produces doomed requests in a different order.
5. **Bounding the queue is not enough — dropping doomed requests is the win.** Cap alone: 0.13 req/s. Cap + deadline drop: **2.47 req/s, a 19× increase**, with p99 TTFT at 1.03 s instead of 123 s. The mechanism is simple: a 20-deep queue at 6 s of wait is 20 slots of guaranteed waste. Both configurations shed ~350 requests; only one of them shed the *right* ones.

Then extend it: sweep `MAX_SEQS` and plot goodput (it has an interior maximum — this is the sweep lesson 5 promised); implement MLFQ with a quantum of 32 tokens and a re-prefill penalty, and see whether it beats FCFS without oracle knowledge; add a second server with one shared queue and confirm lesson 5's `4.3 S` vs `9.0 S` prediction.

---

## Key takeaways

- A scheduler answers four questions: **order** (who next), **budget** (what's in this step), **admission** (who is refused), **preemption** (who is evicted). Only the first is classic scheduling; the other three cause the outages.
- **FCFS is the default because output length is unknown at admission** — it can't starve anyone and it makes latency predictable. SRPT is optimal for mean flow time but needs a length oracle; **MLFQ (FastServe)** approximates it without one, paying in preemptions.
- **Reordering by size must be paired with aging or a reserved share**, or you convert a bad p99 into an infinite one. Set the aging threshold from your measured wait distribution, or it does nothing.
- **Priority moves latency between classes; it never creates capacity.** Measured here: interactive p99 TTFT 9.8 s → 4.4 s, batch 9.6 s → 23.8 s.
- **Fair-share (VTC) is a different axis** — per-tenant token accounting for multi-tenancy, not a p99 tool.
- **The token budget (`max_num_batched_tokens`) is the LLM-specific knob:** prefill and decode tokens compete in one step. Chunked prefill bounds the ITL spike; the chunk size *is* your ITL-vs-TTFT dial, and it's computable from `a`, `b`, and `c`.
- **Three admission gates:** memory (`OK`/`LATER`/`NEVER` + watermark), queue bound (`λ_target × W_slo`, then 429 + `Retry-After`), and **deadline drop**. The third is the cheapest big win in this lesson: 19× goodput under 1.5× overload.
- **Preemption policy** = victim choice (LIFO, or lowest priority) + a livelock guard + a **preemptions/sec alarm**, which is your over-admission detector.
- Every number in a good scheduler config traces back to an SLO or a measured cost coefficient. If you can't say where a knob's value came from, you're guessing.

**Next:** [Measuring it honestly: load generation & percentiles →](07-measuring-honestly.md) — open vs closed loop, coordinated omission, warmup, and how to build a harness whose numbers you'd defend in a design review.
