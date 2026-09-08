# 10 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 2's artifact proved you can *measure* a GPU. Phase 3's proves you can *operate a service* — and it is the first thing in this roadmap that looks like a job.

Everything here runs on a laptop (CPU or Apple Silicon) with GPT-2 124M. A Colab T4 makes the numbers bigger, not the lessons different. **Do not wait for hardware.**

---

## Warm-up exercises (run them, write down the numbers)

Each maps to one lesson. **Predict first, then measure, then explain the gap** — that third step is the whole exercise.

1. **Draw the timeline** ([1](01-what-a-serving-system-is.md)): for a 500-token prompt / 200-token output on a 7B model, compute prefill ms, decode ms, TTFT, TPOT, e2e from first principles. Then say which of the five segments a queue of 40 requests lands in, and by how much.
2. **Static-batch waste** ([2](02-static-batching.md)): for outputs `[20, 20, 50, 800]` in one batch of 4, compute the fraction of decode slot-steps spent on already-finished rows. Then measure the same thing in lesson 8's server with `--max-tokens 8 16 32 128`.
3. **The window that does nothing** ([3](03-dynamic-batching.md)): run lesson 3's simulator, then sweep `MAX_WAIT` ∈ {0, 5, 10, 50, 200} ms on the real server. Report mean batch size against `1 + λT` and explain why the timer barely matters under load.
4. **Trace the loop** ([4](04-continuous-batching.md)): run lesson 4's static-vs-continuous simulator. Report p99 TTFT and throughput for both, then change one thing (max seqs, prompt length, output skew) and explain which metric moved and why.
5. **KV arithmetic** ([4](04-continuous-batching.md)): compute bytes/token and max concurrent sequences for Llama-2-7B (MHA) and Llama-3-8B (GQA) at 2k and 8k context on 24 GB, 40 GB and 80 GB. Say at which point compute stops being the constraint.
6. **The `1/(1−ρ)` curve** ([5](05-queueing-theory.md)): run the M/M/1 simulator; confirm `W ≈ S/(1−ρ)`, `p99/W ≈ 4.6`, and that `C_s² = 4` roughly doubles the wait at fixed ρ. Then find the same curve in your own server's data.
7. **Little's Law on your own server** ([5](05-queueing-theory.md)): log in-flight count `L` every 100 ms and arrival rate λ during a bench run; check `W ≈ L/λ` against measured e2e. Report the agreement. Disagreement = an instrumentation bug you just caught for free.
8. **Policies and gates** ([6](06-scheduling-policies-and-admission-control.md)): run the scheduling simulator. Report FCFS vs SJF vs priority vs aging, then the overload triple (no control / queue cap / cap + deadline drop). State the goodput multiplier you got from dropping doomed requests.
9. **Coordinated omission, on purpose** ([7](07-measuring-honestly.md)): run the CO experiment; report the lie factor at ρ = 1.2 and ρ = 2. Then reproduce it against your *real* server by capping client concurrency and comparing send-time vs intended-time latency.
10. **Percentile stability** ([7](07-measuring-honestly.md)): reproduce the sample-size table. Then compute how long your bench run must be, at your QPS, to quote a p99 honestly.
11. **Both knees** ([8](08-build-naive-vs-batched-server.md)): sweep naive and dynamic-batched endpoints; plot p50/p99 TTFT vs QPS and goodput vs QPS. Mark each maximum. State peak throughput and peak goodput and note that they are at different loads.
12. **The batch-cap dial** ([8](08-build-naive-vs-batched-server.md)): sweep `MAX_BATCH` ∈ {1, 2, 4, 8, 16, 32} at your best QPS. Plot throughput and p99 TTFT together; pick the value your SLO implies, not the value that maximizes tokens/sec.
13. **Correctness before speed** ([8](08-build-naive-vs-batched-server.md), [9](09-build-continuous-batching-engine.md)): run the batch-invariance test on both servers. Then break `position_ids` deliberately, record the corrupted output, and restore it.
14. **Uniform vs ragged** ([9](09-build-continuous-batching-engine.md)): benchmark the dynamic-batched server and the continuous-batching engine on **both** `--max-tokens 32` and `--max-tokens 8 16 32 128`. Report all four curves. Explain, with slot-step arithmetic, why the ranking flips.
15. **Force a preemption** ([9](09-build-continuous-batching-engine.md)): shrink `KV_BUDGET` until `preemptions` climbs. Report preemptions/sec, the victims' ITL spike, and the wasted prefill tokens; then argue recompute vs swap for your prompt length.
16. **Read a real scheduler** ([4](04-continuous-batching.md), [6](06-scheduling-policies-and-admission-control.md)): in vLLM, trace one request from `add_request` through `schedule()`. Answer in writing: *what happens to a request that doesn't fit the current token budget?* and *where exactly do finished sequences release their blocks?* Cite file and function names.

Keep everything in `labs/phase3/`. The numbers feed the writeup below.

---

## Conceptual self-check (no notes)

If any answer is fuzzy, re-read the linked lesson before Phase 4.

1. Name the five segments of a request's life and the metric that owns each. Which metric includes queue wait, and why does that matter more than anything else on this list? ([1](01-what-a-serving-system-is.md))
2. Why must TTFT and TPOT be reported separately? Give a change that improves one and worsens the other. ([1](01-what-a-serving-system-is.md))
3. Static batching wastes decode steps two different ways. Name both. ([2](02-static-batching.md))
4. `B ≈ min(N, 1 + λT)` — when is a batching window worth having, and when is it theatre? ([3](03-dynamic-batching.md))
5. What exactly changes between dynamic and continuous batching, expressed as *when* the scheduling decision is made? ([4](04-continuous-batching.md))
6. What three capabilities must the model layer provide for continuous batching to be possible? ([4](04-continuous-batching.md))
7. A 4,000-token prefill arrives while 60 sequences are decoding. What happens, and what are three fixes in increasing order of ambition? ([4](04-continuous-batching.md))
8. State Little's Law and give three distinct uses: sizing, deriving, and catching a liar. ([5](05-queueing-theory.md))
9. Why is p99 ≈ 4.6× the mean in M/M/1, and what does that imply about ever reporting an average latency? ([5](05-queueing-theory.md))
10. Using Little's Law *and* `S(B) = a + b·B`, explain why raising the batch cap raises p99 even while throughput improves. ([5](05-queueing-theory.md))
11. Why do latency-sensitive services run at 60-75% utilization instead of 95%? ([5](05-queueing-theory.md))
12. Why is FCFS the default in LLM engines when SRPT provably minimizes mean flow time? What does MLFQ do about it? ([6](06-scheduling-policies-and-admission-control.md))
13. Name the three admission gates and how you'd size each. Which one is worth the most and why? ([6](06-scheduling-policies-and-admission-control.md))
14. What is coordinated omission, how big can the error be, and what are the two ways to eliminate it? ([7](07-measuring-honestly.md))
15. How many samples does an honest p99 need, and why is averaging percentiles across shards meaningless? ([7](07-measuring-honestly.md))
16. Why is throughput a bad headline metric and goodput a good one? Sketch both curves against offered load. ([7](07-measuring-honestly.md), [8](08-build-naive-vs-batched-server.md))
17. **The keystone question:** *"Your chat service's p99 TTFT went from 400 ms to 6 s at the same QPS. Walk me through the diagnosis."* Answer as a **procedure**: check whether the run is stable at all (drift), split TTFT into queue wait vs prefill, check in-flight count and mean batch size against Little's Law, check preemption rate, check whether prompt-length distribution or `max_tokens` changed, check whether utilization crossed the `1/(1−ρ)` knee — and name the fix each verdict implies. ([1](01-what-a-serving-system-is.md), [5](05-queueing-theory.md), [6](06-scheduling-policies-and-admission-control.md), [7](07-measuring-honestly.md))

Question 17 is close to verbatim a real inference-engineering interview question. The strong answer is a procedure with evidence at each step, not a list of optimizations.

---

## Exit artifact (this is what "finishing Phase 3" means)

Produce **at least Option A** and commit it. A + B together is the strongest portfolio pair in the whole roadmap so far: A proves you can operate and measure a service, B proves you can build the thing that makes it fast.

### Option A — Project 01: the tiny inference server (required)

`projects/01-tiny-inference-server/README.md`, containing:

- **The two servers** (`server.py` with `/generate_naive` and `/generate_batched`) and the **harness** (`bench.py`), committed and runnable.
- **The results table** in the [benchmarking playbook](../../playbooks/benchmarking.md) format: one row per (endpoint, offered QPS) with n, p50/p90/p99 TTFT, p50 TPOT, p99 ITL, e2e p99, output tok/s, goodput, shed count, and the drift flag.
- **Two plots**: p50/p99 TTFT vs offered QPS (both endpoints, knees marked) and **goodput vs offered QPS** (maxima marked).
- **The `MAX_BATCH` sweep** with throughput and p99 TTFT on one chart, and one sentence naming the value your SLO picks.
- **A correctness section**: the batch-invariance test passing, plus the corrupted output from the deliberately broken `position_ids`.
- **A methodology section** stating hardware, model, dtype, prompt/output distributions, warmup, settle, run length, repeats — and one honest paragraph on what your setup could not isolate.
- **A 300-word conclusion** answering self-check 17 with *your* numbers: your peak throughput, your peak goodput, the load you'd actually run at, and why they're three different numbers.

### Option B — Project 02: the continuous-batching engine (strongly recommended)

`projects/02-continuous-batching-engine/README.md` with the [lesson 9](09-build-continuous-batching-engine.md) deliverables:

- The engine (`engine.py`): per-sequence KV, chunked prefill under a token budget, immediate eviction, recompute preemption, `/metrics`.
- **The four-curve comparison**: dynamic vs continuous × uniform vs ragged workload. This is the centerpiece — it's the experiment that shows you understand *why* continuous batching exists rather than just that it's faster.
- **The preemption experiment**: preemptions/sec vs `KV_BUDGET`, victims' ITL spike, wasted prefill tokens.
- **A vLLM mapping table**: your components against theirs, plus a paragraph on the specific reasons vLLM beats your engine (varlen kernels, paged KV, CUDA graphs, C++/Rust hot paths).
- **The optimization writeup**: what re-batching KV every step cost you, what caching it saved, and why that experience explains PagedAttention.

### Option C — Deep read (optional, cheap, high signal)

A 600-word piece: *"What a scheduler decides, every 20 milliseconds."* Walk through one iteration of vLLM's scheduler with real function names, the budget it enforces, what it does when memory runs out, and the one config pair (`max_num_seqs`, `max_num_batched_tokens`) you'd tune first for a stated SLO. Cite the lines you read.

---

## How you know you're ready for Phase 4

- [ ] You can draw a request's five segments and name the metric and failure mode of each.
- [ ] You can state Little's Law and use it three ways without notes.
- [ ] You can explain why p99 explodes at high utilization and quote the `1/(1−ρ)` numbers at 80/90/95%.
- [ ] You can explain continuous batching in terms of *when* the scheduling decision happens — and name what breaks (prefill interference) and the three fixes.
- [ ] You can compute KV-cache bytes per token and turn it into a concurrency limit for a given GPU.
- [ ] You have built a server whose numbers you'd defend in a design review: open loop, warmed up, percentiles, stated workload, stated caveats.
- [ ] You have seen your own goodput curve rise, peak, and collapse.
- [ ] You have written a scheduler loop with admission, eviction and preemption — and its output matches single-sequence generation token for token.
- [ ] Your exit artifact is committed.

Then go to **[Phase 4 — Inference Optimization Techniques](../../ROADMAP.md#phase-4--inference-optimization-techniques)**. Phase 3's verdict: batching is the difference between a toy and a service (measured here: ~8-10× goodput), but you hit a wall made of **KV-cache memory** and **per-token cost**. Phase 4 attacks both — quantization, PagedAttention, prefix caching, speculative decoding — and every one of those techniques is evaluated with the harness and the vocabulary you just built.

---

## Where these ideas come back

| Phase 3 idea | Comes back as |
|---|---|
| TTFT / TPOT / ITL / goodput | Every benchmark, dashboard and SLO for the rest of the track |
| Continuous batching loop | vLLM/TGI/SGLang internals; your first OSS contribution (Phases 5, 10) |
| KV-cache as the binding constraint | PagedAttention, KV quantization, prefix caching, offloading (Phase 4) |
| Prefill/decode collision | Chunked prefill tuning, P/D disaggregation (Phases 4, 6) |
| Token budget & batch cap | `max_num_batched_tokens` / `max_num_seqs` tuning in production (Phase 5) |
| Admission control & shedding | Autoscaling, rate limits, multi-tenant fairness, cost control (Phases 6-8) |
| Little's Law & `1/(1−ρ)` | Capacity planning, replica sizing, cost per million tokens (Phases 6-8) |
| Open-loop benchmarking discipline | Every performance claim you will ever make or evaluate |
| Preemption & recompute vs swap | Block managers, KV offloading, long-context serving (Phases 4-6) |

Phase 2 taught you why one request wastes a GPU. Phase 3 taught you how to keep it fed without wrecking anyone's tail. Phase 4 makes each token itself cheaper.
