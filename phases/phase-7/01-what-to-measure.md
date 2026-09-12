# 1 — What to Measure: SLIs for an Inference Service

> **You'll be able to say:** "The four golden signals map onto inference as TTFT/TPOT/E2E latency (latency), requests-per-second *and* tokens-per-second (traffic), a taxonomy of five distinct error classes (errors), and KV-cache occupancy plus queue depth — **not** GPU utilization (saturation). I can name, for each signal, the unit, the aggregation, the place in the stack it must be measured, and the decision it drives. And I know why averages are useless here: a streaming response has two latencies, and the one users feel is a *per-token* distribution, not a request total."

This is the first lesson of [Phase 7](README.md), and it is the one that makes the rest possible. Everything downstream — dashboards, alerts, SLOs, cost accounting, canary gates — is a function over the metrics you chose to emit. Choose badly and no amount of Grafana skill recovers it.

Phases 3-6 taught you to *measure during a benchmark*. This phase is about measuring *forever*, in production, while real users are attached, at a cost in cardinality and CPU you can afford.

---

## The one-slide model

```
  CLIENT                ROUTER / GATEWAY            ENGINE (vLLM/TGI/...)        GPU
    │                        │                            │                      │
    ├── request ────────────►│                            │                      │
    │                        ├── queue (admission) ──────►│                      │
    │                        │                            ├── waiting queue      │
    │                        │                            ├── scheduled ────────►│ prefill
    │◄────── first token ────┤◄─────── first token ───────┤◄─────────────────────┤
    │                        │                            │                      │
    │◄── token ── token ── token ── ... ─────────────────  ├── decode steps ─────►│ decode
    │◄────── last token ─────┤                            │                      │
    │                        │                            │                      │
    └── E2E latency ─────────┘                            └── engine-side only   └── DCGM

  TTFT   = request admitted → first token emitted      (prefill + all queueing)
  TPOT   = (E2E − TTFT) / (output_tokens − 1)          inter-token pace, the "reading speed"
  ITL    = the *distribution* of individual gaps        (TPOT is its mean; p99 ITL catches stalls)
  E2E    = request in → last token out                 = TTFT + TPOT × (N_out − 1)
```

Three consequences that most teams discover the hard way:

1. **A single "latency" metric is a lie for streaming.** A 20-second response is excellent if TTFT was 200 ms and tokens streamed steadily; it is awful if the user stared at nothing for 19 seconds. Measure TTFT and TPOT separately, always. ([Phase 3 lesson 1](../phase-3/01-what-a-serving-system-is.md) defined these; here they become production SLIs.)
2. **E2E latency is contaminated by output length**, which the *user* controls. A p99 E2E of 40 s may mean "someone asked for 2000 tokens," not "we are slow." This is why E2E is a bad SLO target on its own and why you normalize: TTFT, TPOT, and optionally `E2E / output_tokens`.
3. **Where you measure changes the number.** Client-side includes network and TLS; router-side includes queueing you control; engine-side excludes both. You need at least two vantage points, or you cannot tell "we are slow" from "the internet is slow."

---

## Signal 1: Latency

| SLI | Definition | Unit | Aggregation | Drives |
|---|---|---|---|---|
| **TTFT** | admit → first token byte | seconds | p50 / p95 / p99 histogram | the primary user-facing SLO for chat; scale-out trigger |
| **TPOT** (a.k.a. ITL mean) | mean gap between output tokens | seconds/token | p50 / p95 / p99 | "does it feel stuck"; batch-size pressure |
| **ITL p99** | worst single inter-token gap | seconds | p99 / max | detects stalls: preemption, CUDA sync, GC, a long prefill hogging a step |
| **E2E** | admit → last token | seconds | p50 / p95 / p99 | capacity planning, timeout setting |
| **queue wait** | admit → first schedule | seconds | p95 | isolates *our* queueing from model compute |
| **prefill time** | first schedule → first token | seconds | p95, bucketed by prompt length | isolates compute; the denominator of prefill throughput |

**Normalize TTFT by prompt length and TPOT by batch size when you diagnose**, not when you alert. A p95 TTFT regression is meaningless until you know whether prompt lengths grew; `TTFT vs prompt_tokens` as a heatmap answers it in one look ([lesson 6](06-dashboards-and-diagnosis.md)).

### Percentiles, and why p99 of *tokens* matters

A request emitting 500 tokens contains 499 inter-token gaps. If 0.2% of decode steps stall for 2 seconds (a preemption, [Phase 4 lesson 5](../phase-4/05-paged-attention.md)), then **roughly one gap per request stalls** — a p99-of-steps problem becomes a p63-of-requests problem. Tail latency compounds over token count. This is the single most counterintuitive measurement fact in LLM serving: *per-step tails are amplified by output length*, so step-level SLIs must be much tighter than request-level intuition suggests.

---

## Signal 2: Traffic

You need **two** traffic metrics because the unit of work is not the request.

| SLI | Why it exists |
|---|---|
| requests/sec (by route, by model, by tenant) | the classic load number; drives concurrency and connection limits |
| **prompt tokens/sec** | prefill load — the compute-bound half ([Phase 1 lesson 6](../phase-1/06-prefill-vs-decode.md)) |
| **output tokens/sec** | decode load — the memory-bandwidth-bound half, and the denominator of cost per 1M tokens ([lesson 9](09-cost-per-million-tokens.md)) |
| concurrent requests (running + waiting) | the actual saturation driver; what an autoscaler should see ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md)) |
| output-length distribution | a histogram, not a mean; the shape that makes your batch memory footprint unpredictable |

Two requests/sec of 8k-token prompts is ~30× the work of two requests/sec of 200-token prompts. **RPS without token rates cannot explain a single capacity graph.** Emit both or you will mis-size the fleet.

---

## Signal 3: Errors — five classes, not one

Inference failures are not "5xx." They are five distinct classes with different owners, different alerts, and different user impact:

| Class | Examples | Detect via | Correct response |
|---|---|---|---|
| **Rejections (intentional)** | admission control 429, queue full, rate limit | counter by reason, labelled `shed` | expected under overload; alert on *rate*, not existence ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)) |
| **Capacity failures** | CUDA OOM, KV exhaustion, preemption-to-failure | engine logs + counter | fix config (`gpu_memory_utilization`, `max_num_seqs`), not retries |
| **Timeouts / aborts** | client disconnect, gateway timeout mid-stream | counter + *partial* token accounting | must still bill/attribute the tokens already generated |
| **Hardware / infra** | XID errors, ECC DBE, NVLink down, node eviction | DCGM + kubelet events ([lesson 3](03-gpu-and-host-telemetry.md)) | drain the replica; page if replicas < N |
| **Quality failures (silent)** | empty completion, repetition loop, truncated JSON, refusal spike, finish_reason distribution shift | output-side metrics + canary diffing ([lesson 8](08-canary-and-shadow-traffic.md)) | **the class nobody instruments, and the one users actually notice** |

The last class is where inference differs most from web serving. A 200 OK carrying garbage is a full outage from the user's perspective and a green dashboard from yours. Minimum viable instrumentation: **`finish_reason` counter** (`stop` / `length` / `abort` / `error`), **output token count = 0 counter**, and an n-gram repetition check on a sample of responses.

---

## Signal 4: Saturation — the part everyone gets wrong

```
  ✗ GPU utilization (DCGM_FI_DEV_GPU_UTIL)
       = fraction of time ≥1 kernel was resident.
       Batch 1 decode: ~95%.  Batch 256 decode: ~99%.  100× the work, same number.
       → USEFUL FOR: "is the process alive."  USELESS FOR: saturation, scaling, cost.

  ✓ KV-cache occupancy (vllm:gpu_cache_usage_perc)
       = fraction of KV blocks allocated. THE memory-side saturation metric.
       >0.9 sustained ⇒ preemption imminent ⇒ ITL tail explodes.

  ✓ queue depth (vllm:num_requests_waiting)
       = requests admitted but not running. Non-zero and rising ⇒ ρ→1 ⇒ latency diverges
         (Phase 3 lesson 5). The cleanest scale-out trigger that exists.

  ✓ batch size (vllm:num_requests_running / max_num_seqs)
       = how much of your configured concurrency you're using.

  ~ DCGM_FI_PROF_SM_ACTIVE / PIPE_TENSOR_ACTIVE / DRAM_ACTIVE
       = real work fractions. Good for "am I memory-bound or compute-bound"
         (Phase 2 lesson 5, the roofline, as a live metric). Sampling cost is non-zero.
```

The general rule from the USE method (*Systems Performance*, Gregg): for each resource, track **U**tilization, **S**aturation, **E**rrors. For a GPU inference replica the honest mapping is:

| Resource | Utilization | Saturation | Errors |
|---|---|---|---|
| GPU compute | `SM_ACTIVE`, achieved TFLOPs | step time vs floor | XID, kernel failures |
| GPU memory (capacity) | `FB_USED` / total | **KV occupancy**, preemption count | CUDA OOM |
| GPU memory (bandwidth) | `DRAM_ACTIVE` | tokens/sec plateau | — |
| Engine scheduler | batch size / `max_num_seqs` | **queue depth**, queue wait p95 | admission rejects |
| Interconnect (TP/PP) | NVLink TX/RX bytes | collective wait time | NCCL timeouts ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)) |
| Host | CPU%, tokenizer thread time | request accept backlog | 5xx from the HTTP layer |

---

## The "measure it here" table

Same metric, different meaning by vantage point. Pick deliberately:

| Vantage | Sees | Misses | Use for |
|---|---|---|---|
| Client / synthetic prober | DNS, TLS, network, CDN, everything | internal attribution | the SLO you promise users; black-box availability |
| Gateway / router | auth, rate limit, routing, cross-replica queueing | client network | per-tenant SLIs, traffic split during canary ([lesson 8](08-canary-and-shadow-traffic.md)) |
| Engine `/metrics` | queueing, batch, KV, preemption, tokens | anything before it | capacity, scaling, engine diagnosis |
| DCGM / node exporter | hardware truth | request semantics | hardware faults, power/thermal, fleet-level cost |

**A working minimum is three**: a synthetic prober (is it up, from outside), the router (per-tenant SLI), and the engine (why). Adding tracing joins them per-request ([lesson 4](04-tracing-and-logging.md)).

---

## Worked example: naming the metric set for a chat product

```
  SLI CONTRACT (what the SLO in lesson 5 will be written against)
  ────────────────────────────────────────────────────────────────
  availability   : router 2xx+streamed-first-token / total non-shed requests
  TTFT           : histogram, labels {model, route}          buckets 0.05..30 s
  TPOT           : histogram, labels {model}                 buckets 0.005..1 s
  E2E            : histogram, labels {model}                 buckets 0.1..300 s
  output tokens  : histogram, labels {model}                 buckets 1..8192
  prompt tokens  : histogram, labels {model}                 buckets 1..131072
  tokens total   : counters {direction=prompt|output, model, tenant_class}
  finish_reason  : counter  {reason=stop|length|abort|error}
  errors         : counter  {class=shed|capacity|timeout|infra|quality, code}
  queue          : gauge num_waiting, gauge num_running, histogram queue_wait
  kv             : gauge cache_usage_perc, counter preemptions_total
  cost inputs    : gauge replicas, gauge gpus_per_replica, $ per GPU-hour (static config)
```

That is ~15 metric families. It fits in a single scrape, answers every question in lessons 5-10, and — critically — contains **no per-request-ID labels**, which is what keeps it affordable ([lesson 2](02-instrumenting-with-prometheus.md)).

### What you deliberately do *not* measure as a metric

- **Per-request-ID or per-user anything.** That is tracing and logging, not metrics ([lesson 4](04-tracing-and-logging.md)).
- **Prompt/response text.** That is a privacy liability and a storage bill; sample it, redact it, and keep it out of the metrics path entirely.
- **Averages of latency.** A mean TTFT hides exactly the behaviour you care about; if you keep one aggregate, keep p95.

---

## The answer key: read a real engine's metrics

vLLM already exposes most of the above, which makes it the fastest way to check your list against production reality. The names you should recognize:

```
vllm:time_to_first_token_seconds        histogram  ← TTFT
vllm:time_per_output_token_seconds      histogram  ← ITL/TPOT
vllm:e2e_request_latency_seconds        histogram  ← E2E
vllm:request_queue_time_seconds         histogram  ← queue wait
vllm:request_prefill_time_seconds       histogram
vllm:request_decode_time_seconds        histogram
vllm:num_requests_running               gauge      ← batch size
vllm:num_requests_waiting               gauge      ← queue depth  (scale trigger)
vllm:gpu_cache_usage_perc               gauge      ← KV occupancy (saturation)
vllm:num_preemptions_total              counter    ← thrashing
vllm:prompt_tokens_total                counter
vllm:generation_tokens_total            counter    ← cost denominator
vllm:request_success_total{finished_reason}  counter
vllm:prefix_cache_queries_total / _hits_total  counter  ← Phase 4/6 hit rate, live
```

Every one of them is a line in the table above. If your own service exposes fewer than these, you are flying with less instrumentation than the open-source engine you're wrapping.

---

## Do this now (45 minutes)

1. **Write your SLI contract** for one service you have (or your Phase-3 server) using the worked-example block as a template: name, type, labels, buckets, vantage point, and the one decision each metric drives. If a metric drives no decision, delete it. Keep this file — lessons 2, 5, 6 and 9 all consume it.
2. **Prove the tail-amplification claim.** From any completion log with per-token timestamps (or generate one: 50 requests × 200 tokens against a local vLLM), compute the ITL distribution, then compute the fraction of *requests* containing at least one gap above p99-of-gaps. Confirm it is far above 1%. That number is why step-level SLIs must be tight.
3. **Classify last month's failures.** Take any incident list, bug tracker, or even your own local crash history and bucket each entry into the five error classes. Whichever class has no metric behind it is your first instrumentation task — and it is usually "quality failures."

---

**Next:** [Instrumenting with Prometheus →](02-instrumenting-with-prometheus.md) — how to emit these SLIs correctly: histogram buckets that survive p99 math, label cardinality that doesn't bankrupt you, and the streaming-aware timing code that gets TTFT right.
