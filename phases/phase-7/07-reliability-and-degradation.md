# 7 — Reliability and Graceful Degradation

> **You'll be able to say:** "In inference, a retry is not cheap — it costs a full prefill and it arrives exactly when the system is already saturated, so naive retry policies are the standard way a degradation becomes an outage. The correct toolkit is: bounded queues plus admission control, timeouts aligned across every hop, retries only with budgets and jitter and never on a partially streamed response, hedging only for TTFT and only below 5% of traffic, circuit breakers per replica, and an explicit degradation ladder — smaller model, shorter max_tokens, no speculative decoding, cached answer, refuse — so that overload sheds quality before it sheds availability."

[Lesson 6](06-dashboards-and-diagnosis.md) diagnosed problems. This lesson is about designing the system so problems stay small. The core asymmetry to internalize: **inference requests are expensive, long-lived, stateful, and non-idempotent-ish** (same input, different sampled output). Every reliability pattern you learned from stateless web services needs re-deriving under those four properties.

---

## The failure catalogue

| Failure | Blast radius | Detection | Correct response |
|---|---|---|---|
| **CUDA OOM mid-batch** | the whole engine step, sometimes the process | engine error + `class=capacity` | preempt/evict one sequence and retry the step; if the process dies, restart and *lower the memory config* — this is a config bug, not bad luck |
| **KV exhaustion / preemption storm** | ITL tail for every in-flight request | KV occupancy, preemption counter | admission control: stop admitting, drain; then reduce `max_num_seqs` or add KV capacity |
| **One slow/degraded GPU** | the replica (and for TP, all its ranks) | per-rank step-time spread, clocks | drain replica; cordon node if XID/ECC ([lesson 3](03-gpu-and-host-telemetry.md)) |
| **GPU falls off the bus (XID 79) / uncontained ECC (95)** | node | DCGM, kernel log | cordon + reboot; never restart the pod in place |
| **NCCL hang in a TP/PP group** | replica hangs *without* dying — the worst kind | no tokens, collective wait rising, `/health` may still pass | watchdog on step progress → kill the replica; see [Phase 6 lesson 9](../phase-6/09-multi-node-operations.md) |
| **Node eviction / spot preemption** | replica, with 30 s-2 min notice | cloud metadata / k8s event | deregister from router, finish in-flight (bounded), exit |
| **Model load failure / bad weights** | new replicas only | readiness never passes | fail deploy fast; registry checksum ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)) |
| **Tokenizer/template mismatch after upgrade** | *all* output quality, silently | quality metrics, finish-reason shift | roll back; this is the top cause of "the model got dumber" ([lesson 8](08-canary-and-shadow-traffic.md)) |
| **Slow client / backpressure** | one stream, plus a held connection and its KV blocks | stream span duration, in-flight gauge | bounded send buffer, then abort the request; KV is too expensive to hold for a stalled reader |
| **Router loses a replica's health state** | traffic to a dead replica | 5xx spike from one pod | passive health checks + circuit breaker |
| **Overload from one tenant** | everyone, unless isolated | per-tenant-class metrics | quotas + fair queueing ([Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md)) |
| **Cascading retry storm** | the entire service, self-inflicted | offered load > client-visible load | retry budgets, circuit breakers, load shedding |

Two of these are inference-specific enough to deserve their own emphasis: the **silent quality failure** (200 OK, useless content) and the **hang without death** (a replica that is up, healthy, and producing nothing). Standard SRE playbooks cover neither well.

---

## Timeouts: the alignment problem

```
  client SDK timeout        60 s   ───────────────────────────────────┐
   > gateway/ingress        60 s   ─────────────────────────────────┐ │
     > router timeout       55 s   ───────────────────────────────┐ │ │
       > engine request TTL 50 s   ─────────────────────────────┐ │ │ │
         > generation cap: max_tokens × expected TPOT ≈ 45 s  ──┘ │ │ │
                                                                  │ │ │
  RULE: inner timeout < outer timeout, at every hop, with a margin.
  VIOLATION SYMPTOM: the gateway returns 504 while the GPU keeps generating for
  another 10 s — you pay for tokens nobody receives, and the retry (also 504)
  doubles that waste. Under load this is a positive feedback loop.
```

Inference-specific rules:

1. **Timeout on TTFT separately from E2E.** "No first token within 10 s" is a real failure; "still streaming at 40 s" may be a perfectly healthy 2000-token response. A single E2E timeout conflates them and either kills legitimate long generations or lets dead requests linger.
2. **Cap `max_tokens` server-side.** An unbounded `max_tokens` is an unbounded timeout, an unbounded KV allocation, and an unbounded bill. Set a hard server-side ceiling and clamp client requests to it.
3. **Propagate cancellation all the way to the scheduler.** A disconnected client must free KV blocks *now*. If your framework doesn't propagate `CancelledError` into the engine's scheduler, you leak the most contended resource you own. Verify it: disconnect 50 streams mid-generation and confirm KV occupancy drops promptly.
4. **Idle-stream timeout.** If no token has been emitted for N seconds while the request is nominally running, the request is stuck — abort it rather than holding blocks indefinitely.

---

## Retries: why the web playbook is wrong here

```
  A STATELESS WEB RETRY                    AN INFERENCE RETRY
  ────────────────────────────────────────────────────────────────────────────
  costs ~1 ms of CPU                       costs a full PREFILL: 10 ms - 10 s of GPU
  idempotent (GET)                         non-deterministic output (sampling)
  arrives when the service is fine          arrives when the service is overloaded
  3 retries = 4x load on a cheap path      3 retries = 4x load on your scarcest resource
  partial response impossible               partial response ALREADY STREAMED to the user
```

The rules that follow:

| Rule | Detail |
|---|---|
| **Never retry after first token** | The user already has partial output. Retrying produces a different continuation; concatenating is corruption, restarting is a visible glitch. Retry is a *pre-first-token* option only. |
| **Retry budget, not retry count** | Cap retries at a small fraction of total requests (e.g. ≤10% — the Envoy/gRPC "retry budget" model). A per-request count of 3 becomes 4× global load during a brownout; a budget cannot. |
| **Retry only on retryable classes** | Connection failures, 503 from a *specific* replica, and "replica draining." Never on 400/422; never on 429 without honoring `Retry-After`; never on capacity errors that will recur. |
| **Exponential backoff with full jitter** | Synchronized retries after a blip are a self-DDoS; jitter is what de-synchronizes them. |
| **Retry to a *different* replica** | A retry to the same overloaded replica is just more load. Prefer a different one, accepting the prefix-cache miss ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)) — availability beats locality here. |
| **Make retries visible** | A separate counter for retried requests. If offered load exceeds client-visible load by a large factor, you have a storm. This is the metric that explains an otherwise inexplicable overload. |

### Hedging (speculative requests)

Send the same request to a second replica after a short delay and take whichever answers first. Effective against tail latency caused by *one unlucky replica*; dangerous as a load amplifier.

```
  hedge after p95 TTFT (e.g. 400 ms) → extra load ≈ 5% of requests
  hedge after p50 TTFT               → extra load ≈ 50% of requests, and during an
                                       incident 50% becomes the incident
  RULES: cap hedged fraction (≤5%), disable hedging when a shedding/brownout signal is
         active, cancel the loser immediately (and verify the cancel frees KV),
         hedge on TTFT only — never re-issue a stream that already produced tokens.
```

Hedging is worth it when your TTFT tail is dominated by per-replica variance (cold caches, stragglers) and not by global saturation. Check which regime you're in with the per-pod TTFT heatmap from [lesson 6](06-dashboards-and-diagnosis.md) before enabling it.

---

## Load shedding and the bounded queue

The single most important reliability property of an inference service: **a bounded queue with an explicit rejection policy.** Unbounded queues convert overload into universal timeout — everybody waits, everybody times out, all the GPU work is wasted.

```
  offered load λ > capacity μ
  ├─ UNBOUNDED QUEUE : latency → ∞, every request eventually times out,
  │                    100% of GPU work discarded. THE WORST OUTCOME.
  ├─ BOUNDED QUEUE + REJECT : (μ/λ) of requests succeed at good latency,
  │                    the rest get a fast, cheap 429 with Retry-After.
  └─ BOUNDED QUEUE + DEGRADE : even more succeed, at reduced quality (below)
```

Practical mechanics, building on [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md):

- **Admit on predicted completion, not on arrival.** With the queue's current depth and measured service rate, estimate TTFT; if it exceeds the SLO, reject now rather than accept-and-miss. Rejecting early is cheaper for both sides.
- **Drop-on-arrival vs drop-oldest.** Under sustained overload, **dropping the oldest queued request** is often better: old requests are likely abandoned already, and their deadline has passed. (Known as "LIFO under overload"; counterintuitive and effective.)
- **Per-tenant fair queueing.** One tenant's burst must not consume the whole queue. Weighted fair queueing by `tenant_class` keeps shedding proportional.
- **Shed cheaply.** The rejection path must not require GPU work, tokenization of a 100k-token prompt, or an auth round trip. Measure the cost of a 429 — if it's not microseconds, overload will still take you down.
- **Return `Retry-After` and honor it client-side**, or your shedding just becomes a retry storm at a different layer.

---

## The degradation ladder

Availability and quality are separate axes. Under pressure, spend quality to keep availability — but *deliberately, in a defined order, with a metric for each rung*.

```
  RUNG 0  normal: full model, full max_tokens, speculative decoding on, full context
  RUNG 1  disable optional extras: turn off speculative decoding (it costs GPU when
          acceptance is low), disable logprobs, disable re-ranking passes
  RUNG 2  reduce generosity: clamp max_tokens (e.g. 2048 → 512), lower default
          temperature-sampling overhead, cap context by truncating middle content
  RUNG 3  cheaper model: route to the INT4/8B variant instead of the FP16/70B
          (a "model cascade" — Phase 4 lesson 1; quality drops measurably, latency and
          capacity improve a lot)
  RUNG 4  serve from cache: exact-match or semantic cache for repeated prompts;
          for RAG, return retrieved passages without generation (Phase 9 lesson 10)
  RUNG 5  static fallback: a canned response / "try again shortly" with a 503 and
          Retry-After, free tier first, paid tier last
  RUNG 6  shed: reject with 429/503 at the edge
```

Design notes:

- **Each rung needs a metric and a label.** `inference_degradation_rung` as a gauge plus `model_version`/`degraded=true` on request logs, so your SLO math and your postmortem can both see it. Silent degradation is indistinguishable from a bug.
- **Rung transitions should be automatic and hysteretic** (enter on 30 s of pressure, exit after 5 min of calm) to avoid flapping between quality levels — users notice oscillation more than a steady lower quality.
- **Decide the order with the product owner in advance.** "Would you rather be slower, dumber, or unavailable?" is a product question. Getting the answer in writing before the incident is the whole point.
- **Cascades are cheap capacity.** Serving 80% of traffic from a distilled/quantized model with fallback to the big model for hard cases is a production pattern, not just a degradation mode ([Phase 4 lesson 1](../phase-4/01-what-to-optimize.md)).

---

## Health checks, drains, and the hang problem

```
  /health      (liveness)  : process alive, event loop responsive
  /ready       (readiness) : weights loaded, CUDA graphs captured, warmed up,
                             AND NOT DRAINING            ← the important conjunct
  progress watchdog        : tokens_generated counter has advanced in the last N seconds
                             while requests are running  ← catches the NCCL hang
```

The hang is the case standard probes miss: the HTTP server answers `/health` happily while the engine is blocked in a collective that will never return. A **progress watchdog** — "if `num_requests_running > 0` and `generation_tokens_total` hasn't moved in 60 s, fail liveness" — converts an infinite hang into a restart. Pair it with NCCL timeouts (`NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_BLOCKING_WAIT`, a finite `NCCL_TIMEOUT`) so the collective itself gives up ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)).

Drain sequence for any planned removal (scale-in, rollout, spot notice, node maintenance):

```
  1. mark NOT ready  → router stops sending new requests (verify: router's healthy-pool metric)
  2. stop admitting  → engine finishes only in-flight sequences
  3. wait ≤ max generation time (bounded by your max_tokens cap), reporting remaining streams
  4. abort stragglers with a clear error the client can retry elsewhere
  5. exit 0
  requires: terminationGracePeriodSeconds > step 3's bound (Phase 8 lesson 4),
            preStop hook performing steps 1-2, and a router that reacts within seconds
```

Every element of that sequence is load-bearing, and the common bug is a `terminationGracePeriodSeconds` of 30 with `max_tokens` allowing 4-minute generations: every rollout then truncates live streams, and your "quality incident" is actually a lifecycle bug.

---

## Multi-replica and fleet-level patterns

| Pattern | Inference-specific note |
|---|---|
| **Circuit breaker per replica** | Passive health: N consecutive failures or a p99 far above peers → eject for a cooldown (Envoy outlier detection). Essential because a degraded-but-alive GPU is common. |
| **Concurrency limits per replica** | The router must cap in-flight requests per replica to its measured `max_num_seqs`. Without it, sticky routing creates hotspots ([lesson 6](06-dashboards-and-diagnosis.md) walkthrough). |
| **Cell/shard isolation** | Partition the fleet so a bad config or poison request takes down one cell, not everything. Roll out per cell. |
| **Poison-request defense** | A single pathological prompt (pathologically long, adversarial grammar, huge `n`) that reliably OOMs a replica will kill replicas in sequence as it retries. Validate limits at the edge, and quarantine requests that have already killed a replica once. |
| **Multi-region / multi-provider** | GPU capacity is scarce; capacity failures are regional. Have a defined "spillover" path even if it's a more expensive provider. |
| **Queue-based async for batch traffic** | Move non-interactive work to a queue with its own SLO; it becomes elastic buffer capacity instead of competing with interactive requests. |

---

## Do this now (60 minutes)

1. **Audit timeouts end-to-end** for one service: list client, ingress, router, engine, and generation-cap limits with numbers, and confirm the strict inner-< outer ordering. Then test cancellation: open 20 streams, kill the clients mid-generation, and verify (a) KV occupancy drops within seconds and (b) `finish_reason="abort"` increments. A failure here is a KV leak that looks like a capacity problem.
2. **Build the bounded queue + reject path** in your Phase-3 server with a predicted-TTFT admission rule, and load-test at 2× capacity. Produce the comparison table: unbounded queue (success rate, p99, wasted GPU work) vs bounded+reject vs bounded+degrade (clamped `max_tokens`). The middle column is the argument for admission control in one screen.
3. **Implement the progress watchdog and a degradation rung.** Add the "no token progress while requests are running" liveness check and verify it trips under an artificial stall (`time.sleep(120)` inside the engine step, or `SIGSTOP` on the engine process). Then implement rung 2 (clamp `max_tokens` under pressure), with the gauge and request label, and confirm both the SLO query and the logs can see when it was active.

---

**Next:** [Canary, shadow traffic, and quality gates →](08-canary-and-shadow-traffic.md) — how to ship a new model, quantization or engine version without betting the SLO on it, and how to detect the quality regression that no latency metric will ever show you.
