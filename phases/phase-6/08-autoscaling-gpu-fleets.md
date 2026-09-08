# 8 — Autoscaling and Capacity for GPU Fleets

> **You'll be able to say:** "GPU inference breaks the standard autoscaler in three specific ways: the signal is wrong (GPU utilization reads ~100% during decode at any load), the actuator is slow (a replica takes 3-15 minutes to become useful — image, weights, warmup, cold cache), and the unit of scale is coarse (a whole TP group, all-or-nothing). So the controller has to be asymmetric — fast out, slow in — driven by queue depth and KV occupancy, with headroom sized as `T_cold × dλ/dt`, a pre-warmed standby pool for bursts, and predictive scaling on the diurnal curve. And the readiness probe must not pass until warmup finishes, or the router will send traffic into a loading replica."

[Lesson 7](07-prefix-aware-routing.md) balanced traffic over `R` replicas. This lesson decides `R`, over time, under cost pressure.

---

## Why the standard autoscaler fails here

```
  A NORMAL WEB SERVICE                       AN LLM REPLICA
  ─────────────────────────────────────────────────────────────────────────────
  scale signal: CPU% tracks load            GPU% is ~100% whenever ANY decode
                                            is running. It tracks *occupancy*,
                                            not saturation. USELESS as a signal.
  new pod ready in: 5-30 s                  3-15 min (image + weights + warmup)
  unit of scale: 1 pod, 0.5 CPU             1 replica = 1..16 GPUs, gang-scheduled
  pod state: none                           in-flight streams + a warm prefix cache
  cost of over-provisioning: cents          $2-40/hr per idle replica
```

The utilization trap deserves emphasis because it is the single most common mistake: **`DCGM_FI_DEV_GPU_UTIL` is the fraction of time at least one kernel was resident.** A decode loop at batch 1 and a decode loop at batch 256 both report ~100%, while the second does 100× the work. Scaling on it gives you a controller that fires at the wrong times in both directions.

### The signals that actually work

| Signal | Source | Meaning | Use as |
|---|---|---|---|
| `vllm:num_requests_waiting` | engine `/metrics` | queue depth — requests admitted but not running | **primary scale-out trigger** ([Phase 3 lesson 5](../phase-3/05-queueing-theory.md): a persistently non-empty queue means ρ→1) |
| `vllm:num_requests_running` | engine | current batch size vs `max_num_seqs` | concurrency-target scaling (the Ray Serve model) |
| `vllm:gpu_cache_usage_perc` | engine | KV occupancy — the real saturation metric | scale-out at >0.8-0.9 sustained; also predicts preemption |
| preemption counter | engine | the engine is thrashing sequences | hard "you are over capacity now" |
| p95 TTFT / TPOT | your harness or engine histograms | the SLO itself | SLO-driven scaling; also the alerting signal ([Phase 7](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)) |
| tokens/sec per replica | engine | throughput being delivered | capacity model + cost per 1M tokens |
| GPU utilization | DCGM | occupancy | **dashboards only, never a trigger** |

Ray Serve's autoscaler is built around `target_ongoing_requests` per replica precisely because concurrency, not utilization, is the meaningful load measure for this workload ([Phase 5 lesson 8](../phase-5/08-ray-serve-and-composition.md)). KEDA/HPA reach the same place through a Prometheus adapter over `num_requests_waiting`.

---

## The cold-start budget

Measure this on *your* stack; it is the number every other decision depends on.

```
  T_cold = node provisioning + image pull + weight fetch + load-to-GPU + warmup + cache warm

  node provisioning (cloud GPU node, if none free)      30 s -  5 min
  container image pull (CUDA + torch + engine, 5-20 GB) 30 s - 10 min   ← pre-pull it
  weight fetch (S3/GCS/HF → local)                      see table below
  load to GPU + shard/quantize                          20 s -  3 min
  warmup: CUDA graph capture, profiling run             20 s -  90 s    ← per shape
  prefix cache warm (traffic-dependent)                 minutes         ← invisible, real
```

Weight fetch, the term that dominates:

| Model | Bytes | 200 MB/s (single-stream S3/HF) | 1 GB/s (tuned, parallel) | 5 GB/s (local NVMe cache) |
|---|---|---|---|---|
| 8B FP16 | 16 GB | 80 s | 16 s | 3 s |
| 70B FP16 | 140 GB | 12 min | 2.3 min | 28 s |
| 405B FP8 | 405 GB | 34 min | 6.8 min | 81 s |

Mitigations, roughly in order of payoff:

1. **Cache weights on the node's local NVMe** (a DaemonSet or an init container that populates a hostPath / local PVC). Turns minutes into seconds for every subsequent replica on that node.
2. **Pre-pull the image** on every GPU node (DaemonSet holding the image, or a node-image with it baked in). Removes the most variable term.
3. **Parallel/streaming loaders** — `tensorizer`, Run:ai Model Streamer, S3 multipart with many streams — which also overlap download with `load_state_dict` instead of serializing them.
4. **Skip re-warmup where safe** (`--enforce-eager` cuts capture time but costs steady-state throughput at high TP — [lesson 3](03-tensor-parallelism.md)). Usually the wrong trade; measure before choosing it.
5. **Snapshot/restore** — the serverless-GPU trick (Modal, Baseten and similar restore a post-warmup memory image). Powerful, and mostly not available unless your platform provides it.

**Then include the invisible term:** a fresh replica has an empty prefix cache, so its first minutes serve worse TTFT than its peers even at low load. That's why lesson 7's router should *ramp* a new replica in rather than treat it as instantly equal, and why "pre-warm with synthetic requests carrying your top-N system prompts" is a real production step.

---

## The control problem: you are always `T_cold` behind

```
  λ(t) rises at  dλ/dt  requests/sec per second
  a replica ordered at t is useful at t + T_cold
  ⇒ unserved demand accumulated during the gap ≈ ∫ (λ − μ_current) dt ≈ T_cold · dλ/dt · T_cold/2

  HEADROOM RULE:  provision spare capacity ≥ T_cold · dλ/dt
                  (i.e. run at a target utilization of 60-75%, not 90%)
```

Worked: `T_cold = 4 min`, morning ramp of `+0.05 QPS/s` (0 → 12 QPS over 4 minutes), per-replica capacity 6 QPS.

```
  demand grows 12 QPS in 240 s.  A purely reactive scaler notices at ρ≈1, orders a
  replica, and it lands 240 s later — by which time demand has grown another 12 QPS.
  Queue during the gap ≈ 0.05 · 240 · 240/2 ≈ 1,440 requests of backlog.
  With 30% headroom (run 2 replicas where 1.4 suffice) the ramp is absorbed and the
  scaler's lag becomes invisible.
```

Controller design that follows from that:

| Rule | Why |
|---|---|
| **Asymmetric: scale out aggressively, scale in slowly** (e.g. out on 30 s of signal, in after 10-15 min) | scaling out costs money; scaling in early costs an SLO breach plus another `T_cold` to undo |
| **Target utilization 60-75%**, not 90% | the headroom rule; also queueing theory — latency explodes as ρ→1 |
| **Stabilization windows / cooldowns on both directions** | prevents thrash; a flapping GPU autoscaler is pure waste (you pay `T_cold` of idle every cycle) |
| **Predictive/scheduled scaling on the diurnal curve** | traffic is periodic and known; scale *before* the ramp instead of chasing it. This is the single biggest win available |
| **Pre-warmed standby pool** (`n` loaded-but-unrouted replicas) | converts `T_cold` into seconds for burst absorption; you pay for idle GPUs, so size it from burst statistics |
| **Queue + shed instead of scale, for spikes shorter than `T_cold`** | you cannot scale into a 60-second spike; admission control is the only correct response ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)) |
| **Drain, don't kill, on scale-in** | stop admitting, let in-flight generations finish (bounded by `max_tokens`), then exit |

`replicas = ceil(λ / (μ_replica × target_util))`, with `μ_replica` measured at SLO by your own harness ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) — not the peak-throughput number from a vendor chart, which is measured at a latency you can't ship.

---

## Coarse units: what model parallelism does to scheduling

A TP8 replica is **8 GPUs on one node, all-or-nothing**:

- **Gang scheduling / all-or-nothing placement.** Seven of eight GPUs is worth zero. On Kubernetes this needs a scheduler plugin (Kueue, Volcano, Ray's placement groups) or you get partially-placed replicas holding GPUs hostage.
- **Fragmentation blocks scale-out even with free GPUs.** Eight nodes with 4 free GPUs each cannot host one TP8 replica. Bin-packing policy is capacity policy ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)).
- **Granularity is expensive.** Scaling by ±1 replica is ±8 GPUs — sometimes ±30% of a small fleet. This is a real argument for smaller replicas (lower TP + quantization) whenever the SLO allows it: finer granularity, cheaper headroom, smaller blast radius ([lesson 9](09-multi-node-operations.md)).
- **Disaggregated deployments have two scalers** with a coupling constraint: the pool ratio from [lesson 6](06-disaggregated-prefill-decode.md) is itself traffic-dependent, so either scale each pool on its own signal (prefill: TTFT + queue; decode: KV occupancy + TPOT) and let the ratio float, or let instances switch roles. Do not hard-code the ratio.

### Readiness, liveness, and the classic bug

```
  ✗ readinessProbe: httpGet /health   → the HTTP server answers while weights load
                                        ⇒ the router sends real traffic into a replica
                                          that has no model yet: TTFT spike + errors
  ✓ readinessProbe: httpGet /health  gated on "engine warmed up, graphs captured"
     (vLLM's server binds /health only after startup completes — verify with
      failureThreshold and a startupProbe generous enough to cover T_cold)
  ✓ startupProbe with failureThreshold ≈ T_cold / periodSeconds  (else k8s kills the
     pod mid-load and you loop forever, paying T_cold every time)
  ✓ terminationGracePeriodSeconds > longest possible generation, or scale-in and
     rollouts truncate live streams
```

---

## Cost: the number the job is actually about

```
  cost per 1M tokens = ( $/GPU-hour × GPUs per replica )
                       ─────────────────────────────────  × 10⁶
                       ( output tokens/sec per replica × 3600 )
```

Every lever in Phases 3-6 lands in that fraction: batching and paging raise the denominator; quantization raises the denominator *and* can shrink `GPUs per replica`; autoscaling and spot pricing shrink the numerator; headroom and idle standby replicas inflate it. Track **cost per 1M tokens at your SLO**, not utilization — utilization can be 100% while you burn money on a badly sized layout.

| Purchasing mode | Discount | Catch |
|---|---|---|
| On-demand | — | the baseline you compare against |
| Committed / reserved | 30-60% | commit to the *baseline* (diurnal trough), not the peak |
| Spot / preemptible | 60-90% | eviction with ~30 s-2 min notice; capacity is not guaranteed when you want it |

**Spot for inference works if you design the drain path**, and the design is short: on the eviction notice, deregister from the router immediately (stop admitting), let in-flight generations finish inside the notice window (cap `max_tokens` so this is bounded), fail over anything that can't finish to an on-demand replica, and never let spot serve more than the fraction of capacity your SLO can lose at once. The standard shape is **on-demand (or reserved) baseline + spot burst**, with the router preferring on-demand for long-generation traffic. Note the interaction with [lesson 7](07-prefix-aware-routing.md): every eviction is a scale event that cold-caches a slice of your sessions.

---

## Do this now (60 minutes)

1. **Measure `T_cold`, broken down.** Time each stage with log timestamps for a real model on your stack: pod scheduled → image pulled → weights fetched → loaded → graphs captured → `/health` OK → first token of the first request → prefix-cache hit rate stabilized. Put the six numbers in a table. Every autoscaling decision you make afterwards is downstream of this table, and most teams have never produced it.
2. **Simulate the controller.** With a diurnal arrival curve (sinusoid + a lunch spike + a 60-second burst), simulate: (a) reactive HPA on queue depth with your measured `T_cold`, (b) the same with 30% headroom, (c) predictive scaling from yesterday's curve, (d) (b) plus a 1-replica warm standby. Report SLO-violating minutes and GPU-hours for each. The ranking is the argument you'd take to a capacity review.
3. **Price two layouts.** For your target model, compute cost per 1M tokens for the two best layouts from [lesson 5](05-hybrid-layouts-and-moe.md) using measured tokens/sec at SLO, then re-price with a 70/30 on-demand/spot mix and 30% headroom. State which is cheaper and *why* in one sentence — that sentence is the deliverable of an inference-cost-optimization role.

---

**Next:** [Multi-node operations and failure modes →](09-multi-node-operations.md) — what breaks when a replica spans eight GPUs and two nodes, and how to debug a collective that never returns.
