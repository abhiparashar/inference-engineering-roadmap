# 5 — Scheduling, Capacity, and Autoscaling Mechanics

> **You'll be able to say:** "Autoscaling GPU inference is two nested control loops with wildly different time constants: pods (seconds, if a warm node exists) and nodes (minutes). I scale pods on a work-backlog signal — queue depth or KV occupancy, never GPU utilization — with a scale-up that is fast and a scale-down that is slow and drain-aware, and I keep a warm pool sized by `cold_start × peak_ramp_rate` because the node loop cannot react inside a traffic spike. Spot capacity is a cost lever with a 30-120 second eviction contract, and cloud GPU quota is a hard wall that no controller can scale through."

[Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md) derived the *policy*: why utilization is the wrong signal, why cold starts destabilize naive controllers, why you pre-warm. This lesson is the configuration that implements it, plus the capacity realities (quota, spot, fragmentation) that decide whether the controller can act at all.

---

## Two loops, two time constants

```
   traffic ↑
      │
      ▼
  ┌─────────────────────────── LOOP 1: PODS ────────────────────────────┐
  │  HPA / KEDA  reads a work metric  →  changes Deployment replicas    │
  │  latency: seconds IF a node with a free GPU and a warm image exists │
  │           otherwise it degenerates into Loop 2's latency            │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │ pod Pending, no capacity
                                     ▼
  ┌─────────────────────────── LOOP 2: NODES ───────────────────────────┐
  │  Cluster Autoscaler / Karpenter  →  provisions a GPU instance       │
  │  latency: 60-180 s provision + 60-600 s cold start (lesson 3)       │
  │  and it can fail outright: quota, capacity, zone exhaustion         │
  └─────────────────────────────────────────────────────────────────────┘
```

**The whole design problem is that Loop 2's latency is longer than most traffic spikes.** Everything below is a way to avoid being *in* Loop 2 when demand arrives.

---

## Scaling pods: the signal

Restating the Phase-6 conclusion because it is the single most-violated rule in GPU autoscaling:

| Signal | Verdict |
|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | **never.** It reads ~100% at batch 1 and at batch 256; it is a busy-flag ([Phase 7 lesson 3](../phase-7/03-gpu-and-host-telemetry.md)) |
| CPU utilization | never; the CPU is not the bottleneck |
| **Queue depth / waiting requests** | **yes** — direct backlog, leads latency |
| **KV-cache occupancy** | **yes** — the real saturation metric ([Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md)) |
| Running-batch size vs `max_num_seqs` | yes — headroom in the scheduler |
| p95 TTFT | usable, but it *lags*; it is the symptom you're trying to prevent |
| Tokens/sec per replica vs known capacity | good for capacity planning; noisy as a controller input |

A KEDA `ScaledObject` on Prometheus, which is the standard implementation:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: { name: llm-inference }
spec:
  scaleTargetRef: { name: llm-inference }
  minReplicaCount: 3                 # never 0 for interactive LLM serving
  maxReplicaCount: 40
  pollingInterval: 15
  cooldownPeriod: 600                # 10 min before scale-down is even considered
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 0      # react immediately
          policies:
            - { type: Percent, value: 100, periodSeconds: 60 }   # up to double per minute
            - { type: Pods,    value: 4,   periodSeconds: 60 }
          selectPolicy: Max
        scaleDown:
          stabilizationWindowSeconds: 900    # 15 min of sustained low load
          policies:
            - { type: Pods, value: 1, periodSeconds: 300 }       # one pod per 5 min
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: |
          sum(vllm:num_requests_waiting) / count(up{job="llm-inference"} == 1)
        threshold: "4"                # target ~4 queued requests per replica
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: avg(vllm:gpu_cache_usage_perc)
        threshold: "0.75"
```

The asymmetry is the point: **scale up fast and aggressively, scale down slowly and grudgingly.** The cost of one extra replica for 15 minutes is ~$0.75; the cost of being short a replica during a spike is SLO burn plus a 6-minute cold start you can't shortcut. Also note `minReplicaCount: 3`, not 0 — scale-to-zero is correct for batch and dev endpoints, and wrong for anything a human waits on, unless you accept multi-minute first-request latency.

**Sticky routing interacts with this.** Adding a replica reshuffles the consistent-hash ring and drops prefix-cache hit rate for a while ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)), which *raises* TTFT right after a scale-up. Expect it, use bounded-load consistent hashing to limit the churn, and don't diagnose it as a regression.

---

## Scaling nodes: Karpenter / Cluster Autoscaler

Cluster Autoscaler works from pre-defined node groups; Karpenter provisions instances directly from a `NodePool` spec and is a better fit for heterogeneous GPU fleets.

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata: { name: gpu-inference }
spec:
  template:
    metadata:
      labels: { workload: inference }
    spec:
      taints:
        - { key: nvidia.com/gpu, value: "present", effect: NoSchedule }
      requirements:
        - { key: node.kubernetes.io/instance-type, operator: In,
            values: ["g6e.xlarge", "g6e.2xlarge", "p5.48xlarge"] }
        - { key: karpenter.sh/capacity-type, operator: In, values: ["on-demand"] }
        - { key: topology.kubernetes.io/zone, operator: In, values: ["us-east-1a","us-east-1b"] }
      expireAfter: 168h
  limits: { "nvidia.com/gpu": 64 }        # a hard cost ceiling — set one
  disruption:
    consolidationPolicy: WhenEmpty        # NOT WhenEmptyOrUnderutilized for stateful serving
    consolidateAfter: 30m
```

Two GPU-specific settings deserve emphasis:

- **`consolidationPolicy: WhenEmpty`.** Aggressive consolidation will evict a perfectly healthy inference pod to repack the cluster — paying a 6-minute cold start and a prefix-cache cold flush to save a few dollars. Let empty nodes go; leave busy ones alone.
- **`limits`.** A runaway metric or a retry storm can otherwise scale you into a five-figure hourly bill. The limit is the circuit breaker; set it, alert at 80% of it.

### The warm pool

The one technique that actually defeats Loop 2's latency:

```
   pool_size ≥ peak_ramp_rate (replicas/min) × cold_start (min) × safety(1.3)

   example: traffic can add 6 replicas/min of demand; cold start 4 min
            → 6 × 4 × 1.3 ≈ 31 warm GPUs standing by
```

That number is often shocking, and it is the honest price of a minute-scale cold start. The levers to make it affordable are exactly [lesson 3](03-image-size-and-cold-start.md)'s: halve the cold start and you halve the pool. Implementations: low-priority **balloon/pause pods** that a real pod preempts (`PriorityClass` with `preemptionPolicy`), Karpenter's static capacity via a `NodePool` with a minimum, or simply an over-provisioned `minReplicaCount` sized to your diurnal trough-to-peak ramp.

Schedule-aware pre-warming beats reactive scaling for anything with a diurnal curve: a CronJob that raises `minReplicaCount` 20 minutes before the known morning ramp is 15 lines of YAML and eliminates the daily 9 a.m. latency incident.

---

## Capacity realities the controller cannot fix

| Reality | What it looks like | What to do about it |
|---|---|---|
| **Cloud GPU quota** | `InsufficientInstanceCapacity` / quota exceeded; pods `Pending` forever | request quota *before* you need it; alert on "pods Pending > 5 min"; know your per-region, per-family limits |
| **Regional stockouts** | H100s unavailable in your zone at peak | multi-zone `NodePool`, a fallback instance family, and a tested plan for serving on the fallback (it has different memory → different `max_model_len`) |
| **Fragmentation** | 12 free GPUs across 6 nodes, but your 8-GPU pod can't schedule | bin-pack by requesting whole nodes for multi-GPU work; use topology-aware placement |
| **Spot preemption** | node vanishes with 30-120 s notice | see below |
| **Reserved/committed capacity** | 40-60% cheaper, but you pay whether or not you use it | baseline on committed, peak on on-demand, batch on spot |
| **MIG partitioning** | a GPU advertised as 7 small devices | good for many small models; wrong for one big one ([Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md)) |

### Spot, concretely

Spot GPUs are typically 50-80% cheaper, which makes them irresistible and dangerous:

| Workload | Spot? |
|---|---|
| Offline/batch inference, evals, benchmark runs | **yes** — near-free capacity, restarts are harmless |
| Async APIs with retries and a queue | yes, with a checkpoint/requeue path |
| Interactive chat serving | only as *surge* capacity on top of an on-demand baseline that alone meets the SLO |
| Multi-node TP groups | **no.** One evicted rank kills the gang ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)) |

The eviction contract is a two-minute (AWS) or 30-second (GCP) notice. Wire the node-termination handler to the same drain path as `SIGTERM` ([lesson 4](04-kubernetes-for-gpu-serving.md)): flip readiness false, stop admitting, finish what fits in the window, and return a retryable error for the rest. Then *test it* by manually terminating an instance during a load test — the number you want is "zero 5xx, N requests retried," and the number you'll get the first time is not that.

---

## Right-sizing: the arithmetic before the controller

Autoscaling multiplies a per-replica capacity number you must first measure honestly ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)):

```
  from the load test, at the SLO (not at max throughput):
     replica_capacity = 850 output tok/s at p99 TTFT ≤ 500 ms
     peak demand      = 22,000 output tok/s
     N_serving        = ceil(22000 / 850)        = 26
     + N-1 redundancy / zone loss                = +3
     + rollout surge headroom (lesson 6)         = +2
     + warm pool for ramp                        = +6
     ────────────────────────────────────────────────
     provisioned                                  = 37 GPUs  (1.42× the "compute" number)
```

That 1.42× multiplier is the gap between benchmark cost and invoice cost from [Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md), made explicit and defensible. Writing this arithmetic down is what a capacity review is; guessing at it is what a cost incident is.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Autoscaler never triggers while latency explodes | scaling on GPU utilization | scale on queue depth / KV occupancy |
| Replica count oscillates (flapping) | scale-down window shorter than cold start; noisy metric | long `stabilizationWindowSeconds`, slow scale-down policy, smoothed metric |
| Scale-up happens but latency stays bad for 6 min | cold start dominates | shrink cold start; warm pool; pre-warm on schedule |
| TTFT worsens right after a scale-up | prefix-cache ring reshuffle | bounded-load consistent hashing; expect and annotate it |
| Pods `Pending` forever | quota / stockout / fragmentation | alert on pending duration; multi-zone; fallback family |
| Bill doubles overnight | no `limits` on the NodePool, or a retry storm scaled the fleet | hard limits + budget alert + retry budgets ([Phase 7 lesson 7](../phase-7/07-reliability-and-degradation.md)) |
| Spot eviction drops in-flight requests | no termination handler | wire notice → drain path; test it |
| Karpenter kills healthy replicas at 2 a.m. | `WhenEmptyOrUnderutilized` consolidation | `WhenEmpty`, plus a PDB |
| Scale-to-zero endpoint: first user waits 7 minutes | zero warm capacity | `minReplicaCount ≥ 1` for interactive; scale-to-zero only for batch/dev |

---

## Do this now (45 minutes)

1. **Compute your provisioning arithmetic** with your own measured per-replica capacity at your SLO. Produce the five-line table above with real numbers, including the multiplier over the naive compute number.
2. **Write a KEDA `ScaledObject`** against a backlog metric your server already exposes, with the asymmetric up/down behaviour. Run a synthetic spike (2× QPS step) and record: detection time, replica ramp, time until p99 recovers. That third number is your true scale-up latency.
3. **Size the warm pool** from `peak_ramp_rate × cold_start`, then compute its monthly cost and compare it against the cost of halving your cold start. One of those two is clearly cheaper for you — knowing which is the point of the exercise.

---

**Next:** [Deploying and rollouts →](06-deploying-and-rollouts.md) — Helm/Kustomize, rolling vs blue-green vs canary on scarce GPU capacity, and how to make rollback a 60-second operation.
