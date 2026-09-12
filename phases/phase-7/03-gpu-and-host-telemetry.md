# 3 — GPU and Host Telemetry: DCGM, XID, Thermals, and the Utilization Decoy

> **You'll be able to say:** "`DCGM_FI_DEV_GPU_UTIL` is a busy-flag, not a utilization. The fields that carry information are the profiling ones — `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE` — which let me read the Phase-2 roofline live and distinguish memory-bound decode from compute-bound prefill on a running server. I can name the hardware failure signals that precede 'random slowness' (clock throttling from power/thermal caps, ECC single-bit storms, row remap pending, NVLink flaps) and the ones that mean 'this replica is dead, drain it now' (XID 48/79/94/95, uncorrectable ECC, fallen-off-the-bus). I also know DCGM profiling metrics cost real GPU cycles and are mutually exclusive with an attached profiler."

[Lesson 2](02-instrumenting-with-prometheus.md) instrumented *your* code. This lesson instruments the hardware underneath it — the layer that explains the 5% of incidents your application metrics call "unexplained variance," and the layer that fleet-level cost accounting ([lesson 9](09-cost-per-million-tokens.md)) is built on.

---

## The stack of GPU telemetry sources

```
  nvidia-smi          human CLI, one-shot; DON'T scrape it in a loop (forks a process,
                      takes an NVML lock, ~100 ms; at 15 s intervals × 8 GPUs it's fine,
                      at 1 s it's a problem and it perturbs what it measures)
     │
  NVML (libnvidia-ml) the C/Python API behind nvidia-smi; pynvml. Cheap counters:
                      memory used, power, temperature, clocks, throttle reasons, XID-ish
     │
  DCGM                NVIDIA's daemon (nv-hostengine) on top of NVML + the profiling
                      interface. Adds SM occupancy/activity, tensor-pipe utilization,
                      DRAM activity, NVLink bytes, health checks, MIG awareness
     │
  dcgm-exporter       DCGM → Prometheus. The standard in every real GPU fleet; ships as
                      a k8s DaemonSet, maps metrics to pod/container via the kubelet
                      device API so you get {pod, namespace, container} labels
     │
  CUPTI / Nsight      per-kernel truth (Phase 2 lesson 8). Development only: high overhead,
                      and it CONFLICTS with DCGM profiling (one consumer at a time)
```

The mapping-to-pod part is what makes `dcgm-exporter` worth deploying over a homegrown NVML scraper: `DCGM_FI_DEV_FB_USED{pod="vllm-7d9-abc", gpu="3"}` is directly joinable to the engine metrics from the same pod, which is how you answer "is this replica's slowness hardware or software."

---

## The decoy field, explained properly

```
  DCGM_FI_DEV_GPU_UTIL  (== nvidia-smi "GPU-Util" == NVML utilization.gpu)
  ────────────────────────────────────────────────────────────────────────
  DEFINITION: percentage of the last sampling period during which ≥1 kernel
              was executing. It says NOTHING about how much of the GPU that
              kernel used.

  decode, batch 1,   ~2% of peak FLOPs   →  reports ~90-100%
  decode, batch 256, ~25% of peak FLOPs  →  reports ~99-100%
  prefill, 8k tokens, ~70% of peak FLOPs →  reports 100%
  a single busy-wait kernel doing nothing →  reports 100%
```

So `GPU_UTIL` = 100% is the normal state of a healthy inference server at *any* load, which is why it is useless as a scaling signal ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md)) and misleading as a capacity signal in a cost review ("we're at 100% GPU, we need more GPUs" is the most expensive sentence in ML infra). It has exactly two legitimate uses: **detecting a totally idle GPU** (it reads ~0, so you're paying for nothing) and **detecting a hung kernel** (it reads 100 while tokens/sec is 0).

### The fields that do carry information

| DCGM field | What it means | Read it as |
|---|---|---|
| `DCGM_FI_PROF_SM_ACTIVE` | fraction of time × fraction of SMs with ≥1 active warp | are we actually filling the machine? low + high GPU_UTIL = tiny kernels / launch-bound ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)) |
| `DCGM_FI_PROF_SM_OCCUPANCY` | resident warps / max warps | occupancy, in the CUDA sense; low means small batches or register/shared-memory limits |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | fraction of cycles the tensor pipes are issuing | **the compute-bound indicator**: high during prefill, low during decode |
| `DCGM_FI_PROF_DRAM_ACTIVE` | fraction of cycles HBM is transferring | **the memory-bound indicator**: decode lives here (it's the roofline, live) |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` / `FB_TOTAL` | framebuffer memory, MiB | capacity headroom; pair with engine KV occupancy |
| `DCGM_FI_DEV_POWER_USAGE` | watts now | real work proxy; also your $ and thermal driver |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | mJ, monotonic | energy per 1M tokens — the honest efficiency metric |
| `DCGM_FI_DEV_GPU_TEMP` / `MEMORY_TEMP` | °C | precedes throttling; HBM temp is the one that limits H100-class parts |
| `DCGM_FI_DEV_SM_CLOCK` | MHz now | dropping clocks + flat load = throttling, not a code regression |
| `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` | bitmask | **why** clocks dropped: power cap, thermal, HW slowdown, SW cap |
| `DCGM_FI_PROF_NVLINK_TX_BYTES` / `RX_BYTES` | bytes over NVLink | TP collective traffic; a flapping link shows up here first ([Phase 6 lesson 2](../phase-6/02-collectives-and-interconnects.md)) |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `RX_BYTES` | host↔device traffic | unexpectedly high = you're copying tensors you shouldn't be |
| `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` / `DBE_VOL_TOTAL` | corrected / uncorrected memory errors | SBE storm = failing HBM; **DBE = drain immediately** |
| `DCGM_FI_DEV_RETIRED_PENDING` / `ROW_REMAP_PENDING` | memory pages/rows awaiting retirement | needs a reset window; schedule drain |
| `DCGM_FI_DEV_XID_ERRORS` | last XID code | the single most important GPU health field |

### The live roofline

This is the payoff of the profiling fields, and it connects Phase 2 directly to production dashboards:

```
  PIPE_TENSOR_ACTIVE high, DRAM_ACTIVE moderate   → compute-bound: prefill-heavy traffic.
                                                    Levers: better kernels, bigger TP,
                                                    chunked prefill tuning.
  DRAM_ACTIVE high, PIPE_TENSOR_ACTIVE low        → memory-bound: decode-dominated (normal).
                                                    Levers: bigger batch, quantization,
                                                    speculative decoding (Phase 4).
  BOTH low, GPU_UTIL ~100%                        → overhead/launch-bound or serialized:
                                                    CUDA graphs, fewer syncs, check the
                                                    tokenizer/detokenizer on the host.
  BOTH low, GPU_UTIL low, queue non-empty         → NOT a GPU problem. Look at the host:
                                                    scheduler, tokenizer, network, locks.
```

That last row is the one that saves the most wall-clock in real debugging: it is the difference between "buy more GPUs" and "your Python detokenizer is single-threaded."

### The cost of profiling metrics

DCGM profiling fields are sampled via the hardware profiling interface and are **not free**: expect low-single-digit percent overhead depending on the field set and sample interval, and note that **DCGM profiling and an attached Nsight/CUPTI profiler are mutually exclusive** — one will fail to initialize. Practical policy: scrape `SM_ACTIVE`, `DRAM_ACTIVE`, `PIPE_TENSOR_ACTIVE` at 10-30 s intervals in production (cheap enough, and you need the roofline signal), keep the rest of the `PROF_*` set for ad-hoc investigation, and be ready to disable the profiling field group on a node when you want to run a real profiler there ([Phase 2 lesson 8](../phase-2/08-profiling-in-practice.md)).

---

## XID errors: the fleet's ground truth

An **XID** is an error the NVIDIA driver reports to the kernel log; it is the most direct statement "something went wrong at the hardware/driver level." They appear in `dmesg`/`journalctl -k` as `NVRM: Xid (PCI:0000:1f:00): 79, ...` and via `DCGM_FI_DEV_XID_ERRORS`.

| XID | Meaning | Usual cause | Action |
|---|---|---|---|
| 13 | graphics/compute engine exception | illegal memory access in a kernel | usually **your** bug (or a bad custom kernel); kills the context |
| 31 | GPU memory page fault | bad address, often a buggy kernel | same as 13; reproduce with `compute-sanitizer` |
| 43 | reset channel verification failure | app-triggered error | app-level, process dies |
| 45 | preemptive channel removal | process killed / OOM-killer | benign in isolation, follows other failures |
| 48 | **double-bit ECC error** | hardware memory fault | **drain node, RMA candidate** |
| 62, 63, 64 | page retirement / row remapping events | HBM degrading | 63/64 pending ⇒ needs reset; drain at next window |
| 74 | NVLink error | link/connector/NVSwitch fault | TP replicas will hang or slow; drain, check the fabric |
| 79 | **GPU has fallen off the bus** | hardware/PCIe/power failure | **replica is dead**; node needs a reboot, often RMA |
| 92 | high single-bit ECC rate | degrading memory | watch; schedule maintenance |
| 94 / 95 | contained / **uncontained** ECC error | memory fault; 95 corrupted other contexts | 94: kill the affected process. **95: reboot the node** |
| 119 / 120 | GSP RPC timeout | firmware-level hang | drain; often needs driver/firmware update |

**The operational rule:** XID 48, 79, 94, 95 and any uncorrectable ECC → **cordon the node and drain the replica**, do not restart the pod in place ([lesson 7](07-reliability-and-degradation.md)). Everything else → alert with a ticket, not a page, and correlate with your own error classes. For a TP replica this is amplified: one bad GPU takes down all `TP` GPUs in the group ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)), so XID alerts must be labelled with the replica, not just the GPU.

Also scrape **`dcgm_health` / run `dcgmi diag -r 1`** (a fast non-invasive check) as a pre-flight in your pod's init phase. Catching a sick GPU at startup is far cheaper than catching it 40 minutes into serving.

---

## Throttling: the "mysterious 20% slowdown"

Clocks are not constant. When a GPU hits a power or thermal limit it reduces clocks, and your tokens/sec drops with no code change and no traffic change.

```
  SYMPTOM: tokens/sec down 15-30%, step time up, no deploy, no traffic shift
  CHECK:   DCGM_FI_DEV_SM_CLOCK  (down?)
           DCGM_FI_DEV_CLOCK_THROTTLE_REASONS (bitmask — which cap?)
           DCGM_FI_DEV_POWER_USAGE vs enforced power limit
           DCGM_FI_DEV_GPU_TEMP / MEMORY_TEMP vs slowdown threshold

  THROTTLE REASON BITS (the ones that matter)
    SW_POWER_CAP        → at the configured power limit; expected on dense nodes
    HW_THERMAL_SLOWDOWN → serious: inlet temp, dust, failed fan, bad rack airflow
    HW_POWER_BRAKE      → PSU/rack power event
    SW_THERMAL_SLOWDOWN → driver-level thermal management
```

Three facts that make this a first-class production concern rather than a datacenter-team concern:

1. **Dense nodes throttle under sustained load by design.** 8×700 W GPUs plus CPUs can exceed what the chassis/rack can cool continuously; sustained prefill-heavy traffic is the worst case, and vendor throughput numbers are usually measured in short bursts.
2. **Throttling is unevenly distributed**, so in a TP group one throttled GPU becomes the straggler that sets the pace for all ranks — every collective waits for the slowest rank. A 10% clock drop on one GPU can cost more than 10% of the replica.
3. **It looks exactly like a software regression** on application-only dashboards. Having `SM_CLOCK` and throttle reasons on the same dashboard row as tokens/sec is what turns a 3-hour investigation into a 30-second one ([lesson 6](06-dashboards-and-diagnosis.md)).

---

## Host-side telemetry you still need

A GPU dashboard that ignores the host misses a surprising share of real incidents, because inference has significant CPU work on the request path.

| Signal | Source | Inference-specific reason |
|---|---|---|
| CPU utilization, per-core and steal | node exporter | tokenization, detokenization, sampling logic, HTTP/JSON — all host-side; a saturated core caps tokens/sec regardless of GPU |
| CPU throttling (cgroup) | cAdvisor `container_cpu_cfs_throttled_seconds_total` | **the classic k8s inference bug**: a low CPU limit throttles the engine's Python loop and shows up as GPU idleness |
| host RAM + page cache | node exporter | weight loading, pinned buffers, CPU-offload KV; OOM-killed pods look like random restarts |
| disk throughput / local NVMe | node exporter | model load time = cold start ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md), [Phase 8 lesson 3](../phase-8/03-image-size-and-cold-start.md)) |
| NIC throughput, retransmits, RDMA counters | node exporter, `ethtool` | streaming responses, multi-node collectives, KV transfer in disaggregated serving |
| open FDs / connection count | node exporter | streaming holds a connection per request for tens of seconds; limits bite differently than in request/response services |
| container restarts, OOMKills | kube-state-metrics | the cheapest early warning that exists |
| shared memory `/dev/shm` | node exporter | NCCL and some engines need it; too small = mysterious hangs at startup |

**Do not skip the cgroup CPU-throttling metric.** "GPU is idle, queue is full, engine looks healthy" with `container_cpu_cfs_throttled_seconds_total` climbing is a complete diagnosis, and the fix is a `resources.limits.cpu` change, not an autoscaling change.

---

## Putting it together: the health model of one replica

```
  IS IT ALIVE?        /health OK, process up, GPU_UTIL > 0, tokens/sec > 0
  IS IT CORRECT?      finish_reason distribution normal, no ECC DBE, no XID 13/31
  IS IT FAST?         step time vs baseline, SM_CLOCK not throttled, no straggler rank
  IS IT FULL?         KV occupancy, queue depth, batch size vs max_num_seqs
  IS IT EXPENSIVE?    output tokens/sec ÷ (GPUs × $/hr), energy per 1M tokens
  IS IT DYING?        SBE rate rising, row remap pending, thermal margin shrinking,
                      NVLink errors, XID 62/63/64/92
```

Those six questions are also the six rows of the replica dashboard in [lesson 6](06-dashboards-and-diagnosis.md) and the six sections of the drain/replace runbook in [lesson 10](10-incident-response-and-chaos.md).

---

## Do this now (45 minutes)

1. **Deploy `dcgm-exporter` (or a 20-line `pynvml` scraper if you have no k8s) against a GPU you can load.** Run a decode-heavy workload at batch 1, then at the largest batch you can, and record `GPU_UTIL`, `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`, power, and tokens/sec for both. Put the two rows side by side: `GPU_UTIL` barely moves while tokens/sec changes 10-100×. That table is the argument you will reuse in every capacity conversation.
2. **Read the live roofline.** Run a prefill-heavy load (long prompts, few output tokens) and a decode-heavy load (short prompts, long outputs). Confirm `PIPE_TENSOR_ACTIVE` and `DRAM_ACTIVE` swap dominance, and write down the two ratios. You now have a production-usable classifier for "which half of my traffic is hurting."
3. **Build the hardware-fault panel and test one branch.** Chart `SM_CLOCK`, throttle reasons, `GPU_TEMP`, `MEMORY_TEMP`, ECC SBE/DBE counters and `XID_ERRORS` alongside tokens/sec. Then induce a throttle you can actually cause: lower the power limit with `nvidia-smi -pl <watts>` (on a machine you own) under load, and watch clocks, throttle reason and tokens/sec move together. Restore the limit afterwards.

---

**Next:** [Tracing and logging a request end-to-end →](04-tracing-and-logging.md) — spans across gateway, router, engine and GPU, the `gen_ai.*` conventions, sampling that keeps the slow requests, and how to log a prompt-bearing service without creating a privacy incident.
