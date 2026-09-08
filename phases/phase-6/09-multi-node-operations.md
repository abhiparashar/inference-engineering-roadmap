# 9 — Multi-Node Operations and Failure Modes

> **You'll be able to say:** "A model-parallel replica is one fate-sharing unit, so its failure rate is `N ×` a single GPU's and its blast radius is all `N`. The signature failure isn't a crash, it's a **hang**: collectives are barriers, so one rank that dies, throws, or gets a mismatched shape leaves every other rank blocked forever with GPUs at 100% 'utilization' and no error in the logs. I can turn that into a rank-attributed error with four environment variables, read a NCCL flight-recorder dump to find which rank never joined which collective, find a straggler from per-rank step times, and roll out a 16-GPU replica without a full outage."

This is the operations lesson: fewer formulas, more of what you do at 2 a.m. It's also the highest-signal interview material in the phase, because everyone has read about tensor parallelism and almost nobody can debug it.

---

## Fate sharing: the arithmetic of blast radius

```
  MTBF(replica) ≈ MTBF(GPU) / N          (any GPU's failure kills the replica)

  GPU MTBF ~10,000 h (hardware faults, ECC, Xid, driver, NVLink)
  ├─ TP1 replica  : ~10,000 h  ≈ 14 months
  ├─ TP8 replica  :  ~1,250 h  ≈ 52 days
  └─ TP8×PP2 (16) :   ~625 h   ≈ 26 days

  A fleet of 50 TP8 replicas (400 GPUs) therefore expects a replica loss
  every ~25 hours. At that rate, recovery is not an incident response — it is
  a routine, automated code path, and it must be tested.
```

Two design consequences, and they're the reason [lesson 1](01-when-one-gpu-isnt-enough.md) said "don't distribute for fun":

1. **Every parallelism dimension multiplies your failure rate and your recovery cost.** Prefer the smallest replica the SLO allows — quantize to lower TP ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)) rather than add GPUs.
2. **Fail fast, whole-replica.** There is no useful degraded mode for a TP group: 7/8 ranks cannot serve. The correct behaviour is to kill the replica, eject it from the router ([lesson 7](07-prefix-aware-routing.md)), restart it, and let clients retry. A replica that limps is worse than one that's gone, because the router keeps sending it traffic.

---

## The failure taxonomy

| Failure | What you see | Root causes | Response |
|---|---|---|---|
| **Collective hang** | server frozen, no logs, GPUs pinned ~100%, no progress | one rank crashed/threw, shape or dtype mismatch across ranks, mismatched collective order, network partition | the playbook below; then fail fast |
| Worker process crash | one rank's traceback; others hang until watchdog | OOM in Python, model bug, driver error | watchdog timeout → kill replica → restart |
| **GPU OOM mid-run** | `CUDA out of memory` after serving fine for a while | activation/workspace spike, not KV — usually too large `max_num_batched_tokens`, or a long prompt with a big chunk; also fragmentation | cap the token budget; lower `gpu_memory_utilization`; check preemption metrics first |
| ECC / Xid error | `nvidia-smi`/`dmesg` Xid, sometimes `unspecified launch failure` | hardware; double-bit ECC needs a reset/RMA | drain node, cordon, replace; do not just restart the pod onto the same GPU |
| NVLink / fabric error | throughput collapse or hang, Xid 74-class errors | cable/connector, switch, driver | `nvidia-smi nvlink -s`, drain node |
| NIC flap / RoCE misconfig | intermittent cross-node hangs, wild latency variance | wrong interface chosen, MTU/PFC misconfig, GDR disabled | pin `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`; verify with `nccl-tests` before serving |
| **Straggler** | everything slow, no errors, all ranks "busy" | thermal/power throttling, down-negotiated PCIe link, one rank doing extra host work, ECC retries, noisy neighbour | per-rank step-time comparison (below) |
| `/dev/shm` too small | cryptic NCCL/`Bus error`/`shm` failures at startup in containers | Docker's 64 MB default `/dev/shm` | `--shm-size=8g` (or a `Memory` `emptyDir` medium in k8s) |
| Silent numeric divergence | one replica's outputs differ subtly | mismatched dtype/kernel/TP degree between replicas | pin versions and layout per deployment; canary compare ([Phase 7](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)) |
| Rendezvous failure | ranks never form the group | wrong `MASTER_ADDR`/port, DNS, firewall, non-stable pod identity | headless Service + StatefulSet ordinals, or Ray for placement |

---

## The hang, and how to make it debuggable

Why it looks like nothing is wrong:

```
  step N:  rank0 ─ all_reduce ──▶ waits for 1,2,3 ...
           rank1 ─ all_reduce ──▶ waits
           rank2 ─ CRASHED (or threw, or entered a DIFFERENT collective)
           rank3 ─ all_reduce ──▶ waits
  ⇒ NCCL kernels spin-wait on the GPU: DCGM reports ~100% utilization,
    the HTTP thread may still answer /health, and no rank prints an error.
    A naive liveness probe sees a perfectly healthy pod. Forever.
```

**Set these before you ever need them:**

| Setting | Effect |
|---|---|
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | the watchdog turns a stuck/errored collective into an exception that tears the process down instead of hanging |
| `TORCH_NCCL_BLOCKING_WAIT=1` | (alternative, higher overhead) synchronous waits with timeouts — easier stack traces |
| process-group `timeout=` (default is minutes) | make it match your step time budget, not the default; a 10-minute hang is a 10-minute outage |
| `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` + `TORCH_NCCL_DUMP_ON_TIMEOUT=1` | **NCCL Flight Recorder**: a ring buffer of recent collectives per rank, dumped on timeout |
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL` | topology and per-collective logging ([lesson 2](02-collectives-and-interconnects.md)) |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` (vLLM) | engine-level watchdog on a step that never completes |

### The playbook

```
  1. Hang or crash?  `kubectl logs --all-containers` on every rank. A single rank
     with a traceback and the rest silent = that rank is your root cause.
  2. Per-rank stacks:  py-spy dump --pid <each rank>   (in-container: needs
     SYS_PTRACE). Compare where each rank is. Look for the odd one out.
  3. Flight recorder dump: each rank's trace lists collectives with a sequence
     number. Find the highest common seq; the rank whose last entry is EARLIER
     (or in a different collective) is the offender.
       - all ranks in the SAME collective, same seq, all waiting → network/fabric
       - one rank in a DIFFERENT collective → control-flow divergence (a Python
         branch that depends on per-rank data: the classic TP bug)
       - one rank missing entirely → it died; find its real exception above
  4. Hardware check on the offending node: `nvidia-smi -q -d PAGE_RETIREMENT,ECC`,
     `dmesg | tail` for Xid, `nvidia-smi nvlink -s`.
  5. Fabric check: run `all_reduce_perf` on the same ranks. If it hangs too, it's
     the fabric/config, not your engine.
```

Xid codes worth recognizing (`dmesg`, `nvidia-smi -q`): **13/31** illegal address or page fault — usually application/kernel bug; **48** double-bit ECC — hardware, drain the node; **63/64** ECC page retirement/row remap — degrading GPU; **74** NVLink/internal error; **79** "GPU has fallen off the bus" — hardware/power, node is done; **119/120** GSP RPC timeout — driver-level, drain.

**The control-flow divergence bug deserves a name**, because it's the one you'll write yourself: any `if` in the model or scheduler path whose condition differs per rank (a rank-local length, a `.item()` on rank-specific data, an exception caught on only one rank) makes ranks issue different collective sequences, and the result is a hang, not an error. Keep every collective on a code path driven by *replicated* control state — which is exactly why engines broadcast the scheduler's decisions to all workers rather than letting each rank decide.

---

## Stragglers: no errors, just slow

A TP group runs at the pace of its slowest rank, every step, because each all-reduce is a barrier. A 20%-slow rank makes the whole replica 20% slow — and nothing logs an error.

```
  detection: instrument per-rank step time (or NCCL wait time) and compare.
             a healthy TP group's per-rank step times sit within a few %.
             persistent skew on one rank = straggler.

  causes to check, in order:
   1. clocks/throttling   nvidia-smi -q -d PERFORMANCE,CLOCK   (SW power cap? thermal?)
   2. PCIe link width     nvidia-smi -q | grep -A2 "Link Width"  (x16 → x4 = bad riser)
   3. NVLink/NIC health   nvidia-smi nvlink -s ; ibstat ; nccl-tests baseline
   4. extra host work     is rank 0 doing tokenization/detokenization/logging that
                          the others aren't? (a real and common asymmetry)
   5. ECC retries / page retirement on that GPU
   6. noisy neighbour     another container sharing the node's PCIe/NUMA path
```

Keep a `nccl-tests` baseline number per node type from [lesson 2](02-collectives-and-interconnects.md). "Is this node slow?" becomes a one-command answer instead of an afternoon.

---

## Rollouts and version skew

Updating a fleet of N-GPU replicas is where model parallelism costs money:

| Strategy | GPU cost | Risk | When |
|---|---|---|---|
| Rolling with surge=1 | **+1 full replica** of spare GPUs (8-16) | low | the default, if you can afford the surge |
| Rolling with surge=0 (drain one first) | none | capacity dip during the roll | small fleets with headroom ([lesson 8](08-autoscaling-gpu-fleets.md)) |
| Blue/green | +100% during the switch | lowest | major version/engine changes, easy rollback |
| Canary % of traffic | +1 replica | lowest quality risk | any weight/quantization change ([Phase 7](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)) |

Rules that specifically matter for distributed, stateful replicas:

- **Drain before terminate.** Deregister from the router, stop admitting, finish in-flight generations, then exit. `terminationGracePeriodSeconds` must exceed your longest possible generation (`max_tokens × TPOT`), or every rollout truncates live streams.
- **Never mix versions within a replica.** Ranks must run identical code and weights; a partial rollout of a TP group is a hang (shape/collective mismatch), not a degraded replica. Deploy the replica as a unit.
- **Across replicas, version skew is a routing problem.** Two engine versions may have incompatible KV layouts and different prefix-cache hashes; pin whole *sessions* to a version, or you get cache misses and, worse, quality inconsistency mid-conversation.
- **Warm before routing.** A new replica is ready-but-cold; ramp its share up ([lesson 7](07-prefix-aware-routing.md)) instead of giving it `1/R` of traffic instantly.
- **Health checks must probe the engine loop, not the HTTP server.** A hang keeps FastAPI answering. Liveness should assert forward progress — e.g. the engine's last-step timestamp is recent, or a tiny 1-token generation succeeds.

---

## Multi-node plumbing that must be right before anything works

| Item | Why it bites |
|---|---|
| Stable rank identity | StatefulSet + headless Service (or Ray placement groups) so rendezvous addresses don't change mid-restart |
| Gang scheduling | Kueue/Volcano/Ray: partially-placed replicas hold GPUs and serve nothing ([lesson 8](08-autoscaling-gpu-fleets.md)) |
| Topology-aware placement | all TP ranks on one NVLink node; PP across nodes ([lesson 4](04-pipeline-parallelism.md)) — get this wrong and you've built cross-node TP by accident |
| RDMA access in containers | device plugin, `IPC_LOCK`, hugepages; without it NCCL silently falls back to TCP sockets (10× slower — check the `NCCL_DEBUG` line) |
| `/dev/shm` sized | 64 MB default breaks multiprocessing workers |
| `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` pinned | otherwise NCCL may pick the management NIC |
| Identical driver/CUDA/engine versions per replica | mixed nodes produce mismatched kernels and mysterious hangs |
| Ray cluster health (if used) | vLLM multi-node uses Ray; a lost Ray worker looks like an engine hang |

---

## Chaos drills (do these before production does them to you)

| Drill | Command | Expected, correct behaviour |
|---|---|---|
| Kill a non-zero rank | `kill -9 <worker pid>` | watchdog fires within your timeout → whole replica exits → restarts → router ejected it in the meantime; in-flight requests fail cleanly (not hang) |
| Kill rank 0 | same | identical outcome; nothing special about rank 0 for availability |
| Sever the fabric | `ip link set <ib dev> down` on one node | timeout → error with rank attribution, not an indefinite hang |
| Straggler injection | `nvidia-smi -lgc <low clock>` on one GPU | throughput drops ~proportionally; your per-rank step-time monitor identifies the rank |
| KV exhaustion | flood long-context requests | preemption + queueing ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)), never OOM-crash |
| Node drain mid-generation | `kubectl drain` | in-flight streams finish or fail fast; no truncated-forever streams |
| Spot eviction | simulate the notice | deregister → drain → exit inside the notice window ([lesson 8](08-autoscaling-gpu-fleets.md)) |

Each drill has one deliverable: **time-to-recovery and requests lost.** Write them in a runbook next to the command that reproduces the failure. That runbook is worth more than any of the throughput numbers in this phase.

---

## Do this now (60 minutes, 1 GPU or CPU-only)

1. **Reproduce the hang.** Two processes, `torch.distributed` (`gloo` is fine): both call `all_reduce` in a loop; make rank 1 `sys.exit(1)` at iteration 10. Observe rank 0 hanging with no error. Then set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (or a short `timeout=` on `init_process_group` for gloo) and observe it becoming a rank-attributed exception. **This 15-line reproduction is the most valuable thing in the lesson.**
2. **Reproduce control-flow divergence.** Make rank 0 call `all_reduce` twice and rank 1 once inside the loop. Confirm the hang, then enable the flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE=2000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, NCCL backend) and read the dump: identify the divergent sequence number. Now you can debug someone else's engine.
3. **Write the runbook.** One page: the four env vars you always set, the five-step playbook, the Xid table, the straggler checklist, and the seven drills with expected behaviour and measured time-to-recovery. Keep it in the repo next to the [exit artifact](11-exercises-and-artifacts.md) — "here is my distributed-inference runbook" answers a whole class of interview questions in one artifact.

---

**Next:** [Build: manual TP + a prefix-aware router →](10-build-manual-tp-and-router.md) — shard a layer by hand, prove it matches the unsharded forward pass, then serve replicas behind a sticky router and measure the hit-rate delta.
