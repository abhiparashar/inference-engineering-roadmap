# 4 — Kubernetes for GPU Serving

> **You'll be able to say:** "GPUs are an *extended resource* surfaced by the NVIDIA device plugin: integer-only, non-overcommittable, `requests` must equal `limits`, and a pod either gets whole GPUs or stays `Pending`. I write a serving pod spec from memory — node selector for the GPU type, shm sizing for NCCL, a readiness probe gated on warmup, a liveness probe that cannot be satisfied by a hung engine, `preStop` plus a generous `terminationGracePeriodSeconds` so streams drain, and a PodDisruptionBudget so a node drain doesn't take the service with it."

[Lesson 3](03-image-size-and-cold-start.md) got the container to start fast. This lesson is how a scheduler that was designed for stateless 200 MB web pods handles a stateful 15 GB process that owns a $30,000 accelerator.

You do not need to be a Kubernetes expert to be an excellent inference engineer. You *do* need to own the ~60 lines of YAML that decide whether your service drains cleanly, gets scheduled on the right silicon, and is detected as dead when it hangs.

---

## How a GPU becomes schedulable

```
  NVIDIA GPU Operator (or manual install) on each GPU node
      ├── driver (or pre-installed on the node image)
      ├── nvidia-container-toolkit          → containers can see the GPU
      ├── DEVICE PLUGIN (DaemonSet)         → advertises  nvidia.com/gpu: 8
      ├── GPU Feature Discovery             → node labels: product, memory, driver
      └── DCGM exporter                     → Prometheus metrics ([Phase 7 L3](../phase-7/03-gpu-and-host-telemetry.md))

  kubelet reports:  Capacity: nvidia.com/gpu: 8
  scheduler:        places a pod requesting nvidia.com/gpu: 1 onto a node with a free one
  device plugin:    sets NVIDIA_VISIBLE_DEVICES=GPU-<uuid> in the container
```

The rules that follow from GPUs being an **extended resource**:

| Rule | Consequence |
|---|---|
| Integer values only | no `nvidia.com/gpu: 0.5`. Sharing needs MIG or time-slicing ([Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md)) |
| `requests` must equal `limits` | no burst, no overcommit. Omitting `requests` and setting only `limits` works, but write both |
| No fractional autoscaling signal | HPA on GPU count is meaningless; scale on queue depth / KV occupancy ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md)) |
| A pod without the resource request sees **no GPU** | unless someone set `NVIDIA_VISIBLE_DEVICES=all` manually — which double-books GPUs |
| Node labels carry the hardware truth | `nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3`, `.memory`, `.count`, driver version |

**Taint your GPU nodes** (`nvidia.com/gpu=present:NoSchedule`) so CPU workloads can't land on them and fragment your capacity; then tolerate the taint in the serving pod. A $30/hr node running a logging sidecar because someone forgot a `nodeSelector` is a real and recurring cost bug.

---

## A serving Deployment worth copying

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  labels: { app: llm-inference }
spec:
  replicas: 4
  revisionHistoryLimit: 5          # keeps rollback targets around
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }   # see lesson 6
  selector:
    matchLabels: { app: llm-inference }
  template:
    metadata:
      labels: { app: llm-inference, version: v1-4-2 }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      terminationGracePeriodSeconds: 180      # long generations must finish
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      # spread replicas across nodes/zones so one node loss ≠ outage
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector: { matchLabels: { app: llm-inference } }
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
      containers:
        - name: server
          image: registry.example.com/llm-server@sha256:9f3a...    # digest, not tag
          imagePullPolicy: IfNotPresent
          args: ["--model=/weights/llama-3.1-8b", "--max-num-seqs=64",
                 "--gpu-memory-utilization=0.90"]
          ports:
            - { name: http, containerPort: 8000 }
          resources:
            limits:
              nvidia.com/gpu: 1
              cpu: "8"                # tokenization + sampling are CPU work
              memory: 48Gi            # host RAM for weight load, not HBM
            requests:
              nvidia.com/gpu: 1
              cpu: "8"
              memory: 48Gi
          env:
            - { name: HF_HOME,   value: /cache/hf }
            - { name: POD_NAME,  valueFrom: { fieldRef: { fieldPath: metadata.name } } }
          volumeMounts:
            - { name: weights,  mountPath: /weights, readOnly: true }
            - { name: cache,    mountPath: /cache }
            - { name: dshm,     mountPath: /dev/shm }
          startupProbe:                 # tolerates a slow cold start
            httpGet: { path: /health/live, port: http }
            periodSeconds: 10
            failureThreshold: 60        # up to 10 minutes to come up
          readinessProbe:               # gated on warmup completion
            httpGet: { path: /health/ready, port: http }
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:                # must detect a HUNG engine, not just a dead process
            httpGet: { path: /health/live, port: http }
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 4
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 10"]   # let endpoints propagate
      volumes:
        - name: weights
          hostPath: { path: /mnt/nvme/weights, type: Directory }   # node-local cache
        - name: cache
          emptyDir: {}
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 8Gi }             # NCCL/IPC needs this
```

---

## The three probes, and the one everyone gets wrong

| Probe | Question it answers | Failure action | Inference-specific gotcha |
|---|---|---|---|
| `startupProbe` | "has it finished booting?" | restart if it never comes up | **required** — without it, a liveness probe kills the pod mid-model-load. `failureThreshold × periodSeconds` must exceed your worst cold start ([lesson 3](03-image-size-and-cold-start.md)) |
| `readinessProbe` | "should it receive traffic?" | remove from Service endpoints | must be **false until warmup completes**, and should go false under overload/shedding — not just on crash |
| `livenessProbe` | "is it beyond repair?" | kill the container | the dangerous one |

**The liveness trap:** a naive `/health` handler returns 200 from the HTTP event loop, which keeps answering happily while the CUDA stream is deadlocked, an NCCL collective is hung, or the engine loop has died. This is exactly the "alive and producing nothing" failure from [Phase 7 lesson 7](../phase-7/07-reliability-and-degradation.md) and [Phase 6 lesson 9](../phase-6/09-multi-node-operations.md).

Make liveness depend on **forward progress**:

```python
# the engine loop stamps this every iteration
LAST_STEP = time.monotonic()

@app.get("/health/live")
def live():
    idle = ENGINE.num_running == 0 and ENGINE.num_waiting == 0
    stalled = (time.monotonic() - LAST_STEP) > STALL_TIMEOUT_S   # e.g. 60s
    if stalled and not idle:          # work present but no steps → hung
        return Response(status_code=503)
    return Response(status_code=200)
```

Two properties matter: it is **false when there is work and no progress**, and it is **true when idle** (no traffic must never look like a hang). Test it with `kill -STOP` on the engine process — the chaos experiment from [Phase 7 lesson 10](../phase-7/10-incident-response-and-chaos.md). Most stacks fail this test the first time.

The other half is readiness under load: returning `503` from readiness when the queue is beyond your admission threshold removes the replica from the load balancer instead of accepting work it can't serve. Use it carefully — if *every* replica does this simultaneously you have removed the whole service. Bound it: never let readiness-based shedding drop below N healthy replicas, and prefer 429 at the router ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)).

---

## Graceful shutdown: the deploy that truncates every stream

The sequence Kubernetes actually performs on pod deletion is **concurrent**, not ordered, and that is the source of the bug:

```
   t=0    pod marked Terminating
          ├── endpoint removal propagates to kube-proxy/ingress …  (ASYNC, 1-10 s!)
          └── preStop hook runs, then SIGTERM to PID 1             (immediate)
   t=grace  SIGKILL
```

Traffic can still arrive **after** `SIGTERM` because endpoint propagation is eventually consistent. Hence:

1. **`preStop: sleep 5-15 s`** — absorb in-flight routing while still serving. Crude, and correct.
2. **On `SIGTERM`: stop accepting, keep serving.** Flip readiness false, drain the in-flight set, then exit. Never `sys.exit()` on the signal.
3. **`terminationGracePeriodSeconds` ≥ preStop + your longest realistic generation.** A 2000-token response at 40 tok/s is 50 s; the default 30 s guarantees truncation on every deploy. 120-180 s is normal here.
4. **Cap the drain**: if a stream exceeds the budget, end it with a proper finish reason rather than being `SIGKILL`ed mid-token.

Verify by load-testing *through* a rolling update: with correct settings you see zero 5xx and zero truncated streams; with the defaults you'll see both, on every deploy, forever, and blame it on the model.

---

## Multi-GPU and multi-node pods

Single-node TP is easy: request `nvidia.com/gpu: 8` in **one** pod and let the engine use all of them. Note the shm requirement — the default 64 MB `/dev/shm` breaks NCCL and torch dataloaders; the `emptyDir{medium: Memory}` above fixes it.

Multi-node TP/PP is where plain `Deployment` stops being adequate, because you need **gang semantics**: all ranks start together, all ranks die together, and ranks need stable identities.

| Need | Mechanism |
|---|---|
| All-or-nothing scheduling | gang scheduling: Kueue, Volcano, or Coscheduling plugin — without it, 7 of 8 pods run and burn money waiting for the 8th |
| Stable rank identity/DNS | `StatefulSet`, or **LeaderWorkerSet** (purpose-built for multi-node inference) |
| One rank dies → restart the group | LWS restart policy, or a supervisor; a lone surviving rank is a hung collective ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)) |
| High-speed fabric | RDMA/InfiniBand via multus + RDMA device plugin, `IPC_LOCK` capability, hugepages |
| Placement in one NVLink domain / rack | topology-aware scheduling, node affinity on a rack/zone label |

**LeaderWorkerSet** is the current answer for "a replica is N pods." Treat the whole group as the scaling unit: your HPA scales groups, your PDB counts groups, your readiness is group readiness.

---

## Disruption, drains, and the PDB

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: llm-inference }
spec:
  minAvailable: 3                     # or maxUnavailable: 1
  selector: { matchLabels: { app: llm-inference } }
```

A PDB constrains **voluntary** disruptions — node drains, cluster upgrades, autoscaler consolidation. Without it, a routine node upgrade can evict three of your four replicas at once. With it, the drain waits.

What a PDB does **not** protect against: node hardware failure, preemption of spot instances, OOM kills, or `SIGKILL`. For spot, you get a termination notice (2 min on AWS, 30 s on GCP) — wire it to the same drain path as `SIGTERM`, and treat 30 seconds as "finish what you can, reject the rest," because a 90-second generation cannot be saved.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Pod `Pending`, `0/12 nodes available: insufficient nvidia.com/gpu` | no free whole GPU; often a surge rollout with no headroom | reserve surge capacity ([lesson 6](06-deploying-and-rollouts.md)), or `maxSurge: 0` with an extra replica |
| Pod restarts every few minutes during startup | liveness/readiness firing during model load | add a `startupProbe` sized to the real cold start |
| Every deploy truncates user streams | default 30 s grace period, no `preStop` | grace 120-180 s, `preStop` sleep, drain on `SIGTERM` |
| 5xx spike for ~10 s after each pod terminates | endpoint propagation lag | `preStop` sleep ≥ propagation time |
| Replica alive, serving nothing, never restarted | liveness probe answered by the HTTP layer | progress-based liveness (above) |
| NCCL init hangs in a multi-GPU pod | 64 MB `/dev/shm` | `emptyDir{medium: Memory}` at `/dev/shm` |
| 7 pods running, 1 `Pending`, all idle | no gang scheduling | Kueue/Volcano/LWS |
| Two pods on one GPU, both slow | manual `NVIDIA_VISIBLE_DEVICES`, or a missing resource request | always request `nvidia.com/gpu` |
| Works on one node type, `no kernel image` on another | mixed GPU archs, no selector | `nodeSelector` on `nvidia.com/gpu.product`, build for both archs |
| Node drain takes out the service | no PDB | add one; test with `kubectl drain` |
| OOMKilled during weight load | container memory limit below host-RAM peak | raise memory limit; `safetensors` mmap lowers the peak |

---

## Do this now (60 minutes)

1. **Write the Deployment for your own server**, including all three probes, `preStop`, a grace period sized to your longest generation, and a PDB. Run it on `kind`/`minikube` (CPU model is fine — every setting above except the GPU resource works there).
2. **Prove the liveness probe catches a hang**: `kubectl exec` in and `kill -STOP` the engine thread/process while requests are in flight. If the pod is not restarted within your stall timeout, your probe is decorative. Fix it and re-run.
3. **Load test through a rolling update** at fixed QPS and count 5xx + truncated streams, first with the k8s defaults and then with your settings. That before/after pair is a strong artifact and, in an interview, an unusually concrete answer to "how do you deploy safely?"

---

**Next:** [Scheduling, capacity, and autoscaling mechanics →](05-scheduling-capacity-and-autoscaling.md) — turning the Phase-6 autoscaling *policy* into HPA/KEDA/Karpenter configuration that survives minute-scale cold starts, spot preemption, and quota limits.
