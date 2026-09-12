# 11 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 7 proved you can *run* a service. Phase 8 proves you can **ship** it: reproducibly, with gates that catch what tests don't, onto hardware you never log into, and — above all — that you can **undo** it.

Almost everything here works on a laptop with `kind` and a CPU model. A GPU makes three exercises real (the arch-mismatch one, the DCGM-adjacent scheduling one, and the perf gate's absolute numbers) but changes none of the mechanics.

---

## Warm-up exercises

**Containers**

1. **The four artifacts.** For your service, write the identity of each of code / runtime / weights / config, and how each is pinned. Any row answered "whatever is on the machine" is your next task. ([1](01-why-shipping-is-the-job.md))
2. **Driver/runtime archaeology.** Record host driver version, image CUDA runtime, and `torch.cuda.get_device_capability()`. Then state, in one sentence, which of the three you can change without touching the nodes. ([2](02-containers-for-gpu-workloads.md))
3. **Break the arch.** Build with a `TORCH_CUDA_ARCH_LIST` that excludes your GPU and record the exact error text. Then include only `+PTX` for it and measure the JIT startup penalty. ([2](02-containers-for-gpu-workloads.md), [3](03-image-size-and-cold-start.md))
4. **PID 1.** Run the same server with shell-form and exec-form `ENTRYPOINT`; time `docker stop` for each with a 400-token generation in flight. Report both, and the number of truncated responses. ([2](02-containers-for-gpu-workloads.md))
5. **Layer autopsy.** `docker history` a naive build; attribute every layer over 200 MB and state whether it is removable. Then shrink the image ≥2× and show the new table. ([3](03-image-size-and-cold-start.md))

**Cold start**

6. **Five-stage timing.** Instrument image pull / container start / weight fetch / load to HBM / warmup. Produce cold-node and warm-node rows. Name the dominant stage and the specific fix for it. ([3](03-image-size-and-cold-start.md))
7. **Digest-keyed cache.** Add a node-local weight cache keyed by digest; prove pod #2 on the same node skips the fetch, and prove a rollback to a previous digest does **not** hit a stale entry. ([3](03-image-size-and-cold-start.md), [7](07-model-registry-and-artifacts.md))
8. **Warmup that matters.** Compare the first post-deploy request's TTFT with steady-state TTFT, before and after warming over your real prompt-length distribution. The delta you remove is what every deploy currently costs your users. ([3](03-image-size-and-cold-start.md))

**Kubernetes**

9. **Probe triad.** Write all three probes. Then prove each does its job: startup tolerates a slow load, readiness is false during warmup, liveness restarts a `SIGSTOP`ped engine. Most stacks fail the third. ([4](04-kubernetes-for-gpu-serving.md))
10. **Drain under load.** Fixed-QPS load test through a rolling update, counting 5xx and truncated streams — first with k8s defaults (30 s grace, no `preStop`), then with your settings. Two rows. ([4](04-kubernetes-for-gpu-serving.md), [6](06-deploying-and-rollouts.md))
11. **PDB test.** `kubectl drain` a node with and without a PodDisruptionBudget; record how many replicas were simultaneously unavailable each time. ([4](04-kubernetes-for-gpu-serving.md))
12. **The shm bug.** Run a 2-GPU (or 2-process gloo) job with default `/dev/shm` and then with an 8 Gi memory-backed `emptyDir`. Record what failure the first one produces — recognising it later is worth an hour of incident time. ([4](04-kubernetes-for-gpu-serving.md))

**Capacity and rollout**

13. **Provisioning arithmetic.** From your measured per-replica capacity at the SLO, compute serving + redundancy + surge + warm pool, and the multiplier over the naive number. ([5](05-scheduling-capacity-and-autoscaling.md))
14. **Autoscaler response curve.** Step QPS 2× and record detection time, replica ramp, and time-to-p99-recovery. Then do it again with cold start artificially doubled and explain the shape change. ([5](05-scheduling-capacity-and-autoscaling.md))
15. **Warm-pool economics.** Cost of `peak_ramp_rate × cold_start × 1.3` warm GPUs per month vs the engineering cost of halving cold start. State which you'd fund and why. ([3](03-image-size-and-cold-start.md), [5](05-scheduling-capacity-and-autoscaling.md))
16. **Surge deadlock.** Fill your cluster, then deploy with `maxSurge: 1 / maxUnavailable: 0` and watch the rollout stall. Then do it with `maxUnavailable: 25%` and measure the capacity dip. Choose a default and justify it. ([6](06-deploying-and-rollouts.md))
17. **Rollback, four ways.** Time rollback with: warm ReplicaSet + cached image; evicted image; cleared weight cache; a model-only (channel pointer) change. Four numbers. ([6](06-deploying-and-rollouts.md), [7](07-model-registry-and-artifacts.md))

**Artifacts and CI**

18. **Minimum viable registry.** Content-addressed dir + `SHA256SUMS` + `metadata.json` with lineage, eval and `runtime_requirements` + a `channels/prod` pointer. Publish twice from identical bytes and show idempotence. ([7](07-model-registry-and-artifacts.md))
19. **Fail closed.** Corrupt one byte of a shard; confirm the server aborts with a readable message rather than serving. Then flip `runtime_requirements.gpu_arch` to something wrong and confirm it also refuses. ([7](07-model-registry-and-artifacts.md))
20. **Retention audit.** Read your actual bucket/registry lifecycle rules and answer: would last month's rollback target still exist? For most teams the honest answer is no. ([7](07-model-registry-and-artifacts.md))
21. **Noise before thresholds.** Ten identical benchmark runs on `main`; compute σ for throughput and p99 TTFT; set gates at 3σ. Record both numbers in the workflow as comments. ([8](08-ci-cd-and-benchmark-gates.md))
22. **Two regressions, two gates.** Inject a 20% slowdown and a chat-template break. Show each is caught by exactly one gate and missed by the other. This is the phase's single most convincing artifact. ([8](08-ci-cd-and-benchmark-gates.md))
23. **Baseline discipline.** Deliberately regress performance 5% ten times in a row with an auto-updating baseline and plot where the baseline ends up. Then repeat with a committed, reviewed baseline. ([8](08-ci-cd-and-benchmark-gates.md))

**Infrastructure**

24. **Worst-case bill.** For each node pool: `max_size × $/hr × 730`. Put it in a code comment. If the total surprises you, your limits are wrong. ([9](09-infrastructure-as-code.md))
25. **Quota table.** Region × instance family × current quota × peak plan need. Any row where need > quota is a launch blocker. ([9](09-infrastructure-as-code.md))
26. **Drift.** Change something in the cloud console, then run `terraform plan -detailed-exitcode` and read what it wants to do. Decide whether that plan would have been safe to apply blind. ([9](09-infrastructure-as-code.md))

---

## Conceptual self-check (no notes)

1. Why can a container never contain the GPU driver, and what mechanism supplies `libcuda.so.1` at runtime? ([2](02-containers-for-gpu-workloads.md))
2. Give the exact error text you'd expect from (a) a driver too old for the image's CUDA runtime and (b) a binary missing your GPU's `sm_XX`. What is the fix for each? ([2](02-containers-for-gpu-workloads.md))
3. Name the five cold-start stages in order, and the dominant mitigation for each. ([3](03-image-size-and-cold-start.md))
4. Why must weights not be baked into the serving image, and what is the one legitimate exception? ([3](03-image-size-and-cold-start.md))
5. Why is `requests` forced to equal `limits` for `nvidia.com/gpu`, and what does that imply for HPA design? ([4](04-kubernetes-for-gpu-serving.md), [5](05-scheduling-capacity-and-autoscaling.md))
6. What does a naive `/health` liveness probe fail to detect, and what property must the probe have instead? ([4](04-kubernetes-for-gpu-serving.md))
7. Traffic still arrives after `SIGTERM`. Why, and what are the two settings that make a rolling update lossless? ([4](04-kubernetes-for-gpu-serving.md))
8. Why is GPU utilization the wrong autoscaling signal, and what two signals replace it? ([5](05-scheduling-capacity-and-autoscaling.md), [Phase 7 L3](../phase-7/03-gpu-and-host-telemetry.md))
9. Write the warm-pool sizing formula and explain each term. ([5](05-scheduling-capacity-and-autoscaling.md))
10. What does `maxSurge: 1` cost on a GPU fleet, and what happens if that cost isn't reserved? ([6](06-deploying-and-rollouts.md))
11. Name three things that make a rollback impossible, all of which are decisions made weeks earlier. ([6](06-deploying-and-rollouts.md), [7](07-model-registry-and-artifacts.md))
12. What is in a "model artifact" besides weights? Name five components and the failure each causes when it drifts. ([7](07-model-registry-and-artifacts.md))
13. Why is a compiled TensorRT-LLM engine plan keyed to more than the model? List the tuple. ([7](07-model-registry-and-artifacts.md), [Phase 5 L6](../phase-5/06-tensorrt-llm-and-compiled-engines.md))
14. Why can't a quality gate assert exact output strings, and what three cheap checks replace it? ([8](08-ci-cd-and-benchmark-gates.md))
15. Why is `assert p99 < 500ms` the wrong performance gate, and what is the right formulation? ([8](08-ci-cd-and-benchmark-gates.md))
16. Which parts of the system belong in Terraform and which do not, and what is the argument for the boundary? ([9](09-infrastructure-as-code.md))

If any answer takes more than ~60 seconds, re-read the linked lesson. Questions 6, 7, 10, 11 and 15 are asked, nearly verbatim, in senior infrastructure interviews.

---

## Exit artifact

Produce **Option A**. B is the natural extension if you have cloud access; C is cheap and disproportionately useful in design reviews.

### Option A — Project 15: the delivery pipeline (required)

`projects/15-docker-cicd-gate/README.md` containing:

- **Image table**: naive → optimized, with sizes, layer attribution, and cold-pull times.
- **Cold-start table**: five stages, cold node and warm node, before and after your fixes.
- **Drain evidence**: 5xx count and truncated-stream count through a rolling update, with k8s defaults vs your settings.
- **Probe evidence**: the `SIGSTOP` experiment showing liveness restarting a hung engine.
- **CI gates**: the workflow files, the measured benchmark σ, the derived thresholds, and logs showing regression A (slowdown) rejected by the perf gate while passing quality, and regression B (template break) rejected by the quality gate while passing perf.
- **Registry**: one published artifact with `metadata.json` (lineage + eval + runtime requirements), plus the fail-closed checksum demonstration.
- **Rollback**: four timed scenarios, and the one sentence you'd say to a hiring manager about your worst case.

### Option B — Cloud-real infrastructure

Terraform for a sandbox GPU environment: remote locked state, a tainted GPU node pool with `ignore_changes` on desired size, a versioned weights bucket with `prevent_destroy`, a registry with a retention policy matching your rollback window, a budget alert, and Infracost commenting the monthly delta on PRs. Include the `plan` output and the worst-case-cost table.

### Option C — The deployment document set

1. **Deployment runbook**: how to deploy, how to canary, how to roll back (copy-pasteable commands), and a `DO NOT` section.
2. **Capacity plan**: measured per-replica capacity at the SLO, the provisioning arithmetic, the quota table, and the purchase-mode split (committed / on-demand / spot).
3. **Artifact policy**: naming, digest addressing, promotion states, retention window, and the GC rule that reads live manifests.

Update [`projects/README.md`](../../projects/README.md) status for 15 when done, and run the [production-readiness checklist](../../playbooks/production-readiness-checklist.md) — in this phase, most of its rows finally become checkable.

---

## How you know you're ready for Phase 9

- [ ] You can state your image size, cold start (cold and warm), and rollback time from memory.
- [ ] Your rollback requires no build, and you have timed it under a cold cache.
- [ ] A rolling update under load produces zero 5xx and zero truncated streams.
- [ ] Your liveness probe has restarted a deliberately hung engine.
- [ ] Weights are digest-addressed, checksum-verified at load, and retained past your rollback window.
- [ ] Your CI has a quality gate and a performance gate, both with thresholds derived from measured noise.
- [ ] You have seen each gate catch a regression the other one missed.
- [ ] You can compute provisioned GPUs from measured capacity, and explain the multiplier over the naive number.
- [ ] Your GPU capacity has a hard ceiling in code and a forecast-based budget alert.
- [ ] Nothing in your production path was created by clicking in a console.

---

## Where these ideas come back

| Phase 8 idea | Comes back as |
|---|---|
| Multi-stage images and cold-start budgets | edge/mobile packaging, where the budget is megabytes ([Phase 9 lesson 7](../phase-9/07-edge-and-on-device.md)) |
| Execution-environment pinning (arch, runtime, plan) | portable model formats and execution providers ([Phase 9 lesson 8](../phase-9/08-model-formats-and-runtimes.md)) |
| GPU sharing via requests/limits | MIG partitioning and multi-tenant isolation ([Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md)) |
| Per-stage timeouts in probes and drains | per-stage budgets across retrieval + LLM pipelines ([Phase 9 lesson 10](../phase-9/10-rag-and-agentic-serving.md)) |
| CI benchmark gates | the engineering-standards bar every capstone must meet ([Phase 10 lesson 2](../phase-10/02-engineering-standards.md)) |
| Digest-pinned artifacts and rollback timing | reproducibility of your capstone results ([Phase 10 lesson 8](../phase-10/08-writeup-and-portfolio.md)) |
| Committed/spot purchase-mode split | the cost-optimization capstone's biggest single lever ([Phase 10 lesson 5](../phase-10/05-capstone-cost-optimization.md)) |

---

**Next:** [Phase 9 — Beyond LLMs: Recsys, Vision, Speech, Hardware Diversity, Edge, Security, RAG](../phase-9/README.md). Phases 1-8 went deep on one workload shape — big model, few QPS, long outputs, NVIDIA GPUs. Phase 9 is the rest of inference engineering: a ranking model at a million QPS in 10 ms, a video pipeline bottlenecked on JPEG decode, a streaming ASR contract, silicon that isn't NVIDIA, and the security and pipeline concerns that show up the moment your model is part of a product rather than a demo.
