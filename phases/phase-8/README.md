# Phase 8 — MLOps Glue: Containers, Orchestration, CI/CD, IaC (Deep Dive)

> **Goal:** turn a service that works on a machine you `ssh` into, into a service that ships. By the end of this phase you can build a small GPU-aware image, cut a multi-minute cold start into stages you can attack individually, write the Kubernetes spec that drains cleanly and detects a hung engine, roll out behind automated gates, address weights as immutable digest-pinned artifacts, gate merges on quality *and* performance evidence, provision GPU capacity in Terraform with cost guardrails — and roll the whole thing back in under five minutes without a build.

This folder is the long-form version of [Phase 8 in the ROADMAP](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac). Phase 7 told you whether the service is healthy and what it costs; Phase 8 is the machinery that puts it there and takes it away again.

The intellectual core:

> **A deployment is four independently-versioned artifacts — code, runtime, weights, config — and most production incidents are a mismatch between two of them rather than a bug in any one. The engineering answer is immutability plus evidence: pin every artifact by digest, and let nothing be promoted without passing a gate that a human didn't have to remember to run. The metric that measures whether you got this right is not throughput. It is time to rollback.**

---

## Prerequisites

- **[Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)** — honest benchmarking. The CI performance gate is that harness, automated; its thresholds come from measured variance, and a gate built on a dishonest benchmark is worse than none.
- **[Phase 4 lesson 8](../phase-4/08-compilation-and-kernels.md)** — `torch.compile` and CUDA graphs. Both are cold-start costs that you must cache or pay for on every pod start.
- **[Phase 5 lesson 6](../phase-5/06-tensorrt-llm-and-compiled-engines.md)** — compiled engines. An engine plan is an artifact keyed to (arch × library × TP × shapes), which is why lesson 7 exists.
- **[Phase 6 lessons 8-9](../phase-6/08-autoscaling-gpu-fleets.md)** — autoscaling policy and multi-node failure modes. Lesson 5 here is the configuration that implements that policy; lesson 4 is the gang-scheduling mechanics.
- **[Phase 7 lessons 5, 7, 8](../phase-7/05-slos-and-error-budgets.md)** — SLOs, degradation, canary analysis. The rollout gates in lesson 6 are literally Prometheus queries against the SLIs you defined there.

### About hardware

- **Most of this phase runs on a laptop.** `kind` or `minikube` with a CPU model exercises every probe, drain, rollout, gate, and rollback mechanic in lessons 4-8. The YAML is identical; only the `nvidia.com/gpu` resource line is inert.
- **A GPU makes three things real**: the driver/arch compatibility failures in lesson 2, cold-start stage timings in lesson 3, and absolute benchmark numbers in the lesson 8 gate. One rented GPU-hour covers all three.
- **A cloud sandbox** (any provider, a few dollars) is what lesson 9 wants. `terraform plan` alone teaches most of it if you'd rather not spend.

---

## The map of this phase

```
                       CAN YOU SHIP IT, AND UNSHIP IT?
                                    │
   ┌───────────────┬────────────────┼────────────────┬─────────────────┐
   ▼               ▼                ▼                ▼                 ▼
 PACKAGE       SCHEDULE          PROMOTE          IDENTIFY          PROVISION
 lessons 2-3   lessons 4-5       lessons 6, 8     lesson 7          lesson 9
 image, CUDA   device plugin,    rollouts,        digest-addressed  terraform,
 stack, cold   probes, drain,    canary gates,    artifacts,        quota, spot,
 start budget  autoscaling       CI evidence      lineage, GC       cost limits
   │               │                │                │                 │
   └───────────────┴────────┬───────┴────────────────┴─────────────────┘
                            ▼
                    lesson 10: build it
                    acceptance = a timed rollback and
                    two regressions caught by two different gates
  ─────────────────────────────────────────────────────────────────────────────
  THE TWO THINGS THAT MAKE GPU DELIVERY DIFFERENT FROM WEB DELIVERY
    1. Cold start is MINUTES, not seconds  ⇒ autoscaling can't react inside a
                                             spike; you buy warm capacity or you
                                             shrink the five startup stages
    2. Capacity is scarce and expensive    ⇒ maxSurge: 1 means a real GPU must
                                             exist; blue-green means paying twice
```

Two conclusions most engineers reach only after an incident:

1. **Your rollback is only as good as your artifact retention.** Every "we can roll back" claim is really a claim about registry lifecycle rules, mutable tags, and node caches — all decided weeks before the incident, usually by a default ([lesson 7](07-model-registry-and-artifacts.md)).
2. **Tests do not catch model regressions; gates do.** A chat-template break passes every unit test, every smoke test, and every latency metric. Only a distributional quality check sees it ([lesson 8](08-ci-cd-and-benchmark-gates.md)), which is why [lesson 10](10-build-container-and-cicd-gate.md)'s acceptance test injects exactly that.

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [Why shipping is the job](01-why-shipping-is-the-job.md) | "A deployment is code × runtime × weights × config, and I diagnose incidents by asking which pair mismatched. The metric I optimize in this phase is time to rollback." |
| 2 | [Containers for GPU workloads](02-containers-for-gpu-workloads.md) | "The driver is on the host and is injected by the toolkit; the image carries the runtime. I know the compatibility rules, the base-image ladder, and why shell-form `ENTRYPOINT` truncates every stream on deploy." |
| 3 | [Image size and cold start](03-image-size-and-cold-start.md) | "Cold start is five measurable stages. I know which one dominates for my service, and the fix is different for each — and weights never go in the image." |
| 4 | [Kubernetes for GPU serving](04-kubernetes-for-gpu-serving.md) | "GPUs are integer, non-overcommittable extended resources. I write the probes — including a liveness probe that catches a hung engine — plus `preStop`, grace period, PDB, and shm sizing, from memory." |
| 5 | [Scheduling, capacity, and autoscaling](05-scheduling-capacity-and-autoscaling.md) | "Two loops, two time constants. I scale on backlog not utilization, fast up and slow down, and I size a warm pool as `ramp_rate × cold_start` because the node loop can't react inside a spike." |
| 6 | [Deploying and rollouts](06-deploying-and-rollouts.md) | "Rolling, blue-green and canary priced in GPUs. Progressive delivery with SLO queries as gates, digest-pinned manifests, expand/contract for anything stateful, and a rollback I have timed." |
| 7 | [Model registry and artifacts](07-model-registry-and-artifacts.md) | "A model is weights plus tokenizer plus template plus quant metadata plus maybe an arch-pinned engine plan. Content-addressed, checksum-verified, lineage-recorded, retained past the rollback window." |
| 8 | [CI/CD and benchmark gates](08-ci-cd-and-benchmark-gates.md) | "Four gate classes. Quality gates compare distributions, never strings. Performance gates need an exclusive runner, pinned clocks, and a threshold at 3σ of measured noise, expressed relative to a committed baseline." |
| 9 | [Infrastructure as code](09-infrastructure-as-code.md) | "Remote locked state, modules shared across environments, `ignore_changes` on desired size, `prevent_destroy` on the weights bucket, hard `max_size` as a budget control, and quota as a launch prerequisite." |
| 10 | [Build: container, CI gate, deploy](10-build-container-and-cicd-gate.md) | "I built the whole path and proved it: two injected regressions caught by two different gates, zero dropped streams through a rolling update, and four timed rollback scenarios." |
| 11 | [Exercises & exit artifact](11-exercises-and-artifacts.md) | "Here are the image, cold-start, drain, and rollback tables, the gate logs, and the artifact policy." |

---

## How to work through this phase

1. **Fix the server before the YAML.** Graceful `SIGTERM` drain, warmup-gated readiness, progress-based liveness ([lesson 10](10-build-container-and-cicd-gate.md) step 1). Every deployment mechanic downstream assumes these exist, and no amount of Kubernetes configuration compensates for their absence.
2. **Measure before optimizing.** The five cold-start stages and the ten-run benchmark σ are the two measurements this phase is built on. Teams routinely spend a week shrinking an image when weight fetch was 70% of startup.
3. **Test every mechanism by breaking it.** `SIGSTOP` the engine. Corrupt a weight byte. Delete the cached image before a rollback. Fill the cluster before a surge rollout. Each takes minutes and each finds a real defect.
4. **Build gates you'd trust at 4 a.m.** A flaky gate gets disabled within a month, so derive thresholds from measured noise and make failures readable — a PR comment with a table, not an exit code.
5. **Time the rollback under hostile conditions**, not the happy path. The happy-path number is marketing; the cold-cache number is the truth.
6. **Keep `kind` in the loop.** Being able to run the whole rollout locally in 90 seconds is what makes you iterate on this instead of avoiding it.

**Time budget:** 2-3 weeks part-time. Lessons 3, 4, 6 and 8 are load-bearing; lesson 10 produces the artifact.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Explain **why a container cannot contain the GPU driver**, and what supplies `libcuda.so.1` at runtime. ([lesson 2](02-containers-for-gpu-workloads.md))
2. Name the **five cold-start stages** and the dominant fix for each. ([lesson 3](03-image-size-and-cold-start.md))
3. Describe the **liveness probe that catches an engine that is alive and producing nothing**, and why the naive one doesn't. ([lesson 4](04-kubernetes-for-gpu-serving.md))
4. State the **two settings that make a rolling update lossless** for streaming responses. ([lessons 4](04-kubernetes-for-gpu-serving.md), [6](06-deploying-and-rollouts.md))
5. Price **rolling vs blue-green vs canary in GPUs** for a 20-replica service, and say what `maxSurge: 1` requires. ([lesson 6](06-deploying-and-rollouts.md))
6. List **what is in a model artifact besides weights**, and the failure each component causes when it silently drifts. ([lesson 7](07-model-registry-and-artifacts.md))
7. Explain why **`assert p99 < 500ms` is the wrong CI gate** and write the right formulation. ([lesson 8](08-ci-cd-and-benchmark-gates.md))
8. Name **three decisions made weeks earlier that make a rollback impossible.** ([lessons 6](06-deploying-and-rollouts.md), [7](07-model-registry-and-artifacts.md))

## Projects that belong to this phase

- **[15 — Multi-stage Dockerfile + GPU-aware CI/CD benchmark gate](../../projects/README.md)** (small→large): the optimized image, instrumented cold start, k8s manifests that drain cleanly, a digest-addressed artifact with lineage, CI quality + performance gates that each catch a regression the other misses, and four timed rollback scenarios. **This is the Phase 8 exit artifact.**

Also do the [Phase 8 labs](../../labs/README.md#phase-8-lab--mlops) — the image-size comparison, the GPU pod spec, and the failing-benchmark workflow stub pair directly with lessons 3, 4 and 8.

---

Next after this: **[Phase 9 — Beyond LLMs](../phase-9/README.md)**. Everything so far assumed one workload shape: a large language model, tens of QPS, long streaming outputs, NVIDIA GPUs. Phase 9 is the rest of the field — ranking at a million QPS in 10 ms, vision pipelines bottlenecked on decode rather than the model, streaming speech contracts, TPUs and Inferentia and CPUs and phones, and the security, tenancy and pipeline concerns that arrive the moment inference is part of a product.
