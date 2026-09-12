# 1 — Why Shipping Is the Job

> **You'll be able to say:** "An inference deployment is four independent artifacts — code, CUDA/driver stack, model weights, and configuration — and an incident is usually a mismatch between two of them, not a bug in any one. The unit of deployment is an *immutable, digest-pinned* tuple of those four, and the number that matters most in this phase is not throughput: it's **time to roll back**. If that number is 30 minutes, nothing else in Phases 3-7 protects you."

Phases 3-7 made a fast, measured, observable service exist *somewhere*. This lesson is about the gap between "it works on the rented H100 I `ssh`'d into" and "it runs, identically, on 40 machines I have never logged into, and the version that ran last Tuesday can be back in production in 90 seconds."

That gap is where most ML systems actually fail, and it is why this is the phase that gets you paid rather than the phase that gets you upvotes.

---

## The four artifacts

Every running inference replica is the composition of four things that version *independently*:

```
   ┌──────────────────────────────────────────────────────────────┐
   │ 1. CODE          your server, router, pre/post-processing     │  git sha
   │ 2. RUNTIME       python, torch, CUDA runtime, cuDNN, NCCL,    │  image digest
   │                  flash-attn, vLLM/TRT-LLM, the driver ABI     │
   │ 3. WEIGHTS       the checkpoint, tokenizer, quantization      │  artifact digest
   │                  scales, LoRA adapters, compiled engine plan  │
   │ 4. CONFIG        max_num_seqs, gpu_memory_utilization, TP     │  config hash
   │                  size, sampling defaults, prompt template     │
   └──────────────────────────────────────────────────────────────┘
```

**The single most useful mental habit in this phase:** when something breaks in production, ask *which of the four changed* before asking *what is wrong with the code*. In practice the failure distribution looks roughly like this:

| Failing pair | Typical symptom | Real example |
|---|---|---|
| Runtime × driver | `CUDA error: no kernel image is available for execution` | image built for `sm_90`, scheduled onto an A100 (`sm_80`) node |
| Runtime × weights | garbage output, or `size mismatch for lm_head.weight` | tokenizer updated in the registry, model image not rebuilt |
| Weights × config | OOM at some prompt length that used to work | `max_model_len` raised without recomputing the KV budget ([Phase 6 lesson 1](../phase-6/01-when-one-gpu-isnt-enough.md)) |
| Config × code | prompt template drift → quality regression with 200 OK | chat template moved from server to client, both apply it |
| Runtime × hardware | 30% slower with no code change | new base image ships a different flash-attn wheel |

Three of those five are invisible to unit tests and visible only to the CI gates in [lesson 8](08-ci-cd-and-benchmark-gates.md).

---

## Why ML deployment is genuinely harder than web deployment

It's fashionable to say "just treat it like any other service." That is 80% right and the remaining 20% is what breaks people:

| Property | Typical web service | Inference service |
|---|---|---|
| Image size | 50-200 MB | **5-15 GB** (CUDA + torch + kernels) |
| Extra artifact | none | **weights: 15 GB - 1 TB**, fetched at start |
| Cold start | 1-5 s | **1-15 min** (pull + load + warmup + compile) |
| Scheduling unit | fraction of a CPU | **whole GPUs**, non-shareable by default |
| Replica cost | ~$0.01/hr | **$2-40/hr** — idle headroom is real money |
| Correctness gate | tests pass | tests pass **and output quality hasn't moved** |
| Rollback | swap image tag | swap image **and** weights **and** engine plan, coherently |
| Failure of one node | drop it | may kill the whole **gang** (TP/PP group, [Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)) |
| Hardware coupling | none | driver ABI, GPU arch, NVLink topology, MIG profile |

Every one of those rows turns into a concrete engineering decision later in this phase. The cold-start row alone drives lesson 3, half of lesson 4, and the autoscaling mechanics in [lesson 5](05-scheduling-capacity-and-autoscaling.md).

---

## The reproducibility contract

The goal of this phase, stated as one testable property:

> **Given a git sha, any engineer on the team can produce a byte-identical deployment on a machine they have never touched, and can revert to the previously running deployment without a build.**

That decomposes into four rules that everything else in this phase implements:

1. **Immutable artifacts, addressed by digest.** `myrepo/server:v1.4.2` is a *label*; `myrepo/server@sha256:9f3a…` is an *identity*. Tags are mutable — `latest` is a bug, and a floating `v1.4` tag means your rollback target can change under you. Pin by digest in the manifest ([lesson 7](07-model-registry-and-artifacts.md)).
2. **Config is data, not code.** Anything you might change at 3 a.m. — batch caps, admission thresholds, timeouts, the degradation rung — is config with a default, not an `if` in a Python file that requires a build. But config is still *versioned and reviewed*: a `ConfigMap` edited by hand is an unrecorded deploy.
3. **Weights are an artifact with a lifecycle**, not a `wget` in the entrypoint. Checksums, retention, and a rollback window that outlives the rollout ([lesson 7](07-model-registry-and-artifacts.md)).
4. **Every promotion passes the same gates.** Build → test → quality → benchmark → canary. The gates are code in the repo, not steps in someone's head ([lesson 8](08-ci-cd-and-benchmark-gates.md), and the canary machinery you already built in [Phase 7 lesson 8](../phase-7/08-canary-and-shadow-traffic.md)).

### The metric this phase optimizes

Borrowing the DORA framing, but with the inference-specific numbers that matter:

| Metric | What it means here | A good target |
|---|---|---|
| **Time to rollback** | alert fires → previous version serving 100% | **< 5 min**, and it must not require a build |
| Lead time for change | merge → production | hours, not weeks |
| Deploy frequency | how often you can safely ship | at least weekly; daily if gates are real |
| Change failure rate | % of deploys that need rollback | < 15%, measured, not guessed |
| **Cold start** | pod scheduled → serving first token | < 3 min ([lesson 3](03-image-size-and-cold-start.md)) |
| **Capacity to deploy** | spare GPUs needed for a surge rollout | budgeted explicitly, not discovered ([lesson 6](06-deploying-and-rollouts.md)) |

Time to rollback is first on purpose. Every other reliability investment assumes you can undo. A team that can revert in 60 seconds can afford to ship aggressively; a team that needs a 25-minute rebuild will instead spend that budget on review meetings, and ship *less safely* as a result.

---

## What "MLOps" means in this repo

The word is overloaded to the point of uselessness — it's used for feature stores, experiment tracking, drift monitoring, and orchestration DAGs. **This phase deliberately covers only the deployment substrate for inference**, because that is what an inference engineer owns:

| In scope here | Out of scope (a different job) |
|---|---|
| Container images for GPU workloads | training pipeline orchestration (Airflow/Kubeflow) |
| Kubernetes GPU scheduling and probes | experiment tracking, hyperparameter search |
| Rollouts, rollback, progressive delivery | data versioning / feature stores |
| Model + weight artifact registry | labeling and dataset curation |
| CI with quality and benchmark gates | drift detection and retraining triggers |
| Terraform for GPU capacity | the model architecture itself |

If your target role is "ML platform engineer" the right-hand column becomes yours too; [`resources/README.md`](../../resources/README.md) points at where to go for it. For inference specifically, the left column is the whole job.

---

## The honest cost of skipping this phase

A concrete, extremely common failure story, told as a timeline — every step of it is preventable by one lesson in this phase:

```
  14:02  merge "bump vllm 0.5.1 → 0.6.0" ; CI runs pytest, green ; image :latest pushed
  14:09  rolling update begins. maxUnavailable: 25%, no surge headroom reserved
  14:11  new pods Pending — the cluster has no spare GPUs                    [L5, L6]
  14:13  old pods already terminated; capacity halved; queue depth climbs    [L6]
  14:15  first new pod runs; pulls a 14 GB image over a shared NAT gateway   [L3]
  14:23  container starts, downloads 16 GB of weights from S3 per pod        [L7]
  14:31  ready. TTFT p99 is 2.4× baseline — 0.6.0 changed a default          [L8]
  14:33  page fires (burn rate 14.4×)                                        [Phase 7 L5]
  14:35  "roll back" → :latest was overwritten; the old digest is unknown    [L7]
  14:52  someone finds the sha in a Slack scrollback; rebuild starts
  15:20  service restored.  78 minutes, entirely self-inflicted.
```

Nothing in that story is an ML problem. It is four missing engineering practices, each of which costs an afternoon to put in place — and this phase is those four afternoons.

---

## Do this now (30 minutes)

1. **Inventory your own four artifacts.** For the server you built in Phases 3/7, write down exactly what identifies each of code, runtime, weights, config today. Any row whose answer is "whatever was on the machine" is a future incident; mark it.
2. **Time your rollback.** With a stopwatch, from a running server: revert to the previous version and serve a request. Do not skip steps you'd have to do for real. Record the number — you will re-measure it at the end of this phase, and the delta is the phase's value in one figure.
3. **Write your CUDA reality down**: `nvidia-smi` (driver + CUDA driver version), `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`. Those three lines are the compatibility constraint that [lesson 2](02-containers-for-gpu-workloads.md) is entirely about.

---

**Next:** [Containers for GPU workloads →](02-containers-for-gpu-workloads.md) — the driver/runtime split that decides whether your image runs on someone else's GPU, and the `nvidia-container-toolkit` mechanics underneath `--gpus all`.
