# 2 — Containers for GPU Workloads

> **You'll be able to say:** "A container does not contain the GPU driver — it never can, because the kernel module lives on the host. `nvidia-container-toolkit` injects the host's user-mode driver libraries into the container at start, which is why the only hard compatibility rule is *host driver ≥ the CUDA runtime version in the image* (relaxed by forward compatibility packages and CUDA minor-version compatibility). I know the difference between `base`, `runtime` and `devel` images, why `--gpus all` needs a runtime hook, and why an image built for `sm_90` dies on an A100."

[Lesson 1](01-why-shipping-is-the-job.md) said the runtime is one of four artifacts. This lesson is what is actually inside it, because the CUDA stack is the one part of your image that is *not* self-contained, and nearly every "works on my machine" GPU bug traces to that fact.

---

## The layer diagram you must have in your head

```
  ┌───────────────────────────────────────────────────────────────┐
  │ CONTAINER                                                     │
  │   your code, python, torch, vLLM, flash-attn                  │
  │   CUDA *runtime* (libcudart.so), cuDNN, cuBLAS, NCCL          │  ← in the image
  │   ─────────────────────────────────────────────────────────   │
  │   libcuda.so.1, libnvidia-ml.so, nvidia-smi   (INJECTED)      │  ← mounted in at
  └───────────────────────────────────────────────────────────────┘     runtime by the
  ┌───────────────────────────────────────────────────────────────┐     toolkit; NOT
  │ HOST: nvidia.ko kernel module + user-mode driver (e.g. 550.x) │     built by you
  │       /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm           │
  └───────────────────────────────────────────────────────────────┘
  ┌───────────────────────────────────────────────────────────────┐
  │ GPU: compute capability sm_80 (A100) / sm_89 (L40S) /         │
  │      sm_90 (H100) / sm_100 (B200)                             │
  └───────────────────────────────────────────────────────────────┘
```

Three consequences, and they explain most GPU container failures:

1. **Never install a driver in your image.** If a Dockerfile of yours runs `apt install nvidia-driver-*`, it is wrong. The image carries the *runtime*; the host carries the *driver*.
2. **The image is not portable across driver-too-old hosts.** CUDA runtime 12.4 in the image needs a host driver that supports 12.x. This is the single most common scheduling failure in a heterogeneous cluster — and the fix is a node label plus a `nodeSelector` ([lesson 4](04-kubernetes-for-gpu-serving.md)), not a rebuild at 3 a.m.
3. **The image *is* portable across GPU architectures only if it was compiled for them.** Which brings us to the second class of failure.

### Compatibility rules, concretely

| Rule | Statement | What breaks if violated |
|---|---|---|
| Driver ≥ runtime | host driver must support the image's CUDA runtime major version | `CUDA driver version is insufficient for CUDA runtime version` at init |
| Minor-version compat | within CUDA 12.x, a 12.0-capable driver generally runs 12.y runtimes | mostly works; exceptions when new driver APIs are used |
| Forward compat | `cuda-compat-12-x` package lets a *newer* runtime run on an older datacenter driver | only supported on datacenter GPUs/drivers; a real escape hatch when you can't touch nodes |
| Arch coverage | the binary must contain SASS for your `sm_XX`, or PTX that can be JIT'd | `no kernel image is available for execution on the device` |
| PTX JIT | PTX in the fatbin is JIT-compiled to unseen newer archs | works, but adds seconds-to-minutes of startup time — a cold-start tax ([lesson 3](03-image-size-and-cold-start.md)) |

```dockerfile
# Building for exactly the fleet you own — smaller and faster than "everything"
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX"
#                          A100 L40S  H100 + PTX fallback for anything newer
```

Every extra arch multiplies compile time and image size. `+PTX` on the *newest* arch you list is the cheap insurance policy: an unknown future GPU JITs instead of failing.

---

## How `--gpus all` actually works

`docker run --gpus all` is not a Docker feature that talks to the GPU. It is a hook:

```
  docker run --gpus all ...
      │
      ▼
  containerd/runc  ── OCI prestart hook ──►  nvidia-container-runtime-hook
                                                  │
                                                  ├─ reads NVIDIA_VISIBLE_DEVICES
                                                  ├─ bind-mounts /dev/nvidia*
                                                  ├─ mounts host libcuda.so.1 etc.
                                                  └─ sets ldcache inside the container
```

Two environment variables control it, and both are set for you by the CUDA base images:

- `NVIDIA_VISIBLE_DEVICES=all | 0,1 | GPU-<uuid> | none`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (add `video` for NVDEC/NVENC — needed for the vision/video pipelines in [Phase 9 lesson 3](../phase-9/03-vision-serving.md), and a classic "why is hardware decode not working" bug)

On Kubernetes you do not set these yourself: the **NVIDIA device plugin** sets `NVIDIA_VISIBLE_DEVICES` to the GPUs it allocated, in response to a `nvidia.com/gpu: 1` resource request ([lesson 4](04-kubernetes-for-gpu-serving.md)). Setting the variable manually *bypasses* the scheduler's accounting and is how you end up with two pods fighting over one GPU.

**Smoke test any GPU host or node image in one command:**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that prints a GPU table, the host stack is correct and any later failure is yours.

---

## Choosing a base image

| Base | Contains | Size (approx) | Use for |
|---|---|---|---|
| `nvidia/cuda:X.Y-base` | minimal CUDA runtime deps | ~200 MB | you install everything else; smallest runtime footprint |
| `nvidia/cuda:X.Y-runtime` | + cuBLAS, cuFFT, NCCL runtime libs | ~2 GB | **serving images** that don't compile anything |
| `nvidia/cuda:X.Y-devel` | + `nvcc`, headers, static libs | ~6 GB | **build stage only** — compiling custom kernels / flash-attn |
| `pytorch/pytorch:*-cuda*` | torch + CUDA preinstalled | ~7 GB | quick start; you inherit their versions |
| `vllm/vllm-openai:vX` | a whole engine, ready to serve | ~10 GB | production if you don't need custom code — pin the digest |
| `nvcr.io/nvidia/tritonserver:YY.MM-py3` | Triton + backends | ~15 GB (or use `-py3-min` and add one backend) | [Phase 5 lesson 7](../phase-5/07-triton-inference-server.md) deployments |

The default professional choice is **`devel` to build, `runtime` to run** — a multi-stage build, which is [lesson 3](03-image-size-and-cold-start.md)'s entire subject. Deciding to use a vendor image instead is legitimate; deciding it *by accident* because it was in a tutorial is not.

A note on the PyTorch wheel: `pip install torch` from PyPI bundles its own CUDA libraries (cuBLAS, cuDNN, NCCL, ~2-3 GB of them) as `nvidia-*` wheels. So a `torch` wheel on top of a `cuda:runtime` base gives you **two copies** of most CUDA libraries. Either build on `cuda:base` and let the wheel bring CUDA, or use the `--index-url download.pytorch.org/whl/cu124` build matched to your base. Being deliberate here is worth ~2 GB.

---

## A correct serving Dockerfile, annotated

```dockerfile
# syntax=docker/dockerfile:1.7
############################  build stage  ############################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Dependency layer first: changes rarely → cached across code edits
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes -r requirements.txt

############################  runtime stage  ##########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/hf \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv
COPY --chown=10001:10001 src/ /app/src/

USER 10001
WORKDIR /app
EXPOSE 8000
# readiness is checked by the orchestrator; HEALTHCHECK is for plain docker
HEALTHCHECK --interval=15s --timeout=3s --start-period=300s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
ENTRYPOINT ["python", "-m", "src.server"]
```

What each non-obvious line buys you:

| Line | Why |
|---|---|
| `venv` copied between stages | one directory carries all deps; no `pip` needed at runtime |
| deps before code | Docker layer cache: a code change rebuilds one small layer, not 4 GB of wheels |
| `--mount=type=cache` | BuildKit keeps the pip cache *outside* the image — fast rebuilds, no bloat |
| `--require-hashes` | supply-chain pinning; a compromised or re-uploaded wheel fails the build |
| `useradd` + `USER 10001` | non-root. Also forces you to notice every path you write to |
| numeric UID | works with k8s `runAsNonRoot`, which cannot resolve usernames |
| `HF_HOME=/cache/hf` | weights land on a mounted volume, never inside the image ([lesson 7](07-model-registry-and-artifacts.md)) |
| `start-period=300s` | model load takes minutes; without this the container is killed while loading |
| `PYTHONUNBUFFERED` | logs appear in `kubectl logs` in real time instead of on crash |
| `ENTRYPOINT` exec-form | PID 1 is python, so `SIGTERM` reaches it — required for graceful drain ([lesson 6](06-deploying-and-rollouts.md)) |

---

## Signals, PID 1, and why your drain hangs

This bites everyone once. In shell-form (`CMD python -m src.server`), PID 1 is `/bin/sh`, which does **not** forward `SIGTERM`. Kubernetes sends `SIGTERM`, nothing happens, and after `terminationGracePeriodSeconds` your pod is `SIGKILL`ed — dropping every in-flight stream. For a 2000-token generation that's up to a minute of user-visible truncation on *every deploy*.

Rules:
- Exec-form `ENTRYPOINT`/`CMD` always.
- If you truly need a shell wrapper, use `exec python …` as its last line, or add `tini`/`dumb-init` as PID 1.
- Handle `SIGTERM` in the server: stop accepting new requests, finish in-flight ones, then exit. Verify with `docker stop` and a stopwatch — it should exit *before* the 10 s default, not at it.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | `nvidia-container-toolkit` not installed/configured on host | install toolkit, `nvidia-ctk runtime configure`, restart docker |
| `CUDA driver version is insufficient` | host driver older than image runtime | upgrade node, use `cuda-compat`, or pin scheduling by driver label |
| `no kernel image is available` | binary lacks your `sm_XX` | set `TORCH_CUDA_ARCH_LIST`, or use a wheel built for that arch |
| Startup takes 3 extra minutes on new hardware | PTX JIT for an unlisted arch | add the arch to the build; check `CUDA_CACHE_PATH` for JIT cache reuse |
| `nvidia-smi` works, torch sees 0 GPUs | `NVIDIA_VISIBLE_DEVICES` unset/overridden, or missing `compute` capability | let the device plugin set it; check `NVIDIA_DRIVER_CAPABILITIES` |
| Hardware video decode unavailable | `video` missing from driver capabilities | `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` |
| Two processes on one GPU, both slow | manual `NVIDIA_VISIBLE_DEVICES=all` bypassing the scheduler | request `nvidia.com/gpu: N`; never set the env var yourself on k8s |
| NCCL hangs at init in a multi-GPU pod | missing IPC/shm or `--ipc=host`; too-small `/dev/shm` | raise shm (k8s: `emptyDir{medium: Memory}`), see [Phase 6 lesson 9](../phase-6/09-multi-node-operations.md) |
| Works as root, fails as UID 10001 | writes to `/app`, `~/.cache`, `/tmp` | point caches at a writable mounted path; make the rootfs read-only and find out early |

---

## Do this now (45 minutes)

1. **Build the annotated Dockerfile above** for your Phase-3/7 server, then run it with `--gpus all` (or CPU-only if that's what you have) and confirm: `nvidia-smi` inside the container, non-root user (`id`), and `docker stop` returning in under 10 s without dropping an in-flight request.
2. **Break it deliberately, twice.** (a) Switch the entrypoint to shell-form and re-measure `docker stop` — watch it take the full grace period. (b) Set `TORCH_CUDA_ARCH_LIST` to an arch your GPU doesn't have and read the exact error text so you recognise it later.
3. **Record the size** of the resulting image (`docker images`) and the breakdown (`docker history --human`). Keep it — [lesson 3](03-image-size-and-cold-start.md) is about cutting that number, and you need the "before."

---

**Next:** [Image size and cold start →](03-image-size-and-cold-start.md) — where the 14 GB actually comes from, how to get it to 4, and why image size is a *latency* metric on an autoscaled GPU fleet.
