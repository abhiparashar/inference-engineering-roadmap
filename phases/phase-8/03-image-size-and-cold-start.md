# 3 — Image Size and Cold Start

> **You'll be able to say:** "Cold start is a latency SLI on any autoscaled GPU fleet, and it decomposes into five measurable stages: image pull, container start, weight fetch, model load to HBM, and warmup/compile. I can name the dominant stage from a timing table instead of guessing, and I know the fixes are different for each — layer hygiene and lazy pulling for the image, a local NVMe cache and a fast format for the weights, and a captured warmup for the compile tax. A 14 GB image on a shared NAT gateway is not a disk problem; it is minutes of unserved traffic."

[Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md) showed that minute-scale cold starts make the naive autoscaler unstable. This lesson is the engineering that shortens the minutes.

---

## Decompose before you optimize

```
  pod scheduled                                                  first token served
       │                                                                  │
       ├── 1. IMAGE PULL ──────┬── 2. CONTAINER START ──┬── 3. WEIGHT FETCH ──┐
       │   registry → node     │   runtime hook, python │   S3/GCS/registry   │
       │   decompress + unpack │   imports (torch: 3-8s)│   → local disk      │
       │                       │                        │                     │
       └── 4. LOAD TO HBM ─────┴── 5. WARMUP / COMPILE ─┴─────────────────────┘
           disk → host RAM         CUDA ctx, kernel autotune, torch.compile,
           → HBM (PCIe/NVLink)     CUDA graph capture, first-batch JIT
```

A representative measured breakdown for an 8B model, first pod on a fresh node:

| Stage | Cold (nothing cached) | Warm node (image + weights cached) | Dominant fix |
|---|---|---|---|
| 1. Image pull (12 GB) | 90-300 s | 0 s | smaller image, lazy pull, pre-pull DaemonSet |
| 2. Container start + imports | 8-20 s | 8-20 s | fewer imports, no PTX JIT |
| 3. Weight fetch (16 GB) | 60-240 s | 0 s | node-local NVMe cache, parallel ranged GETs |
| 4. Load to HBM | 15-45 s | 10-30 s | safetensors + mmap, `pin_memory`, direct-to-GPU |
| 5. Warmup / compile | 20-600 s | 20-600 s (unless cached) | persistent compile cache, captured graphs |
| **Total** | **3-20 min** | **40-90 s** | — |

**Measure your own five numbers before touching anything.** Emit a log line at each boundary (or a Prometheus gauge per stage, scraped once at startup); the shape of that table decides where the work goes. The most common mistake in this area is spending a week shrinking an image when weight fetch was 70% of the time.

---

## Stage 1 — the image

### Where the gigabytes are

Run `docker history --human --no-trunc <image>` on a naive build and you will typically find:

| Layer | Size | Removable? |
|---|---|---|
| `nvidia/cuda:*-devel` base | ~6 GB | **yes** — use `runtime` (or `base`) in the final stage |
| `torch` + bundled `nvidia-*` CUDA wheels | 2.5-3.5 GB | partly — don't duplicate CUDA libs you already have |
| `pip` cache left in the layer | 1-2 GB | **yes** — `PIP_NO_CACHE_DIR=1` or BuildKit cache mounts |
| `apt` lists / build-essential / git | 0.3-1 GB | **yes** — `--no-install-recommends`, `rm -rf /var/lib/apt/lists/*` |
| flash-attn / vLLM / TRT-LLM binaries | 1-3 GB | no (but arch-trim the fatbin) |
| model weights baked into the image | 5-30 GB | **yes, always** — see below |
| `.git`, tests, datasets, notebooks | 0.1-5 GB | **yes** — `.dockerignore` |

Two rules do most of the work: **multi-stage** ([lesson 2](02-containers-for-gpu-workloads.md)'s Dockerfile) and **never bake weights**.

### Why weights do not belong in the image

It is tempting — one artifact, no fetch stage, trivially reproducible. It is still wrong at scale:

| Reason | Consequence |
|---|---|
| Layer immutability | changing one byte of code re-pushes nothing, but changing the *weights* forces a new 20 GB image; the registry fills up |
| Rollback coupling | you can no longer roll back code without also rolling back weights, or vice versa ([lesson 1](01-why-shipping-is-the-job.md)'s four artifacts) |
| Cache locality | the node cannot share one copy of the weights across two different code versions |
| Build time | CI now moves 20 GB per commit |
| Registry limits | many registries throttle or bill hard above ~10 GB layers |

The exception that proves the rule: **edge and air-gapped deployments**, where a single self-contained artifact is worth every downside, and single-model appliances whose code changes as rarely as the weights.

### Getting from 14 GB to ~4 GB

1. **Multi-stage**: `devel` → `runtime`. Typically −5 GB.
2. **`.dockerignore`** everything not needed: `.git`, `tests/`, `data/`, `*.ipynb`, `docs/`. Cheap, often −1 GB, and it also speeds up the build context upload.
3. **One CUDA copy**: either the base image's or the torch wheel's, not both. Up to −2 GB.
4. **Arch-trim**: `TORCH_CUDA_ARCH_LIST` for exactly your fleet, not `8.0;8.6;8.9;9.0;10.0`. −0.5-2 GB on kernel-heavy packages.
5. **Order layers by change frequency**: base → system deps → python deps → app code. This doesn't shrink the image but makes 95% of pulls hit cache for everything but the last small layer.
6. **Prune build tools** from the runtime stage: no `gcc`, no `git`, no `nvcc`.

A realistic target: **3-5 GB** for a vLLM-based server, **1-2 GB** for a lightweight CPU/ONNX server ([Phase 9 lesson 8](../phase-9/08-model-formats-and-runtimes.md)).

### Making the pull fast even when the image is big

| Technique | Mechanism | Typical gain |
|---|---|---|
| **Pre-pull DaemonSet** | a pod on every node pulls the next image before it's needed | pull time → 0 at scale-up |
| **Lazy pulling** (eStargz / SOCI / Nydus) | pull metadata, fetch blocks on demand; container starts before the whole image arrives | 2-10× faster start on large images |
| **Registry mirror / pull-through cache** in-region | avoids cross-region and NAT bandwidth | 2-5× |
| **`imagePullPolicy: IfNotPresent`** + digest pin | no re-pull on restart, still immutable | avoids the pathological re-pull |
| **Compression choice** (zstd over gzip) | faster decompress, which is often the real bottleneck | 20-40% of unpack time |
| Node image bake (AMI with images pre-loaded) | Packer-built AMI containing the image | pull → 0, at the cost of AMI churn |

Note the decompress point: on a fast network, **unpacking, not downloading, dominates**. `gzip` decompression is single-threaded per layer; many small parallel layers beat one huge layer.

---

## Stage 3 — weights

The choices, in increasing order of sophistication:

| Strategy | First pod on new node | Later pods on same node | Complexity |
|---|---|---|---|
| Download from S3/GCS in entrypoint | 60-240 s | 60-240 s (repeated!) | trivial, and the default mistake |
| Download to a node-local NVMe cache dir (`hostPath`) | 60-240 s | **~0 s** | low — do this one |
| Shared ReadOnlyMany volume (EFS/Filestore/NFS) | 100-400 s | 100-400 s, and slower | medium; convenient, often slow |
| CSI ephemeral volume from a pre-baked disk snapshot | 10-40 s | ~0 s | medium |
| OCI artifact + lazy pull (weights as an image layer) | 20-60 s | ~0 s | medium ([lesson 7](07-model-registry-and-artifacts.md)) |
| Peer-to-peer distribution (Dragonfly, Kraken) | fast at fleet scale | ~0 s | high; pays off above ~50 nodes |

Independent of strategy:

- **Parallelize the download.** `hf_transfer`, `s5cmd`, or ranged GETs saturate 10-25 Gbps NICs; a single-stream `boto3` download will not.
- **Use `safetensors`.** Zero-copy `mmap`, no `pickle` execution, faster and safer than `.bin`. Converting is a one-time cost.
- **Verify the checksum** and refuse to start on mismatch — a truncated weight file produces garbage output with 200 OK, the worst failure class from [Phase 7 lesson 1](../phase-7/01-what-to-measure.md).
- **Cache key must include the digest**, not the model name. `models/llama-3.1-8b/` as a cache key is how a stale checkpoint survives a rollback.

---

## Stages 4-5 — load and warmup

Loading to HBM is bounded by disk → host RAM → PCIe. Practical levers: `safetensors` mmap (avoids a full host-RAM copy), loading shards in parallel across TP ranks (each rank reads only its shard — [Phase 6 lesson 3](../phase-6/03-tensor-parallelism.md)), and avoiding a CPU-side dtype conversion by storing weights in the dtype you serve.

Warmup is where the surprises live:

| Cost | Typical | Mitigation |
|---|---|---|
| CUDA context creation | 1-3 s | unavoidable |
| PTX JIT for an unlisted arch | 30-300 s | build the arch in ([lesson 2](02-containers-for-gpu-workloads.md)); persist `CUDA_CACHE_PATH` |
| `torch.compile` / Inductor | 60-600 s | persistent Inductor cache dir on the node; `TORCHINDUCTOR_CACHE_DIR` |
| CUDA-graph capture (vLLM) | 10-60 s | `enforce_eager` for dev only; keep graphs in prod ([Phase 4 lesson 8](../phase-4/08-compilation-and-kernels.md)) |
| cuBLAS/cuDNN autotune on first shapes | 5-30 s | warm the real shapes, not shape `1×1` |
| TensorRT-LLM engine build | **minutes to hours** | build in CI, ship the engine plan as an artifact — never at pod start ([Phase 5 lesson 6](../phase-5/06-tensorrt-llm-and-compiled-engines.md)) |

**Warm with representative traffic, then flip readiness.** A warmup that sends one 5-token prompt leaves every long-prompt kernel un-autotuned, so the first real user pays for it — and that user's TTFT lands in your SLO. Warm across your actual prompt-length distribution, at your real batch sizes, and only then let the readiness probe pass ([lesson 4](04-kubernetes-for-gpu-serving.md)).

```python
# warmup that actually warms: shapes you serve, batch sizes you run
for n_prompt in (128, 512, 2048):
    for batch in (1, 8, max_batch):
        engine.generate(["x " * n_prompt] * batch, max_tokens=8)
READY.set(True)   # readiness probe reads this, not "process is up"
```

---

## Cold start as an SLI

Treat it like any other latency metric ([Phase 7 lesson 2](../phase-7/02-instrumenting-with-prometheus.md)):

```
  inf_startup_stage_seconds{stage="image_pull|start|weight_fetch|load|warmup"}   gauge, set once
  inf_startup_total_seconds                                                      histogram
  inf_startup_cache_hit{kind="image|weights|compile"}                            counter
```

Then two things become possible: an alert when p90 cold start regresses (a silent tax on every scale-up), and honest autoscaler tuning — your scale-up lead time is *measured*, so the predictive/pre-warm policy from [Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md) has a real number to work from instead of a guess.

Cold start also has a direct dollar cost: every second between "GPU allocated" and "GPU serving" is billed and produces nothing. At $3/GPU-hr with 40 scale-up events a day and a 6-minute cold start, that's ~$12/day/replica-flap of pure waste, plus the SLO damage from serving at reduced capacity during it ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md)).

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Pod stuck `ContainerCreating` for minutes | image pull over shared NAT/cross-region | in-region mirror, pre-pull, smaller image |
| Every pod re-downloads weights | no node-local cache, or cache key includes pod name | `hostPath`/CSI cache keyed by artifact digest |
| First requests after a deploy are slow, then fine | readiness passed before warmup | warm real shapes, gate readiness on warmup |
| Startup slower on the *new* GPU type | PTX JIT | add the arch; persist the JIT cache |
| `torch.compile` cost paid every restart | ephemeral inductor cache | mount a persistent `TORCHINDUCTOR_CACHE_DIR` |
| Fast in staging, slow in prod | staging node had a warm image cache | measure on a *fresh* node; that's the number that matters |
| Weight cache serves stale weights after rollback | cache keyed by model name | key by digest, always |
| OOM during load, not during serving | full host-RAM copy of the checkpoint | `safetensors` mmap; check pod memory limits ([lesson 4](04-kubernetes-for-gpu-serving.md)) |

---

## Do this now (60 minutes)

1. **Instrument the five stages** in your server's startup path and print a table on boot. Run it on a machine with nothing cached, and again warm. Two rows: cold and warm.
2. **Shrink the image.** Apply multi-stage + `.dockerignore` + no-baked-weights + arch trim. Record before/after size and before/after pull time from a real registry. Aim for a ≥2× reduction; write down what the remaining bulk is.
3. **Add a node-local weight cache** keyed by artifact digest, and prove the second pod on the same node skips stage 3 entirely.
4. **Fix warmup**: gate readiness on a warmup loop over your real prompt-length distribution, then verify the first post-deploy request's TTFT matches steady-state TTFT — this is the deploy-visible tail your users currently pay for.

---

**Next:** [Kubernetes for GPU serving →](04-kubernetes-for-gpu-serving.md) — the device plugin, why GPUs are not shareable like CPU, probes that catch a hung engine, and the graceful-shutdown settings that stop a rollout from truncating streams.
