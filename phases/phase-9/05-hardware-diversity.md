# 5 — Hardware Diversity: TPU, Inferentia, CPU, and Apple Silicon

> **You'll be able to say:** "Hardware is a per-workload cost decision, not a default. I compare candidates on cost per unit of useful work at my SLO — not on peak TFLOPs, not on $/hour — and I know what each choice charges me: XLA/TPU wants static shapes and a compile step, Neuron wants pre-compiled buckets and its own SDK, CPU serving trades throughput ceiling for zero scheduling friction and huge RAM, and Apple Silicon gives unified memory with a smaller software ecosystem. I also know the honest conclusion: for large-model LLM serving, NVIDIA usually still wins on software maturity, and the alternatives win decisively for small models, low QPS, batch jobs, and cost-sensitive scale."

Phases 2-8 assumed CUDA. That assumption is right most of the time and expensive the rest of the time. This lesson is the decision procedure plus what actually goes wrong on each alternative.

---

## The only comparison that means anything

```
  cost per unit of useful work =        $/hour of the instance
                                 ─────────────────────────────────────
                                 units/second SUSTAINED at your SLO × 3600

  units = tokens · requests · images · audio-hours — pick one and keep it
```

Three ways this gets faked, all common in vendor material and blog posts:

| Fake | Why it misleads | The fix |
|---|---|---|
| Peak TFLOPs | nobody reaches it; LLM decode reaches 5-15% of peak | measure achieved units/s |
| $/hour | a 3× cheaper instance that is 5× slower is worse | always divide |
| Throughput without a latency constraint | batch 512 "throughput" at 9 s TTFT is not your product | measure at the p99 you must hold |
| Ignoring utilization | 40% average utilization triples effective cost | include your real duty cycle |
| Ignoring engineering cost | 3 weeks of porting ≈ $20-40k ≈ a lot of GPU-hours | amortize it into the comparison |

The engineering-cost row is the one that decides most real cases: below roughly ten accelerators, a port that saves 30% is usually value-negative ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md)'s "engineering time dominates for small fleets"). Above a hundred, the same 30% funds a team.

---

## The landscape

| | NVIDIA GPU | TPU | AWS Inferentia/Trainium | x86 CPU | Apple Silicon | AMD GPU |
|---|---|---|---|---|---|---|
| Programming model | CUDA, mature | XLA via JAX/PyTorch-XLA | Neuron SDK (compiler + runtime) | native / ONNX RT / `llama.cpp` | Metal, MPS, Core ML | ROCm/HIP |
| Compilation | optional (eager works) | **mandatory, AOT/JIT** | **mandatory, AOT** | optional | optional | optional |
| Dynamic shapes | fine | **recompiles** — must bucket | **must bucket** | fine | fine | fine |
| Memory | 24-192 GB HBM | 16-128 GB HBM per chip | 32 GB HBM (Inf2 core pairs) | **TB of DRAM, cheap** | unified, up to 512 GB | 24-192 GB HBM |
| Interconnect | NVLink / InfiniBand | ICI torus (excellent) | NeuronLink | none (NUMA) | none | Infinity Fabric |
| Ecosystem reality | everything works | JAX-first; PyTorch works with care | vLLM/TGI support exists; kernel gaps | broad but throughput-capped | good for local, thin for server | improving fast; vLLM supported |
| Where it wins | anything, especially big + bleeding-edge | large-scale first-party/GCP training+serving, long-lived stable models | steady-state cost per token on AWS | small/quantized models, low QPS, huge RAM, batch | local dev, on-device, Metal `llama.cpp` | price/perf when your stack is supported |
| Where it hurts | price, availability | debugging, dynamic shapes, kernel authoring | op coverage, compile times, SDK churn | throughput ceiling, prefill on big models | not a datacenter answer | long-tail kernel/library gaps |

Read the "Compilation" and "Dynamic shapes" rows together: **on TPU and Neuron, shape dynamism is a first-class engineering constraint, not a detail.** That single property is responsible for most of the porting pain.

---

## TPU and XLA: the static-shape world

A TPU is a systolic-array machine: a huge MXU that streams a weight matrix against activations with very high arithmetic efficiency, fed by a compiler that has planned every buffer in advance. You do not write kernels; you write graphs and let XLA fuse and schedule them (Pallas exists for custom kernels, and it is a real learning curve).

What follows from "the compiler plans everything":

- **Every distinct input shape triggers a compile.** A compile is seconds to minutes. A server that sees 500 different sequence lengths will recompile until it dies. The fix is **bucketing**: pad to a small ladder of shapes (e.g. 128/256/512/1024/2048 tokens), compile one program per bucket, warm them all at startup.
- **Padding waste is real.** Bucketing to powers of two wastes up to 50% of compute on the last bucket. Choose buckets from your measured length distribution, not from aesthetics.
- **Autoregressive decode is awkward** — sequence length grows by one every step. The standard answer is a fixed-size KV buffer allocated for the max length with masking (so the shape never changes), which means you pay max-length memory regardless of actual length. This is the opposite trade from PagedAttention ([Phase 4 lesson 5](../phase-4/05-paged-attention.md)): the compiler wants static allocation; paging wants dynamic. Do not try to port a paged allocator to a static-shape compiler and expect a good time.
- **Debugging is different.** A fused XLA program does not have your op names in the profile. You use the XLA HLO dump and the TPU profiler; `print`-debugging inside a compiled function is a trap (in JAX, tracers are not values).
- **Scaling is a strength.** The ICI torus interconnect and the compiler's collective scheduling make multi-chip sharding (`jax.sharding`, GSPMD) genuinely pleasant compared to hand-written tensor parallelism ([Phase 6 lesson 3](../phase-6/03-tensor-parallelism.md)).

Rule of thumb: **TPU is excellent when the model is stable, the shapes are few, and the scale is large.** It is a poor fit for a research-velocity codebase that changes model structure weekly.

---

## AWS Inferentia/Trainium and the Neuron SDK

Same philosophy, different vendor: a compiler (`neuronx-cc`) turns your graph into a NEFF artifact that the runtime executes on NeuronCores.

Concrete operational facts that surprise people:

- **Compilation is ahead-of-time and slow** (minutes to tens of minutes for large models). It therefore belongs in CI, producing a **cached, digest-pinned artifact** exactly like a TensorRT engine plan ([Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)). Compiling at container start is how you get a 20-minute cold start.
- **Buckets again.** You compile for a set of (batch, sequence-length) combinations; anything outside falls back or errors. `transformers-neuronx` and vLLM's Neuron backend expose this directly.
- **Op coverage is the risk.** An unsupported op either falls back to CPU (a disastrous latency cliff mid-graph) or fails to compile. Check coverage *before* committing to a port; custom attention variants and exotic activations are the usual casualties.
- **The core/instance mapping matters.** `inf2.xlarge` = 1 Inferentia2 chip = 2 NeuronCores; larger sizes scale cores and HBM, and tensor parallelism across cores is configured at compile time, not at runtime.
- **Where it pays off:** steady-state, high-volume serving of a fixed model on AWS, where the per-token cost advantage (commonly quoted 30-50% versus comparable GPU instances — verify for your model) compounds over months. Anthropic, AWS's own Bedrock models, and several large deployments run this way in production, so it is a real path, not a science project.

---

## CPU inference: legitimate, and underrated

CPUs are not a consolation prize. They are the right answer more often than GPU-first engineers expect.

**What makes CPU serving work:**

| Lever | Detail |
|---|---|
| Quantization | INT8/INT4 weights; `llama.cpp` GGUF Q4_K_M is the workhorse. Mandatory, not optional |
| SIMD kernels | AVX2 / AVX-512 / AMX (Intel's matrix extension, a real step change) / ARM NEON+SVE |
| Threading | pin threads, one per physical core, respect NUMA; `OMP_NUM_THREADS`, `--numa` |
| Graph optimization | ONNX Runtime / OpenVINO fuse, fold constants, pick layouts |
| Huge, cheap memory | 512 GB-2 TB of DRAM costs less than 80 GB of HBM; a 70B Q4 model *fits* on a CPU box |

**The physics you cannot escape:** a dual-socket server has roughly 200-500 GB/s of DRAM bandwidth against an H100's ~3.3 TB/s. Since LLM decode is bandwidth-bound ([Phase 4 lesson 1](../phase-4/01-what-to-optimize.md)), **single-stream decode speed is roughly bandwidth-ratio-limited: expect ~5-20 tokens/s for a 7B Q4 model on a good server CPU, versus 50-150 on a mid-range GPU.** Prefill is compute-bound and even worse: long-prompt TTFT on CPU is often seconds, which is the single most common reason a CPU deployment fails its SLO.

**Where CPU wins outright:**

- Low QPS. A GPU idle 95% of the time is pure waste; you are paying $1-3/hour for 20 requests/hour.
- Small models: embedding models, rerankers, classifiers, VAD, small vision nets — these are *routinely* CPU-served in production and GPUs add PCIe latency for no gain.
- Huge models on tiny budgets, where "slow but possible" beats "needs 2×A100".
- Anywhere GPUs aren't available, quota-limited, or worth the scheduling complexity ([Phase 8 lesson 4](../phase-8/04-kubernetes-for-gpu-serving.md): integer resources, no overcommit — CPU has none of that friction).
- Batch/offline jobs on cheap spot CPU fleets.

**Apple Silicon** deserves its own line: unified memory means a 128 GB Mac can hold models that need multiple datacenter GPUs, and `llama.cpp` with Metal plus Core ML/MLX are genuinely good. Memory *bandwidth* (up to ~800 GB/s on the top parts) is the differentiator versus x86. It is an excellent development and local-inference platform and not a server platform — no multi-tenant isolation story, no datacenter operations story ([GETTING-STARTED.md](../../GETTING-STARTED.md) covers the practical setup).

---

## The decision procedure

```
  1. What is the workload shape? (lesson 1 — the four numbers)
  2. Does the model FIT? weights + activations + KV/state at your batch
  3. What throughput do you need at your p99? measure, don't extrapolate
  4. Which candidates does your software stack actually support TODAY?
       - is there a maintained backend in vLLM/TGI/ONNX RT/Triton for it?
       - are your custom kernels/ops available?
  5. Cost per unit of useful work for each survivor, at your real duty cycle
  6. Add the port cost and the ongoing maintenance of a second stack
  7. Choose. Write down the number that would change your mind.
```

Defaults that are usually right, stated bluntly:

| Situation | Default |
|---|---|
| Large LLM, interactive, evolving stack | NVIDIA GPU |
| Large LLM, stable, huge steady volume, on AWS | evaluate Inferentia seriously |
| Large LLM, stable, huge volume, on GCP | evaluate TPU seriously |
| Embedding / reranker / classifier / VAD, any scale | **CPU** |
| Small vision model, high throughput | GPU with hardware decode ([lesson 3](03-vision-serving.md)) or CPU+OpenVINO |
| Ranking model with TB embeddings | **CPU fleet** with DRAM, GPU only if the dense part is large ([lesson 2](02-recommendation-and-ranking.md)) |
| Offline batch, cost-sensitive, no deadline | cheapest per unit: spot GPU or big CPU fleet |
| Local dev, demos, privacy-sensitive single-user | Apple Silicon / consumer GPU |
| On-device, mobile | neither — [lesson 7](07-edge-and-on-device.md) |

---

## The portability tax, itemized

Every alternative accelerator charges you in the same currencies. Budget them explicitly in any migration proposal:

| Tax | Typical magnitude |
|---|---|
| Model export/conversion work | days to weeks ([lesson 8](08-model-formats-and-runtimes.md)) |
| Unsupported ops / custom kernels | the long pole; sometimes fatal |
| Compile times in CI | minutes to tens of minutes per artifact/bucket |
| Numerical differences | different accumulation order and dtypes → output drift; needs a tolerance test |
| A second observability stack | different device metrics, different profiler ([Phase 7 lesson 3](../phase-7/03-gpu-and-host-telemetry.md)) |
| A second deployment path | different base image, device plugin, node pool, taints ([Phase 8 lessons 2, 4](../phase-8/02-containers-for-gpu-workloads.md)) |
| On-call knowledge | your team must debug two stacks at 3 a.m. |
| Capacity risk | your fallback plan when the alternative is unavailable |

**The one non-negotiable:** any port must pass an output-equivalence gate before it serves traffic — a fixed prompt/input set, compared against the reference platform, with a stated tolerance and a distributional quality check ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)). "It ran and the latency was good" is not a port.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Latency spikes every few requests on TPU/Neuron | recompile on a new shape | bucket shapes; warm all buckets at startup; assert no compile after warmup |
| 20-minute cold start | compiling at container start | compile in CI, ship the artifact, cache it ([Phase 8 lesson 3](../phase-8/03-image-size-and-cold-start.md)) |
| One op silently runs on host CPU | unsupported op falling back mid-graph | inspect the compiler report; replace/rewrite the op; assert zero fallbacks |
| Output differs from the GPU reference | different accumulation/dtype/fusion | equivalence gate with a tolerance; decide if the drift is acceptable |
| CPU serving fine at low QPS, dies at 10× | throughput ceiling reached; prefill is compute-bound | horizontal scale-out, smaller model, or GPU for prefill |
| CPU model slower than expected | no quantization, wrong thread count, NUMA-crossing memory | quantize, pin threads, one process per socket |
| Great TFLOPs, terrible tokens/s | bandwidth-bound workload on a compute-heavy chip | compare on achieved units/s, not peak |
| Cost went *up* after migrating | low utilization on the new hardware, or per-request overheads | include duty cycle in the comparison; re-measure at real traffic |

---

## What to read and run

- **Docs:** Google Cloud TPU architecture + [XLA](https://openxla.org/) docs (read the "shapes are static" parts twice); AWS [Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/) developer guide (bucketing, `neuronx-cc`, op support matrix).
- **Code:** `microsoft/onnxruntime` — the execution-provider abstraction, the cleanest expression of "same graph, different silicon" ([lesson 8](08-model-formats-and-runtimes.md)); `ggerganov/llama.cpp` — GGUF plus hand-written AVX2/NEON kernels, the reference for what CPU inference can be; `openvinotoolkit/openvino` for Intel CPU/iGPU/NPU; `ROCm/vllm` or upstream vLLM's ROCm path for AMD.
- **Blogs:** AWS ML blog on Inferentia/Trainium deployments; Google Cloud on TPU inference; any *independent* CPU-vs-GPU serving benchmark, read with the fakery table above in hand.

---

## Do this now (60-90 minutes — this is project 16's first half)

1. **Run one model on three runtimes** you have access to: PyTorch GPU FP16 (or MPS), ONNX Runtime CPU with INT8 dynamic quantization, and `llama.cpp` GGUF Q4 (for an LLM) or OpenVINO INT8 (for a vision/tabular model). Record for each: p50/p99 latency at batch 1, sustained throughput, memory, and an accuracy check.
2. **Convert to cost per unit of useful work.** Look up the real $/hour for the instance class each would run on, divide by sustained units/s × 3600, and produce one table. Include a "realistic utilization" column at 30% and 80%.
3. **Find the crossover QPS** where GPU becomes cheaper than CPU for your model. Plot cost/hour versus QPS for both; the intersection is a genuinely useful number and a great interview answer.
4. **Write the decision, and the falsifier.** One paragraph: which hardware you would pick for this workload and the single measurement that would change your mind.

---

**Next:** [Vector search and ANN →](06-vector-search-and-ann.md) — the retrieval workload that sits in front of half of today's LLM services, where recall is a tunable knob and RAM is the cost line.
