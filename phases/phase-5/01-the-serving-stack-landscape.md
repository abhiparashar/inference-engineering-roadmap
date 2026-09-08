# 1 — The Serving-Stack Landscape

> **You'll be able to say:** "Every inference stack is five layers: API/router, scheduler, KV-cache manager, model executor, kernels. vLLM, TGI and SGLang are full engines that own layers 1-5 for one model; TensorRT-LLM owns 3-5 and compiles them ahead of time; Triton Inference Server owns layer 1 plus a model repository and ensembles, and delegates 2-5 to a backend; Ray Serve owns orchestration *above* layer 1. Once I place a framework on that grid I know what its flags can and cannot change, and 'which framework is fastest' becomes 'whose defaults match my workload' — a question I can answer by leveling the config."

Phase 3 and Phase 4 taught you the mechanisms. This lesson is the map of who implements which mechanism, so that the rest of the phase is reading rather than wandering.

---

## The five layers, and what each costs per token

```
 LAYER            RUNS ...                     COST SCALE            GETS BLAMED FOR
 ───────────────────────────────────────────────────────────────────────────────────
 1 API/router     once per request             ~0.1-2 ms/req         TTFT floor, 429s,
                  (+ once per streamed chunk)  + per-chunk SSE cost   streaming jitter
 2 Scheduler      once per engine step         ~0.1-1 ms/step        low batch size,
                  (every ~10-50 ms)                                   p99 TTFT, starvation
 3 KV-cache mgr   once per step, per seq       ~µs-100 µs/step       OOM, preemption,
                                                                      low concurrency
 4 Model executor once per step                the whole forward     tokens/sec
                                                pass (ms)
 5 Kernels        thousands of launches/step   the forward pass      tokens/sec, and
                                                itself                everything at low B
```

Two consequences worth internalizing before you touch any framework:

- **Layers 1-3 are cheap in absolute terms and enormous in leverage.** A scheduler decision that raises mean batch size from 4 to 32 is worth more than any kernel work you will ever do ([Phase 3 lesson 4](../phase-3/04-continuous-batching.md)). This is why engines written in "slow" Python can be state of the art: Python is running layer 2, not layer 5.
- **Layer 1's cost is per *request* and per *streamed chunk*, not per token of compute.** At 50 concurrent streams × 40 tokens/sec each, that's 2,000 SSE chunks/second going through JSON serialization. This is the specific pressure that made HuggingFace write TGI's router in Rust ([lesson 4](04-tgi-and-the-router-split.md)) and it's invisible in any single-request benchmark.

### What each layer actually does

**1. API / router.** HTTP or gRPC, auth, request validation (max tokens, stop sequences, sampling params), tokenization, applying the chat template, OpenAI-schema translation, SSE/streaming, detokenization of output, and — in multi-replica deployments — choosing which replica gets the request. It's the layer that decides what a "request" *is*.

**2. Scheduler.** Which requests run in the next forward pass. Admission (is there KV space?), continuous batching, chunked prefill, preemption/eviction, priority and fairness, and the token budget that caps a step. This is the layer that determines throughput and tail latency, and it is where you should look first in any codebase.

**3. KV-cache manager.** Physical block pool, per-sequence block tables, allocation on demand, reference counts, prefix cache lookup/insert, eviction policy, and optional swap/offload to CPU. Phase 4 lesson 5 is this layer.

**4. Model executor.** Takes the scheduler's batch, builds the input tensors (flattened tokens, positions, block tables, slot mappings), runs the forward pass, samples, and returns tokens. Owns tensor/pipeline parallelism, LoRA adapters, quantized weight loading, CUDA-graph capture/replay, and warmup.

**5. Kernels.** Paged attention, FlashAttention/FlashDecoding/FlashInfer, fused MLP/RMSNorm/RoPE, quantized GEMMs (Marlin, Machete, CUTLASS, cuBLAS), sampling kernels. Mostly shared across engines — vLLM, SGLang and TGI all pull from the same small set of kernel libraries, which is a large part of why their peak numbers converge.

---

## The grid: who owns what

`●` = owns and is a differentiator · `○` = owns, unremarkable · `–` = delegates or doesn't do it

| Framework | 1 API/router | 2 Scheduler | 3 KV mgr | 4 Executor | 5 Kernels | Written in | Sweet spot |
|---|---|---|---|---|---|---|---|
| **vLLM** | ○ (FastAPI, OpenAI) | ● | ● (paged + prefix) | ● | ● | Python + CUDA/C++ | The default general-purpose LLM engine |
| **SGLang** | ○ | ● (cache-aware) | ● (RadixAttention) | ● | ● | Python + CUDA | Shared-prefix and structured-output workloads |
| **TGI** | ● (Rust router) | ● (Rust queue) | ○ | ○ | ○ (FA/paged) | Rust + Python | HF ecosystem, ops simplicity, Endpoints |
| **TensorRT-LLM** | – (use Triton / `trtllm-serve`) | ● (C++ in-flight batcher) | ● | ● (compiled engine) | ● (fused, hand-tuned) | C++/CUDA + Python API | Peak NVIDIA perf, fixed model set, ops budget |
| **Triton Server** | ● (HTTP/gRPC, repo, ensembles) | ● (dynamic batching, per-model) | – | – (backends) | – | C++ | Multi-model, multi-framework, pipelines |
| **Ray Serve** | ● (ingress + composition) | ○ (per-deployment queues) | – | – | – | Python | Multi-stage apps, autoscaling, heterogeneous fleets |
| **DeepSpeed-MII** | ○ | ○ | ○ | ● (TP kernels, ZeRO-Inference offload) | ● | Python + CUDA | Offload/large-model-on-small-GPU niches |
| **llama.cpp / Ollama** | ○ | ○ | ○ | ● (GGUF, CPU/Metal) | ● | C++ | Local, CPU, Apple Silicon, edge ([Phase 9](../../ROADMAP.md#phase-9--beyond-llms-recsysvisionspeech-hardware-diversity-edge-and-security)) |

Read the table by column, not by row. **Columns 2 and 3 are where the industry innovated in 2023-2025** (continuous batching, paging, prefix caching, cache-aware scheduling); column 5 is a shared commons; column 1 is where operational ergonomics live.

---

## Engine vs server: the distinction people get wrong

```
   ┌──────────────────────────────────────────────────────────────┐
   │  DEPLOYMENT SHAPE A — engine is the server (most LLM work)    │
   │                                                               │
   │   client ──HTTP──▶ vLLM (`vllm serve`) ──▶ GPU                │
   │            ▲ OpenAI-compatible; one model per process         │
   │   scale by replicas + a load balancer (Phase 6)               │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  SHAPE B — Triton Inference Server hosts many models/backends │
   │                                                               │
   │   client ──HTTP/gRPC──▶ Triton ──┬── TRT-LLM backend  (LLM)   │
   │                                  ├── ONNX Runtime     (ranker)│
   │                                  ├── Python backend   (pre/post)
   │                                  └── ENSEMBLE ties them into  │
   │                                      one served graph          │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  SHAPE C — Ray Serve composes independently-scaled services   │
   │                                                               │
   │   client ─▶ Ingress deployment ─▶ Embedder (2 replicas, GPU)  │
   │                    └────────────▶ LLM (4 replicas, GPU)       │
   │                    └────────────▶ Policy (10 replicas, CPU)   │
   │             each autoscales separately on queue depth          │
   └──────────────────────────────────────────────────────────────┘
```

- **Shape A** is what you use for a chat/completions product. Simple, fewest moving parts, best per-GPU LLM performance.
- **Shape B** is what you use when the product is a *pipeline* of models with different frameworks and different batching needs, and you want one endpoint, one process, and in-server tensor passing (no network hop between stages). This is the classic recsys/CV/moderation shape, and it's why Triton is everywhere outside pure-LLM shops.
- **Shape C** is what you use when stages need *independent autoscaling* and arbitrary Python between them, and you're already on Ray/Kubernetes.

They compose: Triton with a vLLM or TRT-LLM backend, Ray Serve with vLLM engines inside deployments, all fronted by a Kubernetes Service ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)).

---

## What actually differs between the LLM engines

Not the ideas — everybody implements continuous batching, paging and prefix caching now. What differs:

| Dimension | Why it decides your choice |
|---|---|
| **Defaults** | The single biggest source of "framework X is faster" claims. Chunked prefill, prefix caching, CUDA graphs, and the token budget default differently across engines and versions. Level them ([lesson 10](10-build-shootout-and-ensemble.md)) or your comparison is noise. |
| **Model coverage & time-to-support** | vLLM/SGLang typically support a new open model within days; TRT-LLM needs a supported architecture and a build. If your product tracks new models, this dominates every perf argument. |
| **Quantization support** | Which checkpoints load at all (GPTQ/AWQ/compressed-tensors/FP8/NVFP4), and which have fast kernels on *your* GPU generation. A format that loads but falls back to a slow dequant path is a trap ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)). |
| **Feature surface** | LoRA multi-adapter serving, structured/grammar output, multimodal inputs, tool calling, beam search, prompt logprobs, embeddings/reranking endpoints. |
| **Operational shape** | Startup time (seconds vs a 10-40 min engine build), memory-config safety, metrics quality, graceful degradation under overload, and how easy it is to run N models on one box. |
| **Hardware** | TRT-LLM is NVIDIA-only by construction. vLLM/SGLang run on NVIDIA + AMD (+ others in varying states). llama.cpp owns CPU/Metal. |
| **Codebase you can actually modify** | If your plan includes patching the scheduler, a 100k-line Python codebase you can `pdb` into is worth more than a faster C++ one you can't. |

**Published performance claims, treated correctly:** the vLLM paper (Kwon et al., SOSP '23) reports 2-4× higher throughput than then-current systems at the same latency, from paging alone; SGLang's RadixAttention paper reports large gains on *shared-prefix* workloads specifically; NVIDIA publishes TRT-LLM numbers on NVIDIA hardware with NVIDIA-selected configs. All three are true and none of them predicts your workload. The number that predicts your workload is the one you measure with your own harness on your own traffic shape — that's lesson 10 and it's the reason Phase 3 lesson 7 exists.

---

## Choosing: a procedure, not a preference

```
  Is the product one LLM behind a chat/completions API?
      │ yes                                     │ no
      ▼                                         ▼
  Do you need max NVIDIA perf on a           Multi-stage pipeline of models?
  model that will not change for months?         │ yes
      │ yes            │ no                      ▼
      ▼                ▼                     Do stages need independent
  TensorRT-LLM     Heavy shared prefixes    autoscaling / arbitrary Python?
  (+ Triton)       or grammar-constrained       │ yes        │ no
                   output?                       ▼            ▼
                       │ yes    │ no        Ray Serve    Triton ensemble
                       ▼        ▼                        (one process, no
                   SGLang     vLLM                        network hop, per-stage
                                                          dynamic batching)
  Special cases:
   • CPU-only / laptop / edge  ─────────────▶ llama.cpp (GGUF), ONNX Runtime
   • Deep in the HF ecosystem, want the
     simplest ops story for one model ──────▶ TGI
   • Model too big for the GPUs you have,
     latency not critical  ─────────────────▶ offload (DeepSpeed ZeRO-Inference,
                                               llama.cpp partial offload)
```

Then sanity-check the choice against three questions that sink more deployments than throughput ever does:

1. **Does it load my exact checkpoint, in my exact quantization, on my exact GPU?** Verify before planning around it.
2. **What happens at 3× my expected load?** Queue and shed ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)), or OOM and drop the whole batch?
3. **Who debugs it at 2 a.m.?** Metrics quality and source readability are operational features. Rank them.

---

## The vocabulary map (same idea, five names)

You will read all of these words this phase. They are mostly the same mechanisms you already built.

| Concept (Phase 3/4 name) | vLLM | TGI | SGLang | TensorRT-LLM | Triton Server |
|---|---|---|---|---|---|
| Continuous batching | continuous batching | continuous batching | continuous batching | **in-flight batching** | (n/a — dynamic batching is *static* per batch) |
| Token budget per step | `max_num_batched_tokens` | `max_batch_prefill_tokens`, `max_batch_total_tokens` | `chunked_prefill_size`, `max_total_tokens` | `max_num_tokens` | — |
| Max concurrent seqs | `max_num_seqs` | `max_concurrent_requests` | `max_running_requests` | `max_batch_size` | `max_batch_size` (per model) |
| KV block pool | block manager / `block_pool.py` | block allocator | token-to-KV pool + allocator | KV cache manager | — |
| Prefix cache | automatic prefix caching (APC) | prefix caching (radix) | **RadixAttention** | KV-cache reuse | — |
| Preemption | preempt + recompute/swap | (re-queue) | retract | pause/evict requests | — |
| Speculative decoding | `speculative_config` | (medusa/n-gram variants) | speculative decoding (EAGLE) | draft-target / Medusa / EAGLE | — |
| Batch wait window | (none — steps are continuous) | `waiting_served_ratio` | schedule policy | — | `max_queue_delay_microseconds` |

That last row is the one to stare at. **Triton's `max_queue_delay_microseconds` is Phase 3's dynamic batching** — wait a bit to form a bigger batch, pay latency for throughput — and it exists because Triton batches *whole requests* of a non-generative model. LLM engines batch *steps*, so they never need a wait window. Knowing which of those two worlds you're in prevents most config mistakes in this phase.

---

## Do this now (30 minutes, no GPU required)

1. `pip install vllm` (or use the CPU/docker image) and run `vllm serve --help | wc -l`. It's a few hundred lines. Skim it and mark every flag you recognize from Phases 3-4. That fraction — usually 60-70% — is what those phases bought you.
2. Clone `vllm-project/vllm`, `huggingface/text-generation-inference` and `sgl-project/sglang`. For each, find the scheduler file and open it. Time yourself. ([Lesson 9](09-reading-engine-source.md) makes this systematic; do it cold first so you can feel the difference.)
3. Write, in your own words, one paragraph per framework: *which layer is this project actually about?* Keep it — you'll grade it against reality at the end of the phase.

---

**Next:** [vLLM architecture: read the code →](02-vllm-architecture.md) — trace one request from HTTP to GPU and back, naming the file at every hop.
