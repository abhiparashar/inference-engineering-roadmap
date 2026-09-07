# Consolidated Resources

Everything referenced across the roadmap, in one place, so you can use it as a reading queue instead of hunting through phases. Phase numbers tell you *when* it becomes relevant.

---

## Books

| Book | Author | Phase | What to read |
|---|---|---|---|
| Designing Data-Intensive Applications | Kleppmann | 0, 3 | Ch. 1-2 early; Ch. 8 during Phase 3 (tail latency, distributed trouble) |
| Computer Systems: A Programmer's Perspective | Bryant & O'Hallaron | 0-2 | Ch. 1, 6 (memory hierarchy), 9 (virtual memory) — the last one makes PagedAttention obvious |
| Programming Massively Parallel Processors | Kirk, Hwu, El Hajj | 2 | Ch. 1-6 minimum (CUDA model, memory, tiling) |
| Designing Machine Learning Systems | Chip Huyen | 5, 8 | Ch. 7 (deployment), 9-10 (infrastructure/MLOps) |
| Systems Performance | Brendan Gregg | 7 | Methodology chapters (USE method); reference the rest |
| Site Reliability Engineering (free online) | Google | 7 | Ch. 4 (SLOs), Ch. 6 (monitoring distributed systems) |

## Papers (roughly in reading order)

| Paper | Why it matters | Phase |
|---|---|---|
| Attention Is All You Need (Vaswani et al., 2017) | The architecture everything else optimizes | 1 |
| FlashAttention / FlashAttention-2 (Dao et al.) | IO-aware attention; the canonical "memory movement is the bottleneck" result | 2 |
| Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22) | Origin of iteration-level / continuous batching | 3 |
| Efficient Memory Management for LLM Serving with PagedAttention (vLLM, SOSP '23) | KV-cache paging; the basis of modern serving | 4 |
| GPTQ (Frantar et al.) | Post-training weight quantization with error compensation | 4 |
| AWQ (Lin et al.) | Activation-aware quantization; widely used in production stacks | 4 |
| Fast Inference from Transformers via Speculative Decoding (Leviathan et al.) | Draft-and-verify decoding | 4 |
| Megatron-LM (Shoeybi et al.) | Tensor/pipeline parallelism mechanics | 6 |
| DistServe (OSDI '24) | Prefill/decode disaggregation, goodput framing | 6 |
| Splitwise (Microsoft, ISCA '24) | Phase-splitting LLM inference across hardware pools | 6 |
| Mooncake (Moonshot AI) | KV-cache-centric disaggregated production architecture | 6 |
| DLRM (Naumov et al., Meta) | Recommendation inference: embedding-dominated, not FLOP-dominated | 9 |

## Engineering blogs worth reading regularly

**Inference-specific (highest signal):**
- [vLLM blog](https://blog.vllm.ai/) — release notes double as inference-optimization tutorials
- [Anyscale blog](https://www.anyscale.com/blog) — the continuous batching explainer is required reading (Phase 3)
- [LMSYS org blog](https://lmsys.org/blog/) — SGLang, RadixAttention, serving benchmarks
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/) — TensorRT-LLM, CUDA, Nsight, quantization
- [HuggingFace blog](https://huggingface.co/blog) — TGI, quantization (bitsandbytes/GPTQ/AWQ), practical deployment
- [PyTorch blog](https://pytorch.org/blog/) — `torch.compile`, CUDA graphs, "Accelerating Generative AI" series
- Inference-cost-focused startups (unusually detailed engineering writeups): **Baseten**, **Modal**, **Together AI**, **Fireworks AI**, **Character.AI** engineering blogs

**Systems/perf fundamentals:**
- [Horace He — "Making Deep Learning Go Brrrr From First Principles"](https://horace.io/brrr_intro.html) (Phase 2 — read this twice)
- [Lilian Weng — "Large Transformer Model Inference Optimization"](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (Phase 1/4 orientation)
- [Marc Brooker (AWS)](https://brooker.co.za/blog/) — queueing, tail latency, distributed systems reality
- [Julia Evans](https://jvns.ca/) — systems/networking/debugging fundamentals explained accessibly
- [Jay Alammar — Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Phase 1)

**Scale/platform case studies:**
- [Uber Engineering — Michelangelo](https://www.uber.com/blog/michelangelo-machine-learning-platform/), Netflix Tech Blog, Meta AI blog, Google Cloud AI/GKE blog (GPU autoscaling), AWS ML blog (Inferentia/Trainium)

## Open-source codebases to read (not just install)

| Repo | Read this specifically | Phase |
|---|---|---|
| `karpathy/nanoGPT` | `model.py` — whole GPT in ~300 lines | 1 |
| `huggingface/transformers` | `models/llama/modeling_llama.py` — KV-cache handling in plain PyTorch | 1 |
| `Dao-AILab/flash-attention` | README algorithm section + kernel tiling comments | 2 |
| `pytorch/pytorch` | `aten/src/ATen/native/cuda/` — real kernel structure | 2 |
| `vllm-project/vllm` | `vllm/core/scheduler.py`, `vllm/core/block_manager.py`, `vllm/entrypoints/openai/`, `benchmarks/benchmark_serving.py` | 3, 4, 5, 7 |
| `huggingface/text-generation-inference` | Rust router queue/batching logic | 3, 5 |
| `triton-inference-server/server` | `config.pbtxt` dynamic batching + ensemble models | 5 |
| `sgl-project/sglang` | RadixAttention prefix-sharing tree | 4, 5 |
| `NVIDIA/TensorRT-LLM` | In-flight batching + quantization toolkit | 5 |
| `ray-project/ray` | Ray Serve autoscaling + deployment graphs | 5, 6 |
| `NVIDIA/Megatron-LM` | `ColumnParallelLinear` / `RowParallelLinear` | 6 |
| `microsoft/DeepSpeed` | Tensor-parallel inference kernels, ZeRO-inference offload | 6 |
| `ggerganov/llama.cpp` | GGUF format, CPU SIMD kernels, Metal backend | 9 |
| `microsoft/onnxruntime` | Execution provider abstraction | 9 |
| `pytorch/executorch` | On-device/edge runtime | 9 |
| `facebookresearch/faiss` / `qdrant/qdrant` | ANN index internals (HNSW/IVF) | 9 |
| `ray-project/llmperf` | Reference LLM load-testing harness | 3, 5 |

## Tools checklist by phase

- **Phase 0-1**: Python, PyTorch, `transformers`, FastAPI, `httpx`
- **Phase 2**: `nvidia-smi`, Nsight Systems, Nsight Compute, `torch.profiler` + Perfetto/`chrome://tracing`, `triton`, `py-spy`
- **Phase 3**: `locust` (or `k6`/`vegeta`), `asyncio`, vLLM's `benchmark_serving.py` as a reference harness
- **Phase 4**: `bitsandbytes`, `auto-gptq`, `autoawq`, `torch.compile`, CUDA graphs
- **Phase 5**: vLLM, TGI, Triton Inference Server, TensorRT-LLM, Ray Serve, Docker
- **Phase 6**: `torch.distributed`/NCCL, NGINX or Envoy (routing), multi-GPU box
- **Phase 7**: `prometheus_client`, Prometheus, Grafana, DCGM exporter, structured logging
- **Phase 8**: Docker (multi-stage), Kubernetes, Helm, GitHub Actions, Terraform
- **Phase 9**: ONNX Runtime, `llama.cpp`, FAISS/Qdrant, ExecuTorch/CoreML/TFLite

---

## Staying current (after the roadmap, forever)

- **Conferences to follow proceedings of** (papers here become production practice within ~12 months): **MLSys**, **OSDI**, **SOSP**, **NSDI**, **ASPLOS/ISCA** (hardware), **NVIDIA GTC** talks (free recordings — often the most practical content available).
- **Watch these repos' releases** on GitHub: `vllm`, `sglang`, `TensorRT-LLM`, `text-generation-inference`, `llama.cpp`, `flash-attention`. Release notes are a free curriculum on what's currently state of the art.
- **Read the issue trackers**, not just the docs. Real performance problems, regressions, and workarounds live in GitHub issues and PR discussions — this is where you learn how experienced engineers reason about these systems.

## Your first OSS contribution (the real top-1% differentiator)

Reading source code makes you good; contributing makes it verifiable. A merged PR in a serving framework is worth more than any certificate. Progression that actually works:

1. **Docs/typos/clarity fixes** in vLLM/TGI/llama.cpp — learn the contribution workflow (CLA, CI, review etiquette) with zero risk.
2. **Reproduce and triage a bug**: pick an open performance/correctness issue, reproduce it locally, post a minimal reproduction with profiler evidence. Maintainers value this enormously and it's often the hardest part of the fix.
3. **Add a benchmark or test** for an existing feature — low review risk, high learning, directly uses your [`playbooks/benchmarking.md`](../playbooks/benchmarking.md) skills.
4. **Fix a small bug** you triaged in step 2.
5. **Implement a feature** (a new quantization backend hook, a scheduler policy option, a metric) — by now you know the codebase and the maintainers know you.

Do steps 1-2 during Phase 5, when you're already deep in these codebases anyway.

## Career layer — what this roadmap qualifies you for

Job titles that map to this skillset: **Inference Engineer**, **ML Systems Engineer**, **Performance Engineer (ML)**, **ML Platform/Infrastructure Engineer**, **GPU/Kernel Engineer** (with more Phase 2 depth), **AI Infrastructure SRE** (with more Phase 7-8 depth).

What interviews for these roles actually probe (map each to a phase):
- "Walk me through what happens when a request hits your inference server." → Phases 1, 3
- "Your p99 latency doubled after a deploy; how do you debug it?" → Phases 2, 7 (methodology matters more than the answer)
- "How would you cut serving cost in half?" → Phases 4, 6, 7 (expect follow-ups on quality tradeoffs)
- "Design a serving system for N QPS with a p99 SLO of X ms." → Phases 3, 6 (capacity math, Little's Law, batching tradeoffs)
- "This model doesn't fit on one GPU. Options?" → Phases 4, 6 (quantize, shard, offload — with tradeoffs)
- Deep-dive on your own projects → this is why the *artifact-per-phase* rule matters. Your capstone writeups with honest numbers are the interview.

Portfolio presentation tips: for each project, publish a short writeup with (1) the problem, (2) baseline numbers, (3) what you changed, (4) after numbers with percentiles, (5) what you'd do next and why. That structure signals seniority far more than the code itself.
