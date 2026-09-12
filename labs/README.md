# Labs — Hands-On Exercises Mapped to Roadmap Phases

These are bite-sized, code-reading + doing exercises. Unlike the projects (which are builds), a lab is "go read this exact file in this exact real codebase, run something, and answer a concrete question." This is how you build the instinct to navigate unfamiliar production codebases fast — a core top-1% skill that pure tutorials never teach.

Do these *alongside* the matching phase in [`ROADMAP.md`](../ROADMAP.md), not instead of the projects.

---

## Phase 0 lab — Systems basics
1. Write a raw-socket Python HTTP server (no framework) answering `GET /health` with `200 OK`. Then `strace`/`dtruss` it while a request comes in (or read `py-spy dump` output) and identify every syscall involved.
2. Run `python -X importtime -c "import torch"` and explain what's slow about importing torch (this matters — cold-start import time is a real production concern for autoscaled inference pods).

## Phase 1 lab — Transformer internals
1. Clone `karpathy/nanoGPT`. Open `model.py`. Find `CausalSelfAttention.forward`. Print the shape of `q`, `k`, `v`, and the attention output at every step for a toy input. Draw it on paper.
2. Open `huggingface/transformers`, find `modeling_llama.py` (or any decoder model). Find where `past_key_value` is read and where it's updated. Answer: at generation step N, what is the shape of the K/V cache tensor, and how does it grow?
3. Load real GPT-2 (`transformers`, `gpt2` checkpoint) and manually step through generation one token at a time using `use_cache=True`, printing the cache tensor shapes at each step, to see the KV-cache literally grow in real time.

## Phase 2 lab — GPU/hardware
1. Look up your GPU's (or a Colab T4/A100) spec sheet: peak FP16 TFLOPs and HBM bandwidth (GB/s). Compute the roofline "knee" arithmetic intensity (FLOPs/byte).
2. Run `torch.profiler` around a `torch.matmul` at increasing sizes (e.g. 128, 512, 2048, 8192) and compute achieved TFLOPs at each size. Find where you approach peak.
3. Read `Dao-AILab/flash-attention`'s README algorithm section. Answer in your own words: what specifically does it avoid writing to HBM that standard attention writes?

## Phase 3 lab — Serving fundamentals
1. Open `vllm-project/vllm`'s `vllm/core/scheduler.py`. Find the method that decides which requests run in the next step. Answer: what happens to a request that doesn't fit in the current step's token budget?
2. Read the Orca paper's Figure explaining iteration-level scheduling. Explain in plain English why this beats request-level batching for tail latency.
3. Using `locust` or a raw asyncio client, hit any local server (yours from Project 1, or a public demo) at increasing QPS and plot the latency-vs-QPS curve. Identify the saturation knee visually.

## Phase 4 lab — Optimization techniques
1. Read `vllm-project/vllm`'s `vllm/core/block_manager.py`. Find the block allocation and the prefix-caching (copy-on-write) logic. Answer: what's stored in the "block table" for one sequence?
2. Quantize any small HF model with `bitsandbytes` `load_in_8bit=True`. Compare memory footprint (`torch.cuda.memory_allocated()`) before/after.
3. Read the Speculative Decoding paper's algorithm box. Answer: why is the expected speedup bounded by the draft model's acceptance rate, and what happens in the worst case (0% acceptance)?

## Phase 5 lab — Production frameworks
1. Install vLLM locally (CPU or GPU) and serve any small model with `vllm serve <model>`. Hit its OpenAI-compatible `/v1/completions` endpoint. Then check `/metrics` — list every Prometheus metric it exposes and explain what each one means.
2. Read Triton Inference Server's example `config.pbtxt` for dynamic batching. Identify the fields controlling max batch size and the batching delay window.
3. Compare TGI's router (Rust) request flow vs vLLM's Python scheduler — same concept (continuous batching), two very different implementation languages. Why might a company choose Rust for this layer specifically?

## Phase 6 lab — Distributed inference
1. Read `Megatron-LM`'s `ColumnParallelLinear`/`RowParallelLinear` source. Answer: for a weight matrix of shape `[d_model, d_ff]` split across 2 GPUs, what shape does each GPU actually hold, and where does the `all_reduce` happen?
2. Read the DistServe or Splitwise paper abstract + system diagram. Answer: what physical resource is transferred between the prefill and decode pools, and over what interconnect?
3. Run vLLM with `--tensor-parallel-size 2` (2 GPUs, or simulate reading the code path if you only have 1) and read `vllm/distributed/` to see where `torch.distributed.all_reduce` is invoked in the forward pass.

## Phase 7 lab — Observability/SRE
1. Read the Google SRE book chapter on SLOs (free online). Write one concrete SLO for a hypothetical chat product (e.g. "99% of requests have TTFT < 500ms over a rolling 28-day window").
2. Set up Prometheus + Grafana locally via Docker Compose against your Project 1 server (once instrumented). Build one dashboard panel per golden signal.
3. Read any GPU DCGM exporter metrics list. Identify which 3 metrics you'd alert on for "GPU about to OOM" vs "GPU underutilized (wasting money)."

## Phase 8 lab — MLOps
1. Write a multi-stage Dockerfile for a PyTorch+CUDA app and compare final image size vs a naive single-stage `pip install` Dockerfile.
2. Read the Kubernetes docs page on GPU scheduling. Write a pod spec requesting 1 GPU with a node selector for a specific GPU type.
3. Write a GitHub Actions workflow stub that runs a benchmark script and fails the build if a latency threshold (passed as an env var) is exceeded.

## Phase 9 lab — Beyond LLMs
1. Read `ggerganov/llama.cpp`'s README on GGUF quantization. Run a small quantized model CPU-only and record tokens/sec. Compare against any GPU FP16 number you already collected in Phase 3/4.
2. Read `microsoft/onnxruntime`'s execution-provider concept docs. Export any small model to ONNX and run it through the CPU execution provider; note what changes (or doesn't) if you swap to a different provider.
3. Read the OWASP "Top 10 for LLM Applications" list. Pick 2 vulnerability classes and write one concrete example input that would trigger each against a naive LLM-backed API.
4. Read `facebookresearch/faiss` or `qdrant/qdrant` docs on HNSW. Answer: what's the recall/latency tradeoff knob, and why can't you get perfect recall and minimal latency simultaneously?

## Phase 10 — capstone labs
No labs — this is project-only. See the [Phase 10 deep dive](../phases/phase-10/README.md), [`ROADMAP.md`](../ROADMAP.md) Phase 10, and [`projects/README.md`](../projects/README.md).
