# Glossary

Every term/acronym used in this repo, in plain English. Alphabetical.

**ANN (Approximate Nearest Neighbor)** — vector search that trades a little accuracy (recall) for a lot of speed. Used in RAG retrieval. Common index types: HNSW (graph-based), IVF (clustering-based).

**Arithmetic intensity** — FLOPs performed per byte of memory moved. Low intensity = memory-bound (you're waiting on memory, not math). The number you compute to decide *which* optimization will help.

**Autoregressive decoding** — generating output one token at a time, each token conditioned on all previous ones. Inherently sequential; the root cause of LLM latency.

**AWQ (Activation-aware Weight Quantization)** — quantization method that protects the small fraction of weight channels that matter most (identified via activation magnitudes).

**BF16 / FP16 / FP32** — brain-float 16-bit, half-precision 16-bit, and single-precision 32-bit floating point. BF16 has FP32's exponent range with less mantissa precision, so it's more robust to overflow than FP16.

**Batching (static / dynamic / continuous)** — grouping requests to use the GPU efficiently. *Static*: fixed groups, wait for the slowest. *Dynamic*: wait a short window to accumulate a batch. *Continuous* (a.k.a. in-flight, iteration-level): swap finished sequences out and new requests in at every decode step — the modern standard (vLLM/TGI).

**CUDA graph** — a captured, replayable sequence of GPU kernel launches. Eliminates per-launch CPU overhead; important for small-batch decode where launch overhead dominates.

**DCGM (Data Center GPU Manager)** — NVIDIA's GPU telemetry system; its Prometheus exporter is the standard way GPU metrics reach dashboards in production fleets.

**Decode** — the phase of LLM inference that generates output tokens one at a time. Memory-bandwidth-bound.

**Disaggregated prefill/decode** — running prefill and decode on separate GPU pools (because they have opposite hardware needs) and shipping the KV-cache between them. See DistServe, Splitwise, Mooncake.

**DLRM (Deep Learning Recommendation Model)** — Meta's reference recommendation architecture; the canonical example of inference dominated by giant embedding tables rather than FLOPs.

**Error budget** — how much SLO violation you're allowed in a window. Spend it on risk (deploys, experiments); when exhausted, freeze changes. From Google SRE practice.

**Execution provider (ONNX Runtime)** — a pluggable backend (CPU, CUDA, TensorRT, CoreML, mobile NPU) that runs the same exported graph on different hardware.

**FlashAttention** — IO-aware exact attention: tiles the computation to keep intermediate values in fast SRAM instead of writing the huge attention matrix to HBM. A memory-movement optimization, not a math approximation.

**GGUF** — the quantized model file format used by `llama.cpp` for efficient CPU/Metal inference.

**Golden signals** — latency, traffic, errors, saturation. The four things you monitor first on any service (Google SRE).

**Goodput** — throughput that actually meets your latency SLO. Raw throughput that violates SLOs is worthless; papers like DistServe optimize goodput specifically.

**GPTQ** — post-training quantization using approximate second-order (Hessian-based) error compensation, layer by layer.

**HBM (High Bandwidth Memory)** — the GPU's main memory (e.g. 80GB on an A100/H100). Large but "slow" relative to on-chip SRAM; most kernel optimization is about touching it less.

**HNSW (Hierarchical Navigable Small World)** — a graph-based ANN index; the common default for vector search. Its knob trades recall against latency/memory.

**KV-cache** — cached Key and Value tensors from previous decode steps so you don't recompute attention over the whole sequence every step. Trades memory for compute; its size (batch × seq_len × layers × heads × dim × dtype) is the main constraint on serving concurrency.

**Little's Law** — L = λW (concurrency = arrival rate × latency). The queueing identity behind capacity planning and why latency explodes near saturation.

**MIG (Multi-Instance GPU)** — NVIDIA feature that partitions one physical GPU into isolated instances with dedicated memory/compute slices. Used for multi-tenant isolation.

**MPS (Metal Performance Shaders)** — PyTorch's Apple Silicon GPU backend. Not CUDA; no Nsight, no CUDA kernels.

**NCCL** — NVIDIA's collective communication library (all-reduce, all-gather) used for multi-GPU tensor/pipeline parallelism.

**Neuron SDK** — AWS's toolkit for compiling/running models on Inferentia/Trainium chips.

**PagedAttention** — vLLM's KV-cache management: store the cache in fixed-size non-contiguous blocks addressed by a per-sequence block table, exactly like OS virtual memory paging. Eliminates fragmentation and enables prefix sharing.

**Pipeline parallelism (PP)** — split a model's *layers* across devices. Introduces idle "bubbles"; needs micro-batching to stay efficient.

**Prefill** — processing the input prompt (all tokens in parallel) before generation starts. Compute-bound. Determines TTFT.

**Prefix caching / prefix sharing** — reusing KV-cache blocks across requests that share a prompt prefix (e.g. the same system prompt). Requires cache-aware ("sticky") routing to be effective across replicas.

**Prompt injection** — untrusted input (user text or retrieved documents) hijacking model behavior. The LLM-era analogue of SQL injection.

**Quantization** — representing weights/activations/KV-cache in fewer bits (INT8, INT4) to cut memory footprint and bandwidth pressure. The highest-leverage single optimization for memory-bound decode.

**RAG (Retrieval-Augmented Generation)** — retrieve relevant documents, inject them into the prompt, then generate. Adds a retrieval latency line item before the model even starts.

**Roofline model** — plot of achievable performance vs arithmetic intensity; shows whether a kernel is limited by compute or memory bandwidth. Your first diagnostic before optimizing anything.

**SLO / SLI / SLA** — Service Level Objective (target, e.g. "p99 TTFT < 500ms"), Indicator (the measurement), Agreement (the contract with consequences).

**SM (Streaming Multiprocessor)** — a GPU's core execution unit; a GPU has many. Occupancy = how well you keep them busy.

**Speculative decoding** — a small "draft" model proposes several tokens; the big model verifies them in one parallel pass and accepts the matching prefix. Converts sequential decode into parallel verification. Speedup is bounded by acceptance rate. Variants: Medusa (extra heads), EAGLE.

**SRAM / shared memory** — tiny, very fast on-chip GPU memory per SM. FlashAttention's whole trick is keeping work here instead of in HBM.

**Tensor parallelism (TP)** — split individual weight matrices across GPUs; requires an all-reduce per layer, so it wants fast intra-node interconnect (NVLink).

**TPOT (Time Per Output Token)** — average inter-token latency during streaming. Dominated by decode step cost. What users perceive as "typing speed."

**TTFT (Time To First Token)** — latency until the first token appears. Dominated by queueing + prefill. What users perceive as "responsiveness."

**Warp** — a group of 32 GPU threads executing in lockstep. Branching within a warp ("warp divergence") wastes cycles.

**XLA** — a compiler for tensor programs (used heavily on TPUs) that fuses ops into optimized kernels ahead of time.
