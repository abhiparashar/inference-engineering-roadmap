# 6 — TensorRT-LLM and Compiled Engines

> **You'll be able to say:** "TensorRT-LLM's original bet was to move work from run time to *build* time: compile a model + GPU + parallelism + max-shape configuration into a serialized engine with fused, autotuned kernels. That bought peak NVIDIA performance and cost you a 10-40 minute rebuild for every config change, per GPU generation. NVIDIA has since made a PyTorch-based runtime the default path — because model velocity beat compile-time specialization — while keeping the parts that were always the real differentiator: hand-tuned kernels, first-class FP8/NVFP4 quantization, and a C++ in-flight-batching runtime whose capacity scheduler exposes the admission policy as an explicit choice (`MAX_UTILIZATION` vs `GUARANTEED_NO_EVICT`)."

This lesson is as much about a *tradeoff pattern* — specialize ahead of time vs stay flexible at run time — as about one framework. That pattern recurs in `torch.compile`, ONNX Runtime, XLA/TPU, and every edge deployment you'll ever do.

---

## The two philosophies

```
   RUN-TIME SPECIALIZATION (vLLM, SGLang, TGI, TRT-LLM PyTorch backend)
   ─────────────────────────────────────────────────────────────────────
   load checkpoint → profile memory → capture CUDA graphs for N batch sizes
                   → serve in ~1-2 minutes
   change a flag → restart, ~1-2 minutes
   new model architecture → often supported within days

   BUILD-TIME SPECIALIZATION (classic TensorRT / TRT-LLM engine workflow)
   ─────────────────────────────────────────────────────────────────────
   checkpoint → convert → BUILD (graph fusion, kernel autotuning per shape,
                                 precision selection, plugin insertion,
                                 TP/PP baked in) → serialized .engine (10-40 min)
                   → serve, with almost no per-step framework overhead
   change TP size / max batch / max seq len / GPU model → REBUILD
   new model architecture → wait for support, or write the definition yourself
```

The build-time bet is the same one a compiler makes: if the shapes and the hardware are known in advance, you can fuse aggressively, pick the best kernel per shape by *measuring* candidates, and delete dispatch overhead. On a fixed model at fixed scale, that wins.

**What actually happened, and why it matters more than either philosophy:** the LLM ecosystem's shape kept changing — new architectures monthly, LoRA, multimodal, spec decode variants, disaggregation — so a workflow whose unit of change is a 30-minute rebuild bled to death against Python engines. NVIDIA's answer (the TRT-LLM 1.x line) is a **PyTorch-based runtime** (`tensorrt_llm/_torch/pyexecutor/`) as the default path, keeping the C++ runtime, the kernels and the quantization stack, and dropping the mandatory ahead-of-time engine build. Read that as an industry verdict: *specialize the kernels, not the whole graph.*

If you find a tutorial built around `trtllm-build` and `.engine` files, you're reading the 2023-2024 workflow. Check the version you have before following it.

---

## What remains, and is genuinely excellent

### 1. Kernels and precision

TRT-LLM is where NVIDIA ships its own hardware's best paths first: FP8 (Hopper) and NVFP4 (Blackwell) attention/GEMM, fused MoE kernels, FlashInfer-class attention, and quantization via **TensorRT Model Optimizer** (ModelOpt): FP8, NVFP4, INT8 SmoothQuant, INT4 AWQ, with calibration. Everything you studied in [Phase 4 lesson 3](../phase-4/03-quantization-methods.md) exists here as a supported toolchain rather than a collection of community repos — and on the newest silicon, the fast path usually appears here first.

If your fleet is H100/H200/B200 and cost per token is the product metric, this is the lever: **FP8 or NVFP4 weights + KV on hardware with native support is a large, measurable win**, and TRT-LLM's implementation is typically the reference one.

### 2. The C++ in-flight batching runtime

`cpp/tensorrt_llm/batch_manager/` is the best-documented *C++* implementation of everything you built in Phase 3-4. Read these:

| File | What it is |
|---|---|
| `capacityScheduler.{h,cpp}` | **Which requests are allowed to run** — admission control, as named policies |
| `microBatchScheduler.{h,cpp}` | Which of those actually go into the next forward pass (context vs generation, token budget) |
| `kvCacheManager.cpp`, `kv_cache_manager_v2/blockRadixTree.cpp` | Paged KV blocks + a radix tree for reuse — the Phase 4 lesson 5-6 pair, in C++ |
| `evictionPolicy.cpp` | Block eviction (LRU and friends) |
| `llmRequest.{h,cpp}` | The request state machine: context phase → generation phase, draft tokens, stop criteria |
| `cacheTransceiver.cpp`, `cacheFormatter.cpp` | KV transfer between instances — prefill/decode disaggregation ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)) |

The capacity scheduler is the highest-value read because it names the policy space that vLLM leaves implicit:

| Policy | Behavior | Consequence |
|---|---|---|
| `MAX_UTILIZATION` | Pack as many requests as possible; accept that some may have to be **paused/evicted** and recomputed later | Highest throughput, non-trivial tail risk — this is the preemption thrash of [lesson 3](03-vllm-in-production.md), chosen deliberately |
| `GUARANTEED_NO_EVICT` | Only admit a request if its worst-case KV (to `max_seq_len`) is guaranteed available | No preemption ever, lower concurrency — literally the "reserve max" allocator from [Phase 4 lesson 5](../phase-4/05-paged-attention.md), with its 30% utilization |
| `STATIC_BATCH` | Classic static batching: batch runs to completion | For non-generative or legacy workloads ([Phase 3 lesson 2](../phase-3/02-static-batching.md)) |
| `MAX_REQUESTS` | Simple count cap | Baseline |

**Being able to explain that `GUARANTEED_NO_EVICT` trades utilization for a hard no-preemption guarantee is a genuinely senior answer** to "how would you protect p99 under load?" — it's a knob, and it's the right one when your SLO punishes latency variance more than it rewards throughput.

### 3. Serving surface

- **`trtllm-serve`** — an OpenAI-compatible server. Your Phase-3 harness works against it unchanged, which is what makes it comparable in the [lesson 10](10-build-shootout-and-ensemble.md) shootout.
- **`trtllm-bench`** — NVIDIA's own benchmark driver. Useful, and to be treated exactly like any vendor harness: read what it measures (closed vs open loop, warmup, what counts as a token) before believing a number, per [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md).
- **The LLM API** (`tensorrt_llm.LLM(...)`) — a Python API deliberately shaped like vLLM's, so migration is mostly config translation.
- **Triton backend** (`tensorrtllm_backend`) — run TRT-LLM as a model inside Triton Inference Server when you need Triton's multi-model/ensemble features ([lesson 7](07-triton-inference-server.md)).

---

## Configuration mapping

| Concept | TRT-LLM | vLLM |
|---|---|---|
| Token budget per iteration | `max_num_tokens` | `--max-num-batched-tokens` |
| Max concurrent requests | `max_batch_size` | `--max-num-seqs` |
| Context length | `max_seq_len` | `--max-model-len` |
| KV pool sizing | `kv_cache_config.free_gpu_memory_fraction` | `--gpu-memory-utilization` |
| Prefix reuse | `enable_block_reuse` | `--enable-prefix-caching` |
| KV quantization | `kv_cache_config.dtype` (fp8) | `--kv-cache-dtype fp8` |
| Chunked prefill | `enable_chunked_context` | `--enable-chunked-prefill` |
| Admission policy | `capacity_scheduler_policy` | (implicit: max-utilization with preemption) |
| CUDA graphs | `cuda_graph_config` (batch-size list) | capture sizes / `--enforce-eager` |
| Parallelism | `tp_size`, `pp_size`, `moe_ep_size` | `--tensor-parallel-size`, … |
| Spec decode | `speculative_config` (EAGLE3, MTP, draft-target, n-gram) | `--speculative-config` |

Same nouns everywhere. That's the payoff of Phases 3-4: **you are configuring mechanisms, not learning products.**

---

## When to choose it — and when not to

**Choose TRT-LLM when:**
- You run a **stable, supported** model on NVIDIA hardware at enough scale that a 10-30% cost delta pays for the operational overhead.
- You need FP8/NVFP4 on Hopper/Blackwell at the earliest maturity available.
- You're already a Triton shop and want the LLM stage of a pipeline to be NVIDIA-native.
- You need features NVIDIA ships first for its own hardware (certain MoE optimizations, disaggregated serving with NIXL/UCX transports, multi-node NVL topologies).

**Don't when:**
- Your model set changes frequently, or you serve many long-tail fine-tunes.
- Your team can't absorb a C++/CUDA debugging surface at 2 a.m.
- You need non-NVIDIA portability. TRT-LLM is NVIDIA-only by construction; that's a strategic lock-in decision, not a technical detail, and it belongs in the design doc.

**The honest performance statement**, which you should be able to defend: *on a supported model, on current NVIDIA hardware, with both configurations properly tuned, TRT-LLM and vLLM/SGLang land in the same neighborhood, with TRT-LLM typically ahead on the newest precision formats and NVIDIA-specific features, and behind on breadth and time-to-support.* Anyone quoting a large fixed multiple in either direction is quoting an unleveled benchmark — usually one where prefix caching, chunked prefill, or CUDA graphs were on for one side and off for the other.

---

## Do this (2-3 hours; only the last step needs an NVIDIA GPU)

1. Read `cpp/include/tensorrt_llm/batch_manager/capacityScheduler.h` (the header alone is enough) and write the four policies with one sentence each on when you'd pick them. Map each to a Phase 3/4 concept.
2. Read `microBatchScheduler.h`: find how it splits context (prefill) requests from generation (decode) requests and how the token budget is spent. Compare to vLLM's single unified loop from [lesson 2](02-vllm-architecture.md) — this is a real design difference worth being able to describe.
3. Skim `kv_cache_manager_v2/blockRadixTree.h`. It's the third radix prefix cache you've now seen (yours, SGLang's, this one). Note what all three agree on: block-aligned keys, refcount/lock protection, LRU over evictable leaves.
4. Read NVIDIA's own perf-overview docs and identify, for one published number: which GPU, which precision, which ISL/OSL, closed or open loop, and whether it's per-GPU or per-node. Practice extracting the assumptions; the numbers are real but they are answers to a specific question.
5. **(GPU)** Run `trtllm-serve` on a small supported model, point your Phase-3 harness at it, and record the same metrics you record for vLLM. Note the startup time and the memory config path — the operational differences will be more informative than the tokens/sec.

---

**Next:** [Triton Inference Server →](07-triton-inference-server.md) — the layer above the engine: model repositories, per-model dynamic batching, and pipelines served as one graph.
