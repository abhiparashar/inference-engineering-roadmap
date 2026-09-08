# Phase 5 — Production Serving Frameworks (read the masters' code)

> **Goal:** stop writing engines and start operating — and extending — the ones the industry actually runs. By the end you can deploy vLLM, TGI, SGLang, TensorRT-LLM and Triton, map every important flag to the mechanism it controls, benchmark them against each other honestly, and read the source well enough to explain a real GitHub issue's root cause and patch it.

This folder is the long-form version of [Phase 5 in the ROADMAP](../../ROADMAP.md#phase-5--production-serving-frameworks-read-the-masters-code). It is the phase where the previous four stop being theory: everything you built by hand — the scheduler, the KV block pool, the prefix cache, the CUDA graph capture — exists in these codebases, written by people who had to make it survive production.

The intellectual core:

> **Every serving framework is the same five layers — API/router, scheduler, KV-cache manager, model executor, kernels. Frameworks differ in which layer they optimize, which they make configurable, and which they hide. Once you can name the five layers in any codebase, "learning a new framework" is a one-day exercise instead of a one-month one.**

---

## Prerequisites

- **[Phase 3 lessons 4, 6, 7](../phase-3/04-continuous-batching.md)** — you must have written a continuous-batching loop and an honest open-loop benchmark harness. Every framework in this phase is a hardened version of that loop, and you will benchmark all of them with *your* harness, not theirs.
- **[Phase 4 lessons 5, 6, 7](../phase-4/05-paged-attention.md)** — block tables, prefix caching, speculative decoding. You will recognize your own `BlockPool` in `vllm/v1/core/block_pool.py`, and that recognition is the entire point of doing Phase 4 first.
- **[Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)** — CUDA graphs and warmup, which explain `--enforce-eager`, TensorRT-LLM's build step, and why the first request after startup is slow.

### About hardware

- **Laptop / Apple Silicon is enough** for: reading source (lessons 2, 4, 5, 9), Triton Inference Server with CPU backends and ensembles (lesson 7), Ray Serve composition (lesson 8), and all the config/flag work.
- **One rented GPU (24 GB, ~$0.30-0.50/hr)** is enough for the framework shootout in lesson 10 Part A with a 1.5-8B model. Budget 4-6 hours of GPU time for the whole phase; that's under $5 on vast.ai/RunPod.
- **TensorRT-LLM (lesson 6) requires NVIDIA hardware** and a long engine build. If you skip the hands-on part, do not skip the mental model — the build-time-vs-runtime split is the interview question.

---

## The map of this phase

```
                        A REQUEST ARRIVES
                               │
   ┌───────────────────────────▼────────────────────────────┐
   │ 1. API / ROUTER   auth, validation, tokenize, OpenAI    │  vLLM: Python FastAPI
   │                   schema, streaming (SSE), multi-replica│  TGI:  Rust router
   │                   routing                               │  Triton: C++ HTTP/gRPC
   ├─────────────────────────────────────────────────────────┤
   │ 2. SCHEDULER      queue, admission, continuous batching, │  vLLM: scheduler.py
   │                   chunked prefill, preemption, priority  │  TGI:  queue.rs
   │                                                          │  TRT-LLM: C++ batch mgr
   ├─────────────────────────────────────────────────────────┤
   │ 3. KV-CACHE MGR   block pool, block tables, prefix cache, │  vLLM: kv_cache_manager
   │                   eviction, offload/swap                  │  SGLang: radix_cache.py
   ├─────────────────────────────────────────────────────────┤
   │ 4. MODEL EXECUTOR forward pass, TP/PP, sampling,          │  vLLM: gpu/model_runner
   │                   CUDA graphs, LoRA, quantized weights    │  TRT-LLM: compiled engine
   ├─────────────────────────────────────────────────────────┤
   │ 5. KERNELS        paged attention, FlashAttention,        │  FlashInfer, FA2/3,
   │                   fused MLP, quant GEMMs                  │  CUTLASS, Triton lang
   └─────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
   SINGLE-MODEL ENGINES                        MULTI-MODEL / PIPELINE LAYER
   vLLM, TGI, SGLang, TRT-LLM                  Triton Server, Ray Serve
   (lessons 2-6)                               (lessons 7-8)
```

Two things fall out of this picture, and they are the two most common mistakes engineers make in this space:

1. **Triton Inference Server is not a competitor to vLLM.** It's layer 1 + a model repository + ensembles; the LLM engine plugs *into* it (TRT-LLM backend, vLLM backend, Python backend). Confusing the two is the tell of someone who's read blog posts, not configs.
2. **"Framework X is 20% faster" is almost always a configuration difference, not an engine difference.** Same layers, same techniques, different defaults for `max_num_batched_tokens`, prefix caching, chunked prefill, and CUDA graphs. Lesson 10 exists to teach you to level those before reporting a number.

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [The serving-stack landscape](01-the-serving-stack-landscape.md) | "Every engine is five layers. I can place vLLM, TGI, SGLang, TensorRT-LLM, Triton, Ray Serve and llama.cpp on that grid, say which layer each one is actually good at, and pick one for a given workload with reasons that aren't vibes." |
| 2 | [vLLM architecture: read the code](02-vllm-architecture.md) | "I traced a request from `/v1/completions` through the API server, the engine core process, the scheduler's token budget, the KV-cache manager's block allocation, into the model runner's batched forward — and I can name the file for each hop." |
| 3 | [Operating and tuning vLLM](03-vllm-in-production.md) | "Every flag maps to a mechanism I built in Phases 3-4. I can size `--gpu-memory-utilization` and `--max-num-batched-tokens` from arithmetic, read `/metrics` to tell a queueing problem from a memory problem, and diagnose the five classic failure modes." |
| 4 | [TGI and the router/server split](04-tgi-and-the-router-split.md) | "TGI splits a fast Rust router (queue, validation, batching decisions) from Python model shards over gRPC. I can explain what that buys, what it costs, and how `waiting_served_ratio`/`max_batch_prefill_tokens` differ from vLLM's token budget." |
| 5 | [SGLang and RadixAttention](05-sglang-and-radixattention.md) | "RadixAttention makes the prefix cache a first-class scheduling input: a radix tree over token prefixes with LRU eviction, plus cache-aware scheduling that reorders the batch to maximize hits. And I know how grammar-constrained decoding is implemented as a token mask from an FSM." |
| 6 | [TensorRT-LLM and compiled engines](06-tensorrt-llm-and-compiled-engines.md) | "TRT-LLM moves work from run time to build time: a per-model, per-GPU, per-config compiled engine with fused kernels and baked-in shapes, plus a C++ in-flight batching runtime. I can say exactly when that tradeoff is worth the operational pain." |
| 7 | [Triton Inference Server](07-triton-inference-server.md) | "Model repository, `config.pbtxt`, instance groups, dynamic batching with a queue delay, and ensembles/BLS for multi-stage pipelines. I can write the config for a 3-stage pipeline and reason about per-stage batching and latency budget." |
| 8 | [Ray Serve and multi-model composition](08-ray-serve-and-composition.md) | "Deployments, replicas, autoscaling on queue depth, and composing models as Python calls between deployments. I know when this beats Triton ensembles and when it's a distributed-systems tax you didn't need." |
| 9 | [Reading and modifying an engine](09-reading-engine-source.md) | "I can enter a 300k-line codebase, find the scheduler in 10 minutes with three greps, trace a request end-to-end, reproduce a real GitHub issue, explain the root cause from the source, and open a patch." |
| 10 | [Build: framework shootout + Triton ensemble](10-build-shootout-and-ensemble.md) | "Two frameworks, one harness, leveled configs, a report with TTFT/TPOT/throughput/memory and an explanation for each delta. Plus a real multi-stage ensemble with per-stage dynamic batching." |
| 11 | [Exercises & exit artifact](11-exercises-and-artifacts.md) | "Here is the comparison report, the ensemble, and the issue write-up with a source-level root cause." |

---

## How to work through this phase

1. **Read source with your Phase-4 code open in the other window.** The value of lesson 2 comes almost entirely from the moment you recognize your own block pool in someone else's production code — and see the eight things they handle that you didn't.
2. **Never report a framework comparison without leveling the config.** Prefix caching on/off, chunked prefill on/off, eager vs graphs, same max batch tokens, same dtype, same client. Lesson 10 gives you the checklist; an unleveled benchmark is worse than no benchmark because it's confidently wrong.
3. **Flags before source, source before opinions.** For each engine: run it, read its own `--help`, map each flag to a Phase 3/4 mechanism, *then* read the code that consumes the flag.
4. **Pick one engine to go deep on** (vLLM is the default choice: largest community, most jobs, clearest code). Be conversant in the rest.
5. **Do the issue exercise for real.** ([lesson 9](09-reading-engine-source.md)) Finding a live batching/memory issue and explaining it from the source is the single highest-signal thing in this phase for interviews, and it's how OSS contributions start.

**Time budget:** 3-4 weeks part-time. Lessons 2, 3 and 9 are load-bearing; lesson 10 produces the artifact.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Given a framework's GitHub issues page, find a real batching/memory bug report and **explain the root cause from the source**. ([lesson 9](09-reading-engine-source.md))
2. Name the five layers of a serving stack and place any framework on them. ([lesson 1](01-the-serving-stack-landscape.md))
3. Map ten vLLM flags to the mechanisms they control, and size two of them with arithmetic. ([lesson 3](03-vllm-in-production.md))
4. Explain why TGI put the router in Rust and vLLM did not, in terms of what each layer actually does per token. ([lesson 4](04-tgi-and-the-router-split.md))
5. Explain what a Triton `config.pbtxt` dynamic-batching block does, and why an ensemble is not the same thing as a client calling three endpoints. ([lesson 7](07-triton-inference-server.md))

## Projects that belong to this phase

- **[09 — Framework shootout](../../projects/README.md)** (small): the same model on two frameworks, the *same* Phase-3 harness, leveled configs, one report. **This is the Phase 5 exit artifact.**
- **[10 — Triton ensemble pipeline](../../projects/README.md)** (large): a real multi-stage pipeline served as one ensemble with per-stage dynamic batching and a per-stage latency budget.

---

Next after this: **[Phase 6 — Distributed Inference at Scale](../phase-6/README.md)**. Phase 5 is one engine on one GPU, operated well; Phase 6 is when the model doesn't fit on one GPU and the traffic doesn't fit on one machine.
