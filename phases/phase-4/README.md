# Phase 4 — Inference Optimization Techniques (Deep Dive)

> **Goal:** make every token cheaper. Phase 3 kept the GPU fed; this phase attacks the two things that are actually expensive — **bytes moved per token** and **tokens generated sequentially** — with the toolbox frontier labs use to get 10-100× cost reductions: quantization, paged KV, prefix caching, speculative decoding, and compilation.

This folder is the long-form version of [Phase 4 in the ROADMAP](../../ROADMAP.md#phase-4--inference-optimization-techniques). It is the phase where the three earlier phases fuse: Phase 1's KV-cache arithmetic, Phase 2's roofline, and Phase 3's scheduler all show up as *constraints that these techniques relax*.

The intellectual core is one sentence, and everything here is a corollary:

> **Decode is memory-bandwidth-bound and sequential. So you either move fewer bytes per token (quantization, KV compression, paging, caching) or stop generating one token at a time (speculative decoding) — and nothing else matters much.**

---

## Prerequisites

- **[Phase 2 lesson 5](../phase-2/05-roofline-model.md)** (the roofline) — you must be able to say why decode at batch 1 has arithmetic intensity ≈ 1 and what that implies. Every claim in this phase is a bytes argument.
- **[Phase 1 lesson 5](../phase-1/05-kv-cache.md)** + **[Phase 3 lesson 4](../phase-3/04-continuous-batching.md)** — KV-cache bytes per token, and why memory (not FLOPs) caps concurrency.
- **[Phase 3 lessons 7-9](../phase-3/07-measuring-honestly.md)** — you will re-use `bench.py` and the continuous-batching engine. **Every technique in this phase must be benchmarked with the harness you already trust**, or you'll believe a vendor's number instead of your own.

### About hardware

Quantization kernels (GPTQ/AWQ/bitsandbytes INT8) generally **require CUDA**. Plan for a Colab T4 or a rented GPU for lessons 2-3's measurements. Everything else — paged KV block management, prefix caching/radix trees, speculative-decoding acceptance math, the scheduler integration — runs fine on a laptop, including Apple Silicon, and those are the lessons with the highest signal per hour anyway.

---

## The map of this phase

```
                     WHERE DOES THE TIME/MONEY GO?
                                  │
        ┌─────────────────────────┴──────────────────────────┐
        ▼                                                    ▼
  BYTES PER TOKEN (decode is bandwidth-bound)          SEQUENTIALITY
        │                                              (one token per step)
   ┌────┴─────┬──────────────┬──────────────┐                │
   ▼          ▼              ▼              ▼                ▼
 weights    KV bytes     KV layout      repeated       speculative
 (quant:    (GQA, KV-    (paging:       prefixes       decoding
  GPTQ/AWQ/  quant,       no frag,      (prefix cache,  (draft + verify:
  INT8/FP8)  windows)     no padding)    RadixAttention) 2-3x, lossless)
 lessons 2-3  lesson 4     lesson 5       lesson 6        lesson 7
        └──────────┬───────────┴──────────────┘                │
                   ▼                                           ▼
          more concurrency = bigger batch          fewer forward passes
                   └───────────────┬───────────────────────────┘
                                   ▼
                    and under all of it: KERNELS & GRAPHS
                     (fusion, FlashAttention/Decoding,
                      torch.compile, CUDA graphs) — lesson 8
```

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [What to optimize, and how to prove it](01-what-to-optimize.md) | "I can take a workload, compute where its bytes and FLOPs go, and name the *one* technique that will help most — plus the number I expect it to move, before I run anything." |
| 2 | [Quantization fundamentals](02-quantization-fundamentals.md) | "Quantization is an affine map from floats to integers with a scale (and maybe a zero point), applied per-tensor, per-channel, or per-group. I can compute the error, the memory saving, and the decode speedup it implies." |
| 3 | [Quantization methods in practice](03-quantization-methods.md) | "GPTQ does second-order error correction, AWQ protects salient channels found from activations, LLM.int8() splits out outlier features, SmoothQuant migrates activation outliers into weights, FP8 does it in hardware. I know which to pick and how to evaluate quality." |
| 4 | [KV-cache optimization](04-kv-cache-optimization.md) | "The KV-cache dwarfs the weights at long context. GQA/MQA, KV quantization, sliding windows and offloading each cut it by a known factor — and each factor converts directly into concurrent sequences." |
| 5 | [PagedAttention](05-paged-attention.md) | "It's OS virtual memory for the KV-cache: fixed-size blocks, a per-sequence block table, no contiguity requirement. That removes 60-80% internal fragmentation and makes copy-on-write prefix sharing possible." |
| 6 | [Prefix caching & RadixAttention](06-prefix-caching-and-radix-attention.md) | "Shared prompts should be computed once. A radix tree over token prefixes turns a 2,000-token system prompt into a cache hit, cutting TTFT and prefill cost by the shared fraction — and it changes how you route requests." |
| 7 | [Speculative decoding](07-speculative-decoding.md) | "A draft model proposes k tokens, the target verifies them in one pass, and rejection sampling makes the output distribution *identical*. Expected speedup is a closed-form function of acceptance rate and cost ratio — and it fails at high batch size." |
| 8 | [Compilation, kernels & graphs](08-compilation-and-kernels.md) | "torch.compile fuses, CUDA graphs delete launch overhead, FlashAttention/FlashDecoding fix attention's IO. I can say which of those a given profile calls for and what each one cannot fix." |
| 9 | [Build: paged KV + prefix sharing + a quantization table](09-build-paged-kv-and-quant-bench.md) | "I implemented a block pool, block tables, copy-on-write prefix sharing, and wired them into my Phase-3 engine — and I have a quantization table with memory, tokens/sec and perplexity for FP16/INT8/INT4." |
| 10 | [Exercises & exit artifact](10-exercises-and-artifacts.md) | "Here is the artifact: measured, reproducible, with the arithmetic that predicted each result." |

---

## How to work through this phase

1. **Predict with arithmetic before every measurement.** Every technique here has a back-of-envelope model (bytes saved, blocks saved, acceptance rate). Write the prediction down, then measure. The gap is where the learning is — and "predicted 2.1×, measured 1.7×, here's the missing 20%" is the most credible sentence you can put in a portfolio.
2. **One variable at a time, with the Phase 3 harness.** Quantization changes memory *and* speed *and* quality. Report all three or you're cherry-picking.
3. **Read the vLLM block manager with lesson 5 open.** It's the single highest-value code read in the roadmap.
4. **Do the large build.** A working paged KV-cache with prefix sharing, integrated into your own engine, is the most convincing artifact in this entire track.

**Time budget:** 3-5 weeks part-time. Lessons 5 and 7 are the load-bearing ones conceptually; lesson 9 is where the portfolio piece comes from.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Explain **PagedAttention's block table** in terms of OS virtual memory paging, in under 60 seconds. ([lesson 5](05-paged-attention.md))
2. Explain why INT4 barely speeds up **prefill** but strongly speeds up **decode**. ([lesson 2](02-quantization-fundamentals.md), [Phase 2 lesson 5](../phase-2/05-roofline-model.md))
3. Derive the **expected speedup** of speculative decoding from acceptance rate and draft/target cost ratio, and say when it stops helping. ([lesson 7](07-speculative-decoding.md))
4. Given a 7B model, 80 GB GPU and 8k contexts, compute how many concurrent sequences you get — and how that number changes with GQA, INT8 KV, and paging. ([lesson 4](04-kv-cache-optimization.md), [lesson 5](05-paged-attention.md))

## Projects that belong to this phase

- **[03 — Quantization benchmark](../../projects/README.md)** (small): FP16 vs INT8 vs INT4 (GPTQ/AWQ) with memory, tokens/sec, TTFT/TPOT and perplexity. One table, fully reproducible.
- **[06 — Simplified PagedAttention](../../projects/README.md)** (large): block pool, block tables, copy-on-write prefix sharing, integrated into your Phase-3 continuous-batching engine, with a fragmentation and prefix-hit-rate study. **This is the Phase 4 exit artifact.**

---

Next after this: **[Phase 5 — Production Serving Frameworks](../phase-5/README.md)**. Phase 4 is where you learn what the frameworks are *doing*; Phase 5 is where you operate and modify them — and you'll recognize every flag.
