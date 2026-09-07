# Phase 2 — GPU Architecture & Low-Level Performance (Deep Dive)

> **Goal:** stop guessing why your model is slow. By the end of this phase, "it's memory-bound" is not an opinion you formed on paper — it's a number you read off a profiler, and you know which of the three fixes (batch it, shrink the bytes, remove the overhead) applies.

This folder is the long-form version of [Phase 2 in the ROADMAP](../../ROADMAP.md#phase-2--gpu-architecture--low-level-performance). Phase 1 told you *what* a transformer does and *why* decode is memory-bound in theory. Phase 2 is about the machine that runs it: what a GPU actually is, where its memory lives, how a kernel executes, and how to measure all of it.

This is the phase that separates people who *use* vLLM from people who can explain *why* vLLM is fast — and eventually contribute to it.

---

## Prerequisites

Finish [Phase 0](../phase-0/README.md) and [Phase 1](../phase-1/README.md) first. Specifically you need:

- **Memory hierarchy + bandwidth** ([Phase 0 lesson 2](../phase-0/02-memory-hierarchy.md)) — Phase 2 is the same story, one level down and 100× faster.
- **Precision / bytes per number** ([Phase 0 lesson 5](../phase-0/05-floating-point-and-precision.md)) — tensor cores and quantization both live here.
- **Arithmetic intensity and "decode is memory-bound"** ([Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md)) — this phase turns that argument into the roofline model and then into profiler output.

### About hardware (read this before you start)

Most of this phase wants an **NVIDIA GPU**. If you're on an Apple Silicon Mac, you have no CUDA, no `nvidia-smi`, and no Nsight — see [GETTING-STARTED](../../GETTING-STARTED.md#free-gpu-options-start-here). The plan that works:

- **Lessons 1-5** (concepts, roofline math, spec sheets): fully doable on any laptop with a calculator. Do them now.
- **Lessons 6-9** (profiling, kernels, benchmarks): use **Google Colab free tier (T4)** or **Kaggle (T4/P100)**. A free T4 is genuinely enough — the roofline shape is the same on a T4 as on an H100, only the numbers move.
- Don't wait for "real" hardware. A measured T4 roofline beats an unmeasured H100 opinion.

---

## The big idea of this phase

There are exactly **three** reasons a GPU program is slow, and every optimization you'll ever do is picking the right one:

```
   1. COMPUTE-BOUND   — the math units are busy; you're near peak FLOPs.       → use better math (tensor cores, lower precision)
   2. MEMORY-BOUND    — the math units are idle, waiting for bytes from HBM.   → move fewer bytes (fusion, quantization, batching)
   3. OVERHEAD-BOUND  — the GPU is idle, waiting for the CPU to tell it what   → launch less (CUDA graphs, bigger kernels,
                        to do next (Python, kernel launches).                     torch.compile)
```

Beginners guess. Top 1% engineers **measure which one it is first**, then fix that one. Fixing the wrong one produces zero speedup and a lot of wasted weeks — this is the single most common failure mode in performance work.

The tool that tells you which one you're in is the **roofline model** (lesson 5), and the tool that confirms it on real hardware is the **profiler** (lesson 8).

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [Why GPUs (CPU vs GPU)](01-why-gpus.md) | "A CPU is a few fast lanes optimized for latency; a GPU is thousands of slow lanes optimized for throughput. Matmul is the perfect workload for the second one." |
| 2 | [The GPU memory hierarchy](02-gpu-memory-hierarchy.md) | "Registers → shared memory/SRAM → L2 → HBM → host RAM, each bigger and much slower. Nearly every kernel optimization is 'make fewer trips to HBM'." |
| 3 | [The CUDA execution model](03-cuda-execution-model.md) | "A kernel launches a grid of blocks of threads; threads run in warps of 32 in lockstep. I know what warp divergence, coalescing, and occupancy mean and why they cost real time." |
| 4 | [Tensor cores & precision on the GPU](04-tensor-cores-and-precision.md) | "Tensor cores do small matrix tiles per instruction and are ~10× the FLOPs of regular CUDA cores — but only if shapes, layout, and dtype cooperate." |
| 5 | [The roofline model](05-roofline-model.md) | "Arithmetic intensity = FLOPs ÷ bytes. Compare it against the hardware's ridge point and you know instantly whether you're compute- or memory-bound." |
| 6 | [Overhead-bound: launches, streams, CUDA graphs](06-overhead-bound-and-cuda-graphs.md) | "At batch 1, the GPU can be idle half the time waiting on Python. CUDA graphs replay a captured launch sequence and reclaim it." |
| 7 | [Kernel fusion & FlashAttention](07-fusion-and-flash-attention.md) | "FlashAttention isn't faster math — it's the same math that never writes the N×N score matrix to HBM." |
| 8 | [Profiling in practice](08-profiling-in-practice.md) | "I can take an unknown model, profile it, and say within 20 minutes whether it's compute-, memory-, or overhead-bound — with evidence." |
| 9 | [Build: roofline benchmark + a fused Triton kernel](09-build-roofline-and-triton-kernel.md) | "I plotted a real roofline on a real GPU, then wrote a fused kernel that beat eager PyTorch, and explained the win in HBM round-trips." |
| 10 | [Exercises & exit artifact](10-exercises-and-artifacts.md) | The concrete proof that Phase 2 is done. |

---

## How to work through this phase

1. **Do the arithmetic by hand.** Lessons 2, 4, and 5 are full of small calculations (bytes moved, FLOPs, intensity). Do them on paper before reading the answer. The whole skill of this phase is estimating a number *before* you measure, then explaining the gap.
2. **Always predict, then measure.** Before every benchmark: write down what you expect and why. A wrong prediction that you then explain teaches more than ten correct ones.
3. **Get on a GPU by lesson 6.** Concepts 1-5 are laptop work; from lesson 6 on you need Colab/Kaggle open.
4. **Read one real kernel.** In lesson 7 you'll read FlashAttention's tiling logic. You don't need to be able to write CUDA — you need to be able to *read the comments and see the SRAM story*.
5. **Produce the exit artifact (lesson 10).** No artifact = phase not finished.

**Time budget:** 2-4 weeks part-time — this is a big phase, and it's the first one where you're fighting real hardware. Lesson 5 (roofline) and lesson 8 (profiling) are the load-bearing ones. If you're short on time, never skip those two.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Given a GPU spec sheet (peak FLOPs + HBM bandwidth), compute the **ridge point** in FLOPs/byte and say what it means. ([lesson 5](05-roofline-model.md))
2. Explain why a batch-1 LLM decode step wastes >90% of the GPU's compute, in terms of arithmetic intensity. ([lessons 5](05-roofline-model.md), [6](06-overhead-bound-and-cuda-graphs.md))
3. Explain FlashAttention as an **IO-aware** algorithm — what it avoids writing to HBM, not what math trick it uses. ([lesson 7](07-fusion-and-flash-attention.md))
4. Take a profiler trace and classify the workload as compute-, memory-, or overhead-bound, with the specific evidence you used. ([lesson 8](08-profiling-in-practice.md))

## Projects that belong to this phase

- **[04 — Roofline profiling](../../projects/README.md)** (small): plot achieved TFLOPs vs matmul size, and show a memory-bound elementwise op sitting on the bandwidth roof.
- **[05 — Fused Triton kernel](../../projects/README.md)** (large): write a fused softmax or bias+GELU in Triton, beat eager PyTorch, explain the win in HBM round-trips.

Both are specified in [lesson 9](09-build-roofline-and-triton-kernel.md).

---

Next after this: **[Phase 3 — Serving Fundamentals: Batching, Queueing, Scheduling](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling)**. Phase 2 tells you the GPU is starved at batch 1; Phase 3 is the systems answer — keep it fed, without wrecking tail latency.
