# 2 — The Memory Hierarchy (Where Data Lives Decides Your Speed)

> **You'll be able to say:** "Data lives in a pyramid — registers, cache, RAM, disk — each level bigger but 10-100x slower. Most performance work is moving data to a faster level and touching the slow levels as little as possible."

This is the most important Phase 0 lesson for everything GPU-related later. FlashAttention, PagedAttention, "decode is memory-bound" — all of it is *this idea* applied to GPUs.

---

## The core problem

CPUs (and GPUs) can do math *far* faster than memory can feed them numbers. A modern CPU core can do billions of additions per second, but fetching a number from main memory (RAM) takes ~100 nanoseconds — during which the CPU could have done *hundreds* of additions. So if your program constantly reaches out to RAM, the expensive processor sits idle, waiting. This is called being **memory-bound**: your speed is limited by how fast data arrives, not by how fast you can compute.

The hardware's answer is a **hierarchy** of memories: small-and-fast close to the processor, big-and-slow far away.

```
        FASTEST, SMALLEST (closest to the math units)
   ┌─────────────────────────────────────────────┐
   │ Registers      ~1 KB      ~0.3 ns   (instant)│
   │ L1 cache       ~64 KB     ~1 ns             │
   │ L2 cache       ~1 MB      ~4 ns             │
   │ L3 cache       ~32 MB     ~15 ns            │
   │ RAM (DRAM)     ~16-256 GB ~100 ns           │
   │ SSD / disk     ~TBs       ~100,000 ns       │
   │ Network        —          ~millions of ns   │
   └─────────────────────────────────────────────┘
        SLOWEST, BIGGEST (farthest away)
```

The exact numbers vary by machine — **the ratios are the point.** Each step down is roughly 5-100x slower. RAM is ~100x slower than L1 cache. Disk is ~1000x slower than RAM. These gaps are enormous, and they dominate real-world performance far more than "how many operations" your algorithm does.

> **Analogy:** you're cooking. Registers are the pinch of salt in your hand. L1/L2 cache is the countertop. RAM is the pantry down the hall. Disk is the grocery store across town. You *can* get anything from the store, but if your recipe sends you there for every ingredient one at a time, dinner takes days. Good cooks stage what they need on the countertop first.

---

## How caching actually works (and why it usually "just works")

You don't manually put things in the CPU cache — the hardware does it automatically using two bets:

1. **Temporal locality:** if you used a piece of data, you'll probably use it again soon. So the cache keeps recently-used data around.
2. **Spatial locality:** if you used one piece of data, you'll probably use its neighbors soon. So the cache doesn't fetch one number from RAM — it fetches a whole **cache line** (typically 64 bytes) at once.

This is why *how you lay out and traverse data* matters enormously. Walk memory in order and you get the neighbors for free (they're already in the cache line you paid for). Jump around randomly and every access is a fresh, slow trip to RAM.

### See spatial locality with your own eyes

Summing a 2D array row-by-row (in memory order) vs column-by-column (jumping around):

```python
import numpy as np, time

a = np.ones((8000, 8000), dtype=np.float32)

t = time.perf_counter()
row_sum = a.sum(axis=1)     # walks each row contiguously — cache-friendly
print("row-major (contiguous):", time.perf_counter() - t)

t = time.perf_counter()
col_sum = a.sum(axis=0)     # strides across memory — cache-unfriendly
print("col-major (strided):   ", time.perf_counter() - t)
```

Same number of additions, often a 2-5x time difference. The *only* variable is whether you respected the cache. This exact effect, scaled up on a GPU, is why "memory access patterns" is a phrase you'll hear constantly in Phase 2.

---

## Bandwidth vs latency (two different "speeds")

People say "memory is slow," but there are two separate meanings:

- **Latency:** how long until the *first* byte arrives after you ask (the ~100 ns trip). Matters when you need one thing and can't continue without it.
- **Bandwidth:** how many bytes per second you can move once the pipe is flowing (e.g. RAM ~50 GB/s, GPU HBM ~2-3 TB/s). Matters when you need to stream *a lot* of data.

Why this split matters for inference: generating one token with an LLM means **reading the model's entire weight matrix out of memory** to do the math. A 7-billion-parameter model in 16-bit is ~14 GB. If your memory bandwidth is 2 TB/s, just *reading the weights once* takes ~7 milliseconds — a hard floor on how fast you can produce a token, no matter how fast the math units are. That is the literal, physical reason **decode is memory-bandwidth-bound** (Phase 1's punchline). You'll compute this exact number yourself in [Phase 1's inference-math lesson](../phase-1/08-inference-math-and-memory.md).

---

## Arithmetic intensity: the number that tells you which problem you have

Here's the single most useful concept from this lesson, stated plainly:

> **Arithmetic intensity = (math operations you do) ÷ (bytes you moved to do them).**

- **High intensity** (lots of math per byte) → you're **compute-bound**. The processor is the bottleneck. Fix it with faster math or fewer bits.
- **Low intensity** (little math per byte) → you're **memory-bound**. Moving data is the bottleneck. Fix it with better data layout, caching, or moving *less* data (e.g. smaller number formats — [lesson 5](05-floating-point-and-precision.md)).

Example: multiplying two big matrices reuses each loaded number many times → high intensity → compute-bound. Adding two big arrays uses each number *once* → low intensity → memory-bound. This one ratio decides which optimization even *can* help you, which is why the "roofline model" in Phase 2 is built entirely on it.

---

## Why this is *the* GPU lesson in disguise

Everything you just learned repeats on the GPU with scarier names and bigger gaps:

| CPU world (this lesson) | GPU world (Phase 2) |
|---|---|
| RAM (big, slow) | **HBM** — the GPU's main memory, 40-80 GB, "slow" at ~2-3 TB/s |
| L1/L2 cache, registers | **SRAM / shared memory** — tiny, ~19+ TB/s, per-SM |
| "keep data in cache" | "keep data in SRAM" — this is *literally* what FlashAttention does |
| cache line / contiguous access | "coalesced memory access" |
| memory-bound vs compute-bound | the **roofline model** |

So if this lesson clicks, Phase 2 is mostly re-learning it with GPU vocabulary. If it doesn't, go back and run the row-vs-column experiment until the *why* is obvious.

---

## Key takeaways

- Memory is a pyramid: small+fast at top, big+slow at bottom, ~10-100x gaps between levels.
- The processor is usually faster than memory can feed it, so "waiting for data" is a huge share of real run time.
- Caches auto-exploit **temporal** and **spatial** locality; you help them by reusing data and walking memory in order.
- **Latency** (time to first byte) and **bandwidth** (bytes/sec) are different bottlenecks.
- **Arithmetic intensity** (math ÷ bytes) tells you whether you're compute-bound or memory-bound — and therefore which optimization will actually help.
- LLM decode is memory-bandwidth-bound because generating each token requires reading all the weights from memory. Remember this sentence; it's half of Phase 1 and Phase 4.

**Next:** [Processes, threads, concurrency vs parallelism →](03-processes-threads-concurrency.md)
