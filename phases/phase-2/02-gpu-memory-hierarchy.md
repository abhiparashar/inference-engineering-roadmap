# 2 — The GPU Memory Hierarchy (Where the Time Actually Goes)

> **You'll be able to say:** "A GPU has registers → shared memory/L1 (SRAM) → L2 → HBM → host RAM, each level bigger and much slower. The math units are so fast that the only thing that usually matters is how many times you touch HBM. Every famous kernel optimization — fusion, tiling, FlashAttention — is a way to touch it fewer times."

This is the most important lesson in Phase 2. Read it slowly.

---

## The same story as Phase 0, one level down

[Phase 0 lesson 2](../phase-0/02-memory-hierarchy.md) taught the CPU version: registers → L1/L2/L3 → RAM → disk, each ~10-100× slower than the last. A GPU has the identical shape, with two twists:

1. **One level is programmer-controlled.** On a CPU, caches are automatic — you influence them only indirectly through access patterns. On a GPU, **shared memory** is a scratchpad you explicitly load and manage. That's why GPU kernel optimization is so much more deliberate: you decide what lives in the fast memory.
2. **The gap is worse.** The GPU's math units are ~100× faster than a CPU's, but its main memory is only ~10× faster than DRAM. So the *relative* penalty for touching main memory is far larger. Starving a GPU is easy.

---

## The hierarchy, with real numbers (H100 class)

```
                        SIZE          BANDWIDTH        LATENCY     WHO CONTROLS IT
 ┌──────────────────┐
 │ Registers        │  ~256 KB/SM    ~100+ TB/s        ~1 cycle    compiler
 ├──────────────────┤
 │ Shared mem / L1  │  ~228 KB/SM    ~20-30 TB/s       ~20 cyc     YOU (the kernel author)
 ├──────────────────┤
 │ L2 cache         │  ~50 MB total  ~7 TB/s           ~200 cyc    hardware
 ├──────────────────┤
 │ HBM (VRAM)       │  80 GB         ~3.35 TB/s        ~400-800    hardware
 ├──────────────────┤                                   cycles
 │ Host RAM (PCIe)  │  TBs           ~64 GB/s          ~microsec   you (explicit copies)
 └──────────────────┘
```

Read the bandwidth column as ratios, because the ratios are what transfer to any GPU generation:

```
shared memory ≈ 7-10 ×  faster than HBM
HBM           ≈ 50    × faster than PCIe
```

Sizes matter as much as speeds. **Shared memory is measured in kilobytes.** You cannot "just put the model in SRAM" — a 7B model in FP16 is 14 GB, roughly **60,000×** larger than one SM's shared memory. Everything must stream through HBM; the only question is *how many times*.

---

## The one number that decides everything: bytes moved

Here's the arithmetic that makes the whole phase click. On an H100:

- Peak dense BF16 matmul throughput: **~1,000 TFLOP/s** ≈ 10¹⁵ FLOP/s
- HBM bandwidth: **~3.35 TB/s** ≈ 3.35 × 10¹² bytes/s

Divide:

```
1,000e12 FLOP/s ÷ 3.35e12 bytes/s ≈ 300 FLOPs per byte
```

**For every single byte you read from HBM, the GPU could have done ~300 floating-point operations in the same time.** (In FP16/BF16 that's ~150 FLOPs per *number*, since a number is 2 bytes.)

So if your kernel does fewer than ~300 FLOPs per byte it moves, the math units are sitting idle waiting for data — you are **memory-bound**, and making the math faster does *nothing*. This ratio is the hardware's **ridge point**, and formalizing it is [lesson 5](05-roofline-model.md).

Now recall [Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md): batch-1 decode does ~2 FLOPs per parameter and reads 2 bytes per parameter → **~1 FLOP/byte**. Against a ridge point of ~300, decode uses roughly **0.3%** of the GPU's math capability. That's not a rounding error; that's the central economic fact of LLM serving.

---

## Why elementwise ops are almost free of math and full of waiting

Take `y = x + 1` on a 1 GB FP16 tensor:

```
bytes moved:  read 1 GB + write 1 GB      = 2 GB
FLOPs:        one add per element         = 0.5e9 elements → 0.5 GFLOP
intensity:    0.5e9 FLOP / 2e9 bytes      = 0.25 FLOP/byte      (ridge point: ~300)

time (HBM-limited) = 2 GB ÷ 3.35 TB/s ≈ 0.6 ms
time (if compute-limited) = 0.5e9 ÷ 1e15 ≈ 0.0005 ms
```

The op takes ~1,200× longer than its math needs. **An elementwise kernel is a memory-copy with a rounding error of arithmetic attached.** Same for LayerNorm, GELU, softmax, residual adds, dropout, and the KV-cache append.

This gives you a rule you can apply immediately, without any profiler:

> **Count the HBM round trips.** For memory-bound ops, `time ≈ bytes_moved ÷ bandwidth`. If a sequence of ops reads and writes the same tensor 5 times, it takes ~5× as long as it needs to.

---

## Kernel fusion, derived from first principles

Now watch how a real optimization falls straight out of that rule. A transformer MLP tail does `y = gelu(x @ W + b)` — in eager PyTorch that's three kernels:

```
UNFUSED (3 kernels, 3 round trips through HBM)
  kernel 1:  t1 = x @ W      read x, W   → write t1 to HBM
  kernel 2:  t2 = t1 + b     read t1     → write t2 to HBM      ← t1 was JUST in registers!
  kernel 3:  y  = gelu(t2)   read t2     → write y  to HBM

FUSED (1 kernel, 1 round trip)
  kernel 1:  y = gelu(x @ W + b)
             read x, W → compute everything while values sit in registers → write y
```

The unfused version writes an intermediate to HBM and immediately reads it back — a ~800-cycle round trip to reload a value that was already in a register. The fused version keeps it on-chip. Nothing about the math changed; you just stopped commuting.

**Fusion is the single most common GPU optimization, and this is all it is.** `torch.compile`, TorchInductor, TensorRT graph optimization, and hand-written Triton kernels are all mostly automated fusion. You'll write one yourself in [lesson 9](09-build-roofline-and-triton-kernel.md), and you'll see the extreme version — FlashAttention — in [lesson 7](07-fusion-and-flash-attention.md).

---

## Tiling: how matmul escapes being memory-bound

If elementwise ops are hopeless, why is matmul compute-bound? Because it **reuses** data. Naively, `C = A @ B` for N×N matrices does 2N³ FLOPs and, if every thread fetched its own operands from HBM, would move ~2N³ × 2 bytes → intensity ~1. Terrible.

The fix is **tiling**: split the output into small blocks and load each input tile into shared memory *once*, then reuse it for every output element in the block.

```
  Load a 128×128 tile of A and a 128×128 tile of B into SRAM   ← one HBM trip
  Compute the full 128×128 output tile from them                ← 128 × more math per byte
  Slide along k, accumulate in registers
  Write the output tile once                                    ← one HBM trip
```

With a tile size of T, each byte loaded is used ~T times, so arithmetic intensity goes up by ~T. With T=128 you land far above the ridge point → **compute-bound** → you actually get to use the tensor cores. That's the whole reason a well-implemented matmul reaches ~80% of peak while a naive one reaches ~3%.

Two lessons generalize from this:

1. **Reuse is what buys you compute-bound behavior.** Matmul has reuse; elementwise ops have none, structurally, and no amount of cleverness will fix that.
2. **Bigger matrices reuse better.** This is *another* reason batching helps: batching turns a matrix-vector product (no reuse, memory-bound) into a matrix-matrix product (reuse, compute-bound). This is [Phase 1 lesson 6](../phase-1/06-prefill-vs-decode.md)'s prefill/decode split, restated in hardware terms.

---

## Where your GPU memory actually goes at serving time

Capacity, not just bandwidth, is a hard constraint. On an 80 GB H100 serving a 7B model in FP16:

```
 ┌─────────────────────────────────────────────────────┐ 80 GB HBM
 │ model weights            14 GB   ████                │
 │ CUDA context + activation                            │
 │ workspace + fragmentation ~4 GB   ██                 │
 │ KV-cache                 ~62 GB   ████████████████   │  ← everything left over
 └─────────────────────────────────────────────────────┘
```

At ~0.5 MB per token for a 7B model ([Phase 1 lesson 5](../phase-1/05-kv-cache.md)), 62 GB ≈ 124,000 cached tokens ≈ 60 concurrent users at 2k context. **KV-cache capacity — not model size — is your concurrency limit.** Check it live with `nvidia-smi`, and in code with:

```python
torch.cuda.memory_allocated()   / 1e9   # tensors you allocated
torch.cuda.memory_reserved()    / 1e9   # what the caching allocator holds from the driver
torch.cuda.max_memory_allocated() / 1e9 # peak — the number that decides if you OOM
```

The gap between `allocated` and `reserved` is PyTorch's caching allocator holding freed blocks for reuse. It's why `nvidia-smi` shows more memory used than your tensors account for, and why fragmentation is a real production failure mode — the problem PagedAttention solves in [Phase 4](../../ROADMAP.md#phase-4--inference-optimization-techniques).

---

## Try it (Colab free tier is fine)

**A. Measure your GPU's real bandwidth.** A big elementwise copy is bandwidth-limited by construction, so it's a bandwidth meter:

```python
import torch, time
n = 256 * 1024 * 1024                      # 256M fp16 elements = 512 MB
x = torch.randn(n, device="cuda", dtype=torch.float16)

def bench(fn, iters=20):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

t = bench(lambda: x + 1.0)                 # read 512 MB + write 512 MB
print(f"{2 * x.numel() * 2 / t / 1e9:.0f} GB/s effective")
```

Compare with your GPU's spec sheet (T4 ≈ 320 GB/s, A100 ≈ 1,555 GB/s, H100 ≈ 3,350 GB/s). You should reach **70-90%** of spec. That fraction has a name — **MBU, Model Bandwidth Utilization** — and it's the metric that matters for decode, far more than "GPU utilization %" ([lesson 8](08-profiling-in-practice.md)).

**B. See fusion win.** Time `x.add(1).mul(2).relu()` (three kernels, three round trips) against `torch.compile`'d version of the same function, and against the theoretical single-round-trip time. Then explain the ratio using only bytes moved. This is the seed of your Phase 2 exit artifact.

---

## Key takeaways

- The hierarchy is **registers → shared memory/L1 (SRAM) → L2 → HBM → host RAM**; shared memory is the one *you* control, and it's only ~200 KB per SM.
- An H100 can do **~300 FLOPs in the time it takes to read one byte** from HBM. Below that intensity you're memory-bound and faster math buys nothing.
- Elementwise/normalization ops have intensity ~0.25-2 FLOP/byte — they are memory-copies. `time ≈ bytes ÷ bandwidth`.
- **Kernel fusion** = do several ops in one kernel so intermediates stay in registers instead of round-tripping HBM. It's the most common optimization there is.
- **Tiling** = load a block into SRAM once and reuse it many times; reuse is what makes matmul compute-bound and lets you touch peak FLOPs.
- Capacity matters too: after weights, **the KV-cache eats all remaining HBM**, and that caps concurrency.

**Next:** [The CUDA execution model →](03-cuda-execution-model.md) — threads, blocks, warps, and the three ways (divergence, uncoalesced access, low occupancy) a kernel silently throws away most of the machine.
