# 3 — The CUDA Execution Model (Threads, Blocks, Warps)

> **You'll be able to say:** "A kernel launch creates a grid of thread blocks; each block runs on one SM; threads execute in **warps of 32** in lockstep. Three things silently destroy performance — warp divergence, uncoalesced memory access, and low occupancy — and I can explain and spot each one."

You don't need to write CUDA to be a great inference engineer. You *do* need to read it, and to understand this execution model — otherwise profiler output ("achieved occupancy 12%", "global memory efficiency 25%") is noise.

---

## The vocabulary, from smallest to largest

```
thread    one execution of your kernel function, on one data element
   ↓ 32 threads
warp      the real unit of execution: 32 threads that run IN LOCKSTEP, one instruction at a time
   ↓ 1-32 warps
block     a group of threads that lives on ONE SM, shares its shared memory, and can synchronize
   ↓ many blocks
grid      all the blocks of one kernel launch, distributed across all SMs
```

A concrete example — add two vectors of 1,000,000 elements:

```python
# in CUDA C this would be:  vecadd<<<3907, 256>>>(a, b, c, n)
threads_per_block = 256
blocks = ceil(1_000_000 / 256) = 3907
# → 3907 blocks × 256 threads = 1,000,192 threads, one per element (last few masked off)
```

You launch **a million threads** to do a million additions. That's normal and correct on a GPU: threads are nearly free; the hardware's job is to schedule them across SMs. Each thread computes its own global index and handles exactly one element:

```c
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // which element am I?
    if (i < n) c[i] = a[i] + b[i];                   // guard: n may not divide evenly
}
```

That's a complete CUDA kernel. Ninety percent of kernels you'll read start with those exact two lines.

### How it maps onto the hardware

- A **block** is assigned to exactly one SM and never migrates. All its threads share that SM's shared memory and can `__syncthreads()` together.
- Blocks are **independent by design** — no ordering guarantees between them. That's what lets the same kernel run on a 20-SM laptop GPU and a 132-SM H100 without changes: more SMs simply means more blocks run at once.
- The SM's schedulers issue instructions **per warp**, not per thread. Warps are where the performance rules live.

---

## Warps: 32 threads, one instruction, no exceptions

The hardware doesn't schedule threads individually. It schedules **warps of 32 threads that all execute the same instruction at the same time** (SIMT). If you launch 100 threads, the hardware runs 4 warps and masks off 28 dead lanes in the last one.

**Rule 1: always make block size a multiple of 32.** A block of 100 threads wastes ~22% of its lanes for free. Standard choices are 128, 256, or 512.

### Warp divergence — the branch tax

What happens when threads in the same warp want to take different branches? The warp can only execute one instruction at a time, so the hardware runs **both** paths and masks off the inactive threads:

```
if (threadIdx.x % 2 == 0)  x = f(x);     // 16 threads active, 16 idle
else                       x = g(x);     // now the other 16 active, first 16 idle
                                         // total time = time(f) + time(g), not max()

  warp lanes:  0 1 2 3 4 5 ... 31
  f-path:      ✓ · ✓ · ✓ · ...          ← half the machine idles
  g-path:      · ✓ · ✓ · ✓ ...          ← the other half idles
```

Divergence within a warp costs you up to 32× if every lane takes a different path. Divergence *between* warps is free — that's just normal scheduling. So the rule is: **branch on data that's uniform across a warp** (e.g. `if (blockIdx.x == 0)` is fine; `if (data[i] > 0)` on random data is not).

Where this bites in inference: variable-length sequences in a batch, MoE expert routing, ragged attention masks, and early-exit logic. The standard mitigations — padding to uniform lengths, sorting requests by length, block-sparse masks — all exist to keep warps uniform. You'll meet them again as scheduling policy in [Phase 3](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling).

---

## Coalesced memory access — the one that costs 32×

This is the highest-value paragraph in the lesson. The GPU's memory system does not fetch individual floats. It fetches **cache lines / transactions of 32 bytes** (and moves 128-byte sectors around). When a warp issues a load, the hardware coalesces the 32 lanes' addresses into as few transactions as possible.

```
COALESCED — lane i reads element i (contiguous):
   addresses: [0][1][2][3] ... [31]        → 1-4 transactions, ~100% of bytes used
   ✅ full bandwidth

STRIDED — lane i reads element i*32:
   addresses: [0]......[32]......[64]...   → 32 transactions, each delivering 128 bytes
                                             of which you use 4
   ❌ ~1/32 of peak bandwidth = 97% of the bytes you paid for are thrown away
```

Both versions do the *same amount of math* and read the *same number of useful floats*. The strided one can be **32× slower** purely because of address layout. Nsight Compute reports this as **global memory efficiency** or *sectors per request*.

This is why layout matters so much in real systems, and it explains things you've probably seen without knowing why:

- **Row-major vs column-major** and why `A @ B.T` can be faster or slower than `A @ B`.
- **`.contiguous()`** calls scattered through PyTorch code — a transposed tensor is strided, and a kernel over it may be uncoalesced.
- **NHWC vs NCHW** layouts in vision inference.
- The KV-cache layout debate in vLLM/TensorRT-LLM: whether to store K as `[num_blocks, num_heads, head_dim, block_size]` or another permutation is *entirely* a coalescing decision for the attention kernel.

**Rule 2: adjacent threads should touch adjacent memory.**

---

## Occupancy — are the SMs even full?

**Occupancy** = (active warps per SM) ÷ (maximum warps per SM). It's a proxy for "does the SM have enough resident work to hide memory latency?" ([lesson 1](01-why-gpus.md)).

Occupancy is limited by whichever resource runs out first:

| Resource | How it limits you |
|---|---|
| **Registers per thread** | Register file is fixed per SM; a kernel using 128 regs/thread fits fewer warps than one using 32 |
| **Shared memory per block** | Ask for 100 KB per block and only 2 blocks fit in a 228 KB SM |
| **Block size** | Blocks are indivisible; awkward sizes leave slots empty |
| **Not enough blocks** | Launch 8 blocks on 132 SMs → **124 SMs literally do nothing** |

That last one is the one that hits inference engineers, not kernel authors. **Batch-1 decode produces tiny kernels that can't fill the GPU.** A matrix-vector product for one token has nowhere near enough parallel work for 132 SMs, so most of the chip is idle no matter how good the kernel is. Batching creates blocks; blocks fill SMs.

> **Important nuance:** higher occupancy is *not* always better. A tiled matmul kernel that uses lots of registers and shared memory per thread deliberately runs at low occupancy and still hits peak FLOPs, because each thread has huge instruction-level parallelism and data reuse. Occupancy is a **diagnostic**, not a goal. Chase it only when your kernel is latency-bound and stalling on memory. Saying this out loud in an interview marks you as someone who's actually profiled things.

---

## Putting it together: why the same math can be 30× apart

Two kernels, same FLOPs, same output:

| | Naive | Optimized |
|---|---|---|
| Memory access | strided / uncoalesced | coalesced |
| Data reuse | none — every thread reloads from HBM | tiled through shared memory ([lesson 2](02-gpu-memory-hierarchy.md)) |
| Branching | per-element `if`s → divergence | branch-free, masked |
| Occupancy | 8 blocks, 6% of SMs busy | thousands of blocks, all SMs busy |
| Tensor cores | not used (wrong dtype/shape) | used ([lesson 4](04-tensor-cores-and-precision.md)) |
| **Result** | ~3% of peak | ~80% of peak |

This gap is why cuBLAS, CUTLASS, FlashAttention, and vLLM's kernels exist — and why "I wrote a CUDA kernel" and "I wrote a *fast* CUDA kernel" are different sentences.

---

## Where you'll actually meet this

You will most likely never write raw CUDA in this roadmap. You *will*:

- **Read it.** [Lesson 7](07-fusion-and-flash-attention.md) has you read FlashAttention's tiling loop; the comments talk about SRAM, tiles, and blocks, and now you know what they mean.
- **Write Triton.** OpenAI's Triton ([lesson 9](09-build-roofline-and-triton-kernel.md)) is Python-like and handles coalescing/scheduling *for* you at the block level — you reason about blocks and memory, not individual threads. This is how most ML engineers write kernels today, and it's how vLLM's own kernels are increasingly written.
- **Read profiler output.** "Achieved occupancy 11%", "memory throughput 18%", "warp stall: long scoreboard (waiting on global memory)" — these are now sentences you can act on ([lesson 8](08-profiling-in-practice.md)).

---

## Try it (Colab free tier)

Watch coalescing cost you 10-30× with no CUDA at all — a strided access pattern in PyTorch is enough:

```python
import torch, time
n = 8192
x = torch.randn(n, n, device="cuda", dtype=torch.float16)

def bench(fn, iters=50):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

rows = bench(lambda: x.sum(dim=1))   # each thread walks a contiguous row  → coalesced-friendly
cols = bench(lambda: x.sum(dim=0))   # strided across rows
print(f"sum along rows: {rows*1e3:.3f} ms   sum along cols: {cols*1e3:.3f} ms")

xt = x.t()                            # a view: same memory, strided
print("contiguous copy of transpose:", bench(lambda: xt.contiguous())*1e3, "ms")
```

Same element count, same additions, different address patterns. Write down the ratio and explain it in one sentence using "transactions per warp." (PyTorch's reduction kernels are smart, so the gap may be smaller than the theoretical 32× — explaining *why the library beat the naive prediction* is itself the exercise.)

Also print your GPU's shape, so the numbers in these lessons stop being abstract:

```python
p = torch.cuda.get_device_properties(0)
print(p.name, "| SMs:", p.multi_processor_count,
      "| shared mem/block:", p.shared_memory_per_block/1024, "KB",
      "| total:", p.total_memory/1e9, "GB")
```

---

## Key takeaways

- **grid → blocks → warps → threads.** A block lives on one SM; a **warp of 32 threads executes in lockstep**; blocks are independent so the same kernel scales across GPU sizes.
- **Warp divergence:** different branches inside one warp are executed serially with lanes masked off — up to 32× loss. Keep branches warp-uniform.
- **Coalescing:** adjacent threads must read adjacent addresses, or you waste up to 32/32 of every memory transaction. This is why layout, `.contiguous()`, and KV-cache layouts matter.
- **Occupancy** = resident warps ÷ max warps; limited by registers, shared memory, block size, or simply too few blocks. Batch-1 decode can't fill the SMs — that's a work-supply problem, not a kernel problem.
- Higher occupancy isn't automatically better; it's a diagnostic for latency-hiding, not a target.
- You'll mostly *read* CUDA and *write* Triton — but profiler output is unreadable without this model.

**Next:** [Tensor cores & precision →](04-tensor-cores-and-precision.md) — the matmul-specific hardware that provides ~90% of a modern GPU's FLOPs, and the shape/dtype rules you must satisfy to actually get them.
