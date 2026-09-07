# 1 — Why GPUs (CPU vs GPU, in Plain Words)

> **You'll be able to say:** "A CPU is a handful of very fast lanes designed to finish *one* thing quickly. A GPU is thousands of slow lanes designed to finish *a million* identical things per unit time. Matrix multiply is a million identical things, which is why models run on GPUs — and why a GPU is useless if you only give it one thing to do."

---

## The one-sentence difference

**CPU = latency machine. GPU = throughput machine.**

- **Latency** = how long one task takes. ("This request took 40 ms.")
- **Throughput** = how many tasks finish per second. ("We serve 3,000 tokens/s.")

You can't maximize both with the same silicon budget, so the two chips made opposite bets.

```
        CPU                                     GPU
 ┌───────────────────────┐          ┌────────────────────────────────┐
 │ ┌────┐ ┌────┐         │          │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
 │ │core│ │core│  ~8-64  │          │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  ~thousands of
 │ └────┘ └────┘  cores  │          │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  tiny cores
 │ ┌────┐ ┌────┐         │          │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
 │ └────┘ └────┘         │          │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
 │ ██████████████████    │          │ ██                             │
 │ HUGE caches +         │          │ small caches, no fancy control │
 │ branch predictors +   │          │                                │
 │ out-of-order engines  │          │ most of the die = math units   │
 └───────────────────────┘          └────────────────────────────────┘
   smart, few, fast                    dumb, many, wide
```

A CPU spends most of its transistors on being *clever*: caches, branch prediction, out-of-order execution, speculation — machinery whose only job is to make a single sequential instruction stream finish sooner. A GPU spends most of its transistors on **arithmetic units** and the memory bandwidth to feed them, and it hides slowness by having so many tasks in flight that something is always ready to run.

---

## The analogy that actually holds up

- **CPU** = 8 Formula-1 cars. Incredibly fast, each driven by a genius. Great for delivering one urgent package across town.
- **GPU** = 5,000 delivery bikes. Each is slow and dumb, they all must be given the *same* route instructions, but together they deliver 5,000 packages in the time the F1 car does 8.

Two consequences fall straight out of the analogy, and they are the whole rest of this phase:

1. **If you only have 1 package, the bikes are worthless.** (This is batch-size-1 LLM decode. Hold that thought — it's [lesson 5](05-roofline-model.md).)
2. **If the bikes must each take a different route with different turns, they stall.** (This is *warp divergence* — [lesson 3](03-cuda-execution-model.md).)

---

## Why matrix multiply is the perfect GPU workload

Recall from [Phase 1](../phase-1/03-transformer-block.md): almost all of a transformer's work is matrix multiplication. Look at what computing one output element of `C = A @ B` involves:

```
C[i][j] = sum over k of A[i][k] * B[k][j]
```

Now notice three things:

1. **Every output element is independent.** `C[0][0]` doesn't need `C[0][1]`. You could compute all million of them at once, in any order.
2. **Every output element runs the exact same code.** Same instructions, different data. This pattern has a name: **SIMD** (Single Instruction, Multiple Data). NVIDIA's variant is **SIMT** (Single Instruction, Multiple Threads).
3. **There are no branches.** No `if` statements that make one element take a wildly different path from another.

Independent + identical + branch-free is precisely the workload the "5,000 bikes" design wins at, by orders of magnitude. That's not a coincidence — modern GPUs were progressively redesigned *around* this workload (tensor cores, [lesson 4](04-tensor-cores-and-precision.md), are literally matmul-specific hardware).

The flip side matters just as much: **workloads with branches, pointer chasing, or long dependency chains belong on a CPU.** Tokenization, request routing, sampling logic, JSON parsing — CPU work in your inference server. Knowing which side of that line a piece of work sits on is a real engineering skill you'll use constantly from Phase 3 onward.

---

## Inside a GPU: SMs, and why you should care

A GPU is not one flat sea of cores. It's an array of **Streaming Multiprocessors (SMs)** — think of each SM as a small independent processor with its own scheduler, its own register file, and its own private scratchpad memory.

```
GPU (e.g. H100: 132 SMs)
 ├── SM 0 ──┬── many CUDA cores (FP32/INT math units)
 │          ├── 4 tensor cores          ← matmul-specific units (lesson 4)
 │          ├── register file (huge, ~256 KB)
 │          ├── shared memory / L1 SRAM (~228 KB, programmer-controlled)  ← lesson 2
 │          └── warp schedulers
 ├── SM 1 ──  (same)
 ├── ...
 └── SM 131
      ▲
      │  all SMs share:  L2 cache (~50 MB)  →  HBM (80 GB @ ~3.35 TB/s)
```

Rough numbers worth memorizing for scale (H100 SXM, public spec sheet):

| Thing | Count / size |
|---|---|
| SMs | 132 |
| Threads in flight | ~200,000+ |
| Peak BF16 tensor-core throughput | ~990 TFLOP/s (with sparsity marketing math; ~half that dense-realistic) |
| HBM3 capacity / bandwidth | 80 GB @ ~3.35 TB/s |
| Shared memory per SM | up to 228 KB |

Compare to a good server CPU: ~64 cores, maybe ~2 TFLOP/s FP32, and ~0.4 TB/s of DRAM bandwidth. **The GPU is ~100-500× the math and ~10× the bandwidth.** Note that ratio carefully — *compute grew far faster than bandwidth*, which is exactly why nearly everything in modern inference is memory-bound and not compute-bound. That gap is the reason this roadmap exists.

**Two things to hold onto:**

- Work is distributed across **SMs**. If your work only fills 4 SMs, 128 SMs sit idle and your "GPU utilization" claim is a lie ([lesson 3](03-cuda-execution-model.md), occupancy).
- Each SM has **tiny, extremely fast private memory** (SRAM) and shares **big, slow memory** (HBM) with everyone. The entire art of kernel optimization is keeping data in the first one ([lesson 2](02-gpu-memory-hierarchy.md)).

---

## How a GPU hides latency (the trick that makes it work)

A CPU hides memory latency by *avoiding* it: giant caches, prefetchers, out-of-order execution to keep working while a load is pending.

A GPU hides memory latency by **oversubscription**. Each SM keeps thousands of threads resident. When a group of threads stalls waiting on HBM (hundreds of cycles), the SM's scheduler instantly swaps in another group that's ready to compute. Context switching costs *zero* cycles because every resident thread's registers are already physically allocated — nothing is saved or restored.

```
CPU thread:   [compute]──stall 300 cycles waiting on RAM──[compute]      ← the stall is dead time
GPU SM:       warp A [compute]  ──stalled──────────────────  [compute]
              warp B          [compute] ──stalled────────
              warp C                   [compute] ──stalled──
              warp D                            [compute]
                              ↑ SM always has someone to run: latency hidden
```

This has a critical implication you will hit constantly: **a GPU needs lots of parallel work to be fast at all.** Not just to "go faster" — to hide latency in the first place. Give it too little work and you don't get a slightly slower GPU; you get a catastrophically slower one, because there's nothing to swap in during stalls and the math units simply idle.

That is the deepest reason batch-1 LLM decode performs so poorly, and the deepest reason **batching** ([Phase 3](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling)) is the highest-leverage lever in all of inference serving.

---

## Where a GPU lives in your machine

The GPU isn't the computer; it's a device on the other end of a wire.

```
   CPU ──── PCIe (~64 GB/s on Gen5 x16) ──── GPU
   RAM                                       HBM (~3,350 GB/s)
   (512 GB)                                  (80 GB)
```

**PCIe is ~50× slower than HBM.** Consequences that bite real inference servers:

- Copying tensors between CPU and GPU (`.cpu()`, `.to("cuda")`, `.item()`, `print(tensor)`) is *expensive* and frequently *synchronizing* — it makes the CPU wait for the GPU to catch up. A single stray `.item()` inside a decode loop can cost more than the model's math.
- "Offload the KV-cache to CPU RAM" is a real technique, and it's exactly this trade: more capacity, PCIe-limited speed.
- Multi-GPU setups add **NVLink** (~900 GB/s on H100) precisely because PCIe is too slow for tensor parallelism — that's [Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale).

Rule of thumb for the rest of the roadmap: **keep the data on the GPU, and keep the CPU out of the inner loop.**

---

## The mental model to keep

```
Give the GPU:   a LOT of identical, independent, branch-free math on data already in HBM
                → it is unbeatable

Give the GPU:   one small task, or branchy logic, or data that has to cross PCIe
                → it is slower than your laptop's CPU, and idle >90% of the time
```

Everything in Phase 2 is a refinement of that sentence. Everything in Phases 3-4 (batching, quantization, PagedAttention, speculative decoding) is a trick to move real LLM serving from the second line to the first.

---

## Try it (5 minutes, works on Colab free tier)

```python
import torch, time

def bench(fn, iters=50):
    fn(); torch.cuda.synchronize()               # warm up, then WAIT for the GPU
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()                     # ← without this you time nothing
    return (time.perf_counter() - t0) / iters

for n in (128, 512, 2048, 8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    secs = bench(lambda: a @ b)
    tflops = (2 * n**3) / secs / 1e12            # matmul FLOPs = 2·N³
    print(f"{n:>5} x {n:<5}  {secs*1e6:9.1f} us   {tflops:6.1f} TFLOP/s")
```

You should see achieved TFLOP/s **climb steeply with size** and then flatten near the hardware's peak. Small matrices don't come close — there isn't enough parallel work to fill the SMs or hide latency. That single table is the "GPUs need lots of work" claim, measured. You'll turn it into a proper roofline plot in [lesson 9](09-build-roofline-and-triton-kernel.md).

> **The `torch.cuda.synchronize()` calls are not optional.** CUDA is asynchronous: `a @ b` only *queues* work and returns immediately. Timing without synchronizing measures how fast Python can enqueue, which is a classic beginner benchmark bug that reports absurd TFLOPs. More on this in [lesson 6](06-overhead-bound-and-cuda-graphs.md) and [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md).

---

## Key takeaways

- **CPU optimizes latency** (few clever cores); **GPU optimizes throughput** (thousands of simple cores + massive bandwidth).
- Matmul is independent, identical, and branch-free — the ideal GPU workload; branchy/sequential logic belongs on the CPU.
- A GPU is an array of **SMs**, each with its own registers, tensor cores, and small fast SRAM, all sharing L2 and HBM.
- GPUs hide memory latency by **oversubscription** (swap in another warp), so **too little parallel work = catastrophically slow**, not just "a bit slower."
- Compute has outgrown bandwidth by a lot; that imbalance is why most inference is memory-bound.
- The GPU is across **PCIe (~50× slower than HBM)** — keep data resident on the device and the CPU out of the inner loop.

**Next:** [The GPU memory hierarchy →](02-gpu-memory-hierarchy.md) — where every byte lives, how fast each level is, and why "minimize trips to HBM" is the one sentence behind every kernel optimization ever written.
