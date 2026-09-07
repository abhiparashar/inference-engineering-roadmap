# 5 — The Roofline Model (Compute-Bound or Memory-Bound?)

> **You'll be able to say:** "Arithmetic intensity is FLOPs divided by bytes moved from HBM. I compare it against the hardware's ridge point — peak FLOP/s divided by bandwidth — and I know, before optimizing anything, whether to fix the math or fix the bytes. Fixing the wrong one buys exactly 0%."

Lessons 1-4 gave you the pieces: a GPU is a throughput machine, HBM is the bottleneck, kernels are grids of warps, tensor cores supply the FLOPs. This lesson collapses all of it into **one division and one plot**. With [profiling](08-profiling-in-practice.md) it's the load-bearing lesson of Phase 2 — everything after it is a technique, this is the decision procedure that picks the technique.

A kernel is limited by whichever ceiling it hits first: the math units can't exceed peak FLOP/s, and the data can't arrive faster than HBM bandwidth. You are always pressed against one of them, and **the fix for one does nothing for the other**. Move a memory-bound kernel from FP32 CUDA cores to FP16 tensor cores and the math gets 15× faster while the runtime doesn't budge — those units were already idle, waiting. So the first question is never "how do I make this faster?" It's **"which ceiling am I on?"**

---

## Arithmetic intensity: FLOPs per byte

```
        FLOPs performed by the kernel
  I  =  ──────────────────────────────────────      units: FLOP / byte
        bytes moved between HBM and the chip
```

The units matter: `time_compute = FLOPs ÷ peak_flops` and `time_memory = bytes ÷ bandwidth` are both seconds, so their ratio is dimensionless — which is exactly why you may compare `FLOPs/bytes` against `peak_flops/bandwidth`. That comparison *is* the model. And **"bytes moved" means HBM traffic, not data touched** — the sentence people get wrong. A tensor read ten times out of shared memory counts **once**; an intermediate that never leaves registers counts **zero**. Only bytes crossing the HBM boundary count.

Which is why fusion changes intensity without changing a single FLOP ([lesson 2](02-gpu-memory-hierarchy.md)): the numerator is fixed by the math, the denominator is a property of your *implementation*. Arithmetic intensity is not a property of an algorithm — it's a property of an algorithm **plus a memory schedule**, and the schedule is the part you control.

---

## The ridge point: where the two ceilings cross

```
  ridge = peak FLOP/s ÷ bandwidth (bytes/s)         units: FLOP / byte
```

Same units as intensity, so compare them directly. Derive it for the three GPUs you'll meet:

| GPU | Peak dense tensor FLOP/s | HBM bandwidth | Ridge = FLOP/s ÷ bytes/s |
|---|---|---|---|
| T4 (Colab free) | ~65 TFLOP/s FP16 | ~320 GB/s | 65e12 ÷ 320e9 ≈ **203 FLOP/byte** |
| A100 80GB SXM | ~312 TFLOP/s BF16 | ~2,039 GB/s | 312e12 ÷ 2.039e12 ≈ **153 FLOP/byte** |
| H100 SXM | ~990 TFLOP/s BF16 | ~3,350 GB/s | 990e12 ÷ 3.35e12 ≈ **296 FLOP/byte** |

The H100's ~300 is the "~300 FLOPs per byte" figure from [lesson 2](02-gpu-memory-hierarchy.md), now with a name. Read it as a rule: **`I < ridge` → memory-bound; `I > ridge` → compute-bound.** Which side you're on matters more than anything else in the plot.

**There is one ridge per dtype.** The compute roof depends on precision ([lesson 4](04-tensor-cores-and-precision.md)): FP32 on H100 CUDA cores peaks at ~67 TFLOP/s, so its ridge is `67e12 ÷ 3.35e12 ≈ 20 FLOP/byte` against BF16's 296. Running FP32 lowers your own ceiling ~15×, which makes far more kernels *look* compute-bound. They aren't — you lowered the roof onto your head.

**And the ridge keeps rising.** A100 → H100, one generation:

```
  compute:    312 → 990 TFLOP/s     = ×3.17
  bandwidth:  2.04 → 3.35 TB/s      = ×1.64
  ridge:      153 → 296 FLOP/byte   = ×1.93
```

Compute grows about twice as fast as bandwidth, every generation. (The T4's ~203 isn't a counterexample — it's a small card pairing respectable FP16 math with cheap, narrow memory.) So each new GPU demands **more arithmetic per byte** to be used well, and a fixed workload drifts further left of the ridge over time. Kernels that were compute-bound in 2018 are memory-bound now and nobody changed the code. That is the structural reason inference engineering is a discipline: the hardware keeps getting better at the thing LLM decode doesn't do.

---

## The plot

Attainable FLOP/s against intensity, log-log. Two straight lines:

```
 attainable                                   compute roof = peak FLOP/s (per dtype)
 FLOP/s (log)
  990 T ┤             ╱ ┌──────────────────────────────────────────────
        ┤          ╱    └─ ridge: I = 296 FLOP/byte  (H100, BF16)
  100 T ┤       ╱
   10 T ┤    ╱           slanted roof: attainable = I × bandwidth
    1 T ┤ ╱              (slope set by 3.35 TB/s)
        └──┬──────┬──────┬───────┬───────┬───────┬───────┬──────→  I (log)
          0.25    1      2      64      296    1024    2048
        elem-  decode   Layer   decode   RIDGE  prefill   fused
        wise    (B=1)    Norm   (B=64)           GEMM   attention

        ←────────────── MEMORY-BOUND ──────────┼──── COMPUTE-BOUND ────→

        attainable FLOP/s = min( peak_flops ,  I × bandwidth )
```

On an H100 an elementwise add can never beat `0.25 × 3.35e12 = 0.84 TFLOP/s` — **0.08% of peak** — no matter how brilliant the kernel. Batch-1 decode at `I ≈ 1` caps at 3.35 TFLOP/s, 0.3% of peak. Not a bug in your code; that's the roof. And note the geometry: on the slant, the only way up is **right**. You cannot go faster at fixed intensity. Every memory-bound optimization in this roadmap — fusion, quantization, batching, FlashAttention — slides a point rightward.

---

## Worked intensities for real inference ops

All FP16 (2 bytes/number), all counting HBM traffic only. Do each division yourself before reading the verdict.

| Op | FLOPs | HBM bytes | I | Verdict (vs ~300) | The fix |
|---|---|---|---|---|---|
| Elementwise `y = x + 1` | 1 per elem | 2 read + 2 write = 4 | **0.25** | memory-bound, hopeless alone | fuse into a neighbour |
| LayerNorm, single pass | ~8 per elem | 4 | **2** | memory-bound | fuse; never 3 kernels |
| Attention block, unfused, L=4096, d=128 | 4L²d = 8.59e9 | Q,K,V,O 4.2e6 + **four passes over the L×L scores** 134e6 = 138e6 | **62** | memory-bound — the N×N matrix *is* the traffic | never materialize the scores |
| Same block, fused | 4L²d = 8.59e9 | 4.2e6 | **2,048** | compute-bound | already optimal |
| Decode matmul, batch 1, 7B | 2P = 14e9 | 2P = 14e9 | **1** | memory-bound by ~300× | batch it |
| Decode matmul, batch B | 2PB | 2P (weights read once) | **≈ B** | memory-bound until B ≈ ridge | batch harder |
| Prefill GEMM, M=2048, K=N=4096 | 2MNK = 6.87e10 | 2(MK+KN+MN) = 67e6 | **1,024** | compute-bound | tensor cores, lower dtype |

Two rows deserve staring at. **Fused vs unfused attention:** identical FLOPs, intensity `62 → 2,048`, a 33× jump across the ridge bought entirely by not writing an L×L matrix to HBM. The algebra is `I_fused = 4L²d ÷ 8Ld = L/2`, so longer context makes fused attention *more* compute-bound. That's all of FlashAttention; [lesson 7](07-fusion-and-flash-attention.md) does the tiling. **Decode at batch B:** weights are read once and serve all B tokens, so `I = 2PB ÷ 2P = B` — the generalization of the `I ≈ 1` result first derived in [Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md).

---

## The batching result, stated crisply

```
  decode arithmetic intensity  ≈  batch size B
  ⟹ compute-bound at  B ≈ ridge point  ≈ 153 (A100) to 296 (H100)
```

Batching *is* the x-axis. Every concurrent request slides decode one FLOP/byte to the right: at B=1 you use ~0.3% of the math, at B=64 about 20%, at B≈300 you finally touch the roof. That single line is the whole quantitative case for **continuous batching** ([Phase 3](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling)) — vLLM and TensorRT-LLM exist largely to keep B large. It also shows why B can't be set to 1,000:

- **KV-cache capacity caps it.** At ~0.5 MB/token, a 7B model on an 80 GB H100 leaves ~62 GB of cache ≈ 124k tokens ([lesson 2](02-gpu-memory-hierarchy.md)) — roughly 60 users at 2k context, not 300.
- **Latency SLOs cap it.** Larger batches raise per-token latency and queueing delay; throughput and TTFT/TPOT pull in opposite directions. You'll spend Phase 3 inside that tension — the roofline is why it exists.

---

## MFU and MBU: the two honest scorecards

```
  MFU = achieved FLOP/s  ÷  peak FLOP/s (for your dtype)
  MBU = achieved bytes/s ÷  peak HBM bandwidth

  achieved FLOP/s  ≈ 2 × params × tokens/s
  achieved bytes/s ≈ (model bytes + KV bytes read per token) × tokens/s
```

```
  7B FP16 on an H100 at a well-tuned 150 tok/s, batch 1:
    bytes/s = 14e9 × 150 = 2.1e12  →  MBU = 2.1 ÷ 3.35  ≈ 63%    ← excellent
    FLOP/s  = 14e9 × 150 = 2.1e12  →  MFU = 2.1 ÷ 990   ≈ 0.21%  ← meaningless
```

Same run, two numbers, 300× apart. **Report MFU for prefill and training-like work; report MBU for decode.** Quoting decode MFU makes a good engine look broken; quoting prefill MBU hides a real problem.

Be blunt about the third number: **`nvidia-smi`'s "GPU utilization %" is nearly useless.** It reports the fraction of time at least one kernel was *resident* on the device — not that any unit did work. A batch-1 decode loop stalling on HBM reads 100% utilization while using 0.3% of the FLOPs. When someone offers "we're at 95% utilization" as evidence of efficiency, ask for MFU or MBU. Interviewers notice when you do.

---

## Where the model lies to you

1. **It assumes perfect overlap.** Real kernels don't fully hide memory latency behind compute, so honest points sit *under* both roofs. 70-80% of a roof is a good kernel.
2. **It ignores caches.** The roof uses *HBM* bandwidth, but an H100 has ~50 MB of L2 at ~7 TB/s. A working set that fits in L2 can **beat the HBM roof** — points above the slant usually mean cache hits, not measurement error.
3. **It can't see latency-bound kernels.** A kernel with 200 blocks on 132 SMs is limited by having no work, not by bytes or FLOPs ([lesson 3](03-cuda-execution-model.md)). It plots as a sad dot below both roofs.
4. **It cannot see an idle GPU at all.** The roofline models only time spent *running a kernel*. If the GPU is waiting on Python between launches, the plot is silent — that's the third regime it can't show you, and [lesson 6](06-overhead-bound-and-cuda-graphs.md) owns it entirely.

Confirm every verdict on real hardware with a profiler ([lesson 8](08-profiling-in-practice.md)): memory-bound shows high DRAM throughput with idle tensor pipes, compute-bound is the reverse. And report percentiles at fixed offered load, never the mean of a hot loop ([benchmarking playbook](../../playbooks/benchmarking.md)).

---

## Try it (Colab free tier; T4 numbers will be smaller)

**Predict first.** Square matmul intensity is `2n³ ÷ 6n² = n/3`. On a T4 the ridge is ~203, so the plateau should start near `n ≈ 600`. Write your predicted crossover down before running.

```python
import torch, time

PEAK_TFLOPS = 65.0     # T4 FP16. A100 BF16: 312.  H100 BF16: 990.
PEAK_GBPS   = 320.0    # T4.      A100: 2039.       H100: 3350.

p = torch.cuda.get_device_properties(0)
print(f"{p.name} | SMs {p.multi_processor_count} | HBM {p.total_memory/1e9:.1f} GB")

def bench(fn, iters=30):
    for _ in range(3): fn()                  # warmup: cuBLAS autotune + allocator
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()                 # without this you time launches, not work
    return (time.perf_counter() - t0) / iters

print(f"{'n':>6} {'ms':>8} {'TFLOP/s':>9} {'MFU':>6} {'I=n/3':>7}")
for n in (128, 256, 512, 1024, 2048, 4096, 8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    t = bench(lambda: a @ b)
    tfs = (2 * n**3) / t / 1e12
    print(f"{n:>6} {t*1e3:>8.3f} {tfs:>9.2f} {tfs/PEAK_TFLOPS:>5.0%} {n/3:>7.0f}")
    del a, b

x = torch.randn(256 * 1024 * 1024, device="cuda", dtype=torch.float16)   # 512 MB
t = bench(lambda: x + 1.0)
gbps = (2 * x.numel() * 2) / t / 1e9         # read 512 MB + write 512 MB
print(f"elementwise add: {t*1e3:.3f} ms  {gbps:.0f} GB/s  MBU {gbps/PEAK_GBPS:.0%}  I=0.25")
```

You should see roughly: TFLOP/s climbing steeply through the small sizes then flattening once `n/3` passes the ridge; MFU topping out in the 50-80% band at large `n`; the elementwise add at 70-90% MBU with an MFU of essentially zero. Small `n` will *undershoot* the slant — those are launch- and latency-bound, failures 3 and 4 above. Then plot both on one chart: intensity on x, achieved FLOP/s on y, log-log, with the two roofs drawn from **your own** spec sheet (`peak_flops` flat, `I × bandwidth` slanted). Matmul points should land near the flat roof or the knee; the elementwise point sits on the slant, far left. That chart is **Project 04 (`04-roofline-profiling`)**, specified in full in [lesson 9](09-build-roofline-and-triton-kernel.md) — explaining each gap between a point and the roofs in one sentence is the actual deliverable.

---

## Key takeaways

- **`I = FLOPs ÷ bytes moved from HBM`** — bytes *moved*, not touched. Registers and shared memory are free, which is why fusion raises intensity without changing FLOPs.
- **`ridge = peak FLOP/s ÷ bandwidth`**, same units. `I < ridge` → memory-bound, `I > ridge` → compute-bound. T4 ≈ 203, A100 ≈ 153, H100 ≈ 296 FLOP/byte.
- **`attainable FLOP/s = min(peak_flops, I × bandwidth)`.** On the slant, the only helpful direction is *right*: raise intensity.
- **One roof per dtype.** FP32 on an H100 lowers your own ceiling ~15× (990 → 67 TFLOP/s) and the ridge to ~20.
- **The ridge keeps rising** (A100→H100: compute ×3.17, bandwidth ×1.64), so unchanged workloads become *more* memory-bound each generation.
- **Decode intensity ≈ batch size**, so compute-bound needs B ≈ 150-300 — the quantitative case for continuous batching, capped by KV-cache capacity and latency SLOs.
- **MFU for prefill, MBU for decode.** `nvidia-smi` utilization only means "a kernel was resident," not "the machine was working."
- The model is blind to **L2 hits, latency-bound tiny kernels, and an idle GPU** — always confirm the verdict with a profiler.

**Next:** [Overhead-bound: launches, streams, CUDA graphs →](06-overhead-bound-and-cuda-graphs.md) — the regime the roofline cannot see, where the GPU has finished and is waiting on Python.
