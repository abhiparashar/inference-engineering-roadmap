# 9 — Build: A Measured Roofline + A Fused Triton Kernel

> **Goal:** produce the two artifacts that make Phase 2 real. **(A)** Plot your own GPU's roofline from measurements, with matmul and elementwise points on it, and explain every gap. **(B)** Write a fused kernel in Triton, prove it's correct, beat eager PyTorch, and explain the win in HBM round trips — not in vibes.

These are [project 04 (`04-roofline-profiling`)](../../projects/README.md) and [project 05 (`05-triton-fused-kernel`)](../../projects/README.md). A free Colab T4 is enough for both; the shapes of the curves are identical on an H100, only the axis labels move.

**Rules:** every number you report must come from a run with warmup and `torch.cuda.synchronize()` ([lesson 6](06-overhead-bound-and-cuda-graphs.md)), every kernel you write must be checked against PyTorch for correctness *before* you time it, and every speedup must come with a predicted value from the byte arithmetic. A fast wrong kernel is worth nothing.

```
projects/04-roofline-profiling/          projects/05-triton-fused-kernel/
├── probe.py     # device + spec sheet   ├── bias_gelu.py   # your fused kernel
├── sweep.py     # matmul + memory scan  ├── softmax.py     # the reduction kernel
├── plot.py      # the roofline chart    ├── test.py        # correctness vs torch
├── results.csv  # raw measurements      ├── bench.py       # vs eager + torch.compile
└── README.md    # the writeup           └── README.md      # bytes math + results
```

---

# Part A — Project 04: your GPU's roofline

## Milestone 1 — The spec sheet and the ridge point

Before measuring, write down the two numbers you're measuring *against*, and derive the ridge point ([lesson 5](05-roofline-model.md)).

```python
import torch
p = torch.cuda.get_device_properties(0)
print(f"{p.name} | SMs {p.multi_processor_count} | HBM {p.total_memory/1e9:.1f} GB "
      f"| capability {p.major}.{p.minor}")
```

Then look up the official peak numbers (NVIDIA's datasheet, not a forum post) and put them in a table in your README: peak FP16/BF16 tensor-core TFLOP/s, peak FP32 CUDA-core TFLOP/s, HBM GB/s, and `ridge = peak_flops ÷ bandwidth` for **each** dtype. For a T4: `65e12 ÷ 320e9 ≈ 203 FLOP/byte` in FP16, and `8.1e12 ÷ 320e9 ≈ 25` in FP32. Two roofs, two ridges — that difference will show up in your plot.

**Predict now, in writing:** square matmul has `I = 2n³ ÷ 6n² = n/3`, so the compute roof should be reached around `n ≈ 3 × ridge` (~600 on a T4). Write your predicted crossover down. You will be graded by yourself against it.

## Milestone 2 — The compute roof (matmul sweep)

```python
import torch, time, csv

def bench(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

rows = []
for dtype in (torch.float32, torch.float16):
    for n in (128, 256, 512, 768, 1024, 2048, 3072, 4096, 6144, 8192):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        t = bench(lambda: a @ b)
        rows.append(dict(op=f"matmul-{dtype}".replace("torch.", ""), n=n,
                         intensity=n / 3, flops=2 * n**3 / t,
                         bytes_per_s=3 * n * n * a.element_size() / t, ms=t * 1e3))
        del a, b; torch.cuda.empty_cache()
        print(rows[-1]["op"], n, f"{rows[-1]['flops']/1e12:7.2f} TFLOP/s")
```

Run it twice: once with `torch.backends.cuda.matmul.allow_tf32 = False`, once `True`, and note what happens to the FP32 curve ([lesson 4](04-tensor-cores-and-precision.md)). That single flag moving your FP32 points up by ~5-8× is the most memorable demonstration of "there is one roof per dtype" you will ever run.

## Milestone 3 — The bandwidth roof (memory sweep)

Memory-bound points are what pin down the slanted roof. Sweep both *op* and *size*:

```python
for name, fn_factory, bytes_per_elem in [
        ("add1",   lambda x: (lambda: x + 1.0),            4),   # read + write (fp16: 2+2)
        ("copy",   lambda x: (lambda: x.clone()),          4),
        ("chain5", lambda x: (lambda: ((x*1.5+0.25).relu()*2.0-1.0).sigmoid()), 4),
        ("sum",    lambda x: (lambda: x.sum()),            2)]:  # read only
    for mb in (1, 4, 16, 64, 256, 512):
        x = torch.randn(mb * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)
        t = bench(fn_factory(x))
        gbps = x.numel() * bytes_per_elem / t / 1e9
        print(f"{name:7} {mb:4} MB  {t*1e6:9.1f} us  {gbps:7.0f} GB/s")
        del x; torch.cuda.empty_cache()
```

Three things you must explain in the writeup:

1. **Small sizes look terrible.** A 1 MB op takes about as long as a 4 MB one — you're launch- and latency-bound, not bandwidth-bound ([lesson 6](06-overhead-bound-and-cuda-graphs.md)). This is the left edge of every real roofline plot and beginners always mistake it for a bandwidth problem.
2. **Sizes that fit in L2 beat the HBM roof.** An H100 has ~50 MB of L2 at ~7 TB/s; a T4 has 4 MB. Points above the slant are cache hits, not errors ([lesson 5](05-roofline-model.md)).
3. **`chain5` should be ~5× slower than `add1`** on large tensors, since it's 5 memory-bound kernels instead of 1. That prediction is the whole thesis of [lesson 7](07-fusion-and-flash-attention.md), and confirming it here is what makes Part B's speedup unsurprising.

## Milestone 4 — The plot

```python
import matplotlib.pyplot as plt
import numpy as np

PEAK_TFLOPS, PEAK_GBPS = 65.0, 320.0            # your GPU, your dtype
I = np.logspace(-1, 4, 200)
plt.loglog(I, np.minimum(PEAK_TFLOPS * 1e12, I * PEAK_GBPS * 1e9), "k-", lw=2,
           label="roofline (min of both roofs)")
plt.axvline(PEAK_TFLOPS * 1e12 / (PEAK_GBPS * 1e9), ls=":", c="gray")  # ridge
for r in rows:
    plt.scatter(r["intensity"], r["flops"], s=18)
plt.xlabel("arithmetic intensity (FLOP/byte)"); plt.ylabel("achieved FLOP/s")
plt.title("Measured roofline — <your GPU>"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("roofline.png", dpi=150, bbox_inches="tight")
```

Every point should sit **on or under** the line. Annotate the interesting ones directly on the chart: the elementwise cluster far left on the slant, the small-matmul points *under* the slant (latency-bound), the large FP16 matmuls near the flat roof, the FP32 matmuls near a *lower* flat roof.

## Milestone 5 — The writeup (this is the actual deliverable)

`README.md` must contain: the spec table with per-dtype ridge points; `roofline.png`; your **predicted** crossover vs the measured one; and one sentence per anomaly explaining the gap between each point and the roof. Target: 400 words. The plot is easy; **the explanations are the artifact.** Aim to state, for each cluster, which of the four lies from [lesson 5](05-roofline-model.md) applies.

---

# Part B — Project 05: a fused Triton kernel

Triton lets you write GPU kernels in Python at **block** granularity: you reason about tiles of memory and let the compiler handle threads, coalescing, and scheduling ([lesson 3](03-cuda-execution-model.md)). It's how a growing share of vLLM's and PyTorch's own kernels are written, and it's the realistic path for an inference engineer who won't be hand-writing CUDA C.

## The mental model, in five lines

```
tl.program_id(0)   which block am I? (the grid is yours to define)
tl.arange(0, B)    a vector of B lane offsets — the block's shape is compile-time constant
tl.load(p+o, mask) load a tile from HBM; mask handles the ragged last block
   ... compute ...  stays in registers/SRAM: this is where fusion happens
tl.store(p+o, y, mask)   write once
```

Everything between `load` and `store` is free of HBM traffic. **The whole art is putting more work between them.**

## Milestone 1 — Fused bias + GELU

Eager PyTorch runs `x + b` (read 2 tensors, write 1) then `gelu` (read 1, write 1): **~5 tensor-sized HBM passes** for large `x`. Fused: read `x`, read a tiny `b`, write `y` → **2 passes**. Predicted speedup ≈ 2.5×.

```python
import torch, triton, triton.language as tl

@triton.jit
def bias_gelu_kernel(x_ptr, b_ptr, y_ptr, n_elem, n_cols, BLOCK: tl.constexpr):
    offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + (offs % n_cols), mask=mask, other=0.0).to(tl.float32)
    z = x + b
    t = 0.7978845608028654 * (z + 0.044715 * z * z * z)     # sqrt(2/pi) * (...)
    tanh_t = 1.0 - 2.0 / (tl.exp(2.0 * t) + 1.0)            # tanh, overflow-safe
    tl.store(y_ptr + offs, (0.5 * z * (1.0 + tanh_t)).to(x_ptr.dtype.element_ty), mask=mask)

def bias_gelu(x, b, BLOCK=1024):
    x = x.contiguous(); y = torch.empty_like(x)
    n = x.numel()
    bias_gelu_kernel[(triton.cdiv(n, BLOCK),)](x, b, y, n, x.shape[-1], BLOCK=BLOCK)
    return y
```

Note the `.to(tl.float32)` on load and the cast back on store: compute in FP32, store in the input dtype. That's the same rule real kernels follow for numerically sensitive math ([lesson 4](04-tensor-cores-and-precision.md)).

**Correctness before speed:**

```python
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, device="cuda", dtype=torch.float16)
ref = torch.nn.functional.gelu(x + b, approximate="tanh")
out = bias_gelu(x, b)
print("max abs err:", (out - ref).abs().max().item())      # expect ~1e-3 in fp16
assert torch.allclose(out, ref, atol=2e-2, rtol=1e-2)
```

Then benchmark against both baselines — eager *and* `torch.compile`, because "beat eager" is easy and "match the compiler" is the real bar:

```python
from triton.testing import do_bench
eager = lambda: torch.nn.functional.gelu(x + b, approximate="tanh")
comp  = torch.compile(lambda x, b: torch.nn.functional.gelu(x + b, approximate="tanh"))
for _ in range(5): comp(x, b)
for name, fn in [("eager", eager), ("compile", lambda: comp(x, b)),
                 ("triton", lambda: bias_gelu(x, b))]:
    ms = do_bench(fn)
    gbps = 2 * x.numel() * 2 / (ms * 1e-3) / 1e9          # 1 read + 1 write, fp16
    print(f"{name:8} {ms:7.3f} ms   {gbps:6.0f} GB/s effective")
```

Report all three, plus effective GB/s as a percentage of your GPU's peak (MBU). A good fused elementwise kernel lands at **70-90% MBU**; if yours does, you are done — you're touching the hardware roof and no further cleverness exists.

## Milestone 2 — Fused softmax (the reduction kernel)

Harder and more instructive: a row-wise reduction needs the whole row on chip. One program per row, block size rounded up to a power of two.

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                    # stability; row now in SRAM/registers
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)                    # both reductions, zero HBM traffic
    tl.store(y_ptr + row * stride + cols, y.to(x_ptr.dtype.element_ty), mask=mask)

def softmax(x):
    x = x.contiguous(); y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK < 2048 else (8 if BLOCK < 4096 else 16)
    softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols, BLOCK=BLOCK, num_warps=num_warps)
    return y
```

Verify against `torch.softmax(x, dim=-1)`, then benchmark at `n_cols` = 128, 512, 1024, 4096, 8192. Two results to explain:

- A naive multi-pass softmax needs ~4 passes over the row; yours needs 2 (read, write). Predicted ~2×; PyTorch's native softmax is *already* fused, so expect to be roughly **at parity** with `torch.softmax` and much faster than a hand-rolled `exp/sum/div` chain. Benchmark that chain too — it's your honest baseline.
- **It breaks when a row exceeds SRAM.** At `n_cols = 65536`, `BLOCK` won't fit and the kernel fails or spills. State the limit you hit. That failure *is* the constraint that forces the online-softmax tiling of FlashAttention ([lesson 7](07-fusion-and-flash-attention.md)) — you have now personally hit the wall the paper was written to climb.

## Milestone 3 — Tune, then explain

```python
for BLOCK in (128, 256, 512, 1024, 2048, 4096):
    print(BLOCK, do_bench(lambda: bias_gelu(x, b, BLOCK=BLOCK)))
```

Block size trades per-launch parallelism against register/SRAM pressure and occupancy — expect a shallow curve with a clear bad end. Then wrap the kernel in `@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (...) for w in (2,4,8)], key=["n_elem"])` and confirm it picks something near your manual best (`TRITON_PRINT_AUTOTUNING=1` shows its choices).

Finally, the deliverable sentence. Your `README.md` must contain a table like this, filled with **your** numbers, and it must be arithmetic — not adjectives:

| Impl | HBM passes | Predicted speedup | Measured | MBU | Why the gap |
|---|---|---|---|---|---|
| eager `gelu(x+b)` | ~5 | 1.0× | 1.0× | | baseline |
| `torch.compile` | 2 | 2.5× | | | |
| your Triton kernel | 2 | 2.5× | | | |

---

## Deliverables checklist

- [ ] `04-roofline-profiling/`: `results.csv`, `roofline.png`, spec table with per-dtype ridge points, predicted-vs-measured crossover, one explanation per anomalous cluster.
- [ ] Both the FP32-with-TF32-off and FP32-with-TF32-on matmul curves, and a sentence on the gap.
- [ ] `05-triton-fused-kernel/`: two working kernels, a passing correctness test, the three-way benchmark (eager / compile / Triton), the block-size sweep, and the HBM-passes table above.
- [ ] The `n_cols` value at which your softmax kernel breaks, and why.
- [ ] One paragraph you could say out loud in an interview: "I measured X GB/s = Y% MBU on an elementwise op, and Z TFLOP/s = W% MFU on a 4096 matmul, so the crossover was at intensity ~I, matching the ridge point of R FLOP/byte I computed from the datasheet."

## Common failure modes

- **Speedup looks impossible (>10×).** You forgot `torch.cuda.synchronize()`, or the baseline included a `.cpu()`/`.item()`, or your tensor fits in L2. Check bytes-per-second against peak: if it exceeds the roof, the measurement is wrong.
- **Triton kernel is slower than eager.** Tensor too small (launch-bound), input non-contiguous, or `BLOCK` too large so occupancy collapsed.
- **Wrong results only at the edges.** A missing or wrong `mask` on the last block. Always test a non-power-of-two size like `n = 4095`.
- **NaNs in FP16.** You computed `exp` in FP16. Cast to FP32 on load.
- **`torch.compile` beats you and you conclude Triton is pointless.** Wrong conclusion: Inductor *emits Triton*. You just met your competition — now read its output with `TORCH_LOGS="output_code"` and compare it to yours. That reading is worth more than the win.

## Stretch goals (pick one if you have time)

- Fused **RMSNorm** (the norm in Llama-class models) and drop it into a real HF model with a monkey-patch; measure end-to-end tokens/s.
- A tiled matmul with `tl.dot` — you'll feel exactly why cuBLAS took years.
- Work through Triton's official **fused attention** tutorial and map each loop to the pseudocode in [lesson 7](07-fusion-and-flash-attention.md).

---

## Key takeaways

- A roofline you **measured** beats a roofline you read about. Every point on or under the line, every gap explained by one of the model's known blind spots.
- Predict first (`I = n/3`, crossover at `3 × ridge`), then measure, then explain the difference. That loop is the entire skill of this phase.
- `allow_tf32` moving your FP32 curve by ~5-8× is proof that the compute roof is **per dtype**.
- Left-edge points are latency/launch-bound; above-the-slant points are L2 hits. Neither is a measurement error.
- In Triton you own the space between `tl.load` and `tl.store`. Fusion is putting more work in it; the payoff is predictable from HBM passes alone.
- **Correctness first, at non-power-of-two sizes; FP32 accumulation; then speed.** Report against `torch.compile`, not just eager.
- Hitting the SRAM limit in your own softmax kernel is the most valuable failure in Phase 2 — it's the exact wall FlashAttention's tiling was invented to get past.

**Next:** [Exercises & exit artifact →](10-exercises-and-artifacts.md) — close out Phase 2 with the self-check and the committed proof that you can diagnose any GPU workload.
