# 4 — Tensor Cores & Precision on the GPU

> **You'll be able to say:** "Almost all of a modern GPU's advertised FLOPs come from **tensor cores** — hardware that multiplies small matrix tiles in one instruction. They only engage for the right dtype, the right shapes, and the right layout; miss any of those and you silently run at ~1/10 speed on the regular CUDA cores."

[Phase 0 lesson 5](../phase-0/05-floating-point-and-precision.md) taught what the bits mean. This lesson is what the *hardware* does with them — and why "use FP16" is two different optimizations wearing one coat.

---

## What a tensor core is

A regular CUDA core does one scalar fused multiply-add per cycle: `d = a*b + c`. A **tensor core** does an entire small **matrix** multiply-accumulate per instruction:

```
   D  =  A  ×  B  +  C          all in ONE instruction, e.g. a 16×8×16 tile
   ────────────────────
   ~2,000 FLOPs per instruction instead of 2
```

That's the whole idea: since deep learning is ~99% matmul, NVIDIA built silicon that speaks matmul natively (Volta 2017 onward). The payoff on an H100:

| Unit | dtype | Peak throughput (dense) |
|---|---|---|
| CUDA cores | FP32 | ~67 TFLOP/s |
| Tensor cores | TF32 | ~495 TFLOP/s |
| Tensor cores | FP16 / BF16 | ~990 TFLOP/s |
| Tensor cores | FP8 | ~1,979 TFLOP/s |
| Tensor cores | INT8 | ~1,979 TOP/s |

(Marketing sheets often quote 2× these using structured sparsity — ignore that number unless you're actually using 2:4 sparsity.)

**Read the table again.** FP32 on CUDA cores is ~67 TFLOP/s; BF16 on tensor cores is ~990. That's a **~15× gap**, from the same chip, decided by dtype. If your model runs in FP32, you are leaving 93% of the GPU on the table before writing a single line of optimization.

---

## The two independent wins from low precision

People say "FP16 makes it 2× faster" and stop. There are really two separate mechanisms, and knowing which one applies is the difference between predicting a speedup and being surprised by one:

1. **Fewer bytes to move** — helps **memory-bound** work (decode, elementwise ops, KV-cache reads). FP16 halves bytes vs FP32 → roughly halves time. This one applies to *everything* memory-bound and is why quantization is so effective for decode.
2. **Faster math units** — helps **compute-bound** work (prefill, large matmuls). Tensor cores give ~15× over FP32 CUDA cores. This one applies only when you're actually near the compute roof.

Which explains a classic interview question from the ROADMAP:

> *"Why does INT4 quantization sometimes barely speed up prefill but dramatically speed up decode?"*

Because **prefill is compute-bound** — you're limited by math units, and INT4 weights usually get dequantized to FP16 for the matmul anyway, so the math throughput doesn't improve (sometimes it gets *worse* from dequant overhead). **Decode is memory-bound** — time ≈ bytes ÷ bandwidth — so cutting weights from 16 bits to 4 bits cuts the bytes ~4× and cuts the time nearly 4×. Same optimization, opposite outcomes, because the two phases sit on opposite sides of the roofline ([lesson 5](05-roofline-model.md)).

---

## The precision zoo, from a hardware point of view

```
FP32   1 sign │ 8 exponent │ 23 mantissa      4 bytes   the old default; CUDA cores only
TF32   1      │ 8          │ 10               (stored as 4B, computed on tensor cores)
BF16   1      │ 8          │ 7                2 bytes   FP32's range, less precision → no scaling needed
FP16   1      │ 5          │ 10               2 bytes   more precision, small range → can overflow
FP8    1      │ 4or5       │ 3or2             1 byte    Hopper+; needs per-tensor scaling
INT8   integer + scale factor                  1 byte    quantization (Phase 4)
INT4   integer + scale factor                  0.5 byte  weight-only quantization (Phase 4)
```

The practical guidance:

- **BF16 is the default for inference today.** Same exponent range as FP32, so values don't overflow/underflow and you don't need loss scaling. Supported on Ampere (A100) and newer.
- **FP16 is fine and slightly more precise**, but its max value is ~65,504 — activations in big models can overflow. On older cards (T4, V100) it's your only tensor-core float option.
- **TF32** is the sneaky one: on Ampere+, `torch.matmul` on FP32 tensors may *silently* run on tensor cores at reduced mantissa. Controlled by `torch.backends.cuda.matmul.allow_tf32`. Great for speed, and an occasional source of "why did my numbers change?"
- **FP8 / INT8 / INT4** are [Phase 4](../../ROADMAP.md#phase-4--inference-optimization-techniques) territory — they need calibration and quality evaluation, not just a dtype flag.

For inference specifically: **you almost never need FP32.** Inference has no gradients to accumulate and no optimizer state ([Phase 0 lesson 4](../phase-0/04-what-is-inference.md)), so the numerical fragility that forces mixed precision in training mostly doesn't apply.

---

## The rules for actually getting tensor cores

Tensor cores are not automatic. Miss a condition and the compiler quietly falls back to CUDA cores — no error, no warning, just 10× slower. The conditions:

**1. Right dtype.** FP16/BF16/FP8/INT8 (or FP32 with TF32 enabled). Pure FP32 without TF32 = no tensor cores.

**2. Shapes aligned to the tile size.** Tensor cores work on tiles; dimensions should be multiples of **8** (FP16 minimum) and ideally **64 or 128** for the best kernels. This is why:

- Vocabulary sizes get padded (GPT-NeoX pads 50,257 → 50,304).
- Hidden dims are 4096, 5120, 8192 — never 4097.
- A `seq_len=1000` prefill can be measurably slower than `seq_len=1024`.

**3. Enough work per tile.** A matmul with an M or N dimension of 1 — i.e. a **matrix-vector product, which is exactly what batch-1 decode is** — cannot fill a 16×8×16 tile. Tensor cores are essentially wasted during batch-1 decode; only batching brings them back into play.

**4. Contiguous, well-aligned memory.** Non-contiguous tensors may force a copy or a slower kernel path ([lesson 3](03-cuda-execution-model.md)).

Turn them on and verify in PyTorch:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True     # let FP32 matmuls use tensor cores
torch.backends.cudnn.allow_tf32 = True

model = model.half()          # FP16
model = model.bfloat16()      # BF16 — preferred on A100/H100
# or, for mixed precision around a region:
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x)
```

To *prove* tensor cores ran, don't trust a flag — measure. If achieved TFLOP/s exceeds the FP32 CUDA-core peak, tensor cores are running. In Nsight Compute the metric is `sm__pipe_tensor_cycles_active` ([lesson 8](08-profiling-in-practice.md)).

---

## Where precision bites in production

- **Numerics change.** FP16 and BF16 produce different results from FP32, and different *batch sizes* produce different results from each other (reduction order changes). This is why `temperature=0` isn't bit-reproducible on a busy server ([Phase 1 lesson 7](../phase-1/07-sampling.md)) and why "deterministic output" is an SLA you must think hard about before promising ([Phase 7](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)).
- **Softmax and LayerNorm accumulate in FP32** even in half-precision models — small dynamic-range-sensitive reductions are the one place people keep the extra bits. You'll see `.float()` inside those kernels in real code; now you know why.
- **KV-cache dtype is its own decision.** Weights in BF16 with the KV-cache in FP8 is a common production config: the cache is what's eating your HBM ([lesson 2](02-gpu-memory-hierarchy.md)), so halving it doubles concurrency.
- **Older GPUs constrain you.** T4/V100 have no BF16; T4 has no FP8. Before you plan a deployment, check the architecture's supported dtypes — a "cheap GPU" that can't run your dtype isn't cheap.

---

## Try it (Colab free tier; T4 works, numbers will be smaller)

Measure the dtype gap yourself — this is the single most convincing experiment in the lesson:

```python
import torch, time

def bench_matmul(n, dtype, iters=30):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    (a @ b); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): a @ b
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters
    return (2 * n**3) / t / 1e12          # TFLOP/s

n = 4096
torch.backends.cuda.matmul.allow_tf32 = False
print(f"FP32 (no TF32): {bench_matmul(n, torch.float32):6.1f} TFLOP/s")
torch.backends.cuda.matmul.allow_tf32 = True
print(f"FP32 (TF32 on): {bench_matmul(n, torch.float32):6.1f} TFLOP/s")
print(f"FP16          : {bench_matmul(n, torch.float16):6.1f} TFLOP/s")
if torch.cuda.is_bf16_supported():
    print(f"BF16          : {bench_matmul(n, torch.bfloat16):6.1f} TFLOP/s")
```

Then the alignment experiment — same order of magnitude of work, wildly different efficiency:

```python
for n in (4096, 4095, 4097, 4088, 4032):
    print(n, f"{bench_matmul(n, torch.float16):6.1f} TFLOP/s")
```

Record both tables. They belong in your Phase 2 exit artifact, and they make the "shapes must be aligned" rule something you've *seen*, not something you've read.

---

## Key takeaways

- **Tensor cores** do a whole matrix-tile multiply-accumulate per instruction and supply ~90%+ of a modern GPU's FLOPs; CUDA-core FP32 is ~15× slower.
- Low precision wins in **two independent ways**: fewer bytes moved (helps memory-bound decode) and faster math (helps compute-bound prefill). Know which one you're claiming.
- That distinction answers the classic question: INT4 speeds up decode a lot and prefill barely.
- **BF16 is the sane inference default** (FP32's range, half the bytes); FP16 for older cards; TF32 can silently apply to FP32 matmuls; FP8/INT8/INT4 are Phase 4.
- Tensor cores require **right dtype + aligned shapes (multiples of 8, prefer 64/128) + enough work + contiguous memory** — otherwise you silently fall back.
- **Batch-1 decode is a matrix-vector product** and can't use tensor cores meaningfully; batching is what brings them back.
- Precision changes numerics: expect non-bit-reproducibility, keep sensitive reductions in FP32, and treat KV-cache dtype as a separate lever.

**Next:** [The roofline model →](05-roofline-model.md) — one plot that takes everything from lessons 1-4 and tells you, for any kernel, whether to fix the math, the bytes, or the launches.
