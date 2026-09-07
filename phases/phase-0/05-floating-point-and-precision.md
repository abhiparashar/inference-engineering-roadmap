# 5 — Floating Point & Precision (Why Fewer Bits = Faster)

> **You'll be able to say:** "A number can be stored in 32, 16, 8, or 4 bits. Fewer bits means less memory and less data to move — and since decode is memory-bound, that directly makes inference faster and cheaper. This is the single highest-leverage optimization in the field."

This lesson is *the* seed of Phase 4. If you only deeply understand one Phase 0 concept, make it this one.

---

## Why computers can't store most numbers exactly

Computers store everything in bits (0s and 1s). Whole numbers are easy. But real numbers — 3.14159…, 0.1, 2/3 — are infinite or awkward in binary, so computers store an *approximation* using a fixed number of bits. The more bits you spend, the closer the approximation. That trade — bits vs accuracy — is the entire topic.

The standard format is **floating point**, which stores a number like scientific notation:

```
value = sign × mantissa × 2^exponent
        │       │            │
        │       │            └── exponent: how big/small (the RANGE)
        │       └── mantissa (a.k.a. fraction): the significant digits (the PRECISION)
        └── sign: positive or negative (1 bit)
```

- **Exponent bits** control the **range** — the biggest and smallest magnitudes you can represent.
- **Mantissa bits** control the **precision** — how many significant digits, i.e. how fine-grained.

Every format below just splits its total bits differently between "sign / exponent / mantissa."

---

## The formats you must know by name

| Format | Total bits | Sign / Exp / Mantissa | Plain-English character |
|---|---|---|---|
| **FP32** (single) | 32 | 1 / 8 / 23 | The "full precision" default. Lots of range *and* precision. The baseline. |
| **FP16** (half) | 16 | 1 / 5 / 10 | Half the memory. Good precision but **small range** — big/small numbers overflow to infinity easily. |
| **BF16** (bfloat16) | 16 | 1 / 8 / 7 | Same **range** as FP32 (8 exponent bits), less precision. Much harder to overflow → the modern default for ML. |
| **INT8** | 8 | integer, 256 levels | Not floating point — a plain integer 0-255 (or -128..127). Needs a scale factor to map to real values. 1/4 the memory of FP32. |
| **INT4** | 4 | integer, 16 levels | Only 16 possible values! 1/8 the memory of FP32. Aggressive; needs care to keep quality. |

Two things surprise beginners:

1. **FP16 and BF16 are both 16 bits but differ a lot.** FP16 spends more bits on precision; BF16 spends more on range. In ML, range matters more (activations can get large), and overflow is catastrophic, so **BF16 is usually preferred** even though it's "less precise." This is why you'll see BF16 everywhere in modern models.
2. **INT8/INT4 aren't "small floats" — they're integers plus a scale.** To store a real weight like `0.037` in INT8, you pick a scale `s` and store the nearest integer `round(0.037 / s)`, then multiply back by `s` when you use it. Choosing good scales *is* the art of quantization (GPTQ, AWQ — Phase 4).

### See the approximation error

```python
import numpy as np
x = 0.1
print(f"{np.float32(x):.20f}")   # already not exactly 0.1
print(f"{np.float16(x):.20f}")   # coarser — bigger error
```

Neither is exactly `0.1`, and FP16's error is larger. Fewer bits → coarser grid → more rounding error. Usually harmless; occasionally it isn't, which is why you *measure* quality after reducing precision (Phase 4).

---

## Why fewer bits makes inference *faster*, not just smaller

This is the part that surprises people, and it's the whole point. Two independent wins:

**Win 1 — Less memory (obvious).** A 7-billion-parameter model:
- FP32: 7B × 4 bytes = **28 GB**
- FP16/BF16: 7B × 2 bytes = **14 GB**
- INT8: 7B × 1 byte = **7 GB**
- INT4: 7B × 0.5 byte = **3.5 GB**

INT4 makes a model that needed a $30k datacenter GPU fit on a gaming card. That alone is huge for cost and access.

**Win 2 — Less data to move (the subtle, bigger win for decode).** Recall from [lesson 2](02-memory-hierarchy.md): generating each LLM token requires **reading the entire model's weights out of memory**, and decode is **memory-bandwidth-bound** (limited by bytes/sec, not math/sec). If you halve the *bytes* of the weights, you halve the data that must be streamed per token → roughly **2x faster decode**, for free, because the bottleneck was moving bytes, not doing math.

```
Decode is memory-bound, so time-per-token ≈ (bytes of weights) ÷ (memory bandwidth)

  FP16 7B model: 14 GB ÷ 2 TB/s ≈ 7 ms  ── floor per token
  INT8 7B model:  7 GB ÷ 2 TB/s ≈ 3.5 ms ── ~2x faster, same GPU, same math
```

That's the killer insight: **quantization speeds up decode precisely *because* decode is bandwidth-bound.** Fewer bytes per weight = fewer bytes to stream = faster tokens. You are converting the memory-hierarchy lesson into money.

> Caveat that shows you *get* it: quantization helps **decode** (bandwidth-bound) a lot, but helps **prefill** (compute-bound — see [Phase 1](../phase-1/06-prefill-vs-decode.md)) less, because prefill's bottleneck is math, not bytes. Being able to say *why* is a Phase 4 self-check question. You now can.

---

## The cost side: what you pay for fewer bits

Nothing is free. Fewer bits = coarser approximation = potential quality loss (the model gets slightly "dumber": higher perplexity, occasional worse answers). The engineering job is to **push bits as low as possible while keeping quality acceptable**, and to *measure* that trade rather than guess. Rough real-world picture:

- **FP32 → BF16/FP16:** almost always free (negligible quality change). Do it by default.
- **→ INT8:** usually a very small quality hit with good methods; big memory/speed win. Common in production.
- **→ INT4:** noticeable if done naively, but modern methods (GPTQ, AWQ) keep it surprisingly good; the win is large. Used heavily for running big models on small hardware.

The reason clever methods exist is that not all weights matter equally — a few "important" weights deserve more bits, and the rest can be crushed. That's the idea behind AWQ ("protect the salient channels") and GPTQ ("correct the error layer by layer"), which you'll implement/benchmark in Phase 4. For now, just hold the shape: **lower precision → smaller + faster, at some quality cost you must measure.**

---

## Where each format shows up in practice

- **Training:** mostly BF16 (with some FP32) — needs range and reasonable precision for gradients.
- **Inference weights:** FP16/BF16 as the safe default; INT8/INT4 when you need cheaper/faster and can afford to validate quality.
- **KV-cache** (Phase 1): can *also* be quantized (e.g. FP8/INT8) — important because at long context lengths the cache can be bigger than the weights.
- **Newer formats:** FP8 (8-bit float) on recent NVIDIA GPUs, and even more exotic ones. Same principle, more points on the trade-off curve.

---

## Tiny experiments

**1. Measure the memory win directly:**

```python
import numpy as np
n = 7_000_000_000   # 7B params
for name, bytes_per in [("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)]:
    print(f"{name}: {n * bytes_per / 1e9:.1f} GB")
```

**2. Round-trip a weight through INT8 by hand** to feel quantization:

```python
import numpy as np
w = np.array([0.037, -0.51, 0.0, 0.99], dtype=np.float32)
scale = np.abs(w).max() / 127          # map max value to 127
q = np.round(w / scale).astype(np.int8)  # store these 8-bit ints
recovered = q.astype(np.float32) * scale  # dequantize when using
print("original: ", w)
print("int8 codes:", q)
print("recovered:", recovered)          # close, not exact — that's the trade
```

You just did quantization. GPTQ/AWQ are smarter versions of exactly this.

---

## Key takeaways

- Floating point stores numbers as `sign × mantissa × 2^exponent`; **exponent = range, mantissa = precision.** Fewer bits = coarser approximation.
- **FP32** = baseline; **FP16** = half memory but small range; **BF16** = half memory with FP32's range (the ML default); **INT8/INT4** = integers + a scale, 1/4 and 1/8 the memory.
- Fewer bits wins twice: **less memory** (fit bigger models on smaller GPUs) and **less data to move**.
- Because **decode is memory-bandwidth-bound**, halving the bytes per weight roughly **halves time-per-token** — quantization's speedup is a direct payoff of the memory-hierarchy lesson.
- The cost is quality loss, so you always **measure** (perplexity / eval) after reducing precision. Smart methods (GPTQ, AWQ) minimize the loss by spending bits where they matter.

**Next:** [Build a server from scratch, then with a framework →](06-build-a-server-from-scratch.md) — make the systems concepts real with code you run.
