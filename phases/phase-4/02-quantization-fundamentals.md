# 2 — Quantization Fundamentals

> **You'll be able to say:** "Quantization is an affine map `w ≈ scale·(q − zero)` with integer `q`. The only interesting question is **how many weights share a scale**. On a real GPT-2 weight matrix, INT4 per-tensor has 91.6% relative error and drives perplexity from 1.61 to 8,815 — the model is destroyed. The *same* INT4 with group-128 scales has 14.0% weight error and +0.13 perplexity, at a storage cost of 4.12 bits per weight. Outliers are why: one weight of 50.0 inflates the shared scale 11× and ruins everyone else's precision. And it speeds up decode because decode is bandwidth-bound — not because integer math is faster."

This lesson is the mechanism. [Lesson 3](03-quantization-methods.md) is the named methods (GPTQ, AWQ, LLM.int8(), SmoothQuant, FP8), and they all make sense *only* once you can see the scale-sharing tradeoff.

---

## The map

```
  float32  w = -0.0731
                 │  quantize:  q = round(w / scale) + zero          (store q as INT4/INT8)
                 ▼
  int4     q = -3                                       ◀── this is what lives in HBM
                 │  dequantize: ŵ = (q − zero) · scale
                 ▼
  float    ŵ = -0.0742          error = 0.0011
```

Two flavours, and the distinction shows up in every library flag:

```
  SYMMETRIC (zero = 0)                   ASYMMETRIC (zero ≠ 0)
  scale = absmax / (2^(b−1) − 1)         scale = (max − min) / (2^b − 1)
  range  [−absmax, +absmax]              range  [min, max]
  cheaper kernels (no zero-point term)   fits skewed distributions (e.g. post-ReLU/GELU)
  standard for WEIGHTS                   common for ACTIVATIONS and for INT4 weights
```

And the axis that actually decides quality — **granularity**, i.e. how many weights share one `scale`:

| Granularity | Scales for a `[768, 3072]` matrix | Extra bits/weight | Used for |
|---|---|---|---|
| per-tensor | 1 | ~0 | activations, INT8 weights (sometimes) |
| per-channel (per output column) | 3,072 | 16/768 = 0.02 | INT8 weights — the standard |
| **per-group of 128** | 18,432 | 16/128 = **0.125** | INT4 weights — the standard (GPTQ/AWQ `group_size=128`) |
| per-group of 32 | 73,728 | 0.5 | aggressive INT4, higher quality, bigger footprint |

**Fewer weights per scale = better fidelity, more metadata.** That's the entire design space, and the numbers below show exactly what it's worth.

---

## Measure it on a real weight matrix

`transformer.h[0].mlp.c_fc.weight` from GPT-2, shape `[768, 3072]`, 2.36 M weights:

```
weight (768, 3072)  mean=-0.0007 std=0.1412 absmax=4.5877  99.9pct=0.5823
```

Note that immediately: **absmax is 4.59 but the 99.9th percentile is 0.58** — an 8× gap. The distribution is a narrow bell with a few far-out weights, and a symmetric per-tensor scale is set by the *worst* weight in the whole matrix. Here's what that costs (round-to-nearest, no error correction):

```
INT8 per-tensor  (1 scale)         bits/weight= 8.00  rel_L2_err= 7.390%  max_abs_err=0.01806  cos=0.998567
INT8 per-group 128                 bits/weight= 8.12  rel_L2_err= 0.780%  max_abs_err=0.01771  cos=1.000289
INT8 per-group 32                  bits/weight= 8.50  rel_L2_err= 0.601%  max_abs_err=0.01724  cos=1.000302
INT8 per-group 128 asym            bits/weight= 8.25  rel_L2_err= 0.663%  max_abs_err=0.00975  cos=1.000306

INT4 per-tensor  (1 scale)         bits/weight= 4.00  rel_L2_err=91.560%  max_abs_err=0.32769  cos=0.514419
INT4 per-group 128                 bits/weight= 4.12  rel_L2_err=13.962%  max_abs_err=0.28987  cos=0.990544
INT4 per-group 32                  bits/weight= 4.50  rel_L2_err=10.845%  max_abs_err=0.17017  cos=0.994329
INT4 per-group 128 asym            bits/weight= 4.25  rel_L2_err=11.237%  max_abs_err=0.16293  cos=0.993887
```

Four readings:

1. **Granularity is worth ~9.5× at INT8 and ~6.6× at INT4** in relative error, for 0.12 extra bits per weight. This is the best deal in the entire phase: you pay 1.5% more storage to remove most of the damage.
2. **INT4 per-tensor is not a quantization scheme, it's vandalism** — 91.6% error, cosine similarity 0.51 (the quantized matrix is barely correlated with the original). Any library that offers it is offering a footgun.
3. **Asymmetric buys about as much as halving the group size**, at half the metadata cost (0.25 vs 0.5 bits/weight), because it doesn't waste range on a side of zero the weights don't use.
4. **INT8 group-128 is essentially free fidelity** (0.78% error). This is why INT8 weight-only quantization is considered safe and why INT4 is where the arguments start.

### The outlier experiment

Add a single weight of value 50.0 to the matrix and re-quantize INT4 per-tensor:

```
with ONE outlier weight (50.0) added, INT4 per-tensor rel_L2_err on the OTHER weights: 97.404%
(scale grew 11x, so every normal weight lost precision)
```

**One weight in 2.36 million degraded everything else**, because the scale is `absmax/7` and absmax is now that outlier. Transformers have exactly this structure — a small number of "outlier features," concentrated in specific channels, that grow with model size. That single fact generates three of the four methods in the next lesson: LLM.int8() isolates outliers, SmoothQuant migrates them from activations into weights, AWQ protects the channels the activations say matter.

---

## Does the weight error matter? Perplexity says yes, sharply

Quantize **every** attention and MLP weight matrix in GPT-2 (48 matrices; embeddings and LayerNorms left in FP32 — they're tiny and sensitive) and measure perplexity on a fixed 1,024-token passage:

```
config                            ppl    delta   (fp32 baseline 1.610)
INT8 per-tensor                1.619   +0.009
INT8 group 128                 1.607   -0.003
INT4 per-tensor             8815.574+8813.964
INT4 group 128                 1.740   +0.129
INT4 group 32                  1.620   +0.010
INT3 group 128              1093.374+1091.764
INT2 group 128             15335.010+15333.399
(quantized 48 weight matrices per config)
```

*(The absolute perplexity is low because the passage is repetitive — read the **deltas**, and reproduce with WikiText-2 for a publishable number.)*

- **INT8 is free.** Both granularities land within noise of FP32; group-128 even edges below, which is measurement noise, not magic.
- **INT4 group-128 costs +0.13 perplexity; group-32 costs +0.01.** Both are shippable, and the choice is a memory-vs-quality dial you can now quantify.
- **INT4 per-tensor produces a broken model** (ppl 8,816). Same bit width, 5,000× worse, purely from scale sharing.
- **INT3 and INT2 with naive rounding are unusable** (1,093 and 15,335). This is the exact gap that GPTQ and AWQ close — they get 3-4 bits working by *choosing* the rounding instead of rounding to nearest. Now you know what those papers are actually fixing.

Run it yourself; it takes a minute on a laptop:

```python
def qdq(W, bits, group=None):                       # quantize -> dequantize, symmetric
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(1, -1) if group is None else W.reshape(-1, group)
    s = Wg.abs().max(1, keepdim=True).values.clamp(min=1e-8) / qmax
    return (torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s).reshape(W.shape)

for name, p in model.named_parameters():
    if any(k in name for k in ("c_fc.weight", "c_proj.weight", "c_attn.weight")):
        p.data = qdq(p.data.float(), bits=4, group=128)
```

---

## Why this speeds up decode (and barely touches prefill)

The kernel does **not** do integer math with your INT4 weights. In weight-only quantization it:

```
  load INT4 block from HBM  ──▶  dequantize in registers/SRAM  ──▶  FP16 matmul  ──▶  accumulate
        ↑ 4× fewer bytes                ↑ a few extra ALU ops        ↑ same FLOPs as FP16
```

So the FLOP count is unchanged and the byte count is divided by 4. Apply [lesson 1](01-what-to-optimize.md)'s model:

- **Decode, batch 1, 7B:** time ≈ `14 GB / 2 TB/s = 7 ms` → INT4 → `3.5 GB / 2 TB/s = 1.75 ms`. **~4× faster**, because it was pure bandwidth.
- **Prefill, 2,000 tokens:** compute-bound (`2·P·T` FLOPs against a compute roof), and dequantization *adds* work. **~1×, sometimes slightly worse.**
- **Decode at large batch with long context:** KV bytes dominate, so weight quantization's share of the total shrinks and the speedup fades toward 1×. Compute the crossover before promising anyone a number.

This is the answer to the Phase 4 self-check question, and it's a favourite interview question precisely because it separates people who understand the roofline from people who memorized "quantization makes it faster."

The second, often larger benefit: **the weights you didn't store are memory you can spend on KV-cache.** 7B FP16 = 14 GB of weights; INT4 = 3.5 GB. On an 80 GB card that's 10.5 GB more KV, which at 0.5 MB/token is ~21,000 more cached tokens — more concurrency, which is more throughput even if the step time were unchanged.

---

## Activation quantization: a different, harder problem

Weight-only (W8A16, W4A16) is the easy, popular case: weights are *static*, so you quantize once, offline, and inspect the error at your leisure. Quantizing **activations** (W8A8, FP8) is harder and has a different payoff:

| | Weight-only (W4A16/W8A16) | Weight+activation (W8A8, FP8) |
|---|---|---|
| Helps | decode (bandwidth) | decode **and prefill** (real INT8/FP8 tensor-core math) |
| Difficulty | offline, deterministic | activations vary per input; outliers are dynamic |
| Calibration | optional (a few hundred samples for GPTQ/AWQ) | required, and outlier handling is essential |
| Hardware | any | INT8 tensor cores; FP8 needs H100/Ada+ |
| Typical use | serving open models on any GPU | frontier-scale, latency- *and* compute-limited serving |

The reason activation quantization is hard has a name: **outlier features.** In large transformers, a handful of hidden dimensions carry values 10-100× the typical magnitude, consistently, across tokens. Per-tensor activation scales then behave exactly like the outlier experiment above. The fixes are lesson 3's subject.

---

## Practical rules

- **Never quantize everything.** Embeddings, LayerNorm/RMSNorm parameters, the LM head, and (usually) the first and last blocks are disproportionately sensitive and tiny. Every real toolkit keeps a skip-list; know that yours has one.
- **Group size 128 is the default for a reason** — the knee of the quality/metadata curve, and the size kernels are tuned for. Use 32 only if you measured that you need it.
- **Report bits-per-weight including metadata.** "INT4" with group-32 asymmetric is 4.5-5 bits/weight, not 4 — a 12-25% difference in the thing you were optimizing.
- **Quality needs two numbers**: perplexity on a standard set (WikiText-2 is the convention) *and* a task metric. Perplexity is a smoke detector, not a safety certificate — instruction-following and code generation degrade before perplexity notices.
- **Measure speed end-to-end, not in the kernel.** A 4× byte reduction with an immature dequant kernel can be *slower* than FP16. This happens routinely on new hardware/format combinations. Trust your Phase 3 harness, not the datasheet.

---

## Try it

1. Reproduce both tables above on a model you care about. Add per-channel (axis-wise) quantization to the comparison.
2. Plot **rel-L2 error vs bits/weight including metadata** for {INT8, INT4, INT3} × {per-tensor, group 128, group 32, asym}. Find the Pareto frontier. Note where INT3-group-32 sits relative to INT4-group-128 at equal bits/weight — that comparison is the actual research question in low-bit quantization.
3. **Find the outliers:** for each layer, compute `absmax / 99.9th-percentile` of the weights, and plot it by depth. Then do the same for *activations* by hooking a forward pass. You'll see the outlier-feature phenomenon with your own eyes, and lesson 3 will read like a solution instead of a list.
4. Quantize progressively deeper prefixes of the network (first 2 layers, first 4, …) and plot perplexity. Sensitivity is not uniform, and this is how mixed-precision assignment is actually chosen.

---

## Key takeaways

- Quantization = affine map `ŵ = scale·(q − zero)`; symmetric for weights, asymmetric where the distribution is skewed. **Granularity — weights per scale — decides quality.**
- Measured on a real GPT-2 matrix: INT8 per-tensor 7.39% error → group-128 **0.78%**; INT4 per-tensor **91.56%** → group-128 **13.96%**, for 0.125 extra bits/weight.
- End-to-end: INT8 is free; **INT4 group-128 costs +0.13 ppl, group-32 +0.01, per-tensor destroys the model** (+8,814). Naive INT3/INT2 are unusable — that's the gap GPTQ/AWQ exist to close.
- **Outliers set the scale.** One weight of 50.0 inflated the scale 11× and pushed error to 97%. Transformers have systematic outlier features; every serious method is an outlier strategy.
- **Weight-only quantization speeds up decode (bandwidth), not prefill (compute)** — it dequantizes to FP16 and does the same FLOPs. ~4× at batch 1; less as KV bytes take over.
- The *second* win is capacity: 10.5 GB of weights freed on an 80 GB card ≈ 21,000 extra KV tokens ≈ more concurrency ≈ more throughput.
- **Activation quantization** (W8A8/FP8) also speeds up prefill but needs calibration and outlier handling, and FP8 needs H100-class hardware.
- Always report **bits/weight including metadata**, perplexity *and* a task metric, and end-to-end speed from your own harness.

**Next:** [Quantization methods in practice →](03-quantization-methods.md) — GPTQ, AWQ, LLM.int8(), SmoothQuant and FP8: what each one actually does about outliers, and how to choose.
