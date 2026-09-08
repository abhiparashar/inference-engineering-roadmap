# 3 — Quantization Methods in Practice

> **You'll be able to say:** "Every method is an answer to outliers. **GPTQ** chooses the rounding instead of rounding to nearest, using an approximate Hessian to compensate each error in the remaining weights. **AWQ** rescales input channels so that activation-salient weights get effective precision back. **LLM.int8()** keeps outlier feature dimensions in FP16 and the rest in INT8. **SmoothQuant** migrates activation outliers into the weights so both sides quantize. **FP8** does it in hardware on H100+. I implemented AWQ's scaling trick and measured it: on GPT-2 it cuts INT4 output error 6.22% → 5.93%, and I can explain why the win is small there and large on a 7B."

Lesson 2 gave you the knobs. This lesson is the state of the art, what each method costs, and how to choose — including the part nobody writes down: **the kernel matters as much as the algorithm.**

---

## The method table

| Method | What it does | Bits | Needs calibration? | Speeds up prefill? | Where you meet it |
|---|---|---|---|---|---|
| **RTN** (round-to-nearest) | naive rounding, per-group scales | 8, 4 | no | no | baseline; fine at INT8 |
| **GPTQ** | layer-wise optimal rounding via approximate second-order (Hessian) error compensation | 4, 3, 2 | yes (~128 samples) | no | `auto-gptq`, `gptqmodel`, vLLM `--quantization gptq` |
| **AWQ** | per-input-channel rescaling that protects activation-salient weights | 4 | yes (activation stats) | no | `autoawq`, vLLM `--quantization awq` |
| **LLM.int8()** | outlier feature dims in FP16, rest INT8, results summed | 8 | no (dynamic) | partly | `bitsandbytes`, HF `load_in_8bit` |
| **NF4 / QLoRA** | 4-bit NormalFloat, information-theoretically matched to a normal distribution + double quantization | 4 | no | no | `bitsandbytes`, `load_in_4bit` — the fine-tuning default |
| **SmoothQuant** | migrates activation outliers into weights via a per-channel scale, enabling **W8A8** | 8/8 | yes | **yes** | TensorRT-LLM, vLLM (`w8a8`) |
| **FP8 (E4M3/E5M2)** | hardware float8 for weights *and* activations | 8 | light (scaling factors) | **yes** | H100/Ada+, TensorRT-LLM, vLLM `--quantization fp8` |
| **k-quants (GGUF)** | mixed per-block schemes (Q4_K_M etc.) tuned empirically | 2-8 | no | n/a (CPU/Metal) | `llama.cpp`, Ollama, LM Studio |

Read the "needs calibration" and "speeds up prefill" columns together: **calibration-free methods are convenient; only the activation-quantizing methods (SmoothQuant, FP8) touch prefill.** If your problem is TTFT, no amount of GPTQ will fix it.

---

## GPTQ: choose the rounding, don't just round

RTN rounds each weight independently and hopes the errors cancel. GPTQ starts from a better objective — minimize the *layer output* error on calibration data:

```
  minimize  || X·W − X·Ŵ ||²      over quantized Ŵ,  given calibration activations X
```

The insight (inherited from Optimal Brain Quantization/Surgeon): weights are not independent. When you round `w_j` up, you can **compensate** by nudging the not-yet-quantized weights, using the inverse Hessian `H⁻¹ = (2XᵀX + λI)⁻¹` to say which nudge fixes the most output error. GPTQ processes columns left to right, quantizing one and pushing the residual into the rest, with a Cholesky-based trick that makes it fast (minutes to hours per model, one pass, no gradient descent).

What you need to know operationally:

- **`group_size=128` and `desc_act=True` (act-order)** are the standard settings; act-order quantizes the most activation-important columns first and typically recovers a chunk of quality at some kernel-speed cost.
- **Calibration set matters less than people fear but not zero**: ~128 sequences of your domain's text. Calibrating a code model on Wikipedia is a real, measurable mistake.
- **It's the workhorse for 4-bit and the only one that credibly reaches 3-bit** on large models.

---

## AWQ: protect the channels the activations care about

AWQ's observation: **not all weights matter equally, and importance is visible in the activations, not the weights.** Roughly 1% of weight channels — those multiplied by large-magnitude activation features — dominate the output error.

The trick is beautifully cheap. For a linear layer `Y = X·W`, insert a diagonal rescaling that cancels:

```
  Y = X·W = (X · diag(1/s)) · (diag(s) · W)
              └─ folded into the previous op    └─ quantize THIS instead
```

Scaling a channel up by `s` before quantization means its quantization error is divided by `s` after the inverse scaling — you have bought that channel effective precision, paid for by the others. Choose `s_j = a_j^α` where `a_j` is the mean absolute activation of input channel `j`, and search `α ∈ [0, 1]` per layer. No backprop, no Hessian; just activation statistics and a grid search.

### Measured, on real weights and real activations

GPT-2 block 0, `mlp.c_fc`, activations captured from a real forward pass, judged by **output** error `‖XW − X'Ŵ'‖/‖XW‖` (the metric that matters, not weight error):

```
activations (512, 768)  per-channel |x| mean: min=0.0454 median=0.0950 max=0.6127  (max/median = 6x)

scheme                                        output rel err
INT4 g128, round-to-nearest                           6.216%
INT4 g128, AWQ-style scaling alpha=0.25               5.929%
INT4 g128, AWQ-style scaling alpha=0.5                6.273%
INT4 g128, AWQ-style scaling alpha=0.75               7.597%
INT4 g128, AWQ-style scaling alpha=1.0                9.936%
INT4 g128, top-1% channels kept FP16                  5.656%   (k=7 of 768)
```

Three honest conclusions:

1. **The mechanism works, and α must be searched.** α = 0.25 improves on RTN; α = 1.0 (fully scaling by activation magnitude) is **60% worse** than doing nothing. This is exactly why AWQ grid-searches α per layer rather than deriving it — and it's a good reminder that a technique applied without its search procedure can hurt.
2. **The win here is small (6.22% → 5.93%) because GPT-2's outliers are mild** — max/median activation magnitude is only **6×**. In 7B+ models that ratio reaches 20-100×, which is where AWQ and LLM.int8() report their large gains. **A method's benefit is proportional to the pathology it targets; measuring it on a small model understates it.** Say this out loud when someone shows you a quantization result on GPT-2.
3. **Keeping the top 1% of channels in FP16 beats both** (5.66%) at a cost of 1% of the matrix staying 16-bit — which is the LLM.int8() idea, and evidence that mixed precision is a genuinely strong baseline.

Reproduce it (the whole experiment is ~30 lines):

```python
a = X.abs().mean(0)                      # per-input-channel activation magnitude
s = (a ** alpha); s = (s / s.mean()).reshape(-1, 1)
Ws = W * s                               # quantize the SCALED weights
Y  = (X / s.reshape(1, -1)) @ qdq(Ws)    # inverse scale folds into the previous layer
```

---

## LLM.int8() and SmoothQuant: two ways to handle activation outliers

**LLM.int8() (bitsandbytes)** — *decompose*. At runtime, find hidden dimensions whose activation magnitude exceeds a threshold (~6.0), compute those columns in FP16, compute everything else in INT8, and sum the two results. Outliers are handled exactly; the other 99.9% of the matrix moves half the bytes. Zero calibration, drop-in via `load_in_8bit=True`, and essentially no quality loss — but the mixed path costs speed, and it is often *slower* than FP16 at small batch. **Use it to fit a model in memory, not to make it fast.**

**SmoothQuant** — *migrate*. Activation outliers are hard to quantize; weights are easy. So move the difficulty:

```
  Y = (X · diag(1/s)) · (diag(s) · W)         same identity as AWQ, different objective
  s_j = max|X_j|^α / max|W_j|^(1−α)           α ≈ 0.5 balances the two difficulties
```

Now both sides are quantizable to INT8, and you get **W8A8** — real INT8 tensor-core math, which speeds up **prefill** as well as decode. That's why it's a staple in TensorRT-LLM. The scales fold into the preceding LayerNorm at export time, so inference pays nothing.

**FP8 (H100/Ada and later)** — the hardware answer. E4M3 (more mantissa, for weights/activations) and E5M2 (more exponent, for gradients) with per-tensor or per-block scaling factors. Because floats degrade gracefully where integers clip, FP8 typically needs far less ceremony than INT8 for the same quality, and it has native tensor-core support: ~2× the FLOP/s of FP16 *and* half the bytes. If you have the hardware, FP8 is usually the first thing to try.

---

## The kernel is half the story

An algorithm that produces 4-bit weights is worthless without a kernel that reads them fast. This is where real deployments go wrong.

- **Dequant-in-kernel is the norm** for W4A16: load packed INT4 + scales into SRAM, dequantize into registers, run FP16 tensor-core math. The win is bytes; the risk is that the dequant overhead eats it at large batch.
- **Kernel maturity varies enormously.** vLLM ships Marlin/Machete for GPTQ/AWQ-style formats because the earlier reference kernels left most of the win on the table. Two runtimes, same checkpoint, very different tokens/sec.
- **Batch size flips the answer.** W4A16 kernels shine at small batch (bandwidth-bound). At large batch, the matmul becomes compute-bound and dequantization is pure overhead — several serving stacks switch strategy above a batch threshold. **Always benchmark at your real batch size.**
- **Shapes must be tensor-core friendly** ([Phase 2 lesson 4](../phase-2/04-tensor-cores-and-precision.md)); quantized kernels are usually *stricter* about alignment than FP16 ones.

---

## Choosing, and evaluating

**A decision procedure that will not embarrass you:**

```
  Have H100/Ada?           ──▶ try FP8 first (weights+activations, prefill too, minimal ceremony)
  Need it to FIT only?     ──▶ bitsandbytes INT8/NF4 — one flag, no calibration, don't expect speed
  Serving W4 on A100/T4?   ──▶ AWQ or GPTQ (group 128); pick by MEASURED quality on your task,
                               and by which kernel your runtime has (Marlin/Machete >> reference)
  Prefill-bound / compute-limited? ──▶ SmoothQuant W8A8 or FP8; W4A16 will not help you
  CPU / Apple Silicon / edge?      ──▶ GGUF k-quants via llama.cpp (Q4_K_M is the sane default)
  Fine-tuning on one GPU?          ──▶ QLoRA (NF4 + LoRA adapters)
```

**The evaluation protocol** — this is the part that makes it an artifact rather than an anecdote:

1. **Memory**: weights on disk, weights in VRAM, peak VRAM at your batch/context. Predicted vs measured.
2. **Speed**: TTFT, TPOT and output tok/s at ≥2 offered loads, from the Phase 3 harness, with warmup. Include a **large-batch** point — that's where quantization wins shrink.
3. **Quality**: perplexity on WikiText-2 **and** a task benchmark (a few hundred GSM8K/HumanEval/MMLU items, or your own eval). Report both; they disagree, and the task metric is the one that matters.
4. **Failure modes**: quantized models degrade unevenly — long-context recall, code, and non-English text go first. Include at least one long-context probe.
5. **One table**, one row per config: `dtype | bits/weight | VRAM | TTFT p50 | TPOT p50 | tok/s @ load | ppl | task metric | $/1M`.

That table *is* Project 03, and it is genuinely portfolio-grade because almost nobody publishes all six columns together.

---

## Try it

1. Reproduce the AWQ experiment on a bigger model if you have the VRAM (Qwen2.5-1.5B, Llama-3.2-1B). Plot `max/median` activation magnitude per layer and compare the AWQ gain to GPT-2's 6×. **You are measuring the outlier pathology directly, which is the real content of the AWQ paper.**
2. Implement the LLM.int8() baseline properly: threshold-based FP16 columns + INT8 for the rest, and sweep the threshold (2, 4, 6, 8). Plot output error vs the fraction of the matrix left in FP16.
3. Quantize a real model with `autoawq` and with `auto-gptq`, serve both with vLLM, and run the Phase 3 harness at 1 QPS and at saturation. Report where the 4-bit advantage shrinks and by how much.
4. Do the **skip-list ablation**: quantize everything including embeddings/LM head, then progressively exclude layers. Plot perplexity vs bits/weight and find which exclusions are worth their bytes.

---

## Key takeaways

- **Every method is an outlier strategy.** GPTQ compensates rounding error via an approximate Hessian; AWQ rescales channels the activations flag as salient; LLM.int8() splits outlier dims into FP16; SmoothQuant migrates activation outliers into weights; FP8 sidesteps clipping with a float format.
- **AWQ's trick is a diagonal rescale that cancels algebraically** — measured here: RTN 6.22% → 5.93% output error at α = 0.25, and **worse** (9.94%) at α = 1.0. Search α; don't assume it.
- **A method's benefit scales with the pathology it targets.** GPT-2's activation max/median is 6×; 7B+ models reach 20-100×, which is where these methods earn their reputation.
- **Keeping the top 1% of channels in FP16 was the best result in the experiment** (5.66%) — mixed precision is a strong, simple baseline.
- **Only activation quantization (SmoothQuant W8A8, FP8) speeds up prefill.** W4A16 is a decode/memory optimization, full stop.
- **The kernel is half the story**: same checkpoint, different runtime, very different speed; and the quantization win shrinks as batch size grows.
- **Evaluate with six columns** — bits/weight, VRAM, TTFT, TPOT/throughput at a stated load, perplexity, and a task metric — plus a long-context probe. Anything less is a press release.

**Next:** [KV-cache optimization →](04-kv-cache-optimization.md) — once weights are small, the cache is the problem: GQA, KV quantization, windows, and offloading, each converted into concurrent sequences.
