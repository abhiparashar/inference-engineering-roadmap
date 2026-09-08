# 1 — What to Optimize, and How to Prove It

> **You'll be able to say:** "Before touching code I write the decode cost model: bytes moved per step = weights + KV, time = max(bytes/bandwidth, FLOPs/peak), tokens/sec = B/step. That model says a 7B FP16 model at batch 1 on an A100 runs ~7.5 ms/step at arithmetic intensity 1 — 300× below the ridge — and it tells me the *ordered* list of what to fix: raise B (Phase 3), shrink weight bytes (quantization), shrink KV bytes (GQA/KV-quant), remove KV waste (paging), skip work entirely (prefix caching), or break sequentiality (speculative decoding). Then I predict the factor before I measure it."

Phase 4 is a toolbox, and the failure mode of a toolbox is reaching for the wrong tool with great enthusiasm. This lesson is the selection procedure. It costs twenty minutes and saves weeks.

---

## The one model you need

Decode, per step, for a dense transformer:

```
  bytes_moved  =  weight_bytes            +  kv_bytes
               =  P · w_bits/8            +  2 · L · H_kv · d_head · kv_bits/8 · ctx · B
                  └─ read ONCE per step,     └─ every sequence's whole history, every step
                     independent of B

  flops        =  2 · P · B

  step_time    ≈  max( bytes_moved / HBM_bandwidth ,  flops / peak_flops )   (+ overhead)
  throughput   =  B / step_time
  intensity    =  flops / bytes_moved       ← compare to the GPU's ridge point (~150-300)
```

Two structural facts fall out immediately, and they explain the entire phase:

1. **Weight bytes are paid once per step regardless of batch size.** That's why batching works (Phase 3) and why *weight-only quantization is a decode optimization*: it divides the dominant term when B is small.
2. **KV bytes scale with `ctx × B`.** So at long context or high concurrency, the KV-cache — not the weights — becomes the thing you're reading. At that point quantizing weights stops helping and you must attack the cache.

### What the model says (A100-80GB: 2.0 TB/s, 312 TFLOP/s FP16, $1.80/hr)

```
model                        ctx  Bmax    B     step    tok/s    bound     AI   KV GB    $/1M
Llama-2-7B  (MHA)           2048    57    1     7.5ms      133   memory      1     1.1   3.768
Llama-2-7B  (MHA)           2048    57   32    24.2ms     1323   memory      9    34.4   0.378
Llama-2-7B  (MHA)           2048    57   57    37.6ms     1516   memory     11    61.2   0.330
Llama-2-7B  (MHA)           8192    14    1     9.1ms      109   memory      1     4.3   4.574
Llama-2-7B  (MHA)           8192    14   14    37.1ms      378   memory      3    60.1   1.324

Llama-3-8B  (GQA)           2048   223    1     8.1ms      123   memory      1     0.3   4.067
Llama-3-8B  (GQA)           2048   223   32    12.3ms     2603   memory     21     8.6   0.192
Llama-3-8B  (GQA)           2048   223  223    37.9ms     5879   memory     47    59.9   0.085
Llama-3-8B  (GQA)           8192    55    1     8.5ms      117   memory      1     1.1   4.268
Llama-3-8B  (GQA)           8192    55   55    37.5ms     1466   memory     12    59.1   0.341

Llama-3-8B  INT4+INT8kv     2048   536    1     2.1ms      484   memory      4     0.1   1.034
Llama-3-8B  INT4+INT8kv     2048   536   32     4.1ms     7716   memory     62     4.3   0.065
Llama-3-8B  INT4+INT8kv     2048   536  536    38.0ms    14116   memory    113    71.9   0.035
Llama-3-8B  INT4+INT8kv     8192   134    1     2.3ms      441   memory      4     0.5   1.134
Llama-3-8B  INT4+INT8kv     8192   134  134    38.0ms     3529   memory     28    71.9   0.142
```

Sit with this table; it is the phase in miniature.

- **Everything is memory-bound.** Even at the largest batch that fits, intensity tops out at 113 against a ridge near 300. **You will not run out of FLOPs. You will run out of bandwidth and capacity.**
- **GQA is worth more than it sounds.** Llama-3-8B's 8 KV heads vs Llama-2-7B's 32 turn 57 concurrent 2k sequences into **223** — and 4× the concurrency is 4× the throughput at the same step time. `$/1M tokens` falls from 0.330 to 0.085.
- **Quantization compounds with concurrency.** INT4 weights + INT8 KV take 8B at 2k context from 5,879 to 14,116 tok/s, and $0.085 → **$0.035 per million tokens**. Not because the math got faster — because there are fewer bytes and therefore room for more sequences.
- **Long context is the tax that eats everything.** The same INT4 model at 8k context does 3,529 tok/s, 4× worse than at 2k, purely because KV bytes scale with context.
- **Batch 1 is a catastrophe you should be able to quote:** 133 tok/s and $3.77 per million tokens for 7B FP16 — **108× the cost** of the same model well-batched and quantized. Anyone benchmarking "tokens/sec at batch 1" is measuring their own configuration mistake.

### The script

Keep it in `labs/phase4/costmodel.py`; you'll re-run it before every experiment in this phase.

```python
"""Predict decode throughput and $/1M tokens BEFORE optimizing anything."""

GPU = dict(name="A100-80GB", hbm_gb=80, bw_tbs=2.0, flops_tfs=312, price_hr=1.80)

def model(name, params_b, layers, kv_heads, head_dim, w_bits=16, kv_bits=16):
    return dict(name=name, params=params_b * 1e9, layers=layers, kv_heads=kv_heads,
                head_dim=head_dim, w_bits=w_bits, kv_bits=kv_bits)

def decode_step(m, batch, ctx, gpu=GPU, overhead_ms=0.0):
    w_bytes  = m["params"] * m["w_bits"] / 8
    kv_tok   = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * m["kv_bits"] / 8
    kv_bytes = kv_tok * ctx * batch                       # every step re-reads the whole cache
    bytes_moved = w_bytes + kv_bytes
    flops = 2 * m["params"] * batch
    t_mem = bytes_moved / (gpu["bw_tbs"] * 1e12)
    t_mat = flops / (gpu["flops_tfs"] * 1e12)
    t = max(t_mem, t_mat) + overhead_ms / 1e3             # roofline: whichever roof binds
    return dict(step_ms=t * 1e3, tok_s=batch / t, bound="memory" if t_mem > t_mat else "compute",
                intensity=flops / bytes_moved, kv_gb=kv_tok * ctx * batch / 1e9,
                cost_per_1m=gpu["price_hr"] / 3600 / (batch / t) * 1e6)

def max_batch(m, ctx, gpu=GPU, reserve_gb=4.0):
    w = m["params"] * m["w_bits"] / 8 / 1e9
    kv_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * m["kv_bits"] / 8
    return max(int((gpu["hbm_gb"] - w - reserve_gb) * 1e9 / (kv_tok * ctx)), 0)
```

**What it deliberately ignores** (say this out loud when you present numbers from it): dequantization overhead, kernel efficiency below peak (assume 60-80% of spec), attention's own FLOPs, prefill entirely, launch overhead at small B, and the fact that real batches are ragged. It's a *ceiling* and a *ranking tool*, not a predictor of your exact tokens/sec. Used that way it is remarkably reliable; used as a promise it will embarrass you.

---

## The selection procedure

Run this in order. Stop when the arithmetic says the next step isn't worth it.

```
  0. Is the batch full?              ── no ──▶ fix scheduling first (Phase 3).
     (mean_batch vs max_num_seqs)              No optimization here beats "actually batch."
              │ yes
              ▼
  1. What dominates bytes_moved?
     weights >> KV  ──▶ WEIGHT QUANTIZATION (lessons 2-3): ÷2 (INT8) or ÷4 (INT4)
     KV >> weights  ──▶ KV WORK (lesson 4): GQA (÷4), KV-quant (÷2), window (÷ctx/W)
              │
              ▼
  2. Is memory *capacity* the limit on batch size?
     yes ──▶ PAGING (lesson 5): reclaims 60-80% wasted by fragmentation/over-reservation
              │
              ▼
  3. Do requests share prefixes (system prompts, few-shot, chat history, agents)?
     yes ──▶ PREFIX CACHING / RadixAttention (lesson 6): prefill cost × (1 − hit rate)
              │
              ▼
  4. Is the SLO per-user latency at LOW batch (interactive, single-user, on-device)?
     yes ──▶ SPECULATIVE DECODING (lesson 7): 1.5-3× on latency, spends spare FLOPs
              │
              ▼
  5. Is the GPU idle between kernels? (sum(kernel time) << wall time)
     yes ──▶ CUDA GRAPHS / torch.compile (lesson 8): removes launch overhead
```

The order is not arbitrary — it's descending expected value per unit of engineering pain, and it front-loads the techniques whose benefit you can *compute in advance*.

---

## What each technique actually divides

| Technique | Divides | Typical factor | Costs you | Helps prefill? |
|---|---|---|---|---|
| Batching (Phase 3) | weight bytes *per token* | up to 10-30× | per-user TPOT, tail latency | no (already compute-bound) |
| **INT8 weights** | weight bytes | 2× | small quality delta, dequant overhead | barely |
| **INT4 weights (GPTQ/AWQ)** | weight bytes | 4× | measurable quality delta, kernel maturity | **no** — prefill is compute-bound |
| **FP8 (H100+)** | weight & activation bytes | 2× | needs hardware support | yes (real FP8 tensor cores) |
| **GQA/MQA** | KV bytes | 4-8× | must be trained that way — a model choice | no |
| **KV quantization** | KV bytes | 2× (INT8), 4× (INT4) | small quality delta at long context | no |
| **Sliding window / eviction** | KV bytes | ctx/W | forgets history — task-dependent correctness risk | no |
| **PagedAttention** | *wasted* KV bytes | 1.5-4× effective concurrency | block-table indirection in the kernel | no |
| **Prefix caching** | prefill FLOPs | 1/(1−hit rate) | cache memory, routing complexity | **yes, enormously** |
| **Speculative decoding** | *number of forward passes* | 1.5-3× at low batch | extra FLOPs, draft model memory, complexity | no |
| **CUDA graphs / compile** | launch overhead | 1.1-2× at small batch | shape rigidity, warmup, capture bugs | some |
| **Distillation / smaller model** | everything | 2-10× | quality — the honest tradeoff nobody likes | yes |

Three consequences worth stating explicitly:

- **Nothing in this table except prefix caching helps TTFT much.** Prefill is compute-bound; INT4 weights must be *dequantized* to compute, so a prefill GEMM sees the same FLOPs. If your problem is TTFT, the answers are prefix caching, chunked prefill scheduling (Phase 3), more compute, or fewer prompt tokens.
- **The KV techniques compound multiplicatively.** GQA (4×) × INT8 KV (2×) = 8× more concurrency at fixed memory, which is 8× more throughput as long as you stay memory-bound. That's the biggest single win available.
- **Speculative decoding is the only one that attacks *sequentiality*.** Everything else moves fewer bytes; it takes fewer steps. That's why it behaves so differently under load (lesson 7).

---

## Prove it: the three numbers every claim needs

An optimization claim without these three is marketing:

1. **Memory** — weights + KV at your context and batch, measured with `torch.cuda.max_memory_allocated()` or the framework's own gauge. Predicted vs measured.
2. **Speed** — TTFT, TPOT and output tok/s **at a stated offered load**, from the Phase 3 harness. Never a single-request number.
3. **Quality** — perplexity on a fixed held-out set *plus* a task metric if you have one. Every quantization method trades quality for bytes; a table without a quality column is hiding the price.

And one meta-number that ties them together: **cost per million tokens** = `GPU $/hour ÷ 3600 ÷ (output tokens/sec) × 1e6`. It is the only metric a business cares about, it collapses all three axes into one, and — as the table above shows — it varies by **100×** across configurations of the *same model on the same GPU*.

---

## Try it (laptop, 5 minutes)

1. Run the cost model for **your** GPU (edit `GPU`) and **your** target model. Record `Bmax`, tok/s at B = 1 / 32 / Bmax, and $/1M at each.
2. Re-run with `w_bits=4`, then with `kv_bits=8`, then both. Write down the predicted factor for each *before* running.
3. Answer with numbers: *at what context length does the KV-cache exceed the weights?* Solve `2·L·H_kv·d·kv_bits/8·ctx·B = P·w_bits/8`. For Llama-3-8B FP16 (kv_tok = 128 KB): weights 16 GB → the crossover is at `ctx·B ≈ 125,000` tokens — i.e. 61 concurrent 2k sequences, or 15 at 8k. **Past that point, quantizing weights is the wrong optimization** and every hour you spend on it is wasted. That single calculation is the most useful thing in this lesson.
4. Now add `overhead_ms=5` (a plausible per-step Python + launch overhead) and re-run at B = 1. Notice that at 2.1 ms/step, INT4's win is *entirely erased* by overhead — which is why lesson 8 exists and why vLLM uses CUDA graphs.

---

## Key takeaways

- **One model, memorized:** `bytes = P·w_bits/8 + kv_tok·ctx·B`, `step = max(bytes/BW, 2PB/FLOPS)`, `tok/s = B/step`. Predict before measuring, always.
- **Decode is memory-bound everywhere** in realistic configurations — intensity 1 at batch 1, still only ~100 at max batch against a ridge of ~300.
- **Weight bytes are per-step-constant; KV bytes scale with `ctx × B`.** Which one dominates decides which optimization is correct, and the crossover is a one-line calculation.
- Measured from the model: batch-1 FP16 7B costs **$3.77/1M tokens**; GQA + INT4 weights + INT8 KV, well batched, costs **$0.035** — a ~100× spread on identical hardware.
- **Order of attack:** fill the batch → quantize the dominant byte source → page the memory → cache shared prefixes → break sequentiality → delete launch overhead.
- **Quantization is a decode optimization**, not a prefill one; prefill is compute-bound and dequantizes anyway. TTFT problems are solved by prefix caching and scheduling.
- **KV wins compound:** GQA × KV-quant × paging can be 8-16× effective concurrency, which is the largest lever in the phase.
- Every claim ships with **memory, speed at a stated load, and quality** — plus `$/1M tokens`, the number that makes the case to anyone who pays for GPUs.

**Next:** [Quantization fundamentals →](02-quantization-fundamentals.md) — what an INT4 weight actually is, where the scale lives, and how to compute the error before you trust a benchmark.
