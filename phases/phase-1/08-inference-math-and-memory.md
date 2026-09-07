# 8 — Inference Math & Memory (The Payoff)

> **You'll be able to say:** "Give me a model config and a GPU spec sheet and I'll tell you, on the back of an envelope: how much memory the weights take, how much the KV-cache takes per token and per user, how many users fit, the floor on time-per-token, and *whether you're compute- or memory-bound* — with the arithmetic intensity to prove it."

This is the lesson that turns Phase 1 from vocabulary into engineering. Everything here is arithmetic you can do in your head or in five lines of Python. Do the calculations yourself as you read; the numbers are the point.

---

## The five formulas

Memorize these. They cover ~90% of real capacity-planning conversations.

```
1.  Weight memory      = num_params × bytes_per_param
2.  KV-cache per token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_param
3.  FLOPs per token    ≈ 2 × num_params                     (decode, dense model)
4.  Decode time floor  = bytes_read_per_step ÷ memory_bandwidth
5.  Arithmetic intensity = FLOPs ÷ bytes_moved              (compare to hardware ridge point)
```

The "2" in formula 2 is K and V. The "2" in formula 3 is one multiply + one add per weight. Formula 4 is the whole reason this field exists.

**Rules of thumb** for FP16: a model needs about `2 × params_in_billions` GB of weights (7B → ~14 GB; 70B → ~140 GB). INT8 halves it, INT4 quarters it.

---

## The running example

Llama-2-7B-class model in FP16 on one A100-80GB.

| Model | | Hardware (A100-80GB SXM) | |
|---|---|---|---|
| Parameters | 6.7 B | HBM capacity | 80 GB |
| Layers | 32 | HBM bandwidth | ~2,039 GB/s |
| Heads (Q) | 32 | FP16 dense peak | ~312 TFLOP/s |
| KV heads | 32 (MHA) | **Ridge point** | 312e12 ÷ 2.039e12 ≈ **153 FLOP/byte** |
| head_dim | 128 | | |
| dtype | FP16 (2 bytes) | | |

The **ridge point** (Phase 2 calls it the roofline knee) is the arithmetic intensity where the hardware stops being memory-limited and starts being compute-limited. Below 153 FLOPs per byte on this GPU, you are memory-bound. Remember that number for the next two sections.

---

## Step 1 — Weight memory

```
6.7e9 params × 2 bytes = 13.4 GB
```

Add ~1-2 GB for CUDA context, activations, and workspace. Call it ~15 GB resident before a single user connects. On an 80 GB card that leaves **~65 GB** — and that remaining space is what you sell, because it holds KV-cache.

## Step 2 — KV-cache size

```
per token = 2 × 32 layers × 32 kv_heads × 128 head_dim × 2 bytes
          = 524,288 bytes = 0.5 MB per token, per sequence

2,048-token conversation  → 2048 × 0.5 MB ≈ 1.0 GB per user
65 GB free ÷ 1.0 GB       → ~65 concurrent users, hard stop
```

**That is your concurrency limit** — not FLOPs, not the model size. Now change one architectural detail: use **GQA** with 8 KV heads instead of 32 (this is exactly what Llama-3-8B does):

```
per token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 0.125 MB   ← 4× smaller
2,048-token conversation → 0.25 GB per user
65 GB free               → ~260 concurrent users              ← 4× the customers
```

One config field, 4× the serving capacity. That is why every modern model ships with GQA, and why "how big is the KV-cache" is the first question to ask about any deployment.

## Step 3 — FLOPs, and the time floor for one token

Decode, batch 1, one token:

```
compute needed = 2 × 6.7e9         = 13.4 GFLOP
bytes to read  = all 13.4 GB of weights + the KV-cache for this sequence

time if compute-limited = 13.4e9  ÷ 312e12 =  0.043 ms
time if memory-limited  = 13.4e9  ÷ 2.039e12 s⁻¹·bytes = 6.6 ms
                                                            ▲
                                        154× slower — memory wins, and it isn't close
```

**Time per output token ≥ 6.6 ms → ≤ ~151 tokens/s for a single sequence, on a $15,000 GPU.** No kernel, no compiler, no amount of tensor cores changes that floor; only reading fewer bytes does. (Good engines reach roughly 60-80% of this bound; naive PyTorch loops land far below it, because per-step kernel-launch overhead starts to matter when the real work is 6 ms.)

Utilization check: 13.4 GFLOP in 6.6 ms = **2.0 TFLOP/s achieved out of 312 → 0.65% of the GPU's math capability.** The card is idle ~99% of the time, waiting on memory. Everything about serving economics follows from this single embarrassing number.

## Step 4 — The proof: arithmetic intensity

```
decode, batch 1:  13.4e9 FLOPs ÷ 13.4e9 bytes  =  1 FLOP/byte
ridge point:                                     153 FLOP/byte
1 ≪ 153  ⟹  MEMORY-BOUND, by ~150×
```

Generalize it. With batch size `B`, you read the weights **once** and do `B` tokens of math with them:

```
intensity ≈ (2 × P × B) ÷ (2 × P) = B FLOPs per byte
```

Arithmetic intensity during decode ≈ **the batch size**. So the crossover to compute-bound is at `B ≈ 153` on an A100 (in practice lower — 64-128 — because attention and KV traffic add bytes and overheads eat headroom). That is the *entire* quantitative justification for continuous batching in Phase 3.

Contrast prefill with a 2,048-token prompt:

```
FLOPs = 2 × 6.7e9 × 2048 = 27.4 TFLOP    bytes ≈ same 13.4 GB of weights
intensity ≈ 2048 FLOP/byte  ≫ 153        ⟹  COMPUTE-BOUND
TTFT floor ≈ 27.4e12 ÷ 312e12 ≈ 88 ms at peak; ~150-250 ms at realistic 40-60% utilization
```

Same model, same GPU, opposite bottleneck — [lesson 6](06-prefill-vs-decode.md) proven with numbers.

---

## Step 5 — What batching actually buys (and what it costs)

Batch 64, each sequence at 2,048 tokens of context. Bytes read **per decode step**:

```
weights          : 13.4 GB                       (read once, shared by all 64)
KV-cache         : 64 × 2048 × 0.5 MB = 65.5 GB  (each sequence reads its OWN cache)
total            : ~79 GB per step
step time        : 79 ÷ 2039 GB/s ≈ 38.7 ms  →  produces 64 tokens
throughput       : 64 ÷ 0.0387 s ≈ 1,650 tokens/s      (vs 151 at batch 1 — 11×)
per-user TPOT    : 38.7 ms  (worse than 6.6 ms, but each user still gets ~26 tok/s — faster than reading)
```

Two lessons in one table. **Throughput scales enormously with batching** — this is why serving is profitable at all. And notice which term blew up: at batch 64 with long contexts, **the KV-cache reads (65.5 GB) are 5× the weight reads (13.4 GB)**. Attention, not the MLP, becomes the bandwidth hog.

That flip is the reason for a whole family of Phase 2/4 techniques: **FlashAttention / FlashDecoding** (don't materialize the N×N grid, stream the KV), **GQA** (4-8× less KV to read *and* store), **KV-cache quantization** (FP8/INT8 halves those bytes), and **PagedAttention** (stop wasting capacity on fragmentation so the batch can be bigger in the first place).

Also note the latency/throughput trade laid bare: batch 1 gives the best TPOT (6.6 ms) and terrible economics; batch 64 gives 11× the throughput and 6× worse TPOT. Choosing a point on that curve *is* the job in Phase 3.

---

## The calculator (write this once, keep it forever)

```python
def inference_math(params_b, layers, kv_heads, head_dim, bytes_per_param=2,
                   gpu_gb=80, bw_gbs=2039, peak_tflops=312):
    weights_gb = params_b * 1e9 * bytes_per_param / 1e9
    kv_per_token_mb = 2 * layers * kv_heads * head_dim * bytes_per_param / 1e6
    free_gb = gpu_gb - weights_gb - 2                       # 2 GB overhead
    flops_per_token = 2 * params_b * 1e9
    decode_ms = weights_gb / bw_gbs * 1000                  # batch-1 floor
    return {
        "weights_GB": round(weights_gb, 1),
        "kv_MB_per_token": round(kv_per_token_mb, 3),
        "users_at_2k_ctx": int(free_gb * 1000 / (kv_per_token_mb * 2048)),
        "decode_ms_floor": round(decode_ms, 2),
        "max_tok_per_s_batch1": round(1000 / decode_ms),
        "intensity_batch1": round(flops_per_token / (weights_gb * 1e9), 2),
        "ridge_point": round(peak_tflops * 1e12 / (bw_gbs * 1e9)),
    }

print(inference_math(6.7, 32, 32, 128))   # Llama-2-7B  (MHA)
print(inference_math(8.0, 32,  8, 128))   # Llama-3-8B  (GQA) — watch the user count jump
print(inference_math(70.0, 80, 8, 128, gpu_gb=80))   # 70B: weights alone don't fit → Phase 6
```

Run it on the three configs. The first prints `weights_GB 13.4, kv_MB_per_token 0.524, decode_ms_floor 6.57, intensity_batch1 1.0, ridge_point 153` — the numbers you just worked through by hand. The second (GQA) drops KV to 0.131 MB/token and quadruples the user count. The third returns a *negative* user count, which is the arithmetic bluntly telling you that 140 GB of FP16 weights don't fit on an 80 GB card — forcing **quantization** (Phase 4) or **tensor parallelism across GPUs** (Phase 6). You just derived the need for both from a spec sheet.

---

## Two utilization metrics you'll live with

- **MFU (Model FLOPs Utilization)** = achieved FLOP/s ÷ peak FLOP/s. The right scorecard for **prefill/training**. Decode's MFU is ~1% and that is *not* a bug.
- **MBU (Model Bandwidth Utilization)** = achieved bytes/s ÷ peak bandwidth. The right scorecard for **decode**. A well-tuned engine hits 60-80% MBU; if yours is at 20%, you have overhead (kernel launches, Python, synchronization), not a hardware problem.

Reporting decode performance as MFU is a classic beginner mistake in benchmark writeups. Use MBU for decode, MFU for prefill.

---

## Sanity-check your intuition

Answer these before moving on (they're the [lesson 10](10-exercises-and-artifacts.md) exercises in miniature):

1. You quantize weights FP16 → INT8. What happens to the batch-1 decode time floor, and why? *(Bytes halve → floor halves → ~2× faster. This is the Phase 0 keystone question, now provable.)*
2. You double the context length. What happens to KV-cache memory, to max concurrency, and to per-step KV bytes read? *(All linear: 2×, ½×, 2×.)*
3. Your GPU shows 3% utilization during decode and someone calls it "underutilized." What do you say? *(It's bandwidth-saturated, not idle — check MBU, not MFU; the fix is a bigger batch or fewer bytes, not more FLOPs.)*

---

## Key takeaways

- **Weights** = `params × bytes`; FP16 ≈ `2 × params_in_billions` GB.
- **KV-cache per token** = `2 × layers × kv_heads × head_dim × bytes`; multiply by context length and batch. It, not the model size, sets **concurrency**.
- **Decode FLOPs per token** ≈ `2 × params`; **decode time floor** = `bytes read ÷ bandwidth`.
- On an A100, batch-1 decode of a 7B model has intensity **≈1 FLOP/byte** against a ridge point of **153** → memory-bound by ~150×, using **<1%** of the GPU's math. Prefill of a 2k prompt is **≈2048 FLOP/byte** → compute-bound.
- Decode arithmetic intensity **≈ batch size** → batching is the only way to reach the ridge point. That's the quantitative case for continuous batching.
- At large batch × long context, **KV-cache reads exceed weight reads** → attention becomes the bandwidth bottleneck → FlashDecoding, GQA, KV quantization, PagedAttention.
- Score decode with **MBU**, prefill with **MFU**.

**Next:** [Build: GPT-2 forward pass from scratch →](09-build-gpt2-from-scratch.md) — implement everything from lessons 1-8 in NumPy, with a real KV-cache, and match HuggingFace token for token.
