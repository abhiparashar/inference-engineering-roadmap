# 4 — KV-Cache Optimization

> **You'll be able to say:** "KV bytes per token = `2 · layers · kv_heads · head_dim · bits/8`. For Llama-2-7B that's 512 KB/token, so an 80 GB card holds 59 concurrent 2k sequences — and **3** at 32k. GQA divides it by 4 (230 sequences), INT8 KV by 2 again (461). I measured what INT8 costs: 0.29% attention-output error per-channel, versus 41% for naive INT4 per-tensor — granularity matters even more for the cache than for weights, and K needs per-channel scales while V is fine per-token."

Once weights are quantized, the cache is the problem. This lesson turns every KV technique into the only unit that matters: **concurrent sequences**, which is throughput.

---

## The arithmetic, and what it does to concurrency

```
  kv_bytes_per_token = 2 (K and V) × layers × kv_heads × head_dim × bits/8
  concurrent_seqs    = (HBM − weights − activations) ÷ (kv_bytes_per_token × context)
```

Computed for real models (2 GB reserved for activations/fragmentation):

```
model                             KV/token  bits    ctx  seqs@80GB  seqs@24GB
Llama-2-7B   MHA  32 kv-heads         512K    16   2048         59          7
Llama-2-7B   MHA  32 kv-heads         512K    16   8192         14          1
Llama-2-7B   MHA  32 kv-heads         512K    16  32768          3          0
Llama-2-7B   MHA  32 kv-heads         256K     8   2048        119         14
Llama-2-7B   MHA  32 kv-heads         256K     8   8192         29          3

Llama-3-8B   GQA   8 kv-heads         128K    16   2048        230         22
Llama-3-8B   GQA   8 kv-heads         128K    16   8192         57          5
Llama-3-8B   GQA   8 kv-heads         128K    16  32768         14          1
Llama-3-8B   GQA   8 kv-heads          64K     8   2048        461         44
Llama-3-8B   GQA   8 kv-heads          64K     8   8192        115         11
Llama-3-8B   GQA   8 kv-heads          64K     8  32768         28          2

Llama-3-70B  GQA   8 kv-heads         320K    16   2048          0          0
```

Five things to take from this table:

1. **Context length is brutal and linear.** Same model, same GPU: 59 sequences at 2k, 3 at 32k. A "128k context" product claim is a *capacity* claim about your fleet, not a model capability claim.
2. **GQA is the single biggest KV lever** — 512 KB → 128 KB per token, 59 → 230 concurrent sequences. It's an architecture decision made at training time, which is why it's now universal (Llama-3, Mistral, Qwen, Gemma). MQA (1 KV head) is the extreme case.
3. **KV quantization stacks multiplicatively with it**: GQA × INT8 = 8× more sequences than MHA/FP16.
4. **A 24 GB consumer card is a different planet.** Llama-3-8B FP16 weights (16 GB) leave room for 22 sequences at 2k and **1** at 32k. This is why 4-bit weights are effectively mandatory on consumer hardware — not for speed, for making room for the cache.
5. **70B doesn't fit at all** (140 GB of weights on an 80 GB card): the zero row is real, and it's the reason Phase 6 exists. Tensor parallelism isn't an optimization there, it's the entry ticket.

**The KV/weights crossover** is the calculation to keep in your head: KV exceeds weights when `kv_per_token × ctx × B > weight_bytes`. Llama-3-8B FP16: `128 KB × ctx × B > 16 GB` → **ctx·B > 125,000 tokens** (61 sequences at 2k, 15 at 8k). Past that point, further weight quantization is close to pointless and every additional byte you save must come from the cache.

---

## Technique 1: fewer KV heads (GQA/MQA)

```
  MHA: every query head has its own K,V          32 heads → 32 KV heads   (baseline)
  GQA: groups of query heads share one K,V       32 heads →  8 KV heads   (4x smaller)
  MQA: all query heads share ONE K,V             32 heads →  1 KV head    (32x smaller)
```

Only the KV projections shrink; query heads and attention math are unchanged, so quality loss is small (and can be recovered with a short uptraining pass — that's how GQA checkpoints were originally produced from MHA ones). MQA is more aggressive and shows quality degradation on some tasks, which is why GQA-8 became the industry default.

**You cannot apply this at serving time** — it's baked into the checkpoint. What you *can* do is choose models by it. When comparing two 7-8B models for a long-context product, `kv_heads` is often a bigger deal than benchmark scores: 4× the concurrency is 4× the throughput and a quarter of the cost.

Related, and worth knowing by name: **MLA (Multi-head Latent Attention)** in DeepSeek-V2/V3 compresses KV into a low-rank latent vector, reporting an order-of-magnitude smaller cache than MHA. Same goal, different mechanism.

---

## Technique 2: quantize the cache

The cache is just tensors, so quantize it — but the statistics differ from weights, and the measurements show it. Real GPT-2 block-0 K/V and real queries, judged by **attention output** error:

```
K stats: absmax=10.50 per-channel absmax spread=3.7x   V absmax=3.58

cache dtype / granularity            attn output rel err  bytes vs FP16
INT8 per-tensor                                  1.4282%          0.50x
INT8 per-token                                   0.4271%          0.50x
INT8 per-channel                                 0.2907%          0.50x
INT4 per-tensor                                 41.0318%          0.25x
INT4 per-token                                  10.9492%          0.25x
INT4 per-channel                                  6.2788%         0.25x
INT4 K (per-channel) + FP16 V                     4.7829%         0.62x
FP16 K + INT4 V (per-token)                       9.7510%         0.62x
```

Read carefully — there are four distinct lessons here:

1. **INT8 KV is nearly free** (0.29-0.43% output error) and buys 2× concurrency. This is why `--kv-cache-dtype fp8` (or int8) is one of the highest value-to-risk flags in vLLM. On H100, FP8 KV is the same idea with hardware support and even better numerics.
2. **INT4 KV is a real tradeoff, not a free lunch**: 6.3% output error at the best granularity, and 41% if you naively use one scale. Research systems (KIVI, KVQuant) make INT4/INT2 work with per-channel K, per-token V, outlier retention and residual buffers — exactly the structure the last two rows hint at.
3. **K and V want different granularities.** K has channel-wise outliers (spread 3.7× here, far larger in big models), so **K wants per-channel scales**; V is well behaved and does fine per-token. The asymmetric split shown in the last two rows makes that concrete: quantizing K alone to INT4 costs 4.8%, quantizing V alone costs 9.8% *at the wrong granularity* — the granularity choice matters more than which tensor you quantize.
4. **Per-tensor is never the answer** for a cache. Same 2-4× storage win, 5-140× the error.

The catch nobody mentions: **quantized KV must be dequantized inside the attention kernel**, so you need a kernel that supports it (FlashAttention/PagedAttention variants do). Without kernel support, "KV quantization" means dequantizing to FP16 before attention — you save capacity but not bandwidth, which is half the point.

---

## Technique 3: store less history

If bytes-per-token can't fall further, reduce the number of tokens.

| Method | Idea | Risk |
|---|---|---|
| **Sliding window** (Mistral, Gemma-2) | attend to the last `W` tokens only; cache is capped at `W` | forgets beyond the window; needs the model trained for it |
| **StreamingLLM / attention sinks** | keep the first few tokens (sinks) + a recent window | near-free for streaming chat; not for long-document recall |
| **H2O / SnapKV / heavy hitters** | evict tokens with historically low attention mass | task-dependent; a dropped token can't come back |
| **Prompt compression** (LLMLingua etc.) | shorten the prompt itself before it becomes cache | quality depends entirely on the compressor |
| **CPU/NVMe offload** | move cold KV to host memory, fetch on demand | PCIe ~25 GB/s vs HBM ~2,000 GB/s — 80× slower; only for cold or huge contexts |

The pattern: cheap and safe (windows on models trained with them, sinks for chat), versus quality-risky (eviction, compression). Try them in that order, and **always test long-context recall specifically** — a needle-in-a-haystack probe is the standard, and eviction schemes fail it in ways perplexity never shows.

Offloading deserves one number so you don't reach for it casually: moving 128 KB/token over PCIe at 25 GB/s is 5.1 µs per token per sequence — for a 2k-token sequence that's **10 ms just to fetch its cache**, versus a ~0.13 ms read from HBM. Offloading is a capacity tool for cold sequences, not a bandwidth tool.

---

## Technique 4: don't store the same thing twice

Two requests that share a system prompt should share its KV. That's **prefix caching**, it needs paged storage to be practical, and it's [lesson 6](06-prefix-caching-and-radix-attention.md). Mentioned here because on the shopping list of "how do I fit more sequences," deduplication is often worth more than every compression trick combined — a 2,000-token system prompt shared by 100 concurrent requests is 200,000 tokens of cache collapsed to 2,000.

---

## Putting it together: a worked capacity plan

*Requirement: Llama-3-8B, 8k contexts, target 100 concurrent sequences, one A100-80GB.*

```
  baseline FP16 weights + FP16 KV :  (80 − 16 − 2) GB ÷ (128 KB × 8192)   =  57 seqs   ✗
  + INT8 KV                       :  (80 − 16 − 2) ÷ (64 KB × 8192)       = 115 seqs   ✓
  + INT4 weights instead          :  (80 −  4 − 2) ÷ (64 KB × 8192)       = 141 seqs   ✓✓
  + paging (recover ~30-60% waste from fragmentation/over-reservation, lesson 5)
                                  :  ~141 usable rather than ~60-90 usable in practice
  + prefix caching if prompts share a 2k system prompt (lesson 6): another large multiple
```

Note the ordering of value: **KV quantization (2×) > weight quantization (adds 12 GB ≈ 1.2×) at this context length** — precisely what the crossover calculation predicted, because at 8k×100 tokens the cache dwarfs the weights. On the same hardware with 512-token contexts, the ranking reverses. **Compute the crossover first; it tells you which lesson to read.**

---

## Try it

1. Reproduce the capacity table for your GPU and target models. Add the model you actually plan to serve, and mark the context length at which you fall below your concurrency target.
2. Reproduce the KV-quantization error table. Then add **per-channel K + per-token V** as one combined scheme (the KIVI layout) and confirm it beats both uniform choices.
3. Extend it end to end: quantize the cache inside your Phase-3 engine (store `int8` tensors + scales, dequantize in `decode_step`) and measure perplexity and tokens/sec. Predict the concurrency gain first, then check `mean_batch` in `/metrics`.
4. Measure the **channel-outlier spread** of K across layers (`per-channel absmax ÷ median`). It grows with depth and model size; that plot is why per-channel K scaling exists.
5. Implement a **sliding window** in your engine (drop KV older than `W`) and measure the memory saving alongside a needle-in-a-haystack recall test. Quantify what you traded.

---

## Key takeaways

- **`kv_bytes/token = 2 · layers · kv_heads · head_dim · bits/8`**, and `concurrency = free_memory ÷ (that × context)`. Every technique in this lesson is a division in that denominator.
- Measured: Llama-2-7B MHA = **512 KB/token** → 59 sequences at 2k on 80 GB, **3** at 32k. Llama-3-8B GQA = 128 KB/token → 230 and 14.
- **GQA (4×) is a model-selection decision** you make before serving; MLA goes further; MQA is the aggressive limit.
- **INT8 KV costs ~0.3-0.4% attention error and buys 2×** — one of the best risk-adjusted flags in serving. **INT4 KV costs ~6%** at good granularity and needs research-grade tricks to be safe.
- **Granularity dominates**: INT4 per-tensor 41% error vs per-channel 6.3%. **K wants per-channel scales (channel outliers), V is fine per-token.**
- Quantized KV only saves *bandwidth* if the attention kernel reads it quantized; otherwise you've only saved capacity.
- **Storing less history** (windows, sinks, eviction) is cheap or risky depending on the method; always validate with a long-context recall probe, not perplexity.
- **Offloading is 80× slower than HBM** (PCIe 25 GB/s): a capacity tool for cold sequences, never a bandwidth fix.
- Compute the **KV-vs-weights crossover** (`kv_per_token × ctx × B` vs `weight_bytes`) before choosing an optimization — it decides whether lesson 3 or lesson 4 is your afternoon.

**Next:** [PagedAttention →](05-paged-attention.md) — even with the cache as small as it can be, the naive layout wastes most of it. Time to steal an idea from operating systems.
