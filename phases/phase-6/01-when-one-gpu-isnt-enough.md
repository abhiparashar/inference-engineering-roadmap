# 1 — When One GPU Isn't Enough

> **You'll be able to say:** "There are exactly four reasons to use more than one GPU — weights don't fit, the KV budget is too small, TPOT is too high, or QPS is too high — and each has a different answer. I can compute the minimum GPU count from `weights + KV + overhead`, compute the aggregate KV budget a layout yields as `TP × usable_per_gpu − weights`, turn that into concurrent sequences, and check divisibility constraints before anyone rents anything. And I know that replicas — not tensor parallelism — are the answer to three of those four reasons."

Every distributed-inference decision starts as an arithmetic problem, and it is almost always the same arithmetic you did in [Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md) and [Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md), now with a GPU count as the unknown.

---

## The four reasons, and their different answers

```
 REASON                        SYMPTOM                          CORRECT FIRST MOVE
 ─────────────────────────────────────────────────────────────────────────────────────────
 1. Weights don't fit       "CUDA out of memory" at load     quantize (Phase 4), then TP
                            time, before any request          (then PP, if cross-node)
 2. KV budget too small     loads fine; concurrency caps at   TP (aggregate KV grows),
                            5-10 seqs; constant preemption    or shrink KV/token
 3. TPOT too high           single-stream tokens/sec too      TP (sublinear), quantize,
                            slow for the product              speculative decoding
 4. QPS too high            queue depth grows without bound   REPLICAS (data parallel)
                            ([Phase 3 lesson 5])
```

Reason 4 is the common case in production, and its answer has nothing to do with model parallelism. Replicate the whole engine, put a router in front ([lesson 7](07-prefix-aware-routing.md)), scale the replica count ([lesson 8](08-autoscaling-gpu-fleets.md)). Throughput scales near-linearly, each replica is an independent failure domain, and rollouts are trivial. **Reach for TP/PP only for reasons 1-3.** People invert this constantly: they run TP=4 on a model that fits in one GPU, get worse throughput-per-dollar than 4 replicas would give, and conclude that distribution is hard.

---

## The per-GPU memory ledger

A GPU's memory is spent, in this order:

```
 ┌─────────────────────────────────────────────────────────────── 80 GB (H100) ──┐
 │ CUDA context + framework  0.5-1.5 GB   allocator, cuBLAS/cuDNN workspaces,   │
 │                                        NCCL buffers (grows with TP)           │
 ├───────────────────────────────────────────────────────────────────────────────┤
 │ WEIGHTS  W / (TP × PP)                 the only term parallelism shrinks       │
 ├───────────────────────────────────────────────────────────────────────────────┤
 │ ACTIVATIONS / peak forward workspace   ∝ max_num_batched_tokens × hidden;     │
 │                                        typically 1-4 GB, NOT sharded by TP    │
 ├───────────────────────────────────────────────────────────────────────────────┤
 │ KV-CACHE  ← everything left over       this is what you are actually buying    │
 └───────────────────────────────────────────────────────────────────────────────┘
   engines don't let you use all 80: `--gpu-memory-utilization 0.90` (vLLM) caps
   the total, and the remainder is deliberate headroom against fragmentation
```

Define, per GPU:

```
usable   = capacity × utilization − context − activation_workspace     (≈ 70 GB on an 80 GB H100)
kv_local = usable − W/(TP × PP)
```

And then the formula that matters, because it's the one people never write down:

> **Aggregate KV budget = `(TP × PP) × usable − W`.**
>
> Every GPU you add to a model-parallel group contributes its *entire* usable memory to the shared KV pool, while the weight cost is paid once across the group. Going from TP=4 to TP=8 on a 70B model doesn't add 4 × 17.5 GB of KV space — it adds 4 × 70 GB.

That superlinearity is why "it fits" and "it serves" are different questions, and why the minimum GPU count for a load is often one step above the minimum for a fit.

### KV bytes per token (the other half)

From [Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md):

```
kv_per_token = 2 × layers × kv_heads × head_dim × bytes_per_element
```

| Model | layers | kv_heads | head_dim | KV/token @ FP16 | @ FP8 |
|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 128 KiB | 64 KiB |
| Llama-3-70B | 80 | 8 | 128 | 320 KiB | 160 KiB |
| Llama-3.1-405B | 126 | 8 | 128 | 504 KiB | 252 KiB |
| Mistral-7B (SWA 4k) | 32 | 8 | 128 | 128 KiB (capped at 4k/seq) | 64 KiB |

TP shards the KV-cache along the `kv_heads` dimension, so per-GPU KV per token is `kv_per_token / TP`; PP shards it along `layers`. Either way the *aggregate* is what the formula above gives you.

---

## Worked sizing table (80 GB GPUs, `usable ≈ 70 GB`, FP16 KV)

| Model + weight dtype | W | Min GPUs to *fit* | Layout | Aggregate KV | KV tokens | Concurrent seqs @ 4k ctx |
|---|---|---|---|---|---|---|
| 8B FP16 | 16 GB | 1 | TP1 | 54 GB | ~442k | ~108 |
| 8B FP8 | 8 GB | 1 | TP1 | 62 GB | ~508k | ~124 |
| 70B FP16 | 140 GB | 3 → use 4 | TP4 | 140 GB | ~459k | ~112 |
| 70B FP16 | 140 GB | — | TP8 | 420 GB | ~1.37M | ~336 |
| 70B FP8 | 70 GB | 1 → infeasible, use 2 | TP2 | 70 GB | ~229k | ~56 |
| 70B FP8 | 70 GB | — | TP4 | 210 GB | ~688k | ~168 |
| 405B FP16 | 810 GB | 12 → use 16 | TP8×PP2 | 310 GB | ~645k | ~157 |
| 405B FP8 | 405 GB | 6 → use 8 | TP8 | 155 GB | ~322k | ~78 |

Read four things off this table:

1. **Quantization is a parallelism decision.** 70B in FP8 fits in 2 GPUs instead of 4 — it is often cheaper and *always* operationally simpler to halve the weights ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)) than to double the TP degree. Check that a fast kernel exists on your GPU generation first; a format that loads but dequantizes in a slow path buys you memory and costs you the tokens/sec you were trying to buy.
2. **"Fits" ≠ "serves."** 70B FP16 on TP4 leaves 140 GB of KV — that is only ~112 concurrent 4k sequences, and continuous batching will start preempting long before you saturate the GPUs' FLOPs. TP8 quadruples concurrency for 2× the GPUs.
3. **Rounding is forced by divisibility, not by taste.** 70B FP16 "fits" in 2.9 GPUs; you use 4 because TP degrees are effectively powers of two (see constraints below).
4. **FP8 KV doubles every token count in the table.** At long context it is the highest-leverage single change you can make, and it competes directly with adding GPUs.

---

## The latency floor, and what parallelism does to it

Decode is memory-bandwidth-bound ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)): each step reads every weight once. So per token:

```
t_step  ≥  W / (TP × BW_effective)          # PP does NOT appear: stages run in series
```

`BW_effective ≈ 0.7-0.8 × spec` for HBM. For 70B FP16 on H100 (3.35 TB/s spec, ~2.7 TB/s effective):

| Layout | Per-GPU weight bytes/step | Bandwidth floor | Floor tokens/sec/seq | Typical measured |
|---|---|---|---|---|
| TP4 | 35 GB | ~13 ms | ~77 | ~25-40 |
| TP8 | 17.5 GB | ~6.5 ms | ~154 | ~40-60 |
| TP8 + PP2 (16 GPUs) | 17.5 GB/stage, 2 stages serial | ~13 ms | ~77 | lower than TP8 |

*(Measured columns are order-of-magnitude figures from public vLLM benchmarks on H100-class hardware; treat them as "2-4× the floor," which is the real lesson.)*

Three consequences:

- **TP is the only lever that lowers the per-token bandwidth floor**, because it is the only one that divides the bytes each GPU must read *per step*. That's why latency-critical deployments over-provision TP relative to what fitting requires.
- **TP's speedup is sublinear**, because the collectives don't shrink with rank count ([lesson 3](03-tensor-parallelism.md)). Doubling TP typically buys 1.6-1.8×, not 2×, and the gap widens with the layer count.
- **PP never improves single-token latency** — it adds serial hops. It buys capacity and throughput ([lesson 4](04-pipeline-parallelism.md)).

---

## Divisibility: the constraints that reject your config at startup

| Constraint | Why | What breaks if violated |
|---|---|---|
| `num_attention_heads % TP == 0` | Q heads are split across ranks | hard error at load |
| `num_kv_heads % TP == 0` | K/V heads are split across ranks | engines either error or *replicate* KV heads (vLLM duplicates when `TP > num_kv_heads`) — memory saving disappears |
| `intermediate_size % TP == 0` | MLP columns are split | hard error at load |
| `vocab_size` padded to a multiple of TP | vocab-parallel embedding/LM head | engines pad silently; changes logits shape |
| quant group size divides the per-rank `K` | group-wise scales must not straddle a shard | "weight shape not divisible" errors with GPTQ/AWQ at high TP |
| `num_layers` divisible-ish by PP | stage balance | works, but the fat stage sets throughput |

**Llama-3-70B has 8 KV heads.** That means TP ∈ {1, 2, 4, 8} is clean, and TP=16 forces KV-head replication — you pay 2× the KV memory per token for the privilege of more GPUs. This single number decides more real deployments than any benchmark.

---

## The procedure

```
  1. kv_per_token = 2 · layers · kv_heads · head_dim · bytes     ← Phase 4 lesson 4
  2. target concurrency C and context L from the traffic shape    ← Phase 3 lesson 5
     required KV = C · L · kv_per_token · (1 + slack≈0.2)
  3. W = params · bytes_per_param                                ← pick dtype FIRST
  4. N_min = smallest (TP·PP) with  N · usable − W ≥ required KV
  5. round N up to satisfy the divisibility table
  6. if N ≤ GPUs per NVLink node:   TP = N, PP = 1
     else:                          TP = node size, PP = N / TP  ← lesson 2 says why
  7. check the latency floor W/(TP · BW) against the TPOT SLO;
     if it fails, raise TP (or quantize, or speculate) — not PP
  8. replicas DP = ceil(offered QPS / measured per-replica QPS)  ← lesson 8
  9. total GPUs = DP × TP × PP
```

Worked end-to-end: a 70B FP16 chat product, 200 concurrent sessions averaging 3k tokens, TPOT SLO 40 ms, offered 12 QPS.

```
kv_per_token = 320 KiB
required KV  = 200 · 3000 · 320 KiB · 1.2 ≈ 220 GB
W            = 140 GB
N · 70 − 140 ≥ 220  →  N ≥ 5.15  →  N = 8 (divisibility: TP ∈ {1,2,4,8})
TP8 fits in one 8-GPU NVLink node → PP = 1
latency floor 17.5 GB / 2.7 TB/s ≈ 6.5 ms ≪ 40 ms SLO ✓ (with ~6× headroom for the
   collectives, attention, sampling and scheduling that the floor ignores)
per-replica throughput: measure it (Phase 3 lesson 7) — say 6 QPS at SLO
DP = ceil(12/6) = 2  →  16 GPUs total, as 2 × (TP8) replicas
```

Now do the alternative and compare, because this is the comparison that gets skipped: **70B FP8, TP4, 4 replicas = 16 GPUs.** Same GPU count, 4 failure domains instead of 2, 210 GB KV per replica (840 GB aggregate vs 840 GB — a wash), higher per-token latency floor (13 ms, still inside SLO), and a quality risk from FP8 that you must measure. That's a real engineering decision with a defensible answer in either direction — which is exactly what an interviewer is listening for.

---

## What distribution costs (the part that isn't in the arithmetic)

- **Blast radius.** A TP=8 replica is one fate-sharing unit: any GPU fault, ECC error or NCCL timeout kills all 8 ([lesson 9](09-multi-node-operations.md)). 8 replicas of 1 GPU lose 12.5% of capacity on the same fault.
- **Startup and rollout.** Load time scales with the *slowest* rank, and every rank must come up before the replica serves. Cold start goes from ~1 min to several minutes — the fact that drives lesson 8's pre-warming.
- **Scheduling granularity.** Kubernetes must place 8 GPUs on one node with the right topology; fragmented clusters can't schedule your pod at all even with free GPUs ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)).
- **Debuggability.** A hang is now a distributed hang. `py-spy dump` on one process tells you nothing; you need per-rank stacks and NCCL logs.
- **Efficiency.** Per-GPU tokens/sec falls monotonically with TP degree. Report throughput *per GPU* in every comparison you make this phase, or you will make TP look free.

---

## Do this now (30 minutes, no GPU required)

1. **Write the sizing script.** Inputs: HF config (`num_hidden_layers`, `num_key_value_heads`, `head_dim`/`hidden_size`, `intermediate_size`, `vocab_size`), weight dtype, KV dtype, GPU capacity, utilization, target concurrency and context. Outputs: `W`, `kv_per_token`, minimum `TP × PP`, aggregate KV, concurrent sequences, latency floor, and every divisibility check. You will use it in every remaining lesson, and it is a legitimately good portfolio artifact on its own.
2. **Run it against three real configs** — `meta-llama/Meta-Llama-3-8B`, `-70B`, and one MoE (`Mixtral-8x7B`: note that `W` counts *all* experts, ~47B params, while per-token FLOPs count only 2 of 8 — the discrepancy is [lesson 5](05-hybrid-layouts-and-moe.md)).
3. **Predict, then check reality.** For 70B on 8×80 GB, predict vLLM's reported `# GPU blocks` (blocks × block_size × kv_per_token should land near your aggregate KV). If you have access to a node, run it and compare; if not, find the number in a public vLLM log or issue and compare against that. Being within 15% means your ledger is right.

---

**Next:** [Collectives and interconnects →](02-collectives-and-interconnects.md) — the seven primitives, the `α + β·bytes` cost model, and why a decode all-reduce is latency-bound while a prefill all-reduce is bandwidth-bound.
