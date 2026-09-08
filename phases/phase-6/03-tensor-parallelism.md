# 3 — Tensor Parallelism, From the Matmul Up

> **You'll be able to say:** "TP splits every weight matrix across ranks: the first matmul of a pair is *column-parallel* (no communication, sharded output), the second is *row-parallel* (partial sums, one all-reduce). Attention needs no communication *inside* it because heads are independent — only the output projection reduces. So a transformer layer costs exactly two all-reduces per forward pass, and for a 70B model that's 160 latency-bound collectives per decode step, about 1.3 ms on NVLink. I can derive every per-rank weight shape, state the GQA and quantization-group divisibility constraints, and explain why TP scaling is sublinear and why it must stay inside the NVLink domain."

This is the mechanism behind `--tensor-parallel-size`. Do the [Phase 6 lab 1](../../labs/README.md#phase-6-lab--distributed-inference) (read Megatron's `ColumnParallelLinear`/`RowParallelLinear`) either just before or just after this lesson.

---

## The primitive: two ways to split `Y = X·A`

`X` is `[tokens, K]`, `A` is `[K, M]`, TP degree `T`.

```
 COLUMN-PARALLEL  (split A by columns / output dim)     RANK r holds A[:, rM/T:(r+1)M/T]
 ───────────────────────────────────────────────────────────────────────────────────────
   X (full, replicated)  ·  A_r  =  Y_r  [tokens, M/T]     ← output is SHARDED
   communication: NONE on the forward pass.
   Every rank needs all of X, and every rank produces a slice of Y.

 ROW-PARALLEL  (split A by rows / input dim)            RANK r holds A[rK/T:(r+1)K/T, :]
 ───────────────────────────────────────────────────────────────────────────────────────
   X_r (sharded on K)  ·  A_r  =  Y_partial  [tokens, M]   ← every rank has a PARTIAL SUM
   communication: ALL-REDUCE (sum) to make Y correct on every rank.
   Every rank needs only its slice of X, and produces a full-shape partial result.
```

The whole design of TP is one observation:

> **A column-parallel layer's output is exactly the sharded input a row-parallel layer wants.** So if you arrange matmuls in column→row pairs, you communicate **once per pair** instead of once per matmul. Megatron calls the two glue operations `f` (forward: identity; backward: all-reduce) and `g` (forward: all-reduce; backward: identity). For inference only `g` exists.

Transformers are conveniently built out of exactly such pairs.

---

## The transformer layer under TP

```
                     ── ATTENTION BLOCK (one all-reduce) ──
  x [tok, H] replicated
     │
     ├─ RMSNorm (weights replicated, tiny)
     │
     ├─ QKV projection ── COLUMN-parallel, split by HEADS ──────────┐
     │     rank r gets  n_q/T query heads, n_kv/T key + value heads │
     │     RoPE applied locally to the rank's own heads             │
     │                                                              │
     ├─ Attention (paged, FlashAttention/FlashDecoding) ────────────┤  NO COMMUNICATION:
     │     rank r attends only over its own heads' KV blocks;       │  heads are independent,
     │     the KV-cache itself is sharded by head → KV/T per rank   │  so softmax is local
     │                                                              │
     ├─ Output projection ── ROW-parallel ──────────────────────────┘
     │     input is the head-sharded attention output; output is a partial sum
     │
     └─▶ ALL-REDUCE  [tok, H]  ◀── collective #1
     │
     + residual add (replicated, post-reduce)

                     ── MLP BLOCK (one all-reduce) ──
     ├─ RMSNorm
     ├─ gate_proj, up_proj ── COLUMN-parallel ──┐   SwiGLU: act(gate)·up, elementwise,
     │     rank r holds I/T columns of each     │   entirely local on the sharded dim
     ├─ down_proj ── ROW-parallel ──────────────┘
     └─▶ ALL-REDUCE  [tok, H]  ◀── collective #2
     + residual add

  ⇒ 2 all-reduces per layer per forward pass, each of size tokens × H × dtype_bytes
```

The two ends of the model:

- **Embedding: vocab-parallel.** Each rank holds `vocab/T` rows; a lookup masks out-of-range IDs to zero and an all-reduce sums the one non-zero contribution. (Cheap: one collective for the whole prefill.)
- **LM head: column-parallel over vocab**, so each rank produces logits for its vocab slice. You then either all-gather logits (simple, and what vLLM does by default) or run a distributed argmax/sampling that only broadcasts the chosen token IDs. **This collective is not small:** batch 32 × 128,256 vocab × 4 B ≈ 16 MB — *bandwidth*-bound by [lesson 2](02-collectives-and-interconnects.md)'s crossover, and a genuine cost at high batch. It's why "sampling overhead" shows up in high-TP profiles.

---

## Per-rank shapes: Llama-3-70B at TP=8

`H = 8192`, `I = 28672`, `n_q = 64`, `n_kv = 8`, `head_dim = 128`, `L = 80`, `vocab = 128,256`.

| Tensor | TP1 shape | Split | TP8 per-rank shape | Bytes/rank (FP16) |
|---|---|---|---|---|
| `q_proj` | [8192, 8192] | column (8 of 64 heads) | [8192, 1024] | 16.8 MB |
| `k_proj` | [8192, 1024] | column (1 of 8 KV heads) | [8192, 128] | 2.1 MB |
| `v_proj` | [8192, 1024] | column (1 of 8 KV heads) | [8192, 128] | 2.1 MB |
| `o_proj` | [8192, 8192] | **row** | [1024, 8192] | 16.8 MB |
| `gate_proj` | [8192, 28672] | column | [8192, 3584] | 58.7 MB |
| `up_proj` | [8192, 28672] | column | [8192, 3584] | 58.7 MB |
| `down_proj` | [28672, 8192] | **row** | [3584, 8192] | 58.7 MB |
| norms | [8192] ×2 | replicated | [8192] ×2 | 32 KB |
| **per layer** | ~2.14 GB | | ~214 MB | ×80 = **17.1 GB** |
| `embed_tokens` | [128256, 8192] | vocab (row) | [16032, 8192] | 263 MB |
| `lm_head` | [128256, 8192] | vocab (column) | [16032, 8192] | 263 MB |
| **total** | 140 GB | | | **~17.6 GB** |

Matches `W/T` from [lesson 1](01-when-one-gpu-isnt-enough.md) — the replicated norms are rounding error. Engines fuse `q/k/v` into one `QKVParallelLinear` and `gate/up` into one `MergedColumnParallelLinear` so each becomes a single larger GEMM; the sharding logic is identical, which is why those two classes exist in `vllm/model_executor/layers/linear.py`.

**The KV-cache shards with the KV heads:** per-rank KV per token = `320 KiB / 8 = 40 KiB`. This is the second half of why TP buys concurrency, not just fit.

---

## Divisibility: what actually rejects your config

| Constraint | 70B @ TP8 | Failure mode |
|---|---|---|
| `n_q % T == 0` | 64/8 = 8 ✓ | hard error at load |
| `n_kv % T == 0` | 8/8 = 1 ✓ | `T > n_kv` → engines **replicate** KV heads: correct output, but per-rank KV/token stops shrinking, so your aggregate KV budget silently stops growing |
| `I % T == 0` | 28672/8 = 3584 ✓ | hard error at load |
| `(K/T) % group_size == 0` for group-wise quant | 3584/128 = 28 ✓ | the classic AWQ/GPTQ failure — e.g. Llama-2-7B has `I = 11008`, so `11008/8 = 1376` and `1376/128 = 10.75` ✗: **GPTQ/AWQ group-128 at TP8 is impossible on that model** |
| `vocab` padded to multiple of `T` | padded to 128,256 | engines pad silently |

**Quantization scale sharding is the subtle one.** Per-output-channel scales (`[M]`) shard cleanly with a column-parallel split. Group-wise scales along the *input* dim (`[K/group, M]`) must not have a group straddling a row-parallel shard boundary — hence the third row. FP8 per-tensor scales are trivially replicated, which is one practical argument for FP8 over INT4 at high TP ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)).

---

## The cost, and why scaling is sublinear

Per decode step, using [lesson 2](02-collectives-and-interconnects.md)'s model with `B` sequences:

```
   compute floor  =  W / (T · BW_eff)           shrinks as 1/T
   comm cost      =  2 · L · (α + 2·B·H·2/B_alg)
                     └─ at B·H·2 bytes ≈ 0.5 MB for B=32, we're below the ~2 MB
                        crossover → comm ≈ 2 · L · α : CONSTANT in T and in B
```

For 70B FP16 on H100s (`α ≈ 8 µs`, `BW_eff ≈ 2.7 TB/s`), batch 32:

| T | Compute floor | Comm (160 × α) | Total | Speedup vs T=1 | Per-GPU efficiency |
|---|---|---|---|---|---|
| 1 | 51.9 ms | 0 | 51.9 ms | 1.0× | 100% |
| 2 | 25.9 ms | 1.3 ms | 27.2 ms | 1.91× | 95% |
| 4 | 13.0 ms | 1.3 ms | 14.3 ms | 3.63× | 91% |
| 8 | 6.5 ms | 1.3 ms | 7.8 ms | 6.65× | 83% |
| 16 (in-node, hypothetical) | 3.2 ms | 1.3 ms | 4.5 ms | 11.5× | 72% |

*(Model-derived, not measured: it ignores attention over the KV-cache, sampling, scheduling and kernel inefficiency, so real numbers are 2-4× the totals — but the* shape *of the efficiency column is what real benchmarks show.)*

That table is the whole argument:

- **Amdahl, with `2·L·α` as the serial term.** The compute halves each time you double `T`; the collective cost doesn't move. Efficiency decays, so **TP throughput-per-GPU is monotonically decreasing** — which is why [lesson 1](01-when-one-gpu-isnt-enough.md) insists you compare against replicas.
- **Bigger batches make TP look better**, because comm is fixed while useful work grows. A TP=8 deployment benchmarked at batch 1 looks terrible and tells you nothing about production.
- **Cross-node TP multiplies the serial term by 3-12×** (α of 25-100 µs instead of 8). At `2·L·α = 4 ms` the collective *is* the step. This is the arithmetic behind "TP ≤ the NVLink domain," and the reason 16-GPU 405B deployments use TP8×PP2, not TP16.
- **Small-message all-reduce is worth its own kernel.** At 0.5 MB you're paying pure launch+sync overhead, so vLLM's `custom_all_reduce` (one-shot/two-shot over NVLink peer memory) beats NCCL in exactly this regime, and CUDA-graph capture of the collectives removes the launch cost per layer ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)). Together they're worth 10-20% of decode at TP8 — which is why `--enforce-eager` hurts more the higher your TP.

### Prefill is the other regime

At 2048 prompt tokens the all-reduce is 32 MB: bandwidth-bound, ~145 µs each, ~23 ms per prompt against a prefill compute cost of roughly `2·N·P/(T·FLOPs_eff)` = `2 × 2048 × 70e9 / (8 × ~700 TFLOP/s)` ≈ 51 ms. So ~30% overhead, and it *scales with the fabric*, not with `α`. Chunked prefill (512-token chunks) cuts each collective to 8 MB and lets it interleave with decode work — one more reason it's on by default in modern engines ([Phase 5 lesson 3](../phase-5/03-vllm-in-production.md)).

---

## Sequence parallelism: the free memory trick

The residual/norm regions between the two blocks are *replicated* across ranks — every rank holds the full `[tokens, H]` activation. Megatron's sequence parallelism shards those regions along the token dimension instead, and replaces:

```
   all-reduce  (2·(N−1)/N·S bytes)
        ⇕  identical total volume
   reduce-scatter (before the norm region) + all-gather (after it)
```

Same bytes on the wire, but activation memory in the norm/residual region drops by `T`. For inference this matters mainly at long context and large prefill chunks, where those activation buffers compete with the KV-cache in [lesson 1](01-when-one-gpu-isnt-enough.md)'s ledger. It's the concrete payoff of the `all-reduce = reduce-scatter + all-gather` identity.

---

## Practical gotchas

- **TP changes your logits.** All-reduce sums in a nondeterministic order, so TP=8 output is not bitwise equal to TP=1 — and with temperature > 0 a tie-break can diverge into a completely different continuation. When you validate a sharded implementation, compare with `torch.allclose(..., atol=1e-2, rtol=1e-2)` on logits under greedy decoding, not string equality of long generations. Expect bug reports of the form "changing TP changed my outputs"; the answer is "yes, by design."
- **Every rank runs the same Python.** In vLLM's V1 architecture, rank 0 drives and workers execute the same model-runner code; a shape mismatch on one rank appears as a *hang* (the others wait in a collective), not an exception. See [lesson 9](09-multi-node-operations.md).
- **All ranks must load weights** — startup is `weights/T` bytes per rank in parallel, so load time improves, but a single slow disk/S3 rank gates the replica.
- **Uneven head counts are a real model-design constraint.** Models with 6 or 12 KV heads (some Qwen/Gemma variants) cap clean TP at those factors; check `num_key_value_heads` before promising a layout.
- **LoRA adapters shard too** — `lora_a` column-parallel, `lora_b` row-parallel, and the adapter's all-reduce merges into the base layer's.

---

## Read the code (in this order, ~45 minutes)

1. **`Megatron-LM/megatron/core/tensor_parallel/layers.py`** — `ColumnParallelLinear`, `RowParallelLinear`. Find the `all_reduce` in `RowParallelLinear.forward` and the `copy_to_tensor_model_parallel_region` / `reduce_from_tensor_model_parallel_region` pair (`f` and `g`). This is the canonical 100 lines.
2. **`vllm/model_executor/layers/linear.py`** — the same two classes, plus `QKVParallelLinear` (note how it handles `num_kv_heads < tp_size` by replicating) and `MergedColumnParallelLinear`. Diff mentally against Megatron: vLLM adds quantization-aware sharding via `weight_loader`.
3. **`vllm/model_executor/layers/vocab_parallel_embedding.py`** — the masked lookup + reduce.
4. **`vllm/distributed/parallel_state.py`** and **`communication_op.py`** — how the TP group is created and what `tensor_model_parallel_all_reduce` dispatches to (custom all-reduce vs NCCL vs pynccl).
5. **`vllm/model_executor/models/llama.py`** — see all of the above wired into `LlamaAttention`/`LlamaMLP`. Count the all-reduces per layer yourself and confirm it's two.

---

## Do this now (60 minutes, CPU is fine)

1. **Shard one linear layer by hand.** Two processes, `gloo` backend, `torch.distributed`. Build `A [512, 2048]`, compute the reference `Y = X @ A`. Then: column-parallel (each rank holds 1024 columns, concatenate outputs with `all_gather`) and row-parallel (each rank holds 256 rows and its slice of `X`, `all_reduce` the partial sums). Assert `allclose` against the reference for both. This is [project 11](../../projects/README.md) Part A and it removes all remaining mystery from TP.
2. **Chain them.** Column-parallel → element-wise GELU → row-parallel, and confirm you needed **exactly one** `all_reduce` for the pair. Count collectives with a wrapper that increments a counter.
3. **Shard attention with GQA.** 8 query heads, 2 KV heads, `T=2`: each rank gets 4 Q heads and 1 KV head, computes attention locally, then row-parallel `o_proj` + all-reduce. Verify against single-process attention. Then set `T=4` and watch the KV-head constraint bite — implement the replication fallback and note that per-rank KV memory stopped shrinking.

---

**Next:** [Pipeline parallelism and the bubble →](04-pipeline-parallelism.md) — splitting layers instead of matrices, one send/recv per boundary, and why this is the tool for crossing a network.
