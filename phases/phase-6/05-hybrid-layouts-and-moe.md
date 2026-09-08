# 5 — Hybrid Layouts, MoE and Expert Parallelism

> **You'll be able to say:** "A layout is `DP × TP × PP (× EP)`, and choosing one is a search with a single hard constraint — the memory ledger — and three communication costs on the critical path: TP's two all-reduces per layer, PP's one send/recv per boundary, EP's two all-to-alls per MoE layer. DP is free. MoE changes the arithmetic in one specific way: memory is set by *total* parameters while FLOPs are set by *active* parameters, and the number of experts a step actually touches grows with batch size — so MoE serving is capacity-bound, batch-hungry, and dominated by all-to-all and expert load imbalance."

Lessons 3 and 4 gave you two mechanisms. This lesson is how to combine them, plus the one model architecture that adds a third.

---

## The layout search

```
  total GPUs per deployment = DP × TP × PP        (each DP replica is TP × PP GPUs)
  per-GPU weight bytes      = W / (TP × PP)       (dense models)
  aggregate KV per replica  = (TP × PP) × usable − W
```

| Dimension | What it splits | Critical-path communication per decode step | Scaling behaviour |
|---|---|---|---|
| **DP** | nothing (full replica) | **none** (only a routing decision, [lesson 7](07-prefix-aware-routing.md)) | ~linear throughput; no latency change; independent failure domains |
| **TP** | every weight matrix | `2 · L` all-reduces of `tokens × H × 2` B — latency-bound in decode | sublinear (83% efficiency at TP8, [lesson 3](03-tensor-parallelism.md)); only lever that cuts per-token latency |
| **PP** | layers | `P − 1` send/recvs of `mb_tokens × H × 2` B | throughput + capacity; **no** latency benefit; bubble `(P−1)/(M+P−1)` |
| **EP** | MoE experts | `2` all-to-alls per MoE layer of `tokens × top_k × H × 2` B | capacity + bigger local GEMMs; exposed to expert load imbalance |

The procedure, which is just [lesson 1](01-when-one-gpu-isnt-enough.md)'s step 6 expanded:

```
  1. memory ledger fixes the minimum N = TP × PP (weights + required KV)
  2. TP first, up to the NVLink domain (8 on HGX; 72 on NVL72-class racks)   ← lesson 2
  3. N still larger? add PP across nodes, never TP across nodes             ← lesson 4
  4. MoE model? consider EP instead of TP for the MoE layers only
  5. everything left over goes to DP (replicas), which is where throughput
     should come from                                                        ← lesson 8
  6. re-check: does the TPOT SLO still pass at the TP degree you chose?
```

### The comparison people skip: same GPUs, different layout

8 H100s, Llama-3-70B FP16 (`W = 140 GB`, `usable ≈ 70 GB/GPU`):

| Layout | Replicas | KV per replica | Aggregate KV | Per-token latency floor | Failure domain | Best for |
|---|---|---|---|---|---|---|
| TP8 | 1 | 420 GB | 420 GB | 6.5 ms | all 8 GPUs | latency SLO, long context, max concurrency in one queue |
| DP2 × TP4 | 2 | 140 GB | 280 GB | 13 ms | 4 GPUs | throughput, cheaper blast radius |
| DP4 × TP2 | 4 | 0 GB — **doesn't fit** | — | — | — | nothing (2×70 − 140 = 0) |

That third row is why "just use more replicas" isn't always available: replicas of a model that barely fits have no KV budget left. Under FP8 weights (`W = 70 GB`) the same table flips — `DP4 × TP2` becomes viable with 70 GB of KV each — which is the concrete version of "quantization is a parallelism decision."

**Report throughput per GPU for every layout you test.** TP8 will win aggregate tokens/sec at high concurrency; `DP2 × TP4` usually wins tokens/sec/GPU. Both facts are true and they answer different questions.

---

## MoE: memory says total, FLOPs say active

A Mixture-of-Experts layer replaces one MLP with `E` expert MLPs plus a router that sends each token to `top_k` of them.

| Model | Total params | Active/token | Layers | Experts | top_k | FP16 weights | FP8 weights |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B | 46.7B | ~12.9B | 32 | 8 | 2 | ~93 GB | ~47 GB |
| Mixtral-8x22B | 141B | ~39B | 56 | 8 | 2 | ~282 GB | ~141 GB |
| Qwen3-235B-A22B | 235B | 22B | 94 | 128 | 8 | ~470 GB | ~235 GB |
| DeepSeek-V3 / R1 | 671B | 37B | 61 | 256 routed + 1 shared | 8 | ~1.34 TB | ~671 GB |

> **You pay for the whole model in memory and use a fraction of it per token.** Mixtral-8x7B needs 93 GB resident — TP2 minimum on 80 GB cards — to do the FLOPs of a 13B model. DeepSeek-V3 in FP8 needs ≥ 8×80 GB just to hold weights, and that's why it ships FP8-native.

The consequence that surprises people:

```
  experts touched by one step ≈ E · (1 − (1 − top_k/E)^B)     B = tokens in the step

  Mixtral (E=8, k=2):  B=1  → 2 experts   → read ~1.6 GB/layer-set of expert weights
                       B=8  → ~6.6        → most of them
                       B≥32 → ~8          → effectively ALL experts, every step
```

- **At batch 1, MoE decode reads only the active weights** → it behaves like a 13B model and is fast.
- **At production batch sizes it reads nearly all weights every step** → it behaves like a 47B model on the memory-bandwidth roofline ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)), while doing 13B-worth of FLOPs. Arithmetic intensity per expert *GEMM* is terrible: each expert sees only `B · top_k / E` tokens, so the local GEMM is skinny.
- **Therefore MoE wants big batches**, and it wants those tokens concentrated per expert — which is precisely what expert parallelism plus a large global batch delivers, and what small-batch latency-critical serving cannot.

---

## Expert parallelism: two all-to-alls

Two ways to shard an MoE layer:

```
  TP on the MoE layer                      EP on the MoE layer
  ────────────────────────────────         ─────────────────────────────────────────
  every rank holds a SLICE of              every rank holds E/EP WHOLE experts
  EVERY expert                             
  → all ranks compute all experts          → tokens must travel to their expert's rank
  → 1 all-reduce (as lesson 3)             → ALL-TO-ALL dispatch, local FFN,
  → tiny, skinny per-rank GEMMs               ALL-TO-ALL combine
  → no imbalance                           → full-width GEMMs, no weight replication
                                           → exposed to router load imbalance
```

```
   EP=4, E=8, top_k=2 — one MoE layer, one step

   ranks     0            1            2            3
   experts  [e0 e1]      [e2 e3]      [e4 e5]      [e6 e7]
              ▲            ▲            ▲            ▲
              └──────── ALL-TO-ALL dispatch ─────────┘     tokens × top_k × H × 2 B
                        (each rank sends each token copy to the owner of its expert)
              ┌──────── local expert FFN (full-width GEMM) ─────────┐
              └──────── ALL-TO-ALL combine  ────────────────────────┘   same volume
                        + weighted sum of the top_k expert outputs
```

Volume, for Mixtral (`H = 4096`, `top_k = 2`, FP16 → 8 KB per token-copy) at batch 32:

| Collective | Size | Regime ([lesson 2](02-collectives-and-interconnects.md)) | Count per step |
|---|---|---|---|
| all-to-all dispatch | 32 × 2 × 4096 × 2 B = 512 KB | latency-bound | 32 layers |
| all-to-all combine | 512 KB | latency-bound | 32 layers |

So ~64 all-to-alls per decode step, and **all-to-all is the worst collective to be latency-bound on**: it is pairwise, so its cost depends on the fabric's *bisection* bandwidth and on the slowest rank pair, and it violates the rail-optimized assumption (rank *i* ↔ rank *i*) that cross-node fabrics are built for. Cross-node EP without kernel-level overlap is the single most fabric-sensitive thing in this phase — which is why DeepSeek published DeepEP (NVLink intra-node + RDMA inter-node, communication overlapped with expert compute) as a separate artifact from the model.

### Load imbalance is the real EP problem

The router decides where tokens go, and it does not promise uniformity. If one expert receives 3× the mean tokens, its rank takes 3× the time and **every other rank waits in the combine all-to-all**. The step time is set by the hottest expert, not the average.

Mitigations, in the order they appear in production:

1. **Large global batch** — imbalance averages out with more tokens per step (another reason MoE wants batching).
2. **Expert replication / redundant experts** — duplicate the hottest experts onto additional ranks and split their traffic. DeepSeek's EPLB (expert-parallel load balancer) does this from measured traffic; it costs memory.
3. **Rebalance periodically** from observed counts — hot experts drift with the traffic mix, so this is a control loop, not a static assignment.
4. **Grouped/limited routing at training time** (DeepSeek's node-limited routing, auxiliary-loss-free balancing) — the model is *designed* to keep each token's experts inside few nodes. You inherit this; check the config.
5. **Capacity factors with token dropping** are a *training* technique. Dropping tokens at inference changes the output, so serving stacks queue or pad instead.

### Attention-DP + Expert-EP: the layout MoE pushed into engines

For models with small per-token KV (DeepSeek's MLA compresses KV to a latent vector), TP on attention is wasteful: it replicates the KV-latent work across ranks for a tensor that was already small. The layout that won instead:

```
  attention layers:  DATA parallel   — each rank owns a DISJOINT SET OF SEQUENCES
                                       and the whole KV for them (no KV sharding)
  MoE layers:        EXPERT parallel — each rank owns E/EP experts
  glue:              an all-gather/all-to-all between the two regimes per layer
```

This is what vLLM's `--data-parallel-size` combined with `--enable-expert-parallel` (and SGLang's DP-attention + EP) implements. Practical consequences: each DP-attention rank runs its own scheduler and its own KV pool, so **the ranks must step in lockstep** — an idle rank still has to participate in the MoE collectives, so engines insert dummy batches. Load skew between DP ranks therefore shows up as wasted work everywhere, which makes routing across DP ranks ([lesson 7](07-prefix-aware-routing.md)) part of the model layout, not just an ingress concern.

---

## Constraints and gotchas

| Constraint | Detail |
|---|---|
| `E % EP == 0` | each rank owns whole experts; `EP > E` is meaningless |
| `EP × TP_moe` ≤ world size | engines usually set `EP = world_size` for MoE layers and TP for attention, or DP-attention as above |
| MoE + PP is natural | experts are per-layer, so a stage owns whole MoE layers; PP hides the all-to-all inside a stage's node |
| Expert weights dominate load time | 671B of FP8 from object storage is minutes; pre-warm ([lesson 8](08-autoscaling-gpu-fleets.md)) |
| Fused MoE kernels expect grouped GEMM | `fused_moe` / grouped-GEMM kernels want tokens sorted by expert; the sort/scatter is a real cost at small batch |
| Quantization of experts is per-expert | scales are per-expert-per-channel; an expert-sharded layout keeps them local, which is one more argument for EP over TP |
| `top_k` multiplies your all-to-all | top_8 models (DeepSeek, Qwen3) move 4× the bytes of top_2 models at the same batch |

---

## Read the code (~40 minutes)

1. **`vllm/model_executor/layers/fused_moe/layer.py`** — `FusedMoE`: find where `ep_size`/`ep_rank` select the local expert range, and where the router's top-k output becomes a scatter into expert GEMMs.
2. **`vllm/model_executor/models/mixtral.py`** then **`deepseek_v2.py`/`deepseek_v3.py`** — compare a top-2/8-expert layer against a top-8/256-expert layer with a shared expert and MLA attention. The second one is where the DP-attention + EP layout comes from.
3. **`vllm/distributed/parallel_state.py`** — find the EP group alongside the TP/PP/DP groups. Four process groups over the same ranks is the whole idea of a hybrid layout.
4. **`deepseek-ai/DeepEP`** README + kernel signatures — read the "normal" vs "low-latency" dispatch modes and note which one is for prefill and which for decode. This is [lesson 2](02-collectives-and-interconnects.md)'s two regimes, appearing as two separate kernels.

---

## Do this now (45 minutes, CPU is fine)

1. **Layout table for your target model.** Extend [lesson 1](01-when-one-gpu-isnt-enough.md)'s sizing script to enumerate every `(DP, TP, PP)` with `DP·TP·PP = N` for `N = 8` and `N = 16`, and print: fits?, KV per replica, aggregate KV, latency floor, failure-domain size. Sort by "tokens/sec per GPU you'd predict." Do it for a dense 70B *and* Mixtral-8x7B — the MoE row set looks completely different because `W` triples while active FLOPs don't.
2. **Measure the expert-coverage curve.** Load Mixtral's (or any MoE's) router weights, or simply simulate: for `B` in `1…256`, sample `top_k` of `E` uniformly per token and count distinct experts touched. Plot against `E · (1 − (1 − k/E)^B)`. This one plot explains every "MoE is fast at batch 1 and disappointing at batch 64" report you'll read.
3. **Simulate imbalance.** With `gloo` and 4 processes, implement a toy EP layer: random router, `all_to_all_single` dispatch, per-rank "expert" = a matmul, all-to-all combine. Then skew the router (80% of tokens to expert 0) and measure step time. Confirm it tracks the *hottest* rank, then implement replication of expert 0 across two ranks and measure again.

---

**Next:** [Disaggregated prefill and decode →](06-disaggregated-prefill-decode.md) — the two phases want opposite hardware, so give them separate pools and ship the KV-cache between them.
