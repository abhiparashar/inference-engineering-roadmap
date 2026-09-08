# 6 — Disaggregated Prefill and Decode

> **You'll be able to say:** "Prefill is compute-bound and decode is memory-bandwidth-bound, so colocating them means every prefill steals a decode step and spikes TPOT. Disaggregation puts them in separate pools with separate hardware, parallelism and scaling, and ships the KV-cache between them — `prompt_tokens × kv_per_token` bytes, which for a 2k prompt on a 70B model is 640 MB, ~16 ms on one 400 Gb NIC and hideable behind prefill itself if you stream it layer by layer. It raises **goodput** — the fraction of requests meeting both TTFT and TPOT SLOs — even when raw throughput is flat, and I can compute the pool ratio from the traffic's prompt:output ratio and say when it isn't worth it."

This is the newest production pattern in the roadmap (2023-2025) and the one most likely to come up as "what's changed recently." The papers: **DistServe** (OSDI '24), **Splitwise** (Microsoft, ISCA '24), **Mooncake** (Moonshot AI, FAST '25).

---

## The problem: prefill/decode interference

From [Phase 1 lesson 6](../phase-1/06-prefill-vs-decode.md) and [Phase 4 lesson 1](../phase-4/01-what-to-optimize.md):

| | Prefill | Decode |
|---|---|---|
| Bottleneck | FLOPs (arithmetic intensity ~hundreds) | HBM bandwidth (intensity ~1) |
| Work per request | `2 · P · params` FLOPs, **once** | `2 · params` bytes read, **per token** |
| Wants | raw FLOPs, big token batches | memory bandwidth, memory *capacity* for KV |
| Latency metric | TTFT | TPOT |
| Ideal GPU | newest, highest TFLOPs | highest bandwidth/$, most VRAM, can be power-capped |
| Parallelism preference | TP for FLOPs; tolerant of big collectives | TP for latency; hates per-step collectives |

Colocated in one continuous-batching engine ([Phase 3 lesson 4](../phase-3/04-continuous-batching.md)), they fight:

```
  COLOCATED, no chunked prefill — one 2k-token prefill lands in a decode-heavy engine
  time →
  decode steps  │20│20│20│                          │20│20│20│   ms
  prefill       │        │███████ 51 ms ███████│
  what a user   │20│20│20│ ←──── 71 ms TPOT spike ──→│20│20│      p99 TPOT blown for
  sees                                                            EVERY running sequence

  COLOCATED, chunked prefill (512-token budget) — the 80% fix
  decode+chunk  │20│33│33│33│33│20│20│   TPOT degraded but bounded; TTFT ~4 chunks

  DISAGGREGATED
  prefill pool  │███████ 51 ms ███████│──KV 640 MB──▶
  decode pool   │20│20│20│20│20│20│20│20│  ← never runs a prefill; TPOT stays flat
```

**Goodput** is the metric this pattern optimizes: *requests/sec that meet both the TTFT and TPOT SLO*. A colocated engine can post excellent mean throughput while failing p99 TPOT on every request that shared a step with a prefill — that throughput is not sellable. DistServe reports **7.4× more requests or 12.6× tighter SLO** vs then-state-of-the-art while keeping >90% of requests inside latency constraints; Splitwise reports **1.4× higher throughput at 20% lower cost**, or **2.35× more throughput at the same power and cost** (its Splitwise-HA config runs prefill on H100s and decode on power-capped A100s); Mooncake reports **50-525% throughput improvement at 16k-128k contexts** and **75% more requests within SLO** in production for Kimi. Read those as direction and magnitude, not as your numbers.

---

## The mechanism, and its one cost

```
   client
     │  POST /v1/chat/completions
     ▼
   ROUTER ─────────────────────────────────────────────────────┐
     │                                                          │
     ▼  (1) prefill request                                     │ (4) stream tokens
   ┌──────────────── PREFILL POOL ────────────────┐             │
   │  TP8 H100, high FLOPs, SMALL KV budget       │             │
   │  runs prompt → produces KV blocks + 1st token│             │
   └───────────────────┬──────────────────────────┘             │
                       │ (2) KV TRANSFER                        │
                       │ prompt_tokens × kv_per_token bytes     │
                       │ RDMA/NVLink, layer-by-layer streamed   │
                       ▼                                        │
   ┌──────────────── DECODE POOL ─────────────────┐             │
   │  high bandwidth + LARGE KV budget, power-    │─────────────┘
   │  capped; runs only decode steps, big batches │
   └──────────────────────────────────────────────┘
```

### The transfer arithmetic

`transfer_bytes = prompt_tokens × kv_per_token` (the KV you already computed, from [lesson 1](01-when-one-gpu-isnt-enough.md)).

Llama-3-70B, `kv_per_token = 320 KiB` FP16:

| Prompt | KV bytes | 1×400 Gb NIC (~40 GB/s) | 8 NICs / rail-parallel (~320 GB/s) | NVLink (450 GB/s) |
|---|---|---|---|---|
| 512 | 160 MB | 4 ms | 0.5 ms | 0.36 ms |
| 2,048 | 640 MB | 16 ms | 2.0 ms | 1.4 ms |
| 8,192 | 2.6 GB | 64 ms | 8.0 ms | 5.7 ms |
| 32,768 | 10.2 GB | 256 ms | 32 ms | 23 ms |

Compare against the prefill compute it follows (2k prompt, TP8 H100 ≈ 51 ms from [lesson 3](03-tensor-parallelism.md)):

- **Transfer scales linearly with prompt length, and so does prefill compute** — so the *ratio* is roughly constant: ~30% on a single NIC, ~4% rail-parallel. That constant ratio is what makes the pattern viable at all context lengths.
- **Layer-wise streaming hides most of it.** Layer `i`'s KV blocks are final the moment layer `i` finishes, so you send them while layer `i+1` computes. Done well, the exposed cost is one layer's worth of KV plus the final handoff.
- **FP8 KV halves every number in that table** ([Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md)) — a second reason to quantize the cache.
- **Amortize against the request, not the prefill.** 300 output tokens at 20 ms is 6 s of decode; 16 ms of transfer is 0.3% of the request's life. Disaggregation's cost is small *because generation is long*. Which gives the first "don't" below.

---

## When it pays, and when it doesn't

| Condition | Disaggregate? | Why |
|---|---|---|
| Long prompts, long outputs, high load | **yes** | max interference removed, transfer well amortized |
| Prompt-heavy with short outputs (classification, reranking, embeddings) | **no** | transfer + hop cost per request, with almost no decode to protect |
| Low load (pool idle much of the time) | **no** | there is no interference to remove; you added a hop and split your memory |
| Tight TPOT SLO with bursty prefill traffic | **yes** | this is exactly the goodput case |
| Fabric < ~25 GB/s effective between pools | **no** | transfer dominates; use chunked prefill instead |
| Very high prefix-cache hit rate | **usually no** | prefill is already nearly free ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)); splitting pools also splits the cache — unless you go KVCache-centric (below) |
| Heterogeneous fleet (mixed GPU generations) | **yes** | the Splitwise argument: give old/power-capped GPUs the decode work |
| Small model that fits with room to spare | **no** | replicas are simpler and cheaper ([lesson 1](01-when-one-gpu-isnt-enough.md)) |

**Chunked prefill is the cheaper 80% solution and should be your baseline.** It bounds the interference inside one engine with zero new infrastructure ([Phase 5 lesson 3](../phase-5/03-vllm-in-production.md)). Disaggregation wins the remaining gap — the part chunked prefill can't fix, because chunks still consume decode-pool bandwidth and still can't give the two phases different hardware or different parallelism.

---

## Sizing the two pools

Each pool has its own capacity unit:

```
  prefill instance:  X tokens/sec of prompt processing
  decode  instance:  Y token-steps/sec  ( = batch_size / step_time )

  per request:  P prompt tokens, O output tokens
  demand at q QPS:  prefill = q·P / X  instances,  decode = q·O / Y  instances

  ratio  n_prefill : n_decode  =  (P/X) : (O/Y)
```

Worked, with `X = 15,000` tok/s and `Y = 2,560` token-steps/s (batch 64, 25 ms step) per TP8 70B instance:

| Traffic shape | `P/X` | `O/Y` | Pool ratio | Comment |
|---|---|---|---|---|
| RAG: 4000 in / 200 out | 0.267 | 0.078 | **3.4 : 1** | prefill-heavy; consider prefix caching first |
| Chat: 2000 in / 300 out | 0.133 | 0.117 | **1.1 : 1** | balanced |
| Agent: 1000 in / 1000 out | 0.067 | 0.391 | **1 : 5.9** | decode-dominated |
| Summarize: 32k in / 500 out | 2.13 | 0.195 | **11 : 1** | prefill pool *is* the system |

Three operational consequences:

1. **The ratio is a property of your traffic, and traffic drifts** — diurnally, and whenever a product ships a longer system prompt. A static split is wrong within a week, so the ratio is an autoscaling input ([lesson 8](08-autoscaling-gpu-fleets.md)), and some systems let instances *switch roles* under load (Splitwise's mixed pool).
2. **Whichever pool saturates first sets your SLO failure mode.** Prefill saturation → TTFT collapse and queueing; decode saturation → TPOT collapse and preemption. Measure and alert on both separately — you can no longer read one queue depth and know your state ([Phase 7](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)).
3. **Under overload, reject early.** Mooncake's prediction-based early rejection exists because admitting a request that will miss its SLO burns *both* pools' capacity — the distributed version of [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)'s admission control.

---

## The KVCache-centric variant

Point-to-point transfer is the simple design. The Mooncake design goes further: make the KV-cache a **shared, pooled, cluster-level resource** in the CPU DRAM and SSDs that the GPU nodes already have.

```
   prefill pool ─┐                                 ┌─ decode pool
                 ├──▶  DISTRIBUTED KVCACHE POOL  ──┤
   prefill pool ─┘   (CPU DRAM + NVMe across nodes) └─ decode pool
                      addressed by prefix hash; RDMA transfer engine
```

What that buys, beyond disaggregation:

- **The prefix cache becomes global.** A prefix computed by any prefill instance is reusable by all of them, so the hit rate stops depending on routing luck ([lesson 7](07-prefix-aware-routing.md) becomes a *lookup* instead of a bet).
- **KV capacity becomes a tier, not a wall.** HBM → CPU DRAM → NVMe, each ~10× bigger and ~10× slower, which is [Phase 2 lesson 2](../phase-2/02-gpu-memory-hierarchy.md)'s hierarchy extended one more time. Long-context reuse that would never fit in HBM now hits in DRAM.
- **Scheduling becomes cache-aware globally**, which is what "KVCache-centric scheduler" means: pick the prefill instance, the decode instance *and* whether to recompute or fetch, from one view of where bytes live.

The cost: a transfer engine and a metadata service to operate, plus a genuinely hard cache-invalidation/eviction problem across tiers. This is the frontier-lab shape, and LMCache + vLLM's connector API bring a usable version of it to open source.

---

## Implementations to know

| Stack | How you turn it on / what to read |
|---|---|
| **vLLM** | `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`; roles `kv_producer`/`kv_consumer`; connector API in `vllm/distributed/kv_transfer/` — read `kv_connector/v1/base.py`, then the Nixl and SharedStorage connectors |
| **NIXL** (NVIDIA) | the transfer library underneath: uniform API over UCX/RDMA/NVLink/file, with layer-wise async transfers |
| **LMCache** | KV offload + reuse layer (GPU → CPU → disk) usable standalone or as a vLLM connector; the practical on-ramp to a shared KV tier |
| **NVIDIA Dynamo** | disaggregated serving with a KV-aware router and a "planner" that scales the two pools; the productized version of this lesson |
| **llm-d** / Gateway API Inference Extension | Kubernetes-native P/D disaggregation + cache-aware endpoint picking ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)) |
| **SGLang** | PD disaggregation using the Mooncake transfer engine; pairs with RadixAttention ([Phase 5 lesson 5](../phase-5/05-sglang-and-radixattention.md)) |
| **Mooncake** (`kvcache-ai/Mooncake`) | the transfer engine + store, open-sourced; the paper is the architecture document |

**Compatibility constraint people hit immediately:** the two pools must agree on model, dtype, KV dtype, block size *and* KV sharding layout. If prefill runs TP4 and decode runs TP8, the KV heads are partitioned differently and blocks cannot be dropped in as-is — they need re-partitioning on the wire or a matching layout. "Different parallelism per phase" is a *DistServe* selling point, so real implementations either constrain it or pay for a reshuffle; check which one yours does before designing around it.

---

## Do this now (60 minutes, no cluster required)

1. **Compute your transfer budget.** For your target model and the p50/p95 prompt length of your traffic, compute `transfer_bytes`, the time on your fabric, and the ratio to prefill compute. If that ratio is over ~30%, disaggregation is a fabric-upgrade project, not a serving project — and knowing that before proposing it is the whole point.
2. **Measure the interference you'd be removing.** Single engine, single GPU: run a steady decode load (Phase 3 harness, closed-loop 16 sessions), record p50/p99 TPOT. Now inject one long prompt every 2 s and re-record. Then set `--enable-chunked-prefill` with `--max-num-batched-tokens 512` and record again. You now have the three numbers — colocated, chunked, and the theoretical disaggregated floor (the no-injection baseline) — that justify or kill the project.
3. **Simulate the pools.** Extend the model with two queues (M/M/c-ish, [Phase 3 lesson 5](../phase-3/05-queueing-theory.md)): prefill service time `P/X`, decode holding time `O × step_time`, plus a transfer delay. Sweep the pool ratio at fixed total GPUs and plot **goodput** (fraction meeting both SLOs), not throughput. The optimum ratio should land near the formula above; the interesting part is how sharply goodput falls off on the prefill-starved side.

---

**Next:** [Routing for stateful serving →](07-prefix-aware-routing.md) — once you have many replicas or pools, round-robin quietly destroys the prefix cache you spent Phase 4 building.
