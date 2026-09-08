# Phase 6 — Distributed Inference at Scale (Deep Dive)

> **Goal:** stop assuming the model fits. By the end of this phase you can size a deployment from arithmetic (how many GPUs, in what parallel layout, with how much KV headroom), explain tensor/pipeline/data/expert parallelism down to the individual collective, justify prefill/decode disaggregation in terms of *goodput* rather than throughput, route stateful prefix-cached traffic without breaking cache locality, and autoscale a fleet whose cold start is measured in minutes.

This folder is the long-form version of [Phase 6 in the ROADMAP](../../ROADMAP.md#phase-6--distributed-inference-at-scale). Phases 3-5 were one engine on one GPU, operated well. This phase is what happens when the weights don't fit in 80 GB, when the KV-cache budget — not the FLOPs — sets your concurrency ceiling, and when the traffic needs more than one machine.

The intellectual core:

> **Replication scales *throughput*. Splitting scales *size*. Only tensor parallelism meaningfully scales *latency*, and it buys that with two collectives per layer on the critical path. Every distributed inference design is a choice about which of those three you're short of — and every collective you add is a tax you pay on every token, forever.**

---

## Prerequisites

- **[Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md)** + **[Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md)** — bytes per parameter, bytes per KV token, and where a GPU's memory actually goes. Lesson 1 of this phase is that arithmetic applied to *N* GPUs; if you can't do it for one, N won't help.
- **[Phase 2 lessons 2 and 5](../phase-2/02-gpu-memory-hierarchy.md)** — the memory hierarchy and the roofline. An interconnect is just the next rung down the hierarchy, and every parallelism decision here is a roofline argument with a slower "memory."
- **[Phase 3 lessons 4-6](../phase-3/04-continuous-batching.md)** — continuous batching, queueing theory, admission control. Load balancing across replicas is queueing theory with N queues, and the sticky-routing lesson is a direct conflict between locality and load balance.
- **[Phase 4 lessons 5-6](../phase-4/05-paged-attention.md)** — block tables and prefix caching. Disaggregation moves *blocks* over a network, and prefix-aware routing exists to protect the hit rate you built in Phase 4.
- **[Phase 5 lessons 2-3](../phase-5/02-vllm-architecture.md)** — you'll read `vllm/distributed/` and set `--tensor-parallel-size` / `--pipeline-parallel-size` for real, and you should already know which layer those flags act on.

### About hardware

- **Laptop / Apple Silicon is enough** for lessons 1, 2 (theory + `gloo` on CPU), 4, 6, 7, 8, 9 and for the tensor-parallel *correctness* build in lesson 10 Part A — `torch.distributed` with the `gloo` backend across 2-4 CPU processes reproduces every sharding and all-reduce detail. Communication *cost* is the only thing you can't measure this way.
- **Two GPUs on one machine (rented, ~$1-1.50/hr)** is enough for real TP: `nccl-tests`, `--tensor-parallel-size 2`, and the TP-vs-replicas comparison in lesson 10. Budget 4-8 GPU-hours for the whole phase.
- **Multi-node is optional and expensive.** You can learn PP and disaggregation on a single node by putting each stage/pool on a different GPU, and you can learn cross-node *behaviour* by throttling with `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` to force the slow path. Do that instead of renting a cluster.
- **The router, autoscaler and disaggregation-simulation work is CPU-only** and is where a surprising amount of the interview signal lives.

---

## The map of this phase

```
                    "ONE GPU IS NOT ENOUGH" — but which kind of "not enough"?
                                        │
   ┌──────────────┬─────────────────────┼──────────────────────┬────────────────────┐
   ▼              ▼                     ▼                      ▼                    ▼
WEIGHTS       KV BUDGET             LATENCY               THROUGHPUT           WORKLOAD MIX
don't fit     too small            TPOT too high          QPS too high      prefill & decode
                                                                            fight each other
   │              │                     │                      │                    │
   ▼              ▼                     ▼                      ▼                    ▼
 SPLIT the model (TP / PP / EP)      TP (sublinear)      REPLICATE (DP)       DISAGGREGATE
 lessons 3, 4, 5                     lesson 3            lessons 7, 8         lesson 6
   │                                                          │
   └────────────── every split adds COLLECTIVES ──────┐       └── every replica adds a
                   on the critical path (lesson 2)    │           ROUTING decision that
                                                      │           can destroy your prefix
                   TP: 2 all-reduce / layer / step ◀──┘           cache hit rate (lesson 7)
                   PP: 1 send-recv / stage / microbatch
                   EP: 2 all-to-all / MoE layer / step

           and all of it runs on a fabric hierarchy that is the Phase-2 memory
           hierarchy with two more rungs:  HBM  ≫  NVLink  ≫  PCIe  ≫  RDMA
                                         3.3 TB/s  450 GB/s  25 GB/s   50 GB/s/NIC
```

Two things fall out of this picture that most people get backwards:

1. **You do not reach for tensor parallelism to go faster; you reach for it because the model doesn't fit, or because you have already spent everything else.** TP at 8 GPUs does not give 8× the tokens/sec of one GPU — it gives roughly 5-6× at best, because the collectives don't shrink as you add ranks. Replicas give a genuine, boring, near-linear 8×.
2. **Distribution changes what "one failure" means.** A TP=8 replica is one fate-sharing unit: one dead GPU kills eight. Every parallelism dimension multiplies your blast radius and your rollout coordination cost, and that — not the throughput number — is what makes a distributed layout expensive to operate ([lesson 9](09-multi-node-operations.md)).

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [When one GPU isn't enough](01-when-one-gpu-isnt-enough.md) | "Given a model, a GPU and an SLO I can compute the minimum GPU count, the aggregate KV budget it yields (`TP × usable − weights`), the concurrency that implies, and which of the four scaling reasons I'm actually solving for." |
| 2 | [Collectives and interconnects](02-collectives-and-interconnects.md) | "I know the seven collectives, the bytes each one moves, and the `α + β·bytes` cost of each fabric. I can say why a decode all-reduce is latency-bound and a prefill all-reduce is bandwidth-bound, and I can measure both with `nccl-tests`." |
| 3 | [Tensor parallelism, from the matmul up](03-tensor-parallelism.md) | "Column-parallel then row-parallel means exactly one all-reduce per MLP and one per attention block. I can derive the per-rank weight shapes, the GQA divisibility constraint, the per-step communication cost, and why TP must stay inside the NVLink domain." |
| 4 | [Pipeline parallelism and the bubble](04-pipeline-parallelism.md) | "PP splits layers, moves one activation tensor per stage boundary, and costs a bubble of `(P-1)/(M+P-1)`. It scales throughput and *size* across nodes for almost no bandwidth — and it never improves per-token latency." |
| 5 | [Hybrid layouts, MoE and expert parallelism](05-hybrid-layouts-and-moe.md) | "`DP × TP × PP × EP` is a search over one memory constraint and three communication costs. MoE turns the MLP into two all-to-alls and makes load imbalance a scheduling problem." |
| 6 | [Disaggregated prefill and decode](06-disaggregated-prefill-decode.md) | "Prefill and decode want opposite hardware. Splitting them into two pools and shipping KV blocks between them raises *goodput* — SLO-meeting throughput — even when raw throughput is flat. I can compute whether the KV transfer pays for itself." |
| 7 | [Routing for stateful serving](07-prefix-aware-routing.md) | "Round-robin destroys a prefix cache. Consistent hashing, session affinity and KV-aware routing trade load balance for hit rate, and I can quantify both sides of that trade with the Phase-3 harness." |
| 8 | [Autoscaling and capacity for GPU fleets](08-autoscaling-gpu-fleets.md) | "Model load takes minutes, so reactive HPA on utilization is the wrong controller. I can design scaling on queue depth/KV occupancy with pre-warmed pools, and price spot vs on-demand against an SLO." |
| 9 | [Multi-node operations and failure modes](09-multi-node-operations.md) | "I can debug a NCCL hang, find the straggler rank, explain what happens when one GPU of a TP group dies, and roll out a new version of a multi-node replica without a full outage." |
| 10 | [Build: manual TP + a prefix-aware router](10-build-manual-tp-and-router.md) | "I sharded a linear layer and an attention block by hand with `torch.distributed`, proved bitwise-comparable output against the unsharded reference, then served multiple replicas behind a sticky router and measured the cache-hit-rate delta vs round-robin." |
| 11 | [Exercises & exit artifact](11-exercises-and-artifacts.md) | "Here is the sharding notebook, the router with its hit-rate/latency table, and a one-page sizing document for a 70B deployment." |

---

## How to work through this phase

1. **Do the arithmetic before you rent anything.** Lesson 1 is the load-bearing lesson: most "we need more GPUs" conclusions are wrong by a factor of two in either direction, and the fix is 10 minutes with a calculator, not a benchmark.
2. **Simulate before you scale.** Two CPU processes with `gloo` teach you sharding correctness; one GPU with `NCCL_P2P_DISABLE=1` teaches you what a bad fabric feels like. Rent multi-GPU time only once you know what number you're going to measure.
3. **Always compare against replicas.** For every TP/PP configuration you test, test "the same total GPUs as independent replicas." That is the honest baseline, it usually wins on throughput-per-dollar, and knowing exactly when it *stops* winning is the whole skill.
4. **Measure the fabric once, on your hardware, and write the numbers down.** `all_reduce_perf` at 512 KB and at 64 MB gives you the two constants (`α`, `β`) that predict every communication cost in this phase.
5. **Keep the Phase-3 harness.** Every claim about routing, autoscaling and disaggregation in lessons 6-8 is a load-test claim, and an open-loop generator with percentile reporting ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) is what makes it credible.
6. **Read `vllm/distributed/` with lesson 3 open.** The recognition moment — your own `all_reduce` after `RowParallelLinear`'s partial sum — is the same payoff as reading the block manager in Phase 4.

**Time budget:** 3-4 weeks part-time. Lessons 1, 3 and 6 are load-bearing; lesson 10 produces the artifact.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Given a model size, GPU memory and GPU count, **calculate whether you need TP and the minimum number of GPUs** — including KV headroom, not just weights. ([lesson 1](01-when-one-gpu-isnt-enough.md))
2. Explain **why prefill/decode disaggregation improves goodput** even when it doesn't improve raw throughput. ([lesson 6](06-disaggregated-prefill-decode.md))
3. Draw the tensor-parallel MLP and attention block and **point at the two all-reduces**, then state the per-decode-step communication cost for a 70B model. ([lesson 3](03-tensor-parallelism.md))
4. Explain why **TP stays within a node and PP crosses nodes**, in bytes per step. ([lessons 2](02-collectives-and-interconnects.md), [3](03-tensor-parallelism.md), [4](04-pipeline-parallelism.md))
5. Explain how **round-robin load balancing breaks prefix caching**, and describe a routing policy that doesn't — including its failure mode. ([lesson 7](07-prefix-aware-routing.md))

## Projects that belong to this phase

- **[11 — Manual tensor parallelism with `torch.distributed`](../../projects/README.md)** (small): shard a feedforward and an attention layer across ranks, verify against the unsharded forward pass. Demystifies TP completely.
- **[12 — Multi-replica serving + prefix-aware sticky load balancer](../../projects/README.md)** (large): TP-served model, multiple replicas, a router implementing consistent-hash/prefix-aware routing, with a measured cache-hit-rate and TTFT comparison against round-robin. **This is the Phase 6 exit artifact.**

Also do the [Phase 6 labs](../../labs/README.md#phase-6-lab--distributed-inference) — the Megatron `ColumnParallelLinear` read pairs directly with lesson 3.

---

Next after this: **[Phase 7 — Observability, Reliability, and Cost](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)**. Phase 6 makes a fleet exist; Phase 7 is how you know it's healthy, what it costs per million tokens, and who gets paged when it isn't.
