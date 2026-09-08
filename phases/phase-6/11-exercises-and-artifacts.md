# 11 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 5 proved you can operate other people's engines. Phase 6 proves you can **make one model into a fleet** — and that when someone asks "how many GPUs, in what layout, and what does it cost," you answer with arithmetic first and a measurement second.

Almost all of this is laptop work: the sizing arithmetic, the sharding proofs (`gloo` across CPU processes), the router, the simulators, the runbook. A rented 2-GPU box for a few hours covers everything else, and `NCCL_P2P_DISABLE=1` simulates a bad fabric on good hardware ([lesson 2](02-collectives-and-interconnects.md)).

Keep the code, configs and numbers in `labs/phase6/`.

---

## Warm-up exercises

**Arithmetic and sizing** — do these before touching a GPU

1. **The sizing script** ([1](01-when-one-gpu-isnt-enough.md)): HF config in, out comes `W`, `kv_per_token`, minimum `TP × PP`, aggregate KV, concurrent sequences at your context length, the decode bandwidth floor, and every divisibility check. Run it on Llama-3-8B, -70B, Llama-3.1-405B and Mixtral-8x7B.
2. **Predict the engine's block count** ([1](01-when-one-gpu-isnt-enough.md)): for one model/GPU pair, predict vLLM's reported `# GPU blocks` before launching. Explain any gap over 15% by naming the term in the ledger you got wrong.
3. **The fit-vs-serve gap** ([1](01-when-one-gpu-isnt-enough.md)): find a model+GPU where TP is *sufficient to load* but leaves under 50 concurrent sequences at 4k context. State the two ways to fix it and their costs.
4. **Quantization as a parallelism decision** ([1](01-when-one-gpu-isnt-enough.md), [Phase 4 lesson 3](../phase-4/03-quantization-methods.md)): for 70B, table TP degree, aggregate KV and concurrency for FP16 / FP8 / INT4 weights × FP16 / FP8 KV. Which cell would you ship, and what quality check gates it?
5. **The layout enumerator** ([5](05-hybrid-layouts-and-moe.md)): all `(DP, TP, PP)` with `DP·TP·PP ∈ {8, 16}`, annotated with fit, KV, latency floor, failure-domain size and predicted tokens/sec/GPU. Do it for a dense 70B and an MoE; explain why the MoE rows look nothing alike.

**Collectives and parallelism**

6. **Your two constants** ([2](02-collectives-and-interconnects.md)): `α` and `B_alg` from `all_reduce_perf` (or `gloo` on CPU). Compute your latency/bandwidth crossover and mark where decode (batch 1, 32, 256) and prefill (512, 2k, 8k) fall.
7. **Break the fabric** ([2](02-collectives-and-interconnects.md)): re-run with `NCCL_P2P_DISABLE=1`, then `+NCCL_SHM_DISABLE=1`. Report the multiplier and translate it into "what cross-node TP would cost me."
8. **Read the NCCL log** ([2](02-collectives-and-interconnects.md)): from `NCCL_DEBUG=INFO`, state the transport actually chosen for each rank pair. Find one setting that changes it.
9. **Shard by hand** ([3](03-tensor-parallelism.md), [10](10-build-manual-tp-and-router.md)): column-parallel, row-parallel, the pair with **one** all-reduce, then GQA attention with the `tp > n_kv` replication path. Verify against the unsharded reference and count collectives.
10. **The negative result** ([10](10-build-manual-tp-and-router.md)): make `down_proj` column-parallel instead of row-parallel. Record that the output is wrong while the collective count is unchanged — and write one sentence on what test would catch this in CI.
11. **Per-rank shape table** ([3](03-tensor-parallelism.md)): for your target model at TP ∈ {2, 4, 8}, every weight's per-rank shape and bytes, summing to `W/TP`. Then find a real model whose `intermediate_size` breaks group-128 quantization at TP8.
12. **TP cost model vs reality** ([3](03-tensor-parallelism.md)): predict decode step time as `W/(TP·BW) + 2·L·α` for TP ∈ {1, 2, 4, 8}, then measure (or measure at the TP degrees you have and extrapolate). Report predicted vs measured and **throughput per GPU**.
13. **The bubble** ([4](04-pipeline-parallelism.md), [10](10-build-manual-tp-and-router.md)): split a small model across two processes, sweep microbatch count `M ∈ {1,2,4,8,16}`, fit against `(P−1)/(M+P−1)`, and explain the residual.
14. **Stage imbalance** ([4](04-pipeline-parallelism.md)): move 20% of layers between stages and show throughput tracking `1/max_stage`. Report the best split you found and the win in %.
15. **PP vs cross-node TP on paper** ([2](02-collectives-and-interconnects.md), [4](04-pipeline-parallelism.md)): for 405B on 16 GPUs across 2 nodes, compute decode step time *and* 2k-prompt TTFT for TP16 vs TP8×PP2 with your measured fabric constants. Which wins on which metric, and why?
16. **MoE expert coverage** ([5](05-hybrid-layouts-and-moe.md)): plot distinct experts touched vs batch size against `E·(1−(1−k/E)^B)`. Then state, in bytes, what a batch-64 MoE decode step reads compared to batch 1.
17. **EP imbalance** ([5](05-hybrid-layouts-and-moe.md)): toy expert-parallel layer with `all_to_all`, skew the router 80% to one expert, show step time tracking the hottest rank, then fix it by replicating that expert.

**Disaggregation and routing**

18. **Measure the interference you'd remove** ([6](06-disaggregated-prefill-decode.md)): steady decode load, p50/p99 TPOT; inject a long prompt every 2 s; re-measure; then enable chunked prefill and re-measure. Three numbers, one conclusion.
19. **The transfer budget** ([6](06-disaggregated-prefill-decode.md)): for your model and p50/p95 prompt lengths, compute KV transfer bytes and time on your fabric, as a ratio to prefill compute. Decide go/no-go and say why.
20. **Pool ratio** ([6](06-disaggregated-prefill-decode.md)): compute `n_prefill : n_decode` for four traffic shapes (RAG, chat, agent, summarization). Then simulate two queues and plot **goodput** vs pool ratio at fixed total GPUs.
21. **Round-robin's true cost** ([7](07-prefix-aware-routing.md)): with mock backends and a session-shaped workload, report prefill tokens computed under round-robin vs sticky for a 6-turn conversation. Predict the ratio first.
22. **The router** ([7](07-prefix-aware-routing.md), [10](10-build-manual-tp-and-router.md)): consistent hash → k candidates → least-loaded, with bounded loads and spill. Four-metric table for four policies including an oracle.
23. **The scale event** ([7](07-prefix-aware-routing.md)): add a replica mid-test with consistent hashing and again with `hash mod R`. Report rehash fraction (expect ≈`1/R` vs ≈`1`), TTFT spike height and duration.
24. **Sabotage the key** ([7](07-prefix-aware-routing.md)): hash a *non*-block-aligned prefix, then hash a prefix that includes a timestamp. Quantify how much hit rate each mistake costs.

**Autoscaling, cost and operations**

25. **`T_cold`, itemized** ([8](08-autoscaling-gpu-fleets.md)): six timestamps from pod-scheduled to prefix-cache-warm. Then name the two cheapest terms to cut on your stack.
26. **Why utilization lies** ([8](08-autoscaling-gpu-fleets.md)): record GPU utilization at batch 1 and at batch ≥64 with the same model. Put both numbers next to tokens/sec and write the one-sentence conclusion.
27. **Controller simulation** ([8](08-autoscaling-gpu-fleets.md)): diurnal curve + burst, four controllers (reactive, +headroom, predictive, +warm standby). Report SLO-violating minutes and GPU-hours for each.
28. **Cost per 1M tokens** ([8](08-autoscaling-gpu-fleets.md)): two layouts, measured tokens/sec at SLO, then re-priced with a 70/30 on-demand/spot mix and 30% headroom. One sentence on which you'd ship.
29. **Reproduce the hang** ([9](09-multi-node-operations.md)): kill one rank mid-collective; observe the hang; then make it a rank-attributed error with a timeout and the async error handler.
30. **Control-flow divergence** ([9](09-multi-node-operations.md)): make one rank issue an extra collective, enable the flight recorder, and identify the divergent sequence number from the dump.
31. **Straggler hunt** ([9](09-multi-node-operations.md)): lock one GPU's clocks low (or `sleep` in one rank), show replica throughput falling proportionally, and identify the rank from per-rank step times alone.
32. **The runbook** ([9](09-multi-node-operations.md)): one page — the env vars you always set, the five-step hang playbook, the Xid table, the straggler checklist, the seven chaos drills with measured time-to-recovery.

---

## Conceptual self-check (no notes)

1. Given a model, a GPU and an SLO, compute the minimum GPU count **including KV headroom**, and state the aggregate-KV formula. ([1](01-when-one-gpu-isnt-enough.md))
2. Name the four reasons to use more than one GPU and the correct first move for each. Which three are answered by replicas? ([1](01-when-one-gpu-isnt-enough.md))
3. Why does going from TP4 to TP8 more than double the KV budget? ([1](01-when-one-gpu-isnt-enough.md))
4. Name the seven collectives and the bytes each moves per rank. Which one does TP use, which PP, which EP? ([2](02-collectives-and-interconnects.md))
5. Why is a decode all-reduce latency-bound and a prefill all-reduce bandwidth-bound, and where is the crossover on your hardware? ([2](02-collectives-and-interconnects.md))
6. Draw the TP transformer layer and point at both all-reduces. Why does attention itself need none? ([3](03-tensor-parallelism.md))
7. Derive the per-rank shapes of `q/k/v/o` and `gate/up/down` at TP8, and state the three divisibility constraints. ([3](03-tensor-parallelism.md))
8. Why is TP scaling sublinear, and why does it look *worse* at batch 1 than at batch 128? ([3](03-tensor-parallelism.md))
9. Why must TP stay inside the NVLink domain? Answer in bytes and microseconds. ([2](02-collectives-and-interconnects.md), [3](03-tensor-parallelism.md))
10. What does PP cost (bubble formula) and what does it *not* buy? ([4](04-pipeline-parallelism.md))
11. Why is PP the right tool for crossing a network while TP isn't? ([4](04-pipeline-parallelism.md))
12. Why is the last pipeline stage usually the bottleneck, and what are the two fixes? ([4](04-pipeline-parallelism.md))
13. For an MoE model, why is memory set by total parameters but FLOPs by active parameters — and what happens to bytes read as batch size grows? ([5](05-hybrid-layouts-and-moe.md))
14. Contrast TP and EP for an MoE layer: collectives, GEMM shapes, and failure mode. ([5](05-hybrid-layouts-and-moe.md))
15. Why does attention-DP + expert-EP exist, and what does it require of the scheduler? ([5](05-hybrid-layouts-and-moe.md))
16. Define goodput, and explain why disaggregation raises it even when raw throughput doesn't move. ([6](06-disaggregated-prefill-decode.md))
17. Compute the KV transfer for a 2k prompt on a 70B model and say when that cost kills the design. Name three cases where disaggregation is the wrong answer. ([6](06-disaggregated-prefill-decode.md))
18. How does round-robin destroy a prefix cache? Give a routing policy that doesn't, and its failure mode. ([7](07-prefix-aware-routing.md))
19. Why is GPU utilization the wrong autoscaling signal, and what three signals replace it? Size the headroom for a `T_cold` of 4 minutes. ([8](08-autoscaling-gpu-fleets.md))
20. **The keystone question:** *"We're launching a 70B chat product: 200 concurrent sessions, 3k average context, p99 TTFT under 1 s, TPOT under 40 ms, and I want the cheapest thing that holds. Design it."* Answer as a derivation: KV per token → required KV → minimum `TP × PP` with divisibility → latency floor check → replicas from measured per-replica QPS → routing policy and why → autoscaling signal, headroom and `T_cold` → cost per 1M tokens → the failure domain you accepted and the drill that proves recovery. Then name the one measurement that would change the design. ([1](01-when-one-gpu-isnt-enough.md), [3](03-tensor-parallelism.md), [7](07-prefix-aware-routing.md), [8](08-autoscaling-gpu-fleets.md), [9](09-multi-node-operations.md))

Question 20 is the phase. A strong answer starts with arithmetic, not with a framework name, and it says out loud which cheaper option it rejected and why.

---

## Exit artifact

Produce **Option A**. A + B is the strongest pair; C is cheap and disproportionately useful in interviews and design reviews.

### Option A — Project 12: multi-replica serving + prefix-aware router (required)

`projects/12-sticky-load-balancer/README.md` with:

- **The router**: `ring.py`, `router.py`, health checking, `/metrics`, and the policy in five readable lines with `k` and `ε` as config.
- **The workload generator**: session-shaped (multi-turn, growing prompts, Zipf system prompts, think time, Poisson arrivals, fixed seed) **plus** the single-turn unique-prompt control.
- **The four-metric table** — prefix hit rate, TTFT p50/p99, load imbalance, goodput at SLO — for round-robin, pure hash, two-level, and an oracle, on mocks **and** on ≥2 real vLLM replicas.
- **Four sweeps**: `k`, `ε`, load ladder, and the scale event with rehash fraction and TTFT-spike duration.
- **Predicted vs measured**: your predicted hit rate from the workload's prefix-sharing structure, next to the engines' `vllm:gpu_prefix_cache_*` counters, with the gap attributed to eviction/staleness.
- **A limits section**: what your router does *not* do (KV-event awareness, disaggregation-aware routing, multi-region, fairness across tenants).
- **Reproducibility**: versions, exact commands, seeds, `results.jsonl`.

### Option B — Project 11: manual tensor parallelism (small, do it first)

`projects/11-manual-tensor-parallel/README.md`: the sharded linear pair, GQA attention with the `tp > n_kv` replication path, verification output at TP ∈ {1, 2, 4}, the collective count and byte count matching `2·L` and `2·L·T·H·dtype`, the wrong-sharding negative result, and — if you had a GPU — your measured per-collective `α` beside the `nccl-tests` number. ([Lesson 10](10-build-manual-tp-and-router.md) Part A.)

### Option C — The sizing document + runbook (cheap, high signal)

Two pages for one real model on real hardware: (1) a **deployment sizing doc** — ledger, layout choice with the rejected alternatives and *why*, KV budget, concurrency, latency floor, replica count, cost per 1M tokens; (2) the **distributed runbook** from [lesson 9](09-multi-node-operations.md) with measured time-to-recovery per drill. This is the document an inference team actually writes, and almost no candidate brings one.

---

## How you know you're ready for Phase 7

- [ ] You can size a deployment from arithmetic — weights, KV, aggregate KV, concurrency, latency floor — and check it against the engine's startup log.
- [ ] You can name the four reasons to scale and pick the right mechanism for each, including when the answer is "just replicas."
- [ ] You can draw the TP layer, point at both all-reduces, and state the per-step communication cost for a real model.
- [ ] You have sharded a layer yourself and proved it matches the unsharded forward pass.
- [ ] You can explain the PP bubble and why PP crosses nodes while TP doesn't, in bytes.
- [ ] You can explain MoE's memory-vs-FLOPs split and what EP's all-to-alls cost.
- [ ] You can justify (or reject) prefill/decode disaggregation with a transfer-budget calculation and a goodput argument.
- [ ] You have a router that beats round-robin on hit rate and can state what it cost you in load balance.
- [ ] You can design a GPU autoscaler that survives a 4-minute cold start, and name the signals it uses.
- [ ] You can debug a NCCL hang: env vars, per-rank stacks, flight recorder, and the fail-fast policy.
- [ ] Your exit artifact is committed.

Then go to **[Phase 7 — Observability, Reliability, and Cost](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)**. Phase 6 built a fleet; Phase 7 is how you prove it's healthy, define the SLOs it's judged against, alert before it breaks, and account for the cost per million tokens that every decision in this phase moved.

---

## Where these ideas come back

| Phase 6 idea | Comes back as |
|---|---|
| The memory ledger and aggregate-KV formula | Every capacity plan, every "how many GPUs" question, forever |
| `α + β·bytes` for collectives | Debugging any distributed slowdown; MoE and long-context design |
| TP/PP/EP mechanics | Reading engine source, model bring-up, hardware selection |
| Goodput over throughput | SLOs and error budgets (Phase 7) |
| Prefix-aware routing | Gateways, multi-tenant fairness, RAG serving (Phases 7, 9) |
| Cold start and pre-warming | Autoscaling, canaries, spot strategy (Phases 7-8) |
| Cost per 1M tokens | The unit economics metric your work is judged on (Phase 7, capstones) |
| Fate sharing and chaos drills | Incident response, rollout design, on-call (Phases 7-8) |
| The runbook habit | The thing that makes you the person who gets paged *and* fixes it |

Phase 5 made you dangerous inside one engine. Phase 6 made one model into a fleet with numbers behind every decision. Phase 7 makes that fleet *accountable*.
