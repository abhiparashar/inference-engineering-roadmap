# 4 — Pipeline Parallelism and the Bubble

> **You'll be able to say:** "PP cuts the model by layers, so each stage boundary moves exactly one activation tensor — `microbatch_tokens × hidden × 2` bytes, point-to-point — which is 100-1000× cheaper than TP's per-layer all-reduce and is why PP is the tool for crossing a network. The price is a bubble of `(P−1)/(M+P−1)`, paid unless you keep at least `P` microbatches in flight, and the fact that PP never reduces per-token latency: stages run in series. It buys *capacity* and *throughput*, and the fat first/last stage is what usually caps it."

TP ([lesson 3](03-tensor-parallelism.md)) splits every matrix; PP splits the stack of layers. They compose, and the standard frontier layout — `TP = NVLink domain, PP = nodes` — falls directly out of comparing their communication volumes.

---

## The mechanism

```
   L = 80 layers, P = 4 stages, one replica spanning 4 GPUs (or 4 nodes)

   STAGE 0            STAGE 1            STAGE 2            STAGE 3
   embed              layers 20-39       layers 40-59       layers 60-79
   layers 0-19                                              final norm
                                                            lm_head + sampling
      │  h [tok, H]        │  h                 │  h            │
      └──── send/recv ─────┴──── send/recv ─────┴─── send/recv ──┘
            ONE tensor per microbatch per boundary; no group sync, no reduction

   per-rank weights:  W / P          (and W / (T·P) if you also use TP inside a stage)
   per-rank KV-cache: the KV for THIS STAGE'S LAYERS ONLY  → aggregate KV still adds up
```

Communication volume, for a 70B (`H = 8192`, FP16 → 16 KB/token):

| Workload | Bytes per boundary crossing | On 400 Gb IB (`α≈20 µs`, 25 GB/s) |
|---|---|---|
| Decode, microbatch of 8 seqs | 128 KB | ~25 µs |
| Decode, microbatch of 32 | 512 KB | ~40 µs |
| Prefill, 2k prompt (chunk 512) | 8 MB | ~340 µs |

Compare against TP's `2 × 80 = 160` all-reduces per step. **A decode step under PP=4 pays 3 hops ≈ 75-120 µs of communication; the same step under cross-node TP pays ~4 ms.** That single comparison is the entire justification for the standard hybrid layout.

---

## The bubble

A pipeline is only busy when every stage has work. Feed it one batch and `P−1` stages idle at any instant.

```
  P = 4 stages, M = 4 microbatches (time →), inference = forward only

  stage0 │ 1 2 3 4 · · ·        │  ← drains, then idles
  stage1 │ · 1 2 3 4 · ·        │
  stage2 │ · · 1 2 3 4 ·        │
  stage3 │ · · · 1 2 3 4        │  ← fills late
         └──────────────────────┘
           fill = P−1 slots      drain = P−1 slots
           bubble fraction = (P−1)/(M+P−1) = 3/7 = 43%
```

| P | M = 1 | M = 4 | M = 8 | M = 16 | M = 32 |
|---|---|---|---|---|---|
| 2 | 50% | 20% | 11% | 6% | 3% |
| 4 | 75% | 43% | 27% | 16% | 9% |
| 8 | 88% | 64% | 47% | 30% | 18% |

**Inference is much luckier than training here**, for two reasons worth stating precisely:

1. **There is no backward pass**, so no 1F1B interleaving is needed and the bubble is only fill/drain, not the deep training bubble.
2. **Continuous batching supplies the microbatches for free.** In a decode-heavy engine there is always a stream of steps to issue; the engine keeps `P` (or more) batches in flight so every stage is working on a different one. vLLM implements exactly this with *virtual engines*: `pipeline_parallel_size` in-flight scheduler batches, each at a different stage.

So the operative rule is not "avoid PP," it's:

> **PP needs at least `P` independent in-flight batches to hide the bubble. If your traffic is too thin to fill `P` microbatches, you pay the bubble as pure latency — and PP at low load is strictly worse than TP.**

And the microbatching tradeoff cuts both ways: splitting a batch of 64 into `M = 8` microbatches of 8 shrinks the bubble but also shrinks every GEMM, which hurts exactly when decode is already memory-bandwidth-bound ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)). Sweep `M`; don't assume more is better.

---

## Latency vs throughput: the honest accounting

```
  per-token latency (pipe full)  =  Σ_stages t_stage  +  (P−1) · t_hop
                                 =  (same total compute as TP1)  +  hops
  steady-state throughput        =  1 / max_stage(t_stage)          ← the SLOWEST stage
```

- **PP does not lower the decode bandwidth floor.** Each stage reads its own `W/P` bytes, but the stages run *in series* for a given token, so the per-token weight traffic is unchanged: `W/(T·BW)` from [lesson 1](01-when-one-gpu-isnt-enough.md), with no `P` in it. **If your problem is TPOT, PP is not the answer.**
- **PP does raise throughput** — up to `P×`, minus the bubble — because different microbatches occupy different stages simultaneously.
- **PP does raise capacity**, exactly like TP: aggregate KV budget is `(T·P) × usable − W`, since each stage stores only its layers' KV.
- **`max_stage` is what you actually tune.** Throughput is set by the slowest stage, so stage balance is the whole game.

### Stage imbalance is the usual bug

Stage 0 owns the embedding; the last stage owns the final norm, the LM head *and* sampling — and the LM head is a `[H, vocab]` GEMM plus a 128k-wide softmax/top-p over every sequence in the microbatch. With an even layer split, the last stage is measurably slower, so it caps throughput and the other `P−1` stages idle proportionally.

Fixes, in order of preference:

1. **Uneven layer partitioning** — give the last stage fewer transformer layers (vLLM honours `VLLM_PP_LAYER_PARTITION`; TensorRT-LLM and Megatron take explicit partition specs).
2. **Combine with TP inside the stage** so the LM head is vocab-parallel and its cost drops by `T`.
3. **Measure per-stage step time before tuning anything else.** A 15% imbalance is 15% off your whole replica's throughput, which is larger than most kernel wins you'll ever chase.

---

## When to use PP (and when not)

| Situation | Use | Why |
|---|---|---|
| Model doesn't fit in one NVLink domain (405B FP16, 16+ GPUs) | `TP = 8, PP = N/8` | cross-node all-reduce is 3-12× slower per collective; send/recv isn't |
| Nodes connected by Ethernet / no RDMA | PP over the slow link, TP inside | PP moves ~0.5 MB per token per boundary; TP would move that 160× |
| Throughput-oriented, high concurrency | PP fine | plenty of in-flight batches to fill the pipe |
| Strict TTFT/TPOT SLO, low concurrency | **avoid PP** | bubble is unhidden; hops add latency; TP or a smaller/quantized model instead |
| Heterogeneous GPUs (mixed fleet) | PP with uneven partitions | assign fewer layers to weaker GPUs — TP requires symmetric ranks |
| Model fits in one GPU | neither | replicate ([lesson 1](01-when-one-gpu-isnt-enough.md), reason 4) |

Worked layout for Llama-3.1-405B FP16 (`W = 810 GB`, `L = 126`, `H = 16384`) on 16×H100:

```
  both layouts: per-rank weights 810/16 ≈ 51 GB → per-GPU read time ≈ 51/2.7 ≈ 18.7 ms
  aggregate KV (either layout) = 16 · 70 − 810 = 310 GB

  DECODE step, batch 32
   TP16 across both nodes: 2 · 126 · α_IB = 252 × 25 µs ≈ 6.3 ms comm
                           → step ≈ 25 ms; one batch in flight
   TP8 × PP2:              2 · 63 · α_NVL = 126 × 8 µs ≈ 1.0 ms comm per stage
                           + 1 send/recv per microbatch (32 × 16384 × 2 B = 1 MB ≈ 60 µs)
                           → 19.7 ms per stage; token latency ≈ 2 × 19.7 ≈ 39 ms,
                             but TWO batches in flight → ~19.7 ms/batch throughput (1.3× TP16)
   ⇒ on decode alone it's a real trade: TP16 has ~1.6× better per-token latency,
     TP8×PP2 has ~1.3× better throughput per GPU.

  PREFILL, 2k prompt — this is what decides it
   all-reduce size = 2048 × 16384 × 2 B = 67 MB  → BANDWIDTH-bound (lesson 2)
   compute ≈ 2 · 2048 · 405e9 / (16 × 700 TFLOP/s) ≈ 148 ms
   TP16:      252 × 67 MB / 25 GB/s  ≈ 676 ms comm  → TTFT ≈ 824 ms (5× compute!)
   TP8 × PP2: 126 × 67 MB / 230 GB/s ≈  37 ms/stage ≈ 74 ms + one 67 MB hop (2.7 ms)
                                                     → TTFT ≈ 225 ms
   ⇒ TP8 × PP2 wins by ~3.7× on TTFT, because cross-node TP puts a 67 MB all-reduce
     on a 25 GB/s fabric 252 times per prompt. One line of arithmetic, not a benchmark.
```

Also note `126 / 2 = 63` layers per stage — check `L % P` before promising a layout; a remainder becomes an imbalanced stage.

---

## Interleaved / virtual stages, and what inference borrows

Training reduces the bubble further by giving each device several *non-contiguous* chunks of layers (virtual pipeline stages), which shrinks the bubble to `(P−1)/(M·v+P−1)` at the cost of `v×` more boundary crossings. Inference rarely needs this: the forward-only bubble is already small once continuous batching supplies microbatches, and the extra crossings are pure latency. The idea worth borrowing is the *scheduling* one — vLLM's virtual engines are the same trick applied to whole scheduler batches instead of layer chunks.

---

## Practical gotchas

- **PP + continuous batching must cooperate.** A naive PP implementation schedules one batch at a time and eats the full bubble. Real engines keep `P` batches in flight, which means `P` sets of scheduler state, `P` sampling buffers, and a subtle consequence: **a request's tokens are produced `P` steps apart in wall-clock terms**, so per-token latency jitter rises with `P`. Watch TPOT *variance*, not just its mean.
- **Recv is a blocking dependency.** If stage 2 is slow, stages 0-1 fill their send queues and block; the whole replica runs at the slowest stage. There is no partial degradation — that's the same fate-sharing property TP has, and it's why [lesson 9](09-multi-node-operations.md)'s straggler detection matters.
- **First token vs later tokens.** Prefill flows through all `P` stages before the first token appears, so TTFT includes `P−1` hops plus any fill bubble. With chunked prefill the chunks themselves microbatch nicely, which is the same reason chunked prefill helps TP.
- **Warmup is per-stage.** CUDA graph capture happens per stage per shape; startup time grows with `P` and the replica isn't ready until the last stage finishes.
- **Debugging looks like a hang.** A shape or dtype mismatch at a boundary manifests as `recv` waiting forever. Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and a finite `NCCL_TIMEOUT` so it becomes an error with a rank number.
- **TGI doesn't do PP**; vLLM (`--pipeline-parallel-size`), TensorRT-LLM (`--pp_size` at build time) and Megatron/DeepSpeed do. If a framework choice constrains your layout, that's a Phase 5 decision leaking into Phase 6 — note it in your sizing doc.

---

## Read the code (~30 minutes)

1. **`vllm/distributed/parallel_state.py`** — find `get_pp_group()` and its `send`/`recv` and `send_tensor_dict`/`recv_tensor_dict` helpers. Note that PP uses point-to-point on the PP group while TP uses collectives on the TP group: two independent process groups over the same ranks.
2. **`vllm/model_executor/models/llama.py`** — find `make_layers` / the `start_layer`/`end_layer` bookkeeping and `IntermediateTensors`. This is literally "which layers do I own, and what do I hand to the next stage."
3. **`vllm/v1/engine/`** (or `vllm/engine/` on older versions) — find where `pipeline_parallel_size` in-flight batches are managed. Confirm the "≥ P in-flight batches" claim in the source rather than trusting this lesson.
4. **Megatron's `schedules.py`** — skim `forward_backward_pipelining_without_interleaving` to see the fill/steady/drain structure explicitly, then note that inference only ever runs the forward half.

---

## Do this now (45 minutes, CPU is fine)

1. **Two-stage pipeline by hand.** Take any small HF decoder model, split its `layers` list in half across two `gloo` processes. Stage 0: embed + first half, then `dist.send` the hidden state. Stage 1: `dist.recv`, second half, norm, LM head. Assert the logits match the single-process model within tolerance. That's PP, complete, in about 60 lines.
2. **Measure the bubble.** Feed it `M = 1, 2, 4, 8, 16` microbatches of the same total token count and plot total wall time. Fit against `(P−1)/(M+P−1)` and explain the residual (per-hop `α`, and shrinking GEMMs at large `M`).
3. **Create imbalance on purpose.** Move 20% of the layers from stage 1 to stage 0 and re-measure throughput. Confirm it tracks `1/max_stage`, then find the split that balances it. Write down both numbers — "we found a 15% throughput win by rebalancing PP stages" is a real production result and it comes from this 10-minute experiment.

---

**Next:** [Hybrid layouts, MoE and expert parallelism →](05-hybrid-layouts-and-moe.md) — `DP × TP × PP × EP` as a search over one memory constraint and three communication costs, and why MoE turns the MLP into two all-to-alls.
