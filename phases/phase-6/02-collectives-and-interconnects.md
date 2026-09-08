# 2 — Collectives and Interconnects

> **You'll be able to say:** "There are seven collectives; I know the bytes each moves per rank and which parallelism dimension uses it. Every fabric costs `α + β·bytes`, where `α` is a per-collective latency floor of ~5-30 µs and `β⁻¹` is an effective bandwidth spanning 25 GB/s (PCIe) to 450 GB/s (NVLink). The crossover between those two terms sits around 1-2 MB, which is why a *decode* all-reduce is latency-bound and pays per layer, while a *prefill* all-reduce is bandwidth-bound and pays per token. I can measure both constants on my hardware with `nccl-tests` and predict any distributed layout's overhead from them."

This is the lesson that makes every later lesson calculable. Parallelism strategies are just different answers to "which collective, how often, how big."

---

## The fabric hierarchy is the memory hierarchy with two more rungs

[Phase 2 lesson 2](../phase-2/02-gpu-memory-hierarchy.md) taught the on-GPU hierarchy. Distribution extends it downward, and the same "each rung is ~10× worse" intuition holds:

| Rung | Typical bandwidth (per GPU) | Latency | Notes |
|---|---|---|---|
| SRAM / shared memory | ~10-20 TB/s | ~ns | Phase 2 |
| HBM3 (H100) | 3.35 TB/s spec, ~2.7 effective | ~350-600 ns | the decode bottleneck |
| NVLink 4 + NVSwitch (H100 SXM) | 900 GB/s bidirectional (450 each way) | ~2-3 µs raw | all-to-all inside an 8-GPU node |
| NVLink 3 (A100 SXM) | 600 GB/s bidirectional | ~2-3 µs | |
| NVLink 5 (B200) | 1.8 TB/s bidirectional | ~2 µs | |
| PCIe Gen5 x16 | 128 GB/s bidi (~50 GB/s practical/direction) | ~5-10 µs | PCIe-only boxes, no NVLink |
| PCIe Gen4 x16 | 64 GB/s bidi (~25 GB/s practical/direction) | ~5-10 µs | the common "cheap 8-GPU server" |
| InfiniBand NDR (400 Gb/s) | 50 GB/s per NIC per direction | ~2-3 µs raw, 15-30 µs per NCCL collective | 8 NICs/node on DGX-class = GPU-to-NIC "rails" |
| RoCE / 100-200 GbE | 12-25 GB/s per NIC | tens of µs, jitter-prone | most cloud non-DGX instances |

*(Spec-sheet numbers; effective throughput is 60-85% of spec. Verify on your own hardware — step 1 of "Do this now.")*

**The ratio that decides your layout: NVLink is roughly 9× the bandwidth of one 400 Gb/s NIC, and PCIe Gen4 is roughly half a NIC.** Everything in lessons 3-6 follows from that one comparison.

Two hardware facts that bite:

- **"8 GPUs" is not a topology.** An 8×H100 SXM node with NVSwitch gives every pair full NVLink bandwidth. An 8×H100 *PCIe* node gives you PCIe, possibly through a host bridge, possibly across NUMA sockets — a 10-20× difference on exactly the collective TP does most. Always run `nvidia-smi topo -m` before you believe a benchmark.
- **GPUDirect RDMA matters as much as link speed.** Without it, cross-node traffic bounces GPU → host memory → NIC, adding copies and latency. `NCCL_DEBUG=INFO` tells you which path you got; the absence of `[send] via NET/IB/x/GDRDMA` in the log is a performance bug worth hours.

---

## The seven collectives

Let `S` = the size of the buffer each rank contributes/holds, `N` = ranks. Ring-algorithm volumes (what NCCL uses for large messages):

| Collective | Semantics | Bytes sent per rank | Used by |
|---|---|---|---|
| **all-reduce** | every rank ends with the elementwise sum of all inputs | `2·(N−1)/N · S` | **TP** (after row-parallel matmuls), gradient sync in training |
| **reduce-scatter** | sum, then each rank keeps `1/N` of the result | `(N−1)/N · S` | sequence parallelism, ZeRO |
| **all-gather** | each rank's shard is concatenated on every rank | `(N−1)/N · S` | sequence parallelism, vocab-parallel logits, weight gathering |
| **all-to-all** | rank *i* sends a distinct chunk to every rank *j* | `(N−1)/N · S` but *pairwise* — bisection bandwidth bound | **EP** (MoE token dispatch/combine) |
| **broadcast** | one rank's buffer copied to all | `S` from the root (tree: `log N` hops) | sampling params, sampled token IDs, weight load |
| **reduce** | sum landing on one rank | `S` toward the root | logprob aggregation, rare in serving |
| **send/recv** (point-to-point) | one rank to one rank | `S` | **PP** (stage boundary activations), disaggregated KV transfer |

Three identities worth memorizing:

1. **`all-reduce = reduce-scatter + all-gather`.** That's why all-reduce moves *twice* the bytes of either half, and why sequence parallelism can replace one all-reduce with a reduce-scatter + all-gather pair at **no extra volume** while cutting activation memory (details in [lesson 3](03-tensor-parallelism.md)).
2. **All-to-all is the only collective whose cost depends on the *fabric's bisection bandwidth*, not just per-link bandwidth.** This is why MoE expert parallelism is far more sensitive to topology than TP ([lesson 5](05-hybrid-layouts-and-moe.md)).
3. **Point-to-point is absurdly cheap by comparison.** PP and disaggregation are built on send/recv precisely because a single directed transfer needs no group-wide synchronization — no straggler couples all ranks together.

---

## The cost model: `α + β·bytes`

```
   t_collective  ≈  α  +  bytes_moved / B_alg

   α       per-collective latency floor: kernel launch + group sync +
           wire latency + (for IB) NIC doorbell/completion.  5-10 µs intra-node,
           15-30 µs inter-node, and it grows with rank count and hop count.

   B_alg   "algorithmic bandwidth" as reported by nccl-tests: buffer size / time.
           For a ring all-reduce, B_alg ≈ B_bus · N / (2(N−1))  ≈ 0.57 · B_bus at N=8.
```

Measured order-of-magnitude constants on current hardware (fill in your own):

| Fabric / group | `α` | `B_bus` | `B_alg` (all-reduce) | Latency/bandwidth crossover |
|---|---|---|---|---|
| 8×H100 NVSwitch | ~6-10 µs | ~350-450 GB/s | ~200-260 GB/s | **~1.5-2.5 MB** |
| 8×A100 NVSwitch | ~7-12 µs | ~230-270 GB/s | ~130-155 GB/s | ~1-2 MB |
| 8×GPU PCIe Gen4 | ~15-25 µs | ~20-25 GB/s | ~12-15 GB/s | ~0.3 MB |
| 2 nodes × 8 GPUs, 8×400 Gb IB | ~20-35 µs | ~40-45 GB/s | ~25 GB/s | ~0.7 MB |
| 2 nodes, single 100 GbE | ~50-150 µs | ~10 GB/s | ~6 GB/s | ~0.5 MB |

**The crossover is the single most useful number in this lesson.** Below it, a collective costs `α` and nothing else — making the message smaller buys you *nothing*, and making it larger is nearly free. Above it, cost is linear in bytes and the fabric's bandwidth is what you're paying for.

### Which regime is inference in?

A TP all-reduce carries `tokens_in_step × hidden × dtype_bytes`. For a 70B model (`hidden = 8192`, FP16 → 16 KB per token):

| Phase | Tokens in step | All-reduce size | Regime | Cost per collective (8×H100) |
|---|---|---|---|---|
| Decode, batch 1 | 1 | 16 KB | latency | ~8 µs (pure `α`) |
| Decode, batch 32 | 32 | 512 KB | latency | ~8-10 µs |
| Decode, batch 256 | 256 | 4 MB | mixed | ~25-30 µs |
| Prefill, 2k prompt | 2048 | 32 MB | **bandwidth** | ~130-160 µs |
| Prefill, chunked 512 | 512 | 8 MB | bandwidth | ~35-45 µs |

Multiply by `2 × num_layers` collectives per forward pass (160 for a 70B):

```
 decode step, batch ≤ 32:   160 × ~8 µs   ≈  1.3 ms   of pure latency, per token,
                                             INDEPENDENT of batch size
 prefill, 2k prompt:        160 × ~145 µs ≈  23 ms    of bandwidth, per prompt
```

Two conclusions that explain most of what TP does:

- **Decode's TP tax is a fixed per-step cost of `2 · L · α`.** It does not shrink when you add GPUs (α is roughly constant) and it does not grow with batch size (still under the crossover). So **TP overhead is amortized by batching**: at batch 1 that 1.3 ms sits against a ~6.5 ms compute floor (≈20% overhead); at batch 128 it sits against a much longer step and becomes noise. "TP is inefficient" is a statement about small batches.
- **Prefill's TP tax is bandwidth**, and it is proportional to prompt tokens — the same scaling as prefill compute, so the *ratio* stays roughly constant (~20-30% on NVLink, and catastrophic off it).

And the reason for the rule "TP inside the node, PP across nodes":

```
  same 70B decode step, 160 collectives:
    NVLink   : 160 × ~8 µs    ≈  1.3 ms      → ~20% over a 6.5 ms compute floor
    IB (x8)  : 160 × ~25 µs   ≈  4.0 ms      → ~60%; TP16 across 2 nodes loses to TP8
    100 GbE  : 160 × ~100 µs  ≈ 16 ms        → the fabric IS your latency
  vs PP across the same nodes: 1 send/recv per stage boundary per microbatch,
    32 seqs × 8192 × 2 B = 512 KB → ~20 µs + 20 µs = ~40 µs per token. 100× cheaper.
```

---

## Topology: what "inside a node" means

```
  DGX/HGX-style 8-GPU node                      Cross-node ("rail-optimized")
  ┌───────────────────────────────┐              node A            node B
  │  GPU0 ─┐                      │            GPU0─NIC0 ══════ NIC0─GPU0
  │  GPU1 ─┤                      │            GPU1─NIC1 ══════ NIC1─GPU1
  │   ...  ├── NVSwitch ── all     │            GPU2─NIC2 ══════ NIC2─GPU2   ← each GPU
  │  GPU7 ─┘   pairs at full      │             ...                          talks to its
  │            NVLink bandwidth   │            GPU7─NIC7 ══════ NIC7─GPU7    peer rank on
  │                               │                    ▲                     a dedicated
  │  each GPU also has a NIC ─────┼──▶ fabric          │                     "rail"
  └───────────────────────────────┘            leaf/spine switches
   TP domain = this box                        PP / DP / disaggregation cross this
```

- **The NVLink domain (usually 8 GPUs; 72 on NVL72-class racks) is the natural TP boundary.** Newer rack-scale NVLink fabrics move that boundary, which is precisely why "TP ≤ 8" is a hardware-generation fact, not a law.
- **Rail-optimized fabrics assume rank *i* on node A mostly talks to rank *i* on node B.** That's the pattern PP and TP-replicated DP produce naturally. All-to-all (MoE) violates it and hits the spine.
- **NUMA and NIC affinity are real.** A GPU pinned to the wrong socket's NIC loses half its cross-node bandwidth. `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` and `numactl` exist for this.

---

## Measuring and debugging

```bash
# 1. Topology first — always.
nvidia-smi topo -m                 # NV# = NVLink hops, PIX/PXB/SYS = PCIe paths

# 2. The two constants, from nccl-tests (github.com/NVIDIA/nccl-tests).
all_reduce_perf -b 8 -e 128M -f 2 -g 8    # read `algbw` at 8B (→ α) and at 64M (→ β)
alltoall_perf   -b 1M -e 64M -f 2 -g 8    # if you plan on MoE

# 3. Verify the path NCCL actually chose.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH python -c "..."   # look for
#   "via NVL" / "via P2P/IPC"        → good, intra-node
#   "via SHM"                        → host shared memory: no P2P, ~5× slower
#   "via NET/IB/3/GDRDMA"            → good, cross-node with GPUDirect
#   "via NET/Socket"                 → TCP: you are 10× off the hardware
```

| Env var | What it's for |
|---|---|
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS` | the first thing to set on any distributed problem |
| `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1` | force slow paths — **use these to simulate a bad fabric on good hardware** |
| `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` | pick the right NIC/interface; the fix for "it used the management NIC" |
| `NCCL_ALGO` / `NCCL_PROTO` | force Ring/Tree/NVLS, Simple/LL/LL128 — for diagnosis, rarely for production |
| `NCCL_NVLS_ENABLE` | NVLink SHARP: in-switch reduction, cuts all-reduce traffic on Hopper+ |
| `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `NCCL_TIMEOUT` | turn silent hangs into errors — see [lesson 9](09-multi-node-operations.md) |

**Two inference-specific realities the training world doesn't share:**

1. **You cannot hide the collectives behind compute.** Training overlaps gradient all-reduce with the backward pass; a TP forward pass has a hard data dependency — the next layer needs the reduced activation. The `1.3 ms` above is on the critical path. (Partial exception: chunked prefill and multi-stream designs can overlap *different requests'* work, which is one reason chunked prefill helps at high TP.)
2. **Collectives must be inside your CUDA graph.** At batch 1, 160 NCCL launches per step is exactly the overhead-bound regime from [Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md). This is why `--enforce-eager` costs *more* at TP8 than at TP1, and why vLLM ships a **custom all-reduce** kernel (`vllm/distributed/device_communicators/custom_all_reduce.py`): a one-shot/two-shot NVLink-only reduction that beats NCCL below a few MB — precisely the latency-bound decode regime this lesson identified. Read that file after [lesson 3](03-tensor-parallelism.md); it is 200 lines and it is this entire lesson made concrete.

---

## Do this now (45 minutes)

1. **Get your constants.** On any 2+ GPU machine, run `all_reduce_perf -b 8 -e 128M -f 2 -g <n>`. Record `algbw` at 8 B, 512 KB, 32 MB. Compute `α` (time at 8 B) and `B_alg` (32 MB / time). Then compute your crossover `α · B_alg` and check it against the table above. No GPU? Do the same with `torch.distributed` + `gloo` across CPU processes — the *shape* of the curve is identical, the constants are worse.
2. **Predict a real overhead.** Using your constants, predict the per-decode-step communication time for Llama-3-70B (`L=80`, `hidden=8192`) at TP=your GPU count, batch 1 and batch 64. Then predict the prefill cost for a 2k prompt. Write both down before lesson 3.
3. **Break the fabric on purpose.** Re-run `all_reduce_perf` with `NCCL_P2P_DISABLE=1` and then `NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1`. Note the multiplier. That multiplier is what "we put TP across nodes" or "we bought the PCIe box" costs you, and now you can quote it from your own measurement.

---

**Next:** [Tensor parallelism, from the matmul up →](03-tensor-parallelism.md) — column-parallel, row-parallel, and exactly where those two all-reduces per layer come from.
