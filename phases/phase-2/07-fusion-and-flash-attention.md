# 7 — Kernel Fusion & FlashAttention

> **You'll be able to say:** "Fusion means doing several operations in one kernel so intermediates never touch HBM. FlashAttention is the extreme case: the *same* attention math, computed in SRAM tiles with an online softmax, so the N×N score matrix is never written to memory at all. It isn't faster arithmetic — it's an IO-aware algorithm."

[Lesson 5](05-roofline-model.md) said the only useful direction on the memory roof is **right**: raise arithmetic intensity. [Lesson 6](06-overhead-bound-and-cuda-graphs.md) said fewer kernels means less launch overhead. Fusion is the one optimization that moves both at once, and this lesson is where you learn to reason about it precisely enough to predict a speedup before you measure it.

Then we do the famous one. If you understand *why* FlashAttention is fast, you understand this entire phase — and you can read the kernels in vLLM, TensorRT-LLM, and SGLang without being lost.

---

## Fusion, stated as arithmetic

[Lesson 2](02-gpu-memory-hierarchy.md) showed the mechanism: keep intermediates in registers instead of round-tripping HBM. Here's the version you can compute with. Take a chain of *k* memory-bound elementwise ops on a tensor of *B* bytes:

```
UNFUSED:  each op reads B, writes B          → traffic = 2kB
FUSED:    read B once, k ops in registers,   → traffic = 2B
          write B once
                                    speedup ≈ k
```

**The expected speedup from fusing a chain of *k* elementwise ops is *k*.** That's a prediction you can write down before running anything, and if you measure much less than *k*, something else is wrong (launch-bound, bad occupancy, non-contiguous input). Predicting then explaining the gap is the skill; the number itself is easy.

A transformer decoder layer in eager mode is full of these chains: `residual + x`, `norm`, `scale`, `silu(gate) * up`, `rope`, cast, mask, dropout. Each is a separate kernel reading and writing a full activation tensor. Fusing the layer's elementwise work is routinely a 1.3-2× end-to-end win on decode, and you now know the win is measured in **round trips**, not FLOPs.

---

## The four kinds of fusion you'll actually meet

| Kind | Example | What it saves |
|---|---|---|
| **Elementwise chain** | `add → mul → silu` in one pass | k-1 round trips; the easy, automatic case |
| **Reduction fusion** | LayerNorm/RMSNorm/softmax as one kernel instead of mean → sub → var → div | multiple passes over the same tensor |
| **Epilogue fusion** | `bias + GELU` folded into the GEMM's output stage | writing the raw GEMM result to HBM at all |
| **Whole-algorithm (IO-aware) fusion** | FlashAttention | the entire O(L²) intermediate |

The first three are what `torch.compile`/TorchInductor, TensorRT, and cuBLASLt epilogues do for you. The fourth requires rethinking the algorithm, which is why there are papers about it and not just compiler passes.

### What blocks fusion

Fusion isn't always possible, and knowing the boundaries is what separates "run `torch.compile` and hope" from engineering:

- **A materialization point.** If an intermediate is needed *whole* before the next step can start — a global reduction, a sort, a `.item()` — the kernel must end. This is why plain softmax needs two passes over the row (max, then sum) unless you use the trick in the next section.
- **Shape or layout changes** that force a real data movement (`transpose().contiguous()`, some reshapes) can't be folded away ([lesson 3](03-cuda-execution-model.md) — coalescing).
- **Graph breaks** in `torch.compile`: data-dependent Python control flow, unsupported ops, printing a tensor. Each break ends a fusion region. Diagnose with `TORCH_LOGS="graph_breaks,recompiles"`, and read the kernels it *did* generate with `TORCH_LOGS="output_code"` — that generated Triton is some of the best free reading material in this phase.
- **SRAM capacity.** A fused kernel needs its whole working tile on-chip in ~228 KB per SM. Too large a tile and you either spill to HBM (defeating the point) or wreck occupancy. This is the real constraint behind every tile-size choice in a real kernel.

---

## Standard attention, counted in bytes

Now the main event. Recall attention from [Phase 1 lesson 2](../phase-1/02-attention-mechanism.md): `O = softmax(QKᵀ / √d) V`. Here is how a naive implementation executes it, one head, sequence length `L`, head dim `d`, FP16:

```
1. S = Q @ Kᵀ          compute L×L scores      → WRITE  L²·2 bytes to HBM
2. S = S / √d + mask   elementwise             → READ  L²·2, WRITE L²·2
3. P = softmax(S)      row-wise                → READ  L²·2, WRITE L²·2
4. O = P @ V           back to L×d             → READ  L²·2
```

Plug in `L = 4096, d = 128`:

```
score matrix S      = 4096² × 2 B          =  33.5 MB   ← per head, per layer
HBM traffic for S/P ≈ 4 passes × 33.5 MB   = 134 MB
Q, K, V, O traffic  = 4 × 4096 × 128 × 2 B =  4.2 MB    ← the actual data
FLOPs               = 4 L² d               = 8.6 GFLOP

intensity = 8.6e9 ÷ 138e6 ≈ 62 FLOP/byte      (H100 ridge: ~296) → MEMORY-BOUND
```

**97% of the memory traffic is an intermediate the caller never asked for.** And it gets worse in two directions at once: the useful data grows like `L`, the garbage grows like `L²`. At `L = 16,384` one head's score matrix is 537 MB; times 32 heads it's 17 GB of scratch — which is the real reason long context used to OOM, not the weights.

---

## FlashAttention: never write S

The idea is embarrassingly simple to state and was hard to do: **compute the output in tiles, keeping each tile of S in SRAM, and never write S to HBM.** Loop over blocks of K/V; for each block, load a tile, compute its slice of the scores, update a running output accumulator, discard the tile.

```
for each block of Q rows (Br rows)         ─── outer loop, one output tile
    load Q_block  → SRAM
    init  O_acc = 0,  m = -inf,  l = 0     ─── running max & running sum, in registers
    for each block of K/V columns (Bc)     ─── inner loop, streams K and V once
        load K_block, V_block → SRAM       ─── ~one HBM trip each
        S_block = Q_block @ K_blockᵀ       ─── lives in SRAM, never in HBM
        m_new   = max(m, rowmax(S_block))
        P_block = exp(S_block - m_new)
        O_acc   = O_acc * exp(m - m_new) + P_block @ V_block     ← rescale, accumulate
        l       = l * exp(m - m_new) + rowsum(P_block)
        m       = m_new
    write O_block = O_acc / l  → HBM       ─── ONE write, size Br×d
```

Two ingredients make it legal:

**1. Online (streaming) softmax.** Softmax normally needs the whole row: you must know the max before exponentiating (for numerical stability) and the full sum before dividing. The trick is to carry a running max `m` and running sum `l`, and when a new block reveals a larger max, **rescale what you've already accumulated** by `exp(m_old - m_new)`. Algebraically identical, one pass, no full row required. That's the materialization point from earlier — dissolved.

**2. Recomputation instead of storage.** Training's backward pass needs P. FlashAttention recomputes it from Q and K in SRAM rather than reading it from HBM — *more* FLOPs, less time, because it was memory-bound. Inference doesn't need the backward pass, but the same logic is why "just recompute it" is a legitimate optimization once you know which side of the ridge you're on.

The result, for the same 4096-token head:

```
traffic:   138 MB  →  ~4.2 MB          (Q,K,V,O only)
intensity:     62  →  ~2,048 FLOP/byte  =  L/2   → COMPUTE-BOUND
memory:     O(L²)  →  O(L)              → long context stops OOMing
```

Note `I_fused = 4L²d ÷ 8Ld = L/2`: **the longer the context, the more compute-bound fused attention becomes.** Exactly backwards from everyone's intuition, and a great thing to be able to derive on a whiteboard.

### What FlashAttention is *not*

- **Not approximate.** Bit-for-bit it differs from a naive implementation only in reduction order, like any other reassociation ([lesson 4](04-tensor-cores-and-precision.md)). It is not sparse attention, not linear attention, not a low-rank trick. Same math, same outputs.
- **Not fewer FLOPs.** It does slightly *more* (rescaling, and recomputation in training). It wins purely on IO.
- **Not a fix for KV-cache size.** The cache still grows with tokens × batch; FlashAttention removes the *scores*, not the cache. Cache capacity is PagedAttention's problem ([Phase 4](../../ROADMAP.md#phase-4--inference-optimization-techniques)).
- **Not one algorithm.** FA-1 (2022) established the tiling; **FA-2** reorganized the work across warps and cut non-matmul operations, roughly doubling throughput; **FA-3** targets Hopper's async copies and FP8 and reaches ~75% of peak. The lineage matters because "which FlashAttention?" is a real production question.

### The decode wrinkle: Flash-Decoding

FlashAttention's parallelism comes from splitting **Q rows** across blocks — perfect for prefill, where there are thousands of them. At decode there is exactly **one** query row, so you get one block, one SM busy, 131 idle ([lesson 3](03-cuda-execution-model.md) occupancy). **Flash-Decoding** fixes this by splitting along the **KV length** instead, computing partial outputs over cache chunks in parallel and combining them with the same rescaling identity. This is why long-context decode kernels look different from prefill kernels, and why engines ship both.

---

## Try it (Colab free tier)

**A. Measure fusion against the prediction.** Chain of 5 elementwise ops → expect ~5×:

```python
import torch, time
def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

x = torch.randn(64 * 1024 * 1024, device="cuda", dtype=torch.float16)   # 128 MB
def chain(t): return torch.nn.functional.silu(t * 1.5 + 0.25) * 2.0 - 1.0
eager = bench(lambda: chain(x))
fused = torch.compile(chain)
for _ in range(5): fused(x)                       # let it trace and codegen
comp = bench(lambda: fused(x))
gb = x.numel() * 2 / 1e9
print(f"eager {eager*1e3:6.2f} ms  ~{2*5*gb/eager/1e3:5.0f} GB/s implied traffic")
print(f"fused {comp *1e3:6.2f} ms  ~{2*1*gb/comp /1e3:5.0f} GB/s   speedup {eager/comp:.2f}x")
```

**B. Watch attention's O(L²) memory appear and vanish.** `F.scaled_dot_product_attention` picks a backend; force each and compare peak memory and time:

```python
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, L, D = 1, 32, 4096, 128
q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.float16) for _ in range(3))
for name, backend in [("math (materializes S)", SDPBackend.MATH),
                      ("flash", SDPBackend.FLASH_ATTENTION),
                      ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION)]:
    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(backend):
        t = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), 20, 5)
    flops = 4 * B * H * L**2 * D * 0.5          # causal ⇒ half the score grid
    print(f"{name:22} {t*1e3:7.2f} ms  {flops/t/1e12:6.1f} TFLOP/s"
          f"  peak {torch.cuda.max_memory_allocated()/1e9:5.2f} GB")
```

Predict first: the `math` backend should allocate roughly `B·H·L²·2 bytes` ≈ 1.07 GB of scores that flash never touches. Then double `L` to 8192 and watch the math backend's memory go **4×** while flash goes 2× — the `O(L²)` vs `O(L)` difference, measured. (On a T4, `math` at L=8192 may simply OOM. That *is* the result; record it.)

---

## Key takeaways

- **Fusion = one kernel, intermediates stay in registers/SRAM.** For a chain of *k* memory-bound ops the predicted speedup is **≈ k**, and it cuts launch count at the same time ([lesson 6](06-overhead-bound-and-cuda-graphs.md)).
- Four flavors: elementwise chains, reduction fusion, **epilogue fusion** into a GEMM, and whole-algorithm IO-aware fusion. The first three are compiler work; the fourth is a research contribution.
- Fusion is blocked by **materialization points, real layout changes, `torch.compile` graph breaks, and SRAM capacity**.
- Naive attention writes and re-reads an **L×L** score matrix: at L=4096 that's 33.5 MB per head and ~97% of the traffic, intensity ~62 → memory-bound.
- **FlashAttention keeps score tiles in SRAM using an online softmax with running max/sum rescaling**, writes only O. Intensity `62 → L/2`, memory `O(L²) → O(L)`. Same math, more FLOPs, far less IO.
- It is **not** approximate, **not** fewer FLOPs, and **not** a fix for KV-cache capacity. FA-2 restructured the warp work; FA-3 targets Hopper/FP8.
- **Flash-Decoding** re-parallelizes over KV length because batch-1 decode has only one query row to split — prefill and decode want different kernels.

**Next:** [Profiling in practice →](08-profiling-in-practice.md) — you now have three hypotheses and three fixes; this is how you prove which one you're actually looking at, on real hardware, in under twenty minutes.
