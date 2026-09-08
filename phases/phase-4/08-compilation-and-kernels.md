# 8 — Compilation, Kernels & Graphs

> **You'll be able to say:** "At batch 1, almost none of the step time is math. I measured a 12-layer decode-shaped graph: 0.300 ms/step at batch 1 versus 1.928 ms at batch 256 — **256× the work for 6.4× the time**, so ~97% of the batch-1 step is fixed per-step cost, not arithmetic. That's what CUDA graphs delete, what fusion reduces, and what `--enforce-eager` gives back. FlashAttention fixes attention's IO; FlashDecoding fixes it again for *long context at small batch* by splitting the KV dimension; and every one of these has a precondition you must check before promising a speedup."

Phase 2 taught these as GPU concepts ([lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md), [lesson 7](../phase-2/07-fusion-and-flash-attention.md)). This lesson is the **serving-specific** application: what to turn on, what it costs, and what breaks.

---

## The overhead you can't see in a FLOP count

A decode step for one sequence is a few hundred tiny kernels, each launched from Python. Measured on the same graph at two batch sizes:

```
device=mps
eager              0.300 ms/step   implied    47.2 GFLOP/s
torch.compile      0.292 ms/step   speedup  1.03x
same graph, batch 256:   1.928 ms/step ->  6.44x the time for 256x the work
  => per-step fixed cost dominates at batch 1: roughly 0.292 ms of the 0.300 ms is NOT
     proportional to work
```

Two readings, one uncomfortable:

- **The batch-1 step is ~97% fixed cost.** Python dispatch, allocator calls, kernel launches, synchronization — the GPU is idle for most of the wall clock. This is *why* a serving engine cares about graphs at all, and it's the same phenomenon as Phase 2's overhead-bound regime, now measured on a decode-shaped workload.
- **`torch.compile` bought 1.03× here** — essentially nothing, on this backend, for this shape. That's an honest result and a useful warning: compilation wins are backend- and shape-dependent (Inductor even reported "Not enough SMs to use max_autotune_gemm mode"). On CUDA with `mode="reduce-overhead"` (which turns on CUDA graphs) the same experiment typically shows a large win. **Never assume; measure on the target hardware.**

---

## The four tools, and what each one actually fixes

| Tool | Fixes | Typical win | Precondition / cost |
|---|---|---|---|
| **Kernel fusion** (`torch.compile`, hand-written) | HBM round trips between elementwise ops | 1.1-2× on memory-bound glue | compile time; graph breaks kill it |
| **CUDA graphs** | per-launch CPU overhead (~5-10 µs × hundreds) | 1.2-2× at small batch | **static shapes and addresses**; capture warmup; no data-dependent control flow |
| **FlashAttention** | attention's `O(L²)` HBM traffic for the score matrix | large at long context; enables long context at all | needs a supported head dim/dtype/GPU |
| **FlashDecoding / split-K** | attention parallelism when `batch × heads` is small but `L` is huge | 2-8× on attention in that regime | only helps that regime |
| **AOT compilers** (TensorRT-LLM, ONNX Runtime) | everything above, ahead of time, plus quantized fused kernels | often best-in-class | build step per model/shape/GPU; least flexible |

### CUDA graphs and why serving engines bucket batch sizes

A CUDA graph records a fixed sequence of kernel launches (with fixed buffers) and replays it with one call — the CPU cost of hundreds of launches collapses to one submission. The catch is in the word *fixed*: shapes and pointers are baked in at capture.

But a serving engine's decode batch changes **every iteration** (Phase 3's whole point). The standard resolution:

```
  capture graphs for a BUCKETED set of batch sizes:  1, 2, 4, 8, 16, 24, 32, 48, 64, …
  at run time: pad the real batch (say 21) up to the next captured bucket (24) and replay
```

You waste a little compute on padded rows and win the launch overhead back many times over. This is exactly what vLLM does (`--enforce-eager` disables it; `cudagraph_capture_sizes` controls the buckets), and it explains two things operators notice: **startup takes tens of seconds** (capture per bucket) and **memory usage jumps at startup** (graph pools). Prefill is usually *not* graphed — its shapes are too variable and it's compute-bound anyway, so there's little to win.

Debugging rule: if you see corrupted output only at certain batch sizes, or "operation not permitted during capture" errors, suspect graph capture; `--enforce-eager` is the first bisection step in any serving bug hunt.

### `torch.compile` in a serving loop

- **`mode="reduce-overhead"`** = Inductor + CUDA graphs; it's the mode that matters for decode.
- **Dynamic shapes cause recompilation.** Every new batch size or sequence length can trigger a fresh compile (seconds to minutes) *during serving*. Mark dynamic dims (`torch._dynamo.mark_dynamic`) or bucket shapes, and always check `torch._dynamo.utils.counters` for recompile counts under load. A production symptom of getting this wrong is periodic multi-second TTFT spikes with no queue growth.
- **Graph breaks** (Python side effects, data-dependent control flow, `.item()`, printing) silently split the graph and delete most of the win. `TORCH_LOGS="graph_breaks"` shows them.
- **Warmup is mandatory** and must be excluded from benchmarks ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) — first-call latency includes compilation and autotuning.

### FlashAttention, and its decode-time sibling

FlashAttention makes attention **IO-aware**: tile Q/K/V into SRAM, use online softmax, never materialize the `L×L` score matrix in HBM. Memory becomes `O(L)` instead of `O(L²)`, and long context becomes feasible at all.

But FlashAttention's parallelization is over `batch × heads × query-blocks`, and **during decode there is exactly one query token**. At batch 1 with a 32k context, that's very few parallel work units against thousands of SMs: the GPU is starved even though there's plenty of memory traffic to do. **FlashDecoding** (and vLLM's `paged_attention_v2`) fixes this by splitting the *KV length* dimension into chunks, computing partial attention with partial softmax statistics in parallel, and combining them — the same online-softmax trick applied along a different axis.

The practical rule: **long context + small batch → make sure your backend uses a split-K decode kernel.** This is one of the few places where a backend flag changes decode latency several-fold.

### Quantized kernels, briefly

From [lesson 3](03-quantization-methods.md): a 4-bit checkpoint is only as fast as its dequant-and-matmul kernel. Marlin/Machete-class kernels exist because the reference implementations left most of the byte-saving on the table, and the right kernel depends on batch size (weight-only kernels win at small batch, lose at large). When you benchmark quantization, you are benchmarking **the kernel**, not the algorithm — say so in your writeup.

---

## Choosing from a profile (the 20-minute triage, serving edition)

```
  1. sum(kernel time) / wall time  ──  << 1 ?  ──▶ OVERHEAD-BOUND
        → CUDA graphs / reduce-overhead, fewer Python ops per step, bigger batch
  2. top kernel is attention, long context, small batch?  ──▶ split-K decode kernel (FlashDecoding)
  3. top kernels are elementwise/norms with high DRAM %?  ──▶ FUSION (compile or hand-fused)
  4. top kernel is a GEMM at high tensor-pipe utilization? ──▶ you are compute-bound:
        quantize activations (FP8/W8A8), or accept it — this is the healthy case
  5. periodic multi-second stalls with a flat queue?      ──▶ RECOMPILATION or graph capture
```

Each branch names a tool *and* the evidence that justifies it. "We turned on `torch.compile`" without step 1 is a guess; with step 1 it's engineering.

---

## Try it

1. **Reproduce the overhead measurement** on your hardware, at batch 1 and batch 256, and compute the fixed per-step cost. On CUDA, add `mode="reduce-overhead"` and a hand-captured `torch.cuda.CUDAGraph` and report all three.
2. **Count launches per decode step** with the PyTorch profiler on your Phase-3 engine. Multiply by ~7 µs and compare to the measured step time — that ratio tells you how much CUDA graphs can possibly buy.
3. **Bucket and pad** in your own engine: round the decode batch up to {1,2,4,8,16,32}, measure the padding waste and the step-time change. You've just reimplemented vLLM's capture-size logic.
4. **Force recompiles**: run with varying batch sizes under `TORCH_LOGS="recompiles"` and watch them fire. Then bucket the shapes and watch them stop. Report the TTFT spikes before and after.
5. **Long-context decode**: benchmark attention at `L = 1k, 8k, 32k` with batch 1 under your backend's available kernels. Confirm the split-K path wins and by how much.
6. **Compare an AOT build**: TensorRT-LLM (or ONNX Runtime) for the same model, same harness. Report tokens/sec, build time, and how much operational flexibility you gave up.

---

## Key takeaways

- **At batch 1 the step is dominated by fixed cost, not math** — measured: 0.300 ms/step at batch 1 vs 1.928 ms for 256× the work, i.e. ~97% overhead. That is the budget CUDA graphs attack.
- **`torch.compile` is not automatically a win** (1.03× in this measurement on MPS). Wins depend on backend, shapes, and whether `reduce-overhead` (CUDA graphs) is on. Measure on target hardware.
- **CUDA graphs need static shapes**, so engines **bucket batch sizes and pad** — which is why startup is slow, memory jumps, and `--enforce-eager` exists as the debug switch.
- **Dynamic shapes cause mid-serving recompiles**; graph breaks silently delete fusion wins. Watch `TORCH_LOGS=recompiles,graph_breaks` and bucket your shapes.
- **FlashAttention** removes the `O(L²)` score matrix; **FlashDecoding/split-K** fixes the *decode* case where one query token can't fill the GPU — the fix for long context at small batch.
- **Quantized speed is a kernel property**, and the right kernel depends on batch size.
- **Triage before tools**: kernel-time/wall-time ratio first, then the top-kernel evidence, then the matching fix.

**Next:** [Build: paged KV + prefix sharing + a quantization table →](09-build-paged-kv-and-quant-bench.md) — implement the block pool, block tables and copy-on-write in your own engine, and produce the quantization artifact.
