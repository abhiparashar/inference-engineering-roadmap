# 10 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. This file is how you *prove* Phase 2 is done — to yourself, and to a future interviewer.

Phase 1 was the last phase you could finish on a laptop with a clear conscience. Phase 2 is the first one where the deliverable is **evidence**: measurements, a profiler trace, a plot with your GPU's name on it. From here on, an inference engineer is someone who shows numbers.

Everything below is doable on a free Colab T4. Don't wait for better hardware.

---

## Warm-up exercises (short, run them)

Each maps to one lesson. Type the code, run it, write down what you saw and whether it matched your prediction. **Predict before running** — a wrong prediction you then explain is worth ten right ones.

1. **Know your machine** ([lesson 1](01-why-gpus.md)): print SM count, HBM size, compute capability, and shared memory per block via `torch.cuda.get_device_properties(0)`. Then find the official datasheet and record peak FP32 TFLOP/s, peak FP16/BF16 tensor TFLOP/s, and HBM GB/s. Compute FLOPs-per-byte for each dtype.
2. **Measure real bandwidth** ([lesson 2](02-gpu-memory-hierarchy.md)): time a large elementwise add and report achieved GB/s as a % of spec (MBU). Then sweep the tensor from 1 MB to 512 MB and find the size where you stop beating the HBM roof — that's your L2 capacity showing up in a measurement.
3. **Count the round trips** ([lesson 2](02-gpu-memory-hierarchy.md)): time `x.add(1).mul(2).relu()` versus the `torch.compile`d version, and versus your own prediction of "3 passes → 1 pass". Report predicted vs measured speedup.
4. **Coalescing costs money** ([lesson 3](03-cuda-execution-model.md)): time `x.sum(dim=0)` vs `x.sum(dim=1)` on an 8192² tensor and `x.t().contiguous()`. Explain the ratio in terms of transactions per warp — including why PyTorch's kernels beat the naive 32× prediction.
5. **The dtype table** ([lesson 4](04-tensor-cores-and-precision.md)): benchmark a 4096² matmul in FP32 (TF32 off), FP32 (TF32 on), FP16, and BF16. Report TFLOP/s and MFU for each. Then run `n = 4096, 4095, 4097, 4088, 4032` in FP16 and explain the alignment effect.
6. **The ridge point, by hand** ([lesson 5](05-roofline-model.md)): compute the ridge point for a T4, an A100 80GB, and an H100 from the datasheets. Then compute the arithmetic intensity of: an elementwise add, a LayerNorm, batch-1 decode of a 7B model, batch-32 decode, and a 2048×4096×4096 prefill GEMM. Classify each.
7. **Prove the sync bug exists** ([lesson 6](06-overhead-bound-and-cuda-graphs.md)): time 50 matmuls without `torch.cuda.synchronize()` and with it. Report both, and the implied (absurd) TFLOP/s of the first. Keep this number — it's the fastest way to explain async execution to a colleague.
8. **Overhead, isolated** ([lesson 6](06-overhead-bound-and-cuda-graphs.md)): time a chain of 100 tiny ops eager, under `torch.compile(mode="reduce-overhead")`, and under a hand-captured `torch.cuda.CUDAGraph`. Derive the implied per-launch cost in µs and compare it to the ~5-10 µs claim.
9. **Attention's missing megabytes** ([lesson 7](07-fusion-and-flash-attention.md)): with `sdpa_kernel`, run `scaled_dot_product_attention` at `L = 2048, 4096, 8192` under the `MATH` and `FLASH_ATTENTION` backends. Record time and `torch.cuda.max_memory_allocated()` for each. Confirm math scales as `O(L²)` and flash as `O(L)`; report the L at which math OOMs.
10. **Twenty-minute triage** ([lesson 8](08-profiling-in-practice.md)): profile prefill and decode separately on a small HF model. Report, for each: GPU-busy fraction of wall time, kernel launch count, top 3 kernels, and the verdict (compute-/memory-/overhead-bound) with the evidence you used.
11. **Read a real kernel** ([lab](../../labs/README.md), [lesson 7](07-fusion-and-flash-attention.md)): in `Dao-AILab/flash-attention`, read the README's algorithm section and find the tiling loop. Answer in your own words, in writing: *what exactly does it avoid writing to HBM that standard attention writes, and where does that data live instead?* Then find `sm__pipe_tensor_cycles_active` or an equivalent tensor-core metric in the Nsight Compute docs and note what value would prove tensor cores ran.
12. **Find the fusion in PyTorch** ([lesson 7](07-fusion-and-flash-attention.md)): run any small model under `TORCH_LOGS="output_code" python x.py` and read one generated Triton kernel. Identify how many original PyTorch ops it fused, and where the `tl.load`/`tl.store` boundaries are.

Keep these in `labs/phase2/`. The *numbers* feed your writeup.

---

## Conceptual self-check (answer without notes)

If any answer is fuzzy, re-read the linked lesson before starting Phase 3.

1. Why is a GPU faster than a CPU for matmul but not for a linked-list traversal? Answer in terms of latency-optimized vs throughput-optimized design. ([1](01-why-gpus.md))
2. Name the GPU memory hierarchy from fastest to slowest with rough sizes and bandwidths, and say which level *you* control. ([2](02-gpu-memory-hierarchy.md))
3. Approximately how many FLOPs can an H100 do in the time it takes to read one byte from HBM? Where does that number come from? ([2](02-gpu-memory-hierarchy.md), [5](05-roofline-model.md))
4. Explain kernel fusion and tiling in one sentence each, in terms of HBM traffic. ([2](02-gpu-memory-hierarchy.md), [7](07-fusion-and-flash-attention.md))
5. What is a warp, and what are the three ways a kernel silently wastes most of the machine? ([3](03-cuda-execution-model.md))
6. Why is high occupancy *not* always the goal? ([3](03-cuda-execution-model.md))
7. What four conditions must hold for tensor cores to actually engage? Why can't batch-1 decode use them? ([4](04-tensor-cores-and-precision.md))
8. Why does INT4 quantization dramatically speed up decode but barely help prefill? ([4](04-tensor-cores-and-precision.md), [5](05-roofline-model.md))
9. Given peak FLOPs and HBM bandwidth, compute the **ridge point** and say what it means. Do it for a GPU you haven't memorized. ([5](05-roofline-model.md))
10. Define MFU and MBU, say which one you report for decode and why, and explain why `nvidia-smi`'s "GPU utilization" is nearly useless. ([5](05-roofline-model.md))
11. Name three things the roofline model cannot see. ([5](05-roofline-model.md))
12. Why can a batch-1 decode step be **overhead-bound** on an H100 but not on a T4? Do the arithmetic. ([6](06-overhead-bound-and-cuda-graphs.md))
13. What do CUDA graphs fix, and what are the four constraints they impose? Why does that force batch-size bucketing in vLLM? ([6](06-overhead-bound-and-cuda-graphs.md))
14. Explain FlashAttention as an **IO-aware** algorithm: what it doesn't write, what makes the online softmax legal, and why longer context makes it *more* compute-bound. Then say what it does **not** fix. ([7](07-fusion-and-flash-attention.md))
15. **The keystone question:** you're handed an unknown model on an unknown GPU and told "it's too slow." Walk through your triage, in order, naming the tool at each step, the specific number you'd read, and the fix each verdict implies. ([8](08-profiling-in-practice.md))

Question 15 is the one that matters, and it's close to verbatim a real inference-engineering interview question. The strong answer is a *procedure with evidence*, not a list of optimizations: sum-of-kernel-time ÷ wall-clock first (overhead), then DRAM % vs SM % vs tensor-pipe % on the top kernel (memory vs compute vs wrong-units), then the matching fix — batch it, shrink the bytes, or remove the launches.

---

## Exit artifact (this is what "finishing Phase 2" means)

Produce **at least Option A** and commit it. A + B together is the strongest portfolio pair in the first half of the roadmap, because A proves you can measure and B proves you can build.

### Option A — "Is it compute-, memory-, or overhead-bound?" (required)

A single writeup, `projects/04-roofline-profiling/README.md`, containing:

- **The measured roofline plot** for your GPU ([lesson 9](09-build-roofline-and-triton-kernel.md) Part A), with the matmul and elementwise points and the per-dtype roofs drawn from the datasheet.
- **The spec table**: peak FLOP/s per dtype, HBM GB/s, ridge point per dtype — and your predicted crossover vs the measured one.
- **A profiler trace screenshot** of a decode loop with the GPU-idle gaps visibly annotated ([lesson 8](08-profiling-in-practice.md)).
- **The triage table** from exercise 10: prefill and decode side by side with GPU-busy %, launch count, top kernels, verdict, evidence.
- **Your MBU and MFU numbers**, and one sentence on which is the meaningful one for each phase.
- **A 300-word conclusion** answering self-check question 15 with your own numbers, ending in the specific optimization you'd do first and the speedup you'd predict from the byte arithmetic.

### Option B — The fused Triton kernel (strongly recommended)

`projects/05-triton-fused-kernel/README.md` with the [lesson 9](09-build-roofline-and-triton-kernel.md) Part B deliverables:

- Two working, correctness-tested kernels (bias+GELU and softmax), tested at non-power-of-two sizes.
- The three-way benchmark: eager vs `torch.compile` vs yours, with effective GB/s and MBU.
- **The HBM-passes table**: passes, predicted speedup, measured speedup, and an explanation of the gap.
- The block-size sweep, and the `n_cols` at which your softmax kernel hits the SRAM wall — plus one paragraph connecting that failure to FlashAttention's tiling.

### Option C — Deep read (optional, cheap, high signal)

A 600-word writeup: *"FlashAttention explained in HBM round trips."* Standard attention's byte count at `L = 4096`, what tiling + online softmax removes, the intensity before and after, and the three things it does **not** solve. Cite the specific lines of the flash-attention repo you read. This is the single best short piece to have in a portfolio when someone asks whether you understand kernels.

---

## How you know you're ready for Phase 3

- [ ] You can compute a ridge point from any datasheet, in your head, and say which side a given op falls on.
- [ ] You have *measured* your GPU's bandwidth and reached ≥70% MBU on an elementwise op.
- [ ] You have seen the FP32-vs-FP16 matmul gap on your own hardware and can quote your two numbers.
- [ ] You can classify an unknown workload as compute-, memory-, or overhead-bound in under 20 minutes, and you have done it once for real.
- [ ] You have seen GPU-idle gaps in a trace with your own eyes, and closed some of them with CUDA graphs or `torch.compile`.
- [ ] You can explain FlashAttention without saying the words "faster math."
- [ ] You wrote a kernel that beat eager PyTorch and you can justify the speedup with byte arithmetic.
- [ ] Your exit artifact is committed.

Then go to **[Phase 3 — Serving Fundamentals: Batching, Queueing, Scheduling](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling)**. Phase 2's verdict is that a single request leaves ~99% of the GPU unused: intensity ≈ 1 against a ridge of ~300, plus hundreds of launches per token. Phase 3 is the systems answer — keep the machine fed with *many* requests at once, without wrecking anyone's tail latency. Everything you measured here is the *why* behind continuous batching.

---

## Where these ideas come back (so you know it wasn't busywork)

| Phase 2 idea | Comes back as |
|---|---|
| Ridge point & arithmetic intensity | Why continuous batching exists; batch-size tuning; capacity planning (Phases 3, 6-7) |
| Decode intensity ≈ batch size | Scheduler design, max-batch-tokens tuning, throughput/latency curves (Phase 3) |
| Bytes-moved reasoning | Quantization, KV-cache compression, weight-only vs activation quant (Phase 4) |
| Fusion & tiling | `torch.compile`, TensorRT graph optimization, custom kernels in vLLM/SGLang (Phases 4-5) |
| FlashAttention / IO-awareness | FlashDecoding, chunked prefill, long-context serving, prefix caching (Phases 4-6) |
| HBM capacity limits | PagedAttention, block managers, preemption and swapping (Phases 4-5) |
| Launch overhead & CUDA graphs | Engine warmup, graph bucketing, `--enforce-eager` tradeoffs (Phase 5) |
| MFU / MBU | The only honest efficiency metrics on a cost dashboard; cost per million tokens (Phases 7-8) |
| Profiler triage | Every production performance investigation you will ever run (Phases 5-10) |
| Tensor-core shape rules | Padded vocabularies, bucketed batch sizes, why odd shapes are slow (Phases 3-5) |

Phase 1 gave you the *what*. Phase 2 gave you the *where* and, more importantly, the *how do you know*. Phase 3 onward you stop optimizing one request and start operating a system.
