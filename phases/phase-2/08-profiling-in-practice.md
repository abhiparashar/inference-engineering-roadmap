# 8 — Profiling in Practice (Proving Which Bottleneck You Have)

> **You'll be able to say:** "Give me an unknown model and a GPU and I'll tell you in twenty minutes whether it's compute-, memory-, or overhead-bound, which kernels own the time, and what the specific evidence was. I don't guess bottlenecks — I read them off a trace."

Lessons 5-7 gave you three hypotheses and three fixes. This lesson is the part that makes you dangerous: turning a hypothesis into a measurement. Along with [lesson 5](05-roofline-model.md), it's the load-bearing lesson of Phase 2. Everything else here is knowledge; this is a skill, and it only develops by doing it on real code.

The discipline is one rule: **measure first, and measure the right thing.** Every hour spent in a profiler saves a week of optimizing something that was never the bottleneck.

---

## The 20-minute triage protocol

Run this in order on any unknown workload. Stop as soon as the answer is obvious — you often don't need step 3.

```
0.  nvidia-smi dmon        →  is the GPU even busy? power draw? memory headroom?     (1 min)
1.  torch.profiler table   →  which ops own the wall clock? how many kernels?        (5 min)
2.  torch.profiler trace   →  is the GPU timeline solid or full of gaps?             (5 min)
3.  ncu on the top kernel  →  DRAM % vs SM % vs tensor-pipe % → the verdict          (10 min)
```

Alongside it, the arithmetic from lesson 5 — you should already have a *prediction* before you open anything. Profiling without a prediction is sightseeing.

---

## Level 0 — `nvidia-smi`, and the number everyone misreads

```bash
nvidia-smi                                  # snapshot: memory, util, power, processes
nvidia-smi dmon -s pucm                     # 1 Hz stream: power, util, clocks, memory
nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm \
           --format=csv -l 1                # scriptable
```

What each column is actually worth:

| Reading | What it really means |
|---|---|
| **`GPU-Util` %** | Fraction of sample windows in which *at least one kernel was resident*. Not efficiency. A batch-1 decode loop stalling on HBM reads ~100%. |
| **`Power draw`** | The honest utilization signal. 90 W on a 300 W card = the machine is idling inside kernels or between them. Near TDP = real work. |
| **`Memory used`** | Includes PyTorch's caching allocator reservation, so it exceeds your tensors ([lesson 2](02-gpu-memory-hierarchy.md)). |
| **`clocks.sm`** | If it's below base clock under load, you're **thermally or power throttled** — a real and frequently missed cause of "the benchmark got slower after 10 minutes." |

**Power draw plus clock is your free bottleneck hint**: low power at full "utilization" means memory-bound or overhead-bound, near-TDP power means compute-bound.

---

## Level 1 — `torch.profiler`: which ops own the time

This is where 80% of inference performance work happens, and it needs no installs.

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

for _ in range(5):                      # WARMUP OUTSIDE THE PROFILER — non-negotiable
    model(inputs)                       # first iters include autotune, JIT, allocator growth
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=False,
             schedule=schedule(wait=1, warmup=1, active=3)) as prof:
    for _ in range(5):
        model(inputs)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")      # open in chrome://tracing or ui.perfetto.dev
```

Read the table in this order:

1. **Sum of `CUDA time` vs your wall-clock time.** This single ratio is the overhead test. If kernels account for 4 ms out of an 11 ms step, the GPU was idle ~60% of the time → **overhead-bound**, go to [lesson 6](06-overhead-bound-and-cuda-graphs.md). No further profiling needed.
2. **`# of Calls`.** Hundreds or thousands of kernel launches per step is the launch-overhead fingerprint. Also look for surprises: a `to`/`copy_` you didn't write means a hidden H2D transfer or a dtype cast.
3. **The top 5 kernels by `cuda_time_total`.** Optimize nothing outside this list. Names tell you a lot:

| Kernel name pattern | What it is |
|---|---|
| `cutlass...gemm...`, `sm80_xmma_gemm`, `*_nn_f16f16_f16` | a real tiled matmul, probably using tensor cores |
| `elementwise_kernel`, `vectorized_elementwise_kernel` | memory-bound elementwise; a fusion candidate ([lesson 7](07-fusion-and-flash-attention.md)) |
| `at::native::(reduce|softmax|layer_norm)` | reduction; memory-bound, fusable |
| `flash_fwd_kernel`, `fmha` | FlashAttention ran — good |
| `gemv`, `*_1x...`, or a GEMM with M=1 | batch-1 matrix-vector: no tensor cores, memory-bound ([lesson 4](04-tensor-cores-and-precision.md)) |
| `Memcpy HtoD` / `DtoH` in the hot loop | a sync point and a PCIe stall; get rid of it |

4. **`Self CUDA %` concentration.** One kernel at 60% = a clear target. Fifty kernels at 2% each = you have a *structural* problem (fusion or graphs), not a kernel problem.

### Reading the trace

Open `trace.json` in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. You get CPU rows and GPU rows on one timeline, and you're looking for exactly one thing:

```
OVERHEAD-BOUND                       HEALTHY
CPU: |op|op|op|op|op|op|             CPU: |op|op|op|  (running ahead, queue deep)
GPU: ·|k|··|k|···|k|··|k|            GPU: |kkkk|kkkkkk|kkkk|kkkkkk|
      ↑ gaps = idle GPU                    ↑ solid, no gaps
```

Gaps → overhead. Solid → the GPU is busy and you now need level 3 to learn *at what*. Also worth spotting: one giant kernel where you expected many (CUDA graphs are active, and individual kernel names disappear inside the replay), and a long first iteration (you forgot warmup).

---

## Level 2 — Nsight Systems (`nsys`): the whole-system timeline

`torch.profiler` sees PyTorch. `nsys` sees everything — CUDA API calls, driver time, memcpys, NCCL collectives, CPU threads, and other processes on the GPU.

```bash
nsys profile -o report --trace=cuda,nvtx,osrt --force-overwrite true python bench.py
nsys stats report.nsys-rep          # CLI summary: kernel and API-call histograms
```

Annotate your own phases so the timeline is readable instead of a wall of kernels:

```python
with torch.cuda.nvtx.range("prefill"):  out = model(prompt_ids)
with torch.cuda.nvtx.range("decode"):   out = model.generate(...)
```

Reach for `nsys` when: latency doesn't add up from kernel times, you suspect CPU-side work (tokenization, scheduler, Python), you're debugging multi-GPU communication ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)), or you need to see H2D/D2H overlap and stream behavior ([lesson 6](06-overhead-bound-and-cuda-graphs.md)).

---

## Level 3 — Nsight Compute (`ncu`): the per-kernel verdict

`ncu` answers "*why* is this one kernel slow" by reading hardware counters. It **replays each kernel many times and serializes execution**, so it can be 100-1000× slower — always filter:

```bash
ncu --set full --kernel-name-base regex:gemm --launch-count 3 -o prof python bench.py
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
--launch-count 5 python bench.py
```

Four metrics decide almost everything:

| Metric | Reads as |
|---|---|
| `dram__throughput...pct_of_peak` | **% of HBM bandwidth achieved** — this is MBU for the kernel |
| `sm__throughput...pct_of_peak` | % of peak compute pipe activity |
| `sm__pipe_tensor_cycles_active...` | **were tensor cores actually used?** 0% on a matmul = you silently fell back to CUDA cores |
| `sm__warps_active...pct` | achieved occupancy ([lesson 3](03-cuda-execution-model.md)) |

### The decision table (memorize this)

| DRAM % | SM % | Tensor % | Gaps on timeline | Verdict | Fix |
|---|---|---|---|---|---|
| **high (>60)** | low | — | none | **memory-bound** | fuse, quantize, batch ([5](05-roofline-model.md), [7](07-fusion-and-flash-attention.md)) |
| low | **high (>60)** | high | none | **compute-bound** | lower precision, better GEMM shapes ([4](04-tensor-cores-and-precision.md)) |
| low | high | **~0** on a matmul | none | **compute-bound on the wrong units** | fix dtype/alignment — free ~10× |
| low | low | — | **many** | **overhead-bound** | CUDA graphs, `torch.compile` ([6](06-overhead-bound-and-cuda-graphs.md)) |
| low | low | — | none | **latency/occupancy-bound** | more parallel work; check stall reasons |

For that last row, `ncu`'s **warp stall reasons** name the culprit: *Long Scoreboard* = waiting on global memory (uncoalesced or just too many bytes), *MIO Throttle* = shared-memory pressure, *Barrier* = `__syncthreads()` imbalance, *Not Selected* = plenty of warps, scheduler oversubscribed (a good problem).

---

## Turning profiler output into MFU / MBU

A verdict is more convincing with a percentage attached. From a decode benchmark you already have the two numbers:

```
measured: 7B FP16 model, batch 1, 62 tok/s on a T4 (320 GB/s)
  bytes/token ≈ model bytes  = 14 GB          (weights dominate; KV is small at short ctx)
  achieved    = 14e9 × 62    = 868 GB/s ...   ← impossible, so the model isn't 7B-in-FP16
                                                 on a 16 GB T4 — it's quantized. Numbers
                                                 that violate the roof mean your MODEL of
                                                 the workload is wrong, not the hardware.
correct for INT4 (≈3.5 GB):   3.5e9 × 62 = 217 GB/s  →  MBU = 217/320 ≈ 68%   ← healthy
```

That's the habit worth building: **when a measurement beats a roof, your assumptions are wrong** — wrong dtype, cache hits ([lesson 5](05-roofline-model.md)), or you counted bytes that never moved. Chasing that contradiction is how you find out what a system is really doing.

MBU near 70-85% on decode means the kernels are fine and your remaining levers are *fewer bytes* (quantization, KV compression) or *more work per byte* (batching). MBU at 20% with a solid timeline means something is reading more than it should; MBU at 20% with gaps means overhead.

---

## The traps that produce wrong conclusions

- **No warmup.** The first iteration includes cuBLAS autotuning, kernel JIT, allocator growth, and `torch.compile` tracing. Always discard 5-10 iterations.
- **No `torch.cuda.synchronize()`** around timers ([lesson 6](06-overhead-bound-and-cuda-graphs.md)) — you measure enqueue time and conclude you have a petaFLOP GPU.
- **Profiling with the profiler's overhead in the number.** `torch.profiler` adds a few %, `ncu` adds orders of magnitude. Report timings from a *clean* run; use the profiler for attribution, not for headline latency.
- **Benchmarking the wrong phase.** Prefill and decode are different workloads with different bottlenecks ([Phase 1 lesson 6](../phase-1/06-prefill-vs-decode.md)). Profile them separately, or NVTX-tag them.
- **Averaging a hot loop.** Real serving has queueing; report percentiles at fixed offered load ([`playbooks/benchmarking.md`](../../playbooks/benchmarking.md)).
- **Clock instability.** Thermal/power throttling and other tenants on a shared GPU. Check `clocks.sm`; on Colab, re-run to confirm.
- **Optimizing a kernel that owns 3% of the time.** Amdahl's law is undefeated. Sort by `cuda_time_total`, work top-down.

---

## Try it (Colab free tier)

Profile a real model end to end and produce the deliverable table. Use any small HF causal LM (`gpt2`, `TinyLlama`, `Qwen2.5-0.5B`):

```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity

name = "gpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
ids = tok("The history of computing" * 40, return_tensors="pt").input_ids.cuda()  # long-ish

@torch.inference_mode()
def prefill(): return model(ids)
@torch.inference_mode()
def decode_step(past): return model(ids[:, -1:], past_key_values=past, use_cache=True)

past = prefill().past_key_values
for _ in range(10): decode_step(past)                 # warmup
torch.cuda.synchronize()

for label, fn in [("prefill", prefill), ("decode", lambda: decode_step(past))]:
    t0 = time.perf_counter()
    for _ in range(20): fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / 20
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    ev = prof.key_averages()
    # PyTorch renamed self_cuda_time_total -> self_device_time_total in 2.5
    KEY = "self_device_time_total" if hasattr(ev[0], "self_device_time_total") \
          else "self_cuda_time_total"
    gpu_us = sum(getattr(e, KEY) for e in ev)
    launches = sum(e.count for e in ev if getattr(e, KEY) > 0)
    print(f"\n=== {label}: wall {wall*1e3:.2f} ms | GPU {gpu_us/1e3:.2f} ms "
          f"| GPU busy {gpu_us/1e3/(wall*1e3):.0%} | ~{launches} kernel launches")
    print(ev.table(sort_by=KEY, row_limit=8))
```

Answer these four questions in writing, with the numbers as evidence:

1. What fraction of wall time was the GPU actually running kernels, for prefill vs decode? Which one is overhead-bound?
2. How many kernel launches per decode step? Multiply by 7 µs — does that explain the gap in question 1?
3. What are the top 3 kernels for each phase, and are they GEMMs or elementwise/reduction kernels?
4. Compute MBU for decode (`model_bytes × tok/s ÷ peak bandwidth`) and MFU for prefill (`2 × params × tokens/s ÷ peak FLOP/s`). Which number is meaningful, and why?

Then export a Chrome trace for the decode loop, open it, and screenshot the gaps. **That screenshot plus these four answers is the core of your Phase 2 exit artifact** ([lesson 10](10-exercises-and-artifacts.md)).

---

## Key takeaways

- Triage in order: **`nvidia-smi` → `torch.profiler` table → trace timeline → `ncu` on the top kernel.** Predict first; profile to confirm.
- **`GPU-Util` is not efficiency.** Power draw and SM clock are the honest quick signals; low power at high "util" means memory- or overhead-bound.
- The **overhead test** is one division: summed kernel time ÷ wall-clock time. Below ~70% with many short kernels → overhead-bound, and no kernel tuning will help.
- Kernel *names* classify themselves: `cutlass/gemm` = matmul, `elementwise/reduce` = fusion candidate, `gemv`/M=1 = batch-1 memory-bound, `flash_fwd` = FlashAttention ran.
- In `ncu`, four metrics decide it: **DRAM % (MBU), SM %, tensor-pipe %, occupancy** — plus stall reasons. Tensor-pipe 0% on a matmul is a free ~10×.
- **When a measurement beats a roof, your assumptions are wrong**, not the hardware. Chase the contradiction.
- Traps that invalidate results: no warmup, no `synchronize()`, profiler overhead in the headline number, mixing prefill with decode, averaging hot loops, throttled clocks, and optimizing a 3% kernel.

**Next:** [Build: roofline benchmark + a fused Triton kernel →](09-build-roofline-and-triton-kernel.md) — stop reading and produce the two artifacts: a measured roofline of your own GPU, and a kernel you wrote that beats eager PyTorch.
