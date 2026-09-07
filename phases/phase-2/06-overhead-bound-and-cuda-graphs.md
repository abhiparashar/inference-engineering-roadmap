# 6 — Overhead-Bound: Kernel Launches, Streams & CUDA Graphs

> **You'll be able to say:** "There's a third regime the roofline can't draw: the GPU doing *nothing* because the CPU can't describe work fast enough. A batch-1 decode step is 200-400 tiny kernels, each costing more CPU time to launch than GPU time to run — so you capture the sequence once as a CUDA graph and replay it with one launch."

[Lesson 5](05-roofline-model.md) gave you two roofs and a ridge point, and both roofs assume the same thing: **the GPU is working.** Now picture the profiler report from a real batch-1 decode loop — *memory throughput 8%, compute throughput 4%.* Both numbers are correct, and neither points at a fix, because neither is the problem. The machine is **empty**: it finished each kernel and then waited for Python. No point on a roofline describes an idle GPU.

---

## CPU and GPU are separate machines connected by a queue

PyTorch runs on the **CPU**. It does not compute your model; it *describes* it, one op at a time, into a queue the GPU drains asynchronously. Every `y = x + 1` walks this path:

```
CPU (one Python thread):  y = x + 1           GPU (asynchronous)
    ↓ ~1-3 µs  PyTorch dispatcher: dtype/device/autograd key lookup
    ↓ ~1-2 µs  ATen kernel selection + argument marshalling
    ↓ ~1-2 µs  cudaLaunchKernel → CUDA runtime → driver  ┌───────────────┐
    ↓          command written to a ring buffer ────────▶ │ command queue │──▶ SMs
  returns IMMEDIATELY; Python runs the next line         └───────────────┘
```

Two numbers set up the whole lesson. A kernel **launch** costs roughly **5-10 µs of CPU work** in eager PyTorch (dispatch + runtime + driver), plus a few µs of queue latency before the SMs start; a batch-1 decode kernel — one projection, one norm, one KV append — **runs for maybe 10-30 µs**, and less on an H100. When CPU work per op is the same order as GPU work per op you are **overhead-bound**: the queue runs dry, the SMs stall between kernels, and your bottleneck is a single Python thread. A faster GPU shortens the kernels and *widens the gaps*. It changes nothing.

---

## Do the arithmetic for one real decode step

A 32-layer decoder, one token, eager PyTorch. Per layer you launch roughly: input norm, Q/K/V projections, RoPE, KV append, attention, output projection, residual, post-norm, two or three MLP matmuls, an activation, another residual — call it **8-12 kernels per layer**, plus embedding, final norm, LM head, sampling.

```
kernels per token   ≈ 32 layers × ~9 kernels ≈ 300   (200-400 is the normal range)
CPU work per token  ≈ 300 launches × 7 µs    ≈ 2.1 ms   ← one core, fully busy
```

Against that, the memory-bound floor from [Phase 1 lesson 8](../phase-1/08-inference-math-and-memory.md): a 7B FP16 model streams all **14 GB** of weights per token.

| GPU | HBM bandwidth | Floor = 14 GB ÷ BW | CPU launch work | Overhead ÷ floor |
|---|---|---|---|---|
| T4 | ~320 GB/s | 14e9 ÷ 320e9 = **43.8 ms** | ~2.1 ms | ~5% — irrelevant |
| A100 80GB | ~2,039 GB/s | 14e9 ÷ 2.039e12 = **6.9 ms** | ~2.1 ms | ~30% |
| H100 SXM | ~3,350 GB/s | 14e9 ÷ 3.35e12 = **4.2 ms** | ~2.1 ms | **~50%** |

Read the last row slowly. The GPU's theoretical best is 4.2 ms; the CPU needs 2.1 ms just to say the words. Decode is a strict serial chain — layer *n* can't start before layer *n-1* finishes — so every microsecond the CPU is late is a microsecond the GPU idles. **At batch 1 on an H100, launch overhead can be half your token latency.** Shrink the model to 1-2B and the floor drops *below* the CPU cost: you are then a Python benchmark with a GPU attached. That's why graph capture isn't a micro-optimization in production decode, why vLLM captures graphs at startup, and why TensorRT-LLM bakes the launch sequence into an engine.

---

## Async execution and the #1 benchmarking bug

Because launches return immediately, a naive timer measures **enqueue** time:

```python
t0 = time.perf_counter()
y = a @ b
t = time.perf_counter() - t0                  # ❌ ~10 µs, implies petaFLOPs. Fiction.
a @ b; torch.cuda.synchronize()               # ✅ warmup: autotune, allocator, JIT
t0 = time.perf_counter()
for _ in range(50): y = a @ b
torch.cuda.synchronize()                      # ← the load-bearing line
t = (time.perf_counter() - t0) / 50
start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
start.record(); y = a @ b; end.record()       # ✅ or time on the GPU's own clock
torch.cuda.synchronize(); print(start.elapsed_time(end), "ms")   # elapsed_time is ms
```

The mirror image of this bug is worse, because it hides in *working* code. Anything that needs a GPU value on the CPU forces an **implicit sync** — `.item()`, `.cpu()`, `.numpy()`, `float(t)`, `print(t)`, `if t > 0:`, `t.tolist()` — draining the queue and stopping the CPU from running ahead. A sampling loop that calls `next_token.item()` every step to check for EOS looks harmless and costs you the whole pipeline: the CPU can no longer be five kernels ahead, so *every* launch gap becomes exposed. Real production bug, not a toy one — keep the EOS check on-device, or check once every *k* steps. Full recipe in [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md).

---

## How to tell you're overhead-bound

- **Latency barely moves when you change the work.** Double the batch, latency rises 10%. Halve the precision, nothing. Both roofs predict a change; overhead doesn't care what's in the kernels.
- **The GPU timeline is mostly gaps**, and the **kernels are short and numerous** — hundreds per step, 5-30 µs each, separated by idle stretches as long as themselves. A compute-bound step is a handful of kernels of hundreds of µs.
- **`nvidia-smi` shows low power draw** (90 W on a 300 W card) with a plausible-looking 60-80% "GPU-Util". That counter reports whether *any* kernel was resident in the sample window, not whether the machine was busy — the most misread number in GPU work.
- **One CPU core pinned at ~100%** while the rest idle. That's your launch thread; `py-spy dump --pid <pid>` shows it inside the dispatcher.
- **A faster GPU changes nothing.** Same code, A100 → H100, same tokens/s. Definitive. Any two of these together is a diagnosis; reading the timeline itself — where the gaps are, which op owns them — is [lesson 8](08-profiling-in-practice.md)'s job.

---

## Streams: real, useful, and not the fix for decode

All PyTorch work lands on the **default stream**, which is ordered: kernel *n+1* starts after kernel *n*. Separate streams are independent queues the hardware may overlap.

```python
copy_s, compute_s = torch.cuda.Stream(), torch.cuda.Stream()
host = torch.empty(4096, 4096, pin_memory=True)          # pinned: required for overlap
with torch.cuda.stream(copy_s): dev = host.to("cuda", non_blocking=True)  # copy engine
with torch.cuda.stream(compute_s): y = w @ w                              # SMs, at once
torch.cuda.current_stream().wait_stream(compute_s)       # explicit ordering, not luck
```

Two honest caveats. **Copies overlap compute only from pinned (page-locked) host memory** — a `non_blocking=True` copy out of pageable memory is silently synchronous, which is why data-loader code allocates pinned buffers. And streams **cannot break a serial dependency**: autoregressive decode has nothing to overlap with itself, so streams buy ~zero there. They earn their keep on H2D/D2H overlap, multi-model serving on one GPU, and overlapping a prefill with an unrelated decode.

---

## CUDA graphs — the actual fix

The launch sequence is *identical every step*. You re-describe the same 300 kernels twenty times a second and the description never changes. So describe it once. **Capture** records the sequence — kernels, arguments, stream dependencies — into a graph without really executing it. **Replay** hands the whole DAG to the driver in one call, and the CPU leaves the loop.

```
EAGER — CPU is the bottleneck, GPU idles between kernels
CPU:  |L1|L2|L3|L4|L5|L6|       each L ≈ 7 µs of dispatch + driver work
GPU:  ···|k1|···|k2|···|k3|     each k ≈ 4 µs of real work
         ↑gap ↑gap ↑gap         ← ❌ the machine is EMPTY here
CUDA GRAPH — one launch, no re-description
CPU:  |R|                       one replay ≈ 5-10 µs total
GPU:  ···|k1|k2|k3|k4|k5|k6|    ✅ back-to-back, gaps collapsed
```

```python
static_in = torch.randn(1, 4096, device="cuda", dtype=torch.float16)   # fixed address
s = torch.cuda.Stream()                       # warm up on a side stream FIRST:
s.wait_stream(torch.cuda.current_stream())    # allocator, cuBLAS handles, autotuning
with torch.cuda.stream(s): [model(static_in) for _ in range(3)]
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): static_out = model(static_in)   # capture: nothing really runs
static_in.copy_(next_embedding)               # steady state: write INTO the same buffer
g.replay(); result = static_out.clone()       # ONE launch; read OUT of the same buffer
```

The mechanism is easy. **The constraints are what hurt people**, and knowing them cold separates someone who has shipped graphs from someone who has read about them:

- **Static shapes.** Dimensions are baked in; a new batch size or sequence length needs a *different* graph. Hence the bucketing in real engines — capture for batch 1, 2, 4, 8, 16… and pad the real batch up to the next bucket. Padding wastes work; too many buckets waste capture time and memory.
- **Static addresses.** Every pointer is baked in. Inputs must be copied into the same buffers, outputs read from the same buffers, and **weights and the KV-cache must not move** — no reallocation, no `.to()`, no allocator reshuffle. Source of most "replay produced garbage" bugs.
- **No data-dependent control flow and no CPU-GPU sync inside the region.** `if logits.max() > threshold` can't be captured — the branch was resolved once, at capture, and is now frozen — and an `.item()` during capture fails outright. Sampling included: everything stays on-device.
- **Capture costs time and memory** at warmup: seconds of startup and a private memory pool per graph.

---

## The one-line version, and why batching helps here too

```python
model = torch.compile(model, mode="reduce-overhead")   # fusion + CUDA graphs underneath
```

`mode="reduce-overhead"` is TorchInductor wiring up CUDA graphs for you, static buffers and all. Plain `torch.compile(model)` helps for a second reason: it **fuses** many small ops into fewer kernels, so there are fewer launches to pay for. Notice what that means — fusion attacks **bytes moved and launch count simultaneously**, which is why it's the answer in this lesson *and* the memory-bound one; internals are [lesson 7](07-fusion-and-flash-attention.md). Two things you can now explain about real systems: vLLM's slow startup includes a **graph-capture phase** across its batch-size buckets, and enforcing eager mode makes it measurably slower at low batch sizes ([Phase 5](../../ROADMAP.md#phase-5--production-inference-frameworks)).

Launch cost is **per kernel, not per token**, so decoding a batch of *B* sequences runs the same ~300 kernels and overhead per token falls like **1/B** — 2.1 ms/token at B=1 becomes 2.1 ms ÷ 32 ≈ **66 µs/token** at B=32. Combine that with [lesson 5](05-roofline-model.md)'s result — decode's arithmetic intensity rises roughly with *B*, walking you up the memory roof toward the ridge point — and you get the most important sentence in serving: **batching raises arithmetic intensity and amortizes launch overhead with one knob.** That's why continuous batching is the highest-leverage lever you own, and it's all of [Phase 3](../../ROADMAP.md#phase-3--serving-fundamentals-batching-queueing-scheduling).

---

## Try it (Colab free tier)

```python
import torch, time
def bench(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters
# (a) the sync bug — the SAME 50 matmuls, timed two ways
n = 4096; f = 2 * n ** 3
a = torch.randn(n, n, device="cuda", dtype=torch.float16); b = torch.randn_like(a)
a @ b; torch.cuda.synchronize()                        # warmup
t0 = time.perf_counter()
for _ in range(50): c = a @ b
wrong = (time.perf_counter() - t0) / 50; torch.cuda.synchronize()    # enqueue time only
right = bench(lambda: a @ b, iters=50, warmup=5)
print(f"(a) no sync {wrong*1e6:7.1f}us = {f/wrong/1e12:7.1f} TFLOP/s (fiction)")
print(f"(a) synced  {right*1e6:7.1f}us = {f/right/1e12:7.1f} TFLOP/s (real)")
# (b) 100 launches of ~zero work: 50 x (mul + add) on 1024 floats
x = torch.randn(1024, device="cuda"); N = 100
def chain(t):
    for _ in range(50): t = t * 1.0001 + 0.0001
    return t
eager = bench(lambda: chain(x))
compiled = torch.compile(chain, mode="reduce-overhead")
for _ in range(10): compiled(x)                        # trace, codegen, THEN capture
torch.cuda.synchronize()
comp = bench(lambda: compiled(x))
# (c) capture the same chain yourself and replay it
static_in = x.clone()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): [chain(static_in) for _ in range(3)]
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): static_out = chain(static_in)
graph = bench(lambda: g.replay())
print(f"(b) eager    {eager*1e6:8.1f}us/iter -> {eager*1e6/N:5.2f}us per launch")
print(f"(b) compiled {comp *1e6:8.1f}us/iter -> {eager/comp :5.2f}x speedup")
print(f"(c) graph    {graph*1e6:8.1f}us/iter -> {eager/graph:5.2f}x; implied launch cost {(eager-graph)*1e6/N:5.2f}us/op")
```

Record those three numbers — they belong in your Phase 2 exit artifact. You should see roughly this shape: the unsynced matmul is absurdly fast and the synced one believable; (b) and (c) land far below eager; and the implied per-launch cost lands in the **~5-10 µs** range predicted at the top of this lesson. GPU work in (b)/(c) is near zero, so what you're measuring *is* the overhead — that's the point. Then rerun (b) with `x = torch.randn(4096, 4096, device="cuda")` and watch the speedup vanish: with real work per kernel, there's no overhead left to remove.

---

## Key takeaways

- The **third regime**: the GPU idles because one CPU thread can't describe work fast enough — no roofline plots it, and "memory 8% / compute 4%" is its signature. A kernel **launch** costs ~5-10 µs of CPU work while a batch-1 decode kernel runs ~10-30 µs, so the queue runs dry and a faster GPU only widens the gaps.
- **~300 kernels/token × 7 µs ≈ 2.1 ms of CPU work** against a **4.2 ms** memory-bound floor for 7B FP16 on an H100 — launch overhead can be **half your token latency**.
- CUDA is async: time with **`torch.cuda.synchronize()` or CUDA events**, and treat `.item()`, `.cpu()`, `print(t)`, `if t > 0` as pipeline stalls hiding in working code.
- **Streams** overlap independent work and need pinned memory for copy/compute overlap; they do **not** fix a serial dependency chain like decode.
- **CUDA graphs** replay a captured launch sequence in one call. The price: **static shapes, static addresses, no data-dependent branches, no internal syncs**, plus warmup capture cost — hence batch-size bucketing in vLLM and TensorRT-LLM.
- `torch.compile(model, mode="reduce-overhead")` is the one-line version; and since launch cost is per **kernel**, overhead per token also falls ~**1/B** — batching fixes intensity and overhead with the same knob.

**Next:** [Kernel fusion & FlashAttention →](07-fusion-and-flash-attention.md) — the optimization that attacks the last two lessons at once, turning many HBM-round-tripping kernels into one that never leaves SRAM.
