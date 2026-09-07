# Playbook: Profiling Inference Systems

Rule zero: **measure before you optimize, measure after you optimize, never trust intuition about where time goes.** Systems performance work is dominated by surprises.

## 1. Know which layer you're profiling
Inference latency is the sum of several layers — profile the *right* one:
1. **Network/HTTP layer** — connection handling, serialization (JSON/protobuf), TLS.
2. **Python/application layer** — request parsing, queueing, scheduling logic, GIL contention.
3. **Framework/tensor layer** — PyTorch op dispatch overhead, data transfer host↔device.
4. **GPU kernel layer** — actual CUDA kernel execution time, memory bandwidth utilization.

Most people jump straight to GPU profiling and miss that their bottleneck is Python-level queueing overhead (layer 2) — profile top-down, not bottom-up.

## 2. Tools per layer
| Layer | Tool | What to look for |
|---|---|---|
| Network/HTTP | `wrk`, `locust`, browser devtools / `curl -w` | connection reuse, TLS handshake overhead |
| Python app | `py-spy top` / `py-spy record` (sampling profiler, works on a running process without restart) | GIL-bound loops, unexpected blocking calls in an async server |
| PyTorch ops | `torch.profiler` (`torch.profiler.profile(...)`, export Chrome trace, open in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/)) | host-device sync points (`.item()`, `.cpu()` calls mid-loop — these silently serialize your pipeline), op-level time breakdown |
| GPU kernels | **Nsight Systems** (`nsys profile`) for timeline view across CPU+GPU; **Nsight Compute** (`ncu`) for deep single-kernel analysis (occupancy, memory throughput %, achieved vs peak FLOPs) | kernel launch gaps (CPU not keeping GPU fed — "launch-bound"), low achieved memory bandwidth vs peak, warp divergence |
| Fleet-level GPU | `nvidia-smi dmon`, **DCGM exporter** → Prometheus | GPU utilization %, memory used, power draw, ECC errors across many machines |

## 3. The workflow
1. **`nvidia-smi dmon -s u`** while your load test runs — if GPU utilization is well below 100% under load, your bottleneck is *not* the GPU. Look at layers 1-2 first.
2. **`py-spy top --pid <pid>`** on your running server process — instantly shows which Python function is hot, with zero code changes or restarts. Great first move on any "why is this slow" question.
3. **`torch.profiler`** around the forward pass specifically — look for `cudaStreamSynchronize` / `cudaMemcpy` entries that indicate accidental host-device syncs breaking async execution (a very common bug: any `.item()`, `print(tensor)`, or Python `if` on a GPU tensor's value forces a sync).
4. **`nsys profile python your_script.py`** for the full-timeline picture — spot gaps between kernel launches (CPU-bound launch overhead, fixable with CUDA graphs) vs back-to-back kernels with low throughput (genuinely compute/memory bound, fixable with better kernels/quantization).
5. **`ncu --set full python your_script.py`** (or target a specific kernel) when you need to know *why* a specific kernel is slow — check "Achieved Occupancy" and "Memory Throughput" vs theoretical peak.

## 4. Common findings (in rough order of how often they turn out to be the real culprit)
1. Accidental CPU-GPU synchronization points killing pipeline overlap (`.item()`, logging tensor values, `.cpu()` in a hot loop).
2. Small batch sizes leaving the GPU launch-bound rather than compute-bound (fix: batching, CUDA graphs).
3. Python-level scheduling/queueing overhead dominating at high QPS (fix: move hot path logic out of the GIL, e.g. Rust router like TGI, or `asyncio` structured correctly).
4. Unnecessary data movement/copies (tokenization on CPU blocking the main loop, unpinned memory for host-device transfer).
5. Genuinely memory-bandwidth-bound decode — the "good" finding, meaning your next lever is quantization/batching, not micro-optimization.

## 5. Reference reading
- NVIDIA Nsight docs (official) — work through the "Nsight Systems Quickstart" once, hands-on, on any script.
- PyTorch docs: ["PyTorch Profiler"](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) recipe.
- Blog: [Horace He — "Making Deep Learning Go Brrrr"](https://horace.io/brrr_intro.html) for the mental model of compute/memory/overhead-bound classification before you even open a profiler.
