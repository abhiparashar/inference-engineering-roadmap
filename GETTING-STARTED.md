# Getting Started — Prerequisites and Compute Access

Read this before Phase 0. It answers the two questions that actually block beginners: *"do I know enough?"* and *"what hardware do I run this on?"*

---

## Part 1 — Prerequisites (and what to do if you're missing them)

You do **not** need an ML degree. You need these five things; each has a concrete, free way to close the gap.

| Prerequisite | Bar you need to clear | If you're missing it |
|---|---|---|
| **Python** | Comfortable with classes, decorators, `asyncio` basics, virtualenvs, `pip` | Any solid Python course; specifically practice `asyncio` (you'll use it in every serving project) |
| **Command line / Linux basics** | Navigate, `ssh` into a remote box, edit files, read logs, manage processes | Practice by renting a cloud GPU box for an hour and living in it |
| **Git** | Branch, commit, PR — you'll be reading OSS repos constantly and eventually contributing | GitHub's own Git docs |
| **PyTorch basics** | Tensors, shapes, `.to(device)`, `nn.Module`, no-grad inference mode | Official PyTorch "Learn the Basics" tutorial (~4-6 hours). Skip training-heavy content; you need *inference* mechanics |
| **Math minimum** | Matrix multiply (shapes!), dot products, softmax, basic probability for sampling | 3Blue1Brown "Essence of Linear Algebra" (visual, ~3 hours) — enough for this entire roadmap |

**What you don't need**: backprop derivations, optimizer theory, model architecture research, or a strong statistics background. This is a *systems* discipline. You care about bytes moved, kernels launched, requests queued, and dollars per million tokens.

---

## Part 2 — Compute access (the real blocker)

Most of this roadmap targets **NVIDIA GPUs**, because that's what production inference runs on and what every tool (`nvidia-smi`, Nsight, vLLM, TensorRT-LLM, bitsandbytes) assumes. Plan your compute deliberately — this is the #1 reason people stall at Phase 2.

### If you're on an Apple Silicon Mac (M-series)
Be clear-eyed: **there is no CUDA on a Mac.** What that means concretely:

**Works on your Mac:**
- All of Phase 0, Phase 1 (transformer internals, from-scratch GPT-2 — CPU/MPS is fine for tiny models)
- PyTorch with the **MPS backend** (`torch.device("mps")`) for small-model inference
- `llama.cpp` with **Metal** acceleration — genuinely good on Apple Silicon; excellent for Phase 9's CPU/edge work
- ONNX Runtime (CPU + CoreML execution providers) — Phase 9
- All server/architecture work: FastAPI servers, batching schedulers, queueing, load testing, Prometheus/Grafana, Docker, Terraform configs — Phases 3, 7, 8 are largely hardware-agnostic
- Reading source code and doing every lab in [`labs/`](labs/README.md)

**Does NOT work on your Mac:**
- `nvidia-smi`, Nsight Systems/Compute (all Phase 2 GPU profiling)
- CUDA kernels, Triton kernels (Phase 2 large project)
- vLLM/TensorRT-LLM/TGI on GPU, `bitsandbytes` INT8/INT4 GPU paths (Phases 4, 5, 6)
- Tensor parallelism across GPUs (Phase 6)

**Conclusion**: you can do roughly Phases 0-1, 3, 7, 8, and much of 9 locally. For Phases 2, 4, 5, 6 you must rent or borrow an NVIDIA GPU. Budget for it; it's cheap if you're disciplined about shutting instances down.

### Free GPU options (start here)
- **Google Colab (free tier)** — T4 GPU, session/idle limits, occasional unavailability. Perfect for Phase 2's profiling exercises and Phase 4's quantization experiments on small models.
- **Kaggle Notebooks** — a weekly GPU quota (T4/P100 class) with longer sessions than Colab free. Good for anything that takes 1-9 hours.
- Both are notebook-first, which is fine for profiling/benchmarking experiments but awkward for multi-service work (Phase 5-7) — use rentals for those.

### Paid rentals (for Phases 5-6 and capstones)
Marketplace/cloud options, roughly cheapest → most expensive: **vast.ai**, **RunPod**, **Lambda Labs**, then hyperscalers (AWS/GCP/Azure). Consumer-class cards (3090/4090) are dramatically cheaper than datacenter cards (A100/H100) and are sufficient for everything in this roadmap except large-model multi-GPU work.

> Pricing changes constantly and varies by region/availability — **check live pricing before you plan a budget.** As a rough order of magnitude at time of writing: consumer 24GB cards are commonly under $0.50/hr on marketplaces, A100 80GB is typically a few dollars per hour, H100 more. Treat these as ballpark only.

**Cost discipline rules (these are the same habits that make you good at Phase 7):**
1. Always use hourly/spot instances, never monthly commitments while learning.
2. Set an alarm/timer *before* you start. Idle GPUs are the #1 way learners waste money.
3. Bake your environment into a Docker image or a setup script so a fresh box is productive in <5 minutes — you will spin boxes up and down constantly.
4. Download model weights to persistent storage/volume when the provider offers it; re-downloading a 15GB model every session wastes both time and money.
5. Track what you spent per experiment. This is literally the Phase 7 skill (cost per token) applied to your own learning.

### Choosing model sizes while learning
Don't start with 70B models. Use the smallest model that still exhibits the phenomenon you're studying:
- Phase 1-3 mechanics: GPT-2 small / distilgpt2 / any ~0.5-1.5B model (Qwen2.5-0.5B, TinyLlama class)
- Phase 4 quantization effects: ~1-7B (small enough to iterate fast, big enough that quantization matters)
- Phase 5-6 framework/parallelism work: 7-8B (fits one 24GB card in FP16 borderline / comfortably quantized), then artificially constrain memory to force multi-GPU behavior instead of paying for a 70B run

You learn the same lessons for 1/100th the cost. Frontier-size runs prove nothing extra about your *engineering*.

---

## Part 3 — Environment setup checklist

- [ ] Python 3.10+ with a virtualenv per project (or `uv`/`conda` if you prefer)
- [ ] PyTorch installed (MPS build on Mac, CUDA build on rented GPU boxes)
- [ ] `git` configured with your GitHub account
- [ ] Docker Desktop (Mac) / Docker Engine (Linux boxes)
- [ ] `py-spy`, `httpx`, `locust` (load testing), `prometheus_client` — your recurring toolkit
- [ ] On GPU boxes: verify `nvidia-smi` works, install Nsight Systems, confirm `torch.cuda.is_available()`
- [ ] An account on at least one GPU rental provider, with billing alerts enabled

## Part 4 — How to not stall

1. **Timebox reading.** If a paper isn't clicking in 45 minutes, move to the code or the lab and come back.
2. **Always produce an artifact.** Every phase must end with something committed to this repo — a benchmark table, a writeup, working code. No artifact = phase not done.
3. **Prefer the smallest reproduction.** Most inference concepts (batching, KV-cache, paging, speculative decoding) can be demonstrated on a tiny model on a laptop. Rent GPUs for *scale*, not for *understanding*.
4. **Log your numbers honestly**, including the ones that disprove your assumption. That habit is most of what makes a top-1% performance engineer.
