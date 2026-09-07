# Projects Index

Every project below maps to a phase in [`ROADMAP.md`](../ROADMAP.md). Status is tracked manually — update it as you go. "Small" projects should take days; "Large" projects should take 1-3 weeks each.

| # | Project | Phase | Size | Status | Path |
|---|---|---|---|---|---|
| 01 | Tiny inference server: naive vs dynamic-batched, benchmarked | 3 | Small | Not started | `01-tiny-inference-server` |
| 02 | Continuous-batching engine (in-flight scheduling) | 3 | Large | Not started | `02-continuous-batching-engine` |
| 03 | GPT-2 forward pass from scratch (NumPy/raw tensors, manual KV-cache) | 1 | Small→Large | Not started | `03-gpt2-from-scratch` |
| 04 | Roofline profiling: matmul + elementwise op benchmarking | 2 | Small | Not started | `04-roofline-profiling` |
| 05 | Fused Triton kernel (softmax or bias+GELU) vs eager PyTorch | 2 | Large | Not started | `05-triton-fused-kernel` |
| 06 | Quantization shootout: FP16 vs INT8 (bitsandbytes) vs INT4 (GPTQ/AWQ) | 4 | Small | Not started | `06-quantization-shootout` |
| 07 | Mini PagedAttention: block-based KV-cache manager with prefix sharing | 4 | Large | Not started | `07-mini-paged-attention` |
| 08 | Speculative decoding end-to-end (draft + verify) | 4 | Large | Not started | `08-speculative-decoding` |
| 09 | Framework shootout: vLLM vs TGI on identical benchmark harness | 5 | Small | Not started | `09-framework-shootout` |
| 10 | Triton Inference Server multi-stage ensemble pipeline | 5 | Large | Not started | `10-triton-ensemble-pipeline` |
| 11 | Manual tensor parallelism with `torch.distributed` | 6 | Small | Not started | `11-manual-tensor-parallel` |
| 12 | Multi-replica serving + prefix-aware sticky load balancer | 6 | Large | Not started | `12-sticky-load-balancer` |
| 13 | Prometheus + Grafana observability stack for an inference server | 7 | Small | Not started | `13-observability-stack` |
| 14 | Canary/shadow deployment harness with automated quality diff | 7 | Large | Not started | `14-canary-shadow-harness` |
| 15 | Multi-stage Dockerfile + GPU-aware CI/CD benchmark gate | 8 | Small→Large | Not started | `15-docker-cicd-gate` |
| 16 | CPU-only serving shootout: ONNX Runtime quantized vs `llama.cpp` GGUF vs GPU FP16 | 9 | Small | Not started | `16-cpu-vs-gpu-serving` |
| 17 | Mini RAG pipeline with per-stage latency budgets, rate limiting, and prompt-injection guardrail | 9 | Large | Not started | `17-rag-pipeline-guardrails` |
| 18 | **Capstone**: nano-vLLM (continuous batching + PagedAttention + streaming API) | 10 | Capstone | Not started | `18-nano-vllm` |
| 19 | **Capstone**: multi-modal production pipeline (STT → LLM → TTS or similar) | 10 | Capstone | Not started | `19-multimodal-pipeline` |
| 20 | **Capstone**: cost-optimization case study with real cloud GPU numbers | 10 | Capstone | Not started | `20-cost-optimization-study` |
| 21 | **Capstone**: production-grade RAG service (retrieval + observability + CI gate + guardrails) | 10 | Capstone | Not started | `21-production-rag-service` |

## How to work a project
1. Read the matching phase in `ROADMAP.md` fully (Learn + Study-this-code sections) before writing code.
2. Do the matching lab(s) in `labs/README.md` first — they prime you with the exact real-world code patterns you're about to reimplement.
3. Build. Run the [`playbooks/benchmarking.md`](../playbooks/benchmarking.md) methodology on your result — every project should end with a real, honest numbers table, not just "it works."
4. Run [`playbooks/production-readiness-checklist.md`](../playbooks/production-readiness-checklist.md) against large projects before considering them done.
5. Update this table's Status column and commit.

## Notes
Nothing is pre-built here on purpose — build each project yourself once you've studied the matching phase. Folders get created when you start a project, not before.
