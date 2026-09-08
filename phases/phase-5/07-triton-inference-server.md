# 7 — Triton Inference Server

> **You'll be able to say:** "Triton is layer 1 plus a model repository: it serves many models, from many frameworks, in one process, each with its own `config.pbtxt` — instance groups (how many copies, on which device), dynamic batching (`preferred_batch_size` + `max_queue_delay_microseconds`, which is Phase 3's latency-for-throughput trade as two numbers), and optional sequence batching for stateful models. An **ensemble** wires several models into one served graph so intermediate tensors never leave the server, and each stage still batches independently across concurrent requests — which is why an ensemble beats a client orchestrating three HTTP calls."

Triton is the least glamorous and most widely deployed thing in this phase. Outside pure-LLM shops — recsys, vision, speech, fraud, ranking — it is *the* serving layer, and knowing it is a large fraction of what "inference engineer" means at companies whose models aren't LLMs ([Phase 9](../../ROADMAP.md#phase-9--beyond-llms-recsysvisionspeech-hardware-diversity-edge-and-security)).

---

## The model repository is the API

Triton doesn't have a "load this model" call in the usual sense; it watches a directory.

```
  model_repository/
  ├── preprocess/                    # Python backend: resize/normalize, or feature extraction
  │   ├── config.pbtxt
  │   └── 1/model.py
  ├── classifier/                    # ONNX Runtime / TensorRT / PyTorch backend
  │   ├── config.pbtxt
  │   └── 1/model.onnx
  ├── postprocess/
  │   ├── config.pbtxt
  │   └── 1/model.py
  └── pipeline/                      # ensemble: no weights, just a graph
      ├── config.pbtxt
      └── 1/                         # empty dir, but it must exist
```

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:<tag>-py3 \
  tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=30
```

Ports, in the order you'll need them: **8000** HTTP inference, **8001** gRPC, **8002** Prometheus metrics. Health/readiness at `/v2/health/ready` — which is what your Kubernetes probes hit ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)).

**Backends** are plugins implementing the same C API: `tensorrt`, `onnxruntime`, `pytorch` (TorchScript), `python`, `openvino`, `vllm`, `tensorrtllm`, plus DALI for GPU-accelerated preprocessing. One process, several frameworks, one protocol — that's the entire pitch, and it's why a company with 40 heterogeneous models runs Triton and not 40 bespoke Flask services.

---

## `config.pbtxt`, field by field

```protobuf
name: "classifier"
backend: "onnxruntime"
max_batch_size: 32                 # 0 = batching disabled; >0 = leading dim is the batch axis

input [ { name: "pixel_values"  data_type: TYPE_FP32  dims: [3, 224, 224] } ]
output [ { name: "logits"       data_type: TYPE_FP32  dims: [1000] } ]
                                   # dims EXCLUDE the batch dimension when max_batch_size > 0
                                   # use -1 for a variable extent (e.g. sequence length)

instance_group [
  { count: 2  kind: KIND_GPU  gpus: [0] }    # two concurrent copies of the model on GPU 0
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 2000         # wait up to 2 ms to form a preferred batch
  preserve_ordering: false
  default_queue_policy {
    timeout_action: REJECT                   # admission control: shed instead of queueing forever
    default_timeout_microseconds: 50000
    max_queue_size: 256
  }
}

model_warmup [ { name: "warmup" batch_size: 8
                 inputs { key: "pixel_values" value: { data_type: TYPE_FP32
                          dims: [3,224,224] zero_data: true } } } ]

version_policy { latest { num_versions: 1 } }
response_cache { enable: true }              # exact-input response caching, if that's meaningful
```

The four fields that decide your performance:

1. **`max_batch_size`** — the ceiling on a formed batch. Zero disables batching entirely (correct for models whose first dim isn't a batch axis, or for stateful/streaming models).
2. **`instance_group`** — how many *copies* of the model run concurrently, and where. This is the knob that fills a GPU when one model instance can't (small models, or models with CPU-side work between GPU calls) and the knob that oversubscribes it when set carelessly. Two instances × batch 32 means up to 64 in flight and roughly 2× the activation memory.
3. **`dynamic_batching`** — group *whole requests* arriving close in time into one batch. `max_queue_delay_microseconds` is the wait window: this is exactly [Phase 3 lesson 3](../phase-3/03-dynamic-batching.md), and the arithmetic there applies unchanged — the window is a latency tax you pay on every request in exchange for a bigger batch, worth it only when the batch actually grows.
4. **`default_queue_policy`** — timeouts, max queue size, and `timeout_action: REJECT`. This is load shedding ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)) as configuration. Set it. The default of "queue forever" turns overload into a timeout storm at the client.

Also worth knowing:

- **`sequence_batching`** for stateful models (streaming ASR, session-based recommenders): routes all requests of a correlation ID to the *same* model instance and gives it start/end/ready control tensors. It's how you serve a model whose state lives in the server between calls — and the routing constraint it implies is the same "sticky routing" idea that prefix caching forces on LLM fleets ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)).
- **`model_warmup`** — run dummy inferences at load so the first real request doesn't eat cuDNN/TensorRT autotuning and allocator warmup. The Triton equivalent of CUDA-graph capture at startup ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)).
- **`rate_limiter`, `priority_levels`** — cross-model resource arbitration when several models share a GPU.

### Dynamic batching vs continuous batching — do not confuse these

```
  TRITON DYNAMIC BATCHING (non-generative models: rankers, CNNs, embedders)
    requests r1 r2 r3 arrive within the delay window
    → ONE batch [r1,r2,r3] → ONE forward pass → all three responses
    Each request = one forward pass. Batch is static once formed.

  LLM CONTINUOUS BATCHING (vLLM/TGI/SGLang/TRT-LLM)
    each request needs HUNDREDS of forward passes; sequences join and leave
    the batch every step. A queue-delay window would be meaningless.
```

Serving an LLM through Triton therefore means using a backend that does its own continuous batching internally (`tensorrtllm` or `vllm` backend), with Triton's own dynamic batcher **disabled** for that model. Getting this wrong — enabling dynamic batching in front of an LLM backend — is a classic misconfiguration.

---

## Ensembles: a pipeline as one served model

```protobuf
name: "pipeline"
platform: "ensemble"
max_batch_size: 32

input  [ { name: "IMAGE_BYTES" data_type: TYPE_UINT8  dims: [-1] } ]
output [ { name: "LABEL"       data_type: TYPE_STRING dims: [1] } ]

ensemble_scheduling {
  step [
    { model_name: "preprocess"  model_version: -1
      input_map  { key: "raw"          value: "IMAGE_BYTES"   }
      output_map { key: "pixel_values" value: "pp_out"        } },
    { model_name: "classifier"  model_version: -1
      input_map  { key: "pixel_values" value: "pp_out"        }
      output_map { key: "logits"       value: "logits"        } },
    { model_name: "postprocess" model_version: -1
      input_map  { key: "logits"       value: "logits"        }
      output_map { key: "label"        value: "LABEL"         } }
  ]
}
```

What you get, and why it's not the same as three HTTP calls from the client:

| Property | Ensemble | Client orchestrating 3 endpoints |
|---|---|---|
| Intermediate tensors | Stay in server memory (GPU→GPU where possible) | Serialized, sent over the network, deserialized — twice |
| Network round trips | 1 | 3 |
| Per-stage batching | Each stage batches across *different* concurrent pipelines | Only if the client batches, which it can't across users |
| Failure/versioning | One atomic served artifact with versions | Three deployments to keep in sync |
| Per-stage scaling | `instance_group` per stage | Separate services (this is where Ray Serve wins — [lesson 8](08-ray-serve-and-composition.md)) |

**Per-stage batching is the subtle win.** With 20 concurrent pipeline requests, the classifier sees a batch of ~20 even though each user sent one image, because stage 2 batches across all in-flight pipelines. That's throughput you cannot get client-side.

**BLS (Business Logic Scripting)** is the escape hatch: a Python-backend model that calls other models programmatically (`pb_utils.InferenceRequest(...).exec()`), so you can branch, loop, and apply conditions — an ensemble is a static DAG, BLS is code. Cost: the Python process becomes a participant in every request, with its own GIL and copies. Use ensembles for straight-line pipelines, BLS where you genuinely need control flow.

---

## Sizing a pipeline: give every stage a latency budget

For a 3-stage pipeline with a 200 ms p99 SLO, write the budget down before configuring anything:

```
  stage         device  per-item  batch  queue delay  budget  notes
  ─────────────────────────────────────────────────────────────────────────────
  preprocess    CPU      6 ms      8      2 ms         30 ms  raise instance count,
                                                              not batch size (CPU-bound)
  classifier    GPU      0.4 ms   32      3 ms         40 ms  batching pays here: 32×
                                                              in ~13 ms of GPU time
  postprocess   CPU      1 ms      8      1 ms         10 ms
  network+serde  —        —         —      —           20 ms
  ─────────────────────────────────────────────────────────────────────────────
  total p50 ≈ 100 ms, leaving 100 ms of headroom for p99 queueing (Phase 3 lesson 5)
```

Rules that fall out of doing this honestly:

- **Every stage's queue delay is additive on every request.** Three stages at 5 ms of delay is 15 ms of pure latency tax, paid even when there's nothing to batch with. Set the window to the *smallest* value that measurably raises batch size — measure `nv_inference_queue_duration_us` per model, don't guess.
- **The bottleneck stage decides pipeline throughput.** Scale it with `instance_group` count (CPU stages) or batch size (GPU stages), and re-measure; fixing a non-bottleneck stage changes nothing.
- **A CPU preprocessing stage is the usual culprit.** GPU-accelerated preprocessing (DALI backend) or moving preprocessing into the client are the two real fixes.

---

## Observability and tools

Triton's Prometheus metrics on `:8002/metrics`, per model:

```
  nv_inference_request_success / _failure          counters
  nv_inference_count, nv_inference_exec_count      → avg batch size = count / exec_count  ★
  nv_inference_request_duration_us                 end-to-end in-server
  nv_inference_queue_duration_us                   time waiting to be batched  ★
  nv_inference_compute_input_duration_us           H2D + input prep
  nv_inference_compute_infer_duration_us           the actual forward
  nv_inference_compute_output_duration_us          D2H + output prep
  nv_gpu_utilization, nv_gpu_memory_used_bytes     per GPU
```

The two starred lines are the whole tuning loop: **average batch size** tells you whether dynamic batching is doing anything, and **queue duration** tells you what it costs. If avg batch ≈ 1 while queue duration > 0, your delay window is pure loss — either traffic is too sparse to batch or the window is too short.

Tooling worth using: **`perf_analyzer`** (concurrency sweeps, per-stage breakdown, in the SDK container) and **`model_analyzer`** (automated sweeps over batch size and instance count, producing a Pareto report). Treat `perf_analyzer`'s default concurrency mode as closed-loop and use `--request-rate-range` for open-loop numbers, per [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md).

---

## Do this (2 hours, CPU-only is fine)

1. Stand up Triton with **one** trivial Python-backend model that returns its input. Confirm `/v2/health/ready`, run one inference, read `:8002/metrics`.
2. Add `dynamic_batching` with `max_queue_delay_microseconds: 5000` and hit it with a concurrent client. Compute avg batch size from `nv_inference_count / nv_inference_exec_count` at two load levels, and watch queue duration. You have now measured the Phase-3 tradeoff in someone else's server.
3. Build the 3-stage ensemble (any real work: tokenize → tiny model → postprocess). Verify with one request, then measure per-stage `queue_duration` and `compute_infer_duration` under load and identify the bottleneck.
4. Break it on purpose: set `max_queue_delay_microseconds: 100000` (100 ms) and observe latency; then set `instance_group count: 4` on the bottleneck stage and observe throughput. Keep the numbers — they're part of [lesson 10](10-build-shootout-and-ensemble.md)'s deliverable.

---

**Next:** [Ray Serve and multi-model composition →](08-ray-serve-and-composition.md) — the same pipeline problem solved with independently autoscaled Python services, and when that's the better answer.
