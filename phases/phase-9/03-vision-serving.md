# 3 — Vision and Video Serving

> **You'll be able to say:** "A vision request is five stages — fetch, decode, resize/normalize, transfer, infer — and before anyone optimizes, decode and resize on CPU are usually 50-70% of the wall clock while the GPU sits idle. So I profile the pipeline, not the model: batch large (there is no autoregressive penalty), move decode to NVDEC/DALI when the CPU:GPU ratio says to, keep the transfer pinned and overlapped, and compile the model to a static-shape engine. I also know that a resize-semantics mismatch between training and serving is an accuracy bug that no latency test and no unit test will ever catch."

Vision is the workload where Phase 2's roofline finally reports "compute-bound" and Phase 3's *static* batching is the right answer. It is also the workload where the naive implementation wastes the most hardware, because the expensive part isn't where anyone looks.

---

## The five stages

```
  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐
  │ 1 FETCH  │→ │ 2 DECODE │→ │ 3 RESIZE + │→ │ 4 H2D    │→ │ 5 INFER │
  │ network/ │  │ JPEG/    │  │ NORMALIZE  │  │ transfer │  │ forward │
  │ blob/S3  │  │ H.264    │  │ crop, cast │  │ to GPU   │  │ + post  │
  └──────────┘  └──────────┘  └────────────┘  └──────────┘  └─────────┘
   5-100 ms      3-15 ms/img   1-5 ms/img      0.3-2 ms      2-20 ms
   (often the    (CPU: ~50-    (CPU, often     (PCIe, hide   (the part
   real p99)     150 MB/s/core) memcpy-bound)  it or pay it) people tune)

  Typical naive breakdown for a 224×224 classifier at batch 32, 8 CPU cores:
      decode 46%  ·  resize/normalize 18%  ·  H2D 6%  ·  model 22%  ·  fetch 8%
  ⇒ making the model 2× faster buys you 11%. Fixing decode buys you 40%.
```

**The first action in any vision-serving task is to produce this table for your own pipeline.** A `time.perf_counter()` around each stage, plus `torch.cuda.synchronize()` (or CUDA events) so GPU time is attributed honestly, is enough. Without it you will optimize stage 5 forever.

### Why decode is so expensive

A 1920×1080 JPEG is ~300 KB on the wire and 6.2 MB decoded (`1920 × 1080 × 3`). Baseline libjpeg-turbo decodes roughly 100-200 MB/s/core of *output* pixels, so ~25-50 full-HD frames/s/core. At 1000 images/s you need **20-40 dedicated CPU cores just to decode**, which on a GPU node you do not have: a typical 8-GPU box has 8-16 cores per GPU, and your dataloader is competing with the server, the metrics exporter, and tokenization.

```
  cores_needed_for_decode  ≈  images_per_sec × decoded_MB_per_image
                              ──────────────────────────────────────
                                    decode_MB_per_sec_per_core
  Example: 1000 img/s × 6.2 MB ÷ 150 MB/s  ≈  41 cores
```

Run that formula before provisioning. If it exceeds the cores you have per GPU, the answer is hardware decode, not more threads.

---

## Fixing the preprocessing bottleneck

| Option | Mechanism | When it's right | Cost |
|---|---|---|---|
| **More CPU threads / process pool** | parallel libjpeg-turbo | small scale, cores available | contends with the server; GIL if naive Python |
| **libjpeg-turbo + pillow-simd** | SIMD decode, 2-4× over stock PIL | always — free baseline | none; just dependencies |
| **nvJPEG / NVDEC** | decode on the GPU's dedicated hardware engine | image throughput ≥ few hundred/s; video always | uses GPU; needs `video` driver capability |
| **NVIDIA DALI** | whole pipeline (decode + resize + normalize) as a GPU graph, prefetched | production image/video serving | a second graph to learn and pin |
| **Decode microservice** | separate CPU fleet feeding GPU nodes raw tensors | when CPU:GPU ratio is fixed by instance type | network cost: you now ship 6 MB, not 300 KB |
| **Pre-decoded cache** | store resized tensors, not originals | repeated inference over a fixed corpus (backfills) | storage; 20× the bytes |
| **Client-side resize** | resize before upload | mobile/browser clients | trust and quality control |

Two traps in that table:

- **"Decode microservice" moves bytes the wrong way.** Compressed in, uncompressed out — you turned a 300 KB transfer into 6 MB across your network. Only do it if the decode nodes are co-resident (same host/rack) or you resize *before* shipping.
- **Hardware decode needs the driver capability.** In containers, `NVIDIA_DRIVER_CAPABILITIES` must include `video` or NVDEC silently isn't there ([Phase 8 lesson 2](../phase-8/02-containers-for-gpu-workloads.md)). This exact bug costs teams days: `nvidia-smi` works, CUDA works, and hardware decode falls back to CPU with no error.

NVDEC/NVENC also have **fixed engine counts per GPU** (typically 1-3 NVDEC engines), so hardware decode has its own throughput ceiling independent of SM count — check `nvidia-smi -q | grep -i decoder` style utilization, not just SM utilization, before concluding you are GPU-bound.

---

## Batching: the Phase 3 "bad" technique is the right one here

No autoregression, no KV cache, no variable output length: every image in a batch finishes at the same time. That makes **static batching optimal** and **dynamic batching** (queue + max delay) the production default ([Phase 3 lessons 2-3](../phase-3/02-static-batching.md), [Phase 5 lesson 7](../phase-5/07-triton-inference-server.md)).

```
  ResNet-50-class model, fp16, one mid-range GPU (illustrative shape, measure yours):
  batch   latency   throughput   latency/img   GPU util
    1      3.1 ms     320/s        3.1 ms       ~15%   ← launch-overhead-bound
    8      5.0 ms    1600/s        0.63 ms      ~55%
   32     12.0 ms    2660/s        0.38 ms      ~85%
   64     22.5 ms    2840/s        0.35 ms      ~92%   ← knee
  128     44.0 ms    2910/s        0.34 ms      ~95%   ← +2% throughput, 2× latency
```

The shape is always the same: steep gains to the knee, then latency doubles for nothing. Pick the largest batch whose *total* latency (queue delay + compute) fits the budget:

```
  max_queue_delay  +  compute(batch)  ≤  SLO
  and at steady state:  batch ≈ QPS × max_queue_delay
```

A 100 ms budget at 500 QPS with a 10 ms queue delay gives batch ≈ 5 — so you must either accept a smaller batch or admit that at this QPS you are latency-bound, not throughput-bound, and shrink the replica count instead.

For **offline batch jobs** (moderation backfills, catalog embedding) there is no queue-delay constraint at all: use the biggest batch that fits memory, run on spot capacity, and restart on preemption ([Phase 8 lesson 5](../phase-8/05-scheduling-capacity-and-autoscaling.md)). This is where 5-10× cost differences hide.

---

## Model-side optimization, in the order that pays

1. **Compile to a static-shape engine.** TensorRT (or ONNX Runtime with the TensorRT EP, or `torch.compile` + CUDA graphs) typically gives 1.5-3× over eager PyTorch on convnets and ViTs, mostly via layer fusion (conv+BN+ReLU) and kernel autotuning. Static shapes are the point: fix the batch to a small set of buckets (1, 8, 32) and build one engine per bucket ([lesson 8](08-model-formats-and-runtimes.md); the engine is an arch-pinned artifact per [Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)).
2. **Use the right memory layout.** Tensor cores want **NHWC** (channels-last) for convolutions; PyTorch defaults to NCHW and inserts transposes. `model.to(memory_format=torch.channels_last)` plus channels-last inputs is often 20-40% on convnets for one line of code.
3. **fp16/bf16, then INT8.** Vision models quantize far better than LLMs: post-training INT8 with a few hundred calibration images typically costs < 1% top-1 accuracy and buys 2-3× ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)). This is the standard production configuration and one of the few places where INT8 is genuinely uncontroversial.
4. **Overlap H2D with compute.** Pinned host memory + a separate copy stream + double buffering; the transfer then costs ~0 instead of 6% serialized.
5. **Then, and only then**, consider a smaller/faster architecture.

---

## The accuracy bug nobody catches: preprocessing mismatch

This is the highest-value paragraph in the lesson. **Your serving preprocessing must be bit-compatible with your training preprocessing**, and "resize an image" is ambiguous in at least six ways:

| Decision | Options that silently differ | Typical accuracy impact |
|---|---|---|
| Resize interpolation | bilinear / bicubic / nearest / Lanczos; antialias on or off | **1-5% top-1** (antialias alone is famously ~1-2%) |
| Resize target | short-side-resize-then-center-crop vs direct resize vs letterbox | 1-3%, plus geometry errors for detection |
| Channel order | RGB vs BGR (OpenCV reads BGR!) | catastrophic to mild depending on the model |
| Normalization | ImageNet mean/std vs [0,1] vs [-1,1]; per-channel vs global | large |
| Dtype/rounding | uint8→float cast point, rounding of the resize | < 0.5%, but breaks numerical comparisons |
| EXIF orientation | honored by PIL, ignored by raw decoders | some images silently rotated |
| Decoder | libjpeg vs nvJPEG produce slightly different pixels | < 0.2% typically, but nonzero |

None of it raises an error. Latency is unchanged. Your unit tests pass because they check shapes. The model just gets quietly worse — the exact failure class as the chat-template regression in [Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md), and the fix is the same: **a quality gate that compares distributions**.

The discipline:

- **Preprocessing is part of the model artifact.** Either bake it into the graph (ONNX/TensorRT preprocessing nodes, DALI pipeline definition, or a Triton ensemble's preprocess model) or version the config alongside the weights with a hash.
- **Golden-image test in CI.** 100 fixed images, per-image logits stored, assert cosine similarity > 0.999 and identical argmax. It catches decoder swaps, interpolation changes, and channel-order regressions in one second.
- **When you move decode to the GPU, re-run the accuracy eval**, not just the benchmark. nvJPEG + GPU bilinear resize is *not* pixel-identical to PIL, and DALI's default antialias behavior has historically differed. Measure the delta once and record it.

---

## Video: the same pipeline plus time

Video adds three genuinely new problems.

```
  1 hour of 1080p30 video
    = 108,000 frames = 670 GB decoded  (vs ~1-2 GB compressed)
  Inference on every frame is almost always the wrong default.
```

| Problem | Detail | Standard answer |
|---|---|---|
| **Frame selection** | most frames are redundant | sample 1-5 fps; or keyframe/I-frame only; or motion-triggered |
| **Decode cost** | H.264/H.265 decode is expensive; sequential dependency inside a GOP | NVDEC; decode whole GOPs, not random frames |
| **Temporal models** | need windows of frames → state across chunks | fixed-stride sliding windows with cached features |
| **Egress and storage** | streaming *out* multimodal data dominates cost | process near the data; never ship decoded frames ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md)) |
| **Live streams** | arrive in real time; cannot batch across time | batch across *streams*: 32 concurrent streams = batch 32 |
| **Long-tail formats** | a codec/container your decoder can't handle | a hard timeout and a dead-letter path; never let one file wedge a worker |

**Batching across streams instead of across time** is the key structural insight for live video, and it's the same idea as batching across sessions in speech ([lesson 4](04-speech-and-streaming.md)): the parallelism you have is concurrency, not lookahead.

Cost sanity check for an offline video pipeline: at 1 fps sampling, 10,000 hours of video = 36M frames; at 2500 frames/s/GPU that's 4 GPU-hours of *inference* — and likely 10× that in decode if you didn't use NVDEC. The decode:inference ratio is the number to report.

---

## Serving-stack notes

- **Triton Inference Server is the default here**, and vision is what it was designed for: dynamic batcher, multiple model instances per GPU, ensembles for preprocess→infer→postprocess, and per-model instance-group concurrency ([Phase 5 lesson 7](../phase-5/07-triton-inference-server.md)). An ensemble keeps preprocessing on the server where it's versioned with the model instead of scattered across five client codebases.
- **Multiple model instances per GPU** (two or three CUDA streams' worth) fills the gaps left by small batches and H2D stalls; it's the cheapest utilization win after batching. Beyond ~3 instances you mostly add latency variance.
- **Payload format matters.** Accept compressed bytes (JPEG/WebP) over HTTP, not JSON arrays of floats: a 224×224×3 float32 tensor as JSON is ~1.5 MB of text versus ~15 KB as JPEG. gRPC with raw bytes, or Triton's binary tensor extension, avoids a JSON-parsing bottleneck that people mistake for model latency.
- **Postprocessing can be the bottleneck** for detection/segmentation: NMS over thousands of boxes, or a 1024×1024 mask argmax, done in Python per image, easily exceeds the model time. Fuse it into the graph (TensorRT has an NMS plugin; ONNX has `NonMaxSuppression`) or vectorize it on-device.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| GPU utilization < 30%, CPUs pegged | decode/resize-bound | nvJPEG/DALI, more decode cores, or client-side resize |
| Throughput doesn't improve with batch size | preprocessing-bound, or per-instance concurrency = 1 | fix stage 2-3 first; add model instances |
| Hardware decode "not available" in container | missing `video` in `NVIDIA_DRIVER_CAPABILITIES` | set it; verify with a decode smoke test in CI |
| Accuracy dropped after a "perf-only" change | preprocessing semantics changed (interpolation/antialias/decoder) | golden-image logit test; pin preprocessing in the artifact |
| Detection boxes slightly offset | letterbox padding or resize rounding differs from training | replicate training geometry exactly; test with a synthetic grid image |
| p99 >> p50 with stable model time | fetch stage (blob store) tail, or a giant input image | per-stage timeouts, input size limits, hedged fetch |
| OOM on some requests only | unbounded input resolution → huge intermediate activations | validate and cap input dimensions at the edge |
| One worker stuck forever | pathological/corrupt media file | hard decode timeout + dead-letter queue |
| Cost dominated by network | shipping decoded frames between services | process near storage; send compressed bytes |

---

## Do this now (60 minutes)

1. **Build the five-stage table** for any image model you can run locally (a torchvision ResNet is fine), at batch 1 and at your knee batch. Report the percentage per stage. Name the dominant stage and the specific fix.
2. **Find the batch knee.** Sweep batch 1/2/4/8/16/32/64/128; record latency, throughput, and latency-per-image. Then compute the largest batch that fits a 100 ms SLO at 200 QPS, using `batch ≈ QPS × max_queue_delay`.
3. **Break preprocessing on purpose.** Run the same 100 images through (a) PIL bilinear with antialias, (b) PIL bilinear without antialias, (c) OpenCV (note BGR!), and record top-1 agreement and mean logit cosine similarity against (a). The numbers will convince you to write the golden-image gate.
4. **Channels-last, one line.** Convert model and input to `channels_last` and re-measure at your knee batch. Record the delta — free performance you would otherwise never have found.

---

**Next:** [Speech: streaming ASR and TTS →](04-speech-and-streaming.md) — where the latency budget is measured against the wall clock of the audio itself, and you must emit output before the input has finished arriving.
