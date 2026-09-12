# 8 — Model Formats and Runtimes

> **You'll be able to say:** "A model format is a contract about a graph, an operator set and a numerical convention — nothing more. Conversion is therefore a *build step* that can fail in four distinct ways (unsupported op, shape dynamism, numerical drift, semantic mismatch in pre/post-processing), and it must be gated by a tolerance test plus a task-metric check, on the target runtime and hardware. I can name what each of ONNX, GGUF, safetensors, TorchScript, `torch.export`, StableHLO and TensorRT plans is *for*, and I never treat a successful conversion as a validated one."

Every previous lesson in this phase ended with "export it to something else." This lesson is that machinery. It is also the least glamorous and most bug-dense part of non-CUDA inference, and the reason experienced engineers budget weeks, not days, for a port.

---

## What a format actually contains

```
  ┌────────────────────────────────────────────────────────────────┐
  │ WEIGHTS      tensors + dtype + layout                          │
  │ GRAPH        ops, their order, shapes, attributes              │
  │ OPSET        WHICH ops exist and what each one MEANS (version!)│
  │ METADATA     input/output names, shapes, dynamic axes          │
  │ ─────────── and what formats usually DON'T contain ─────────── │
  │ ✗ tokenizer / preprocessing semantics                          │
  │ ✗ sampling parameters, chat template                           │
  │ ✗ the exact numerical behavior (accumulation order, fusion)    │
  └────────────────────────────────────────────────────────────────┘
```

The bottom three lines cause more production incidents than the top four. A format guarantees "the same ops in the same order"; it does not guarantee "the same numbers," and it usually says nothing about the code that turns a user's string or JPEG into a tensor ([lesson 3](03-vision-serving.md)'s preprocessing bug, [Phase 8 lesson 7](../phase-8/07-model-registry-and-artifacts.md)'s artifact-completeness argument).

---

## The format map

| Format | Produced by | Consumed by | What it's for | Watch out |
|---|---|---|---|---|
| **safetensors** | HF ecosystem | everything | **weights only**, zero-copy `mmap`, no code execution | not a graph; you still need the modeling code |
| **PyTorch `.pt`/`.bin` (pickle)** | `torch.save` | PyTorch | checkpoints | **arbitrary code execution on load** — never load untrusted ones ([lesson 9](09-security-and-multi-tenancy.md)) |
| **TorchScript** | `torch.jit.trace/script` | LibTorch (C++), older mobile | frozen graph for C++ serving | legacy path; tracing silently bakes in control flow and shapes |
| **`torch.export` / ExportedProgram** | `torch.export` | AOTInductor, ExecuTorch | the modern PyTorch graph capture | graph breaks / unsupported dynamism surface here, loudly |
| **ONNX** | `torch.onnx.export`, tf2onnx, `optimum` | ONNX Runtime, TensorRT, OpenVINO, mobile NPUs | **the interchange lingua franca** | opset version mismatches; op coverage varies per runtime |
| **StableHLO / HLO** | JAX, PyTorch-XLA | XLA (TPU/GPU/CPU) | the compiler IR of the XLA world | static shapes ([lesson 5](05-hardware-diversity.md)) |
| **TensorRT engine plan** | `trtexec`, TRT/TRT-LLM builder | TensorRT runtime | maximal NVIDIA performance | **not portable**: pinned to GPU arch × TRT version × shapes ([Phase 5 lesson 6](../phase-5/06-tensorrt-llm-and-compiled-engines.md)) |
| **NEFF** | `neuronx-cc` | AWS Neuron runtime | Inferentia/Trainium | pinned to compiler version and buckets |
| **GGUF** | `llama.cpp` converters | llama.cpp, Ollama, LM Studio, many wrappers | **quantized LLM weights + tokenizer + metadata in one file** | LLM-shaped only; quant type names matter (Q4_K_M etc.) |
| **`.tflite` / LiteRT** | TF/JAX converters, ai-edge-torch | TFLite runtimes, NNAPI, delegates | mobile/embedded | static shapes, quantization-first |
| **Core ML `.mlpackage`** | `coremltools` | Apple runtimes | Apple ANE/GPU/CPU | Apple-only; compute-unit placement is the real variable |
| **`.pte`** | ExecuTorch | ExecuTorch runtime | PyTorch on device | newer; delegate coverage evolving |
| **OpenVINO IR (`.xml`/`.bin`)** | OpenVINO converter | OpenVINO runtime | Intel CPU/iGPU/NPU | Intel-centric |

Two structural observations worth carrying:

- **GGUF is the only format in this table that packages the tokenizer and generation metadata with the weights.** That is a large part of why the local-LLM ecosystem is so frictionless: one file, one runtime, it works. Everywhere else, the tokenizer/template is a separate artifact you must version yourself — the mismatch class from [Phase 8 lesson 1](../phase-8/01-why-shipping-is-the-job.md).
- **There are two kinds of format**: *portable* (ONNX, safetensors, GGUF, StableHLO) and *compiled/pinned* (TensorRT plan, NEFF, compiled Core ML, cached delegate blobs). Portable artifacts live in your registry; pinned artifacts are derived, cached, and rebuildable — never your only copy, and always keyed by their full environment tuple.

---

## Execution providers: one graph, many backends

ONNX Runtime's **execution provider (EP)** abstraction is the cleanest expression of the idea, and worth reading in `microsoft/onnxruntime` source:

```
      your ONNX graph
            │
   ┌────────┴─────────────────────────────────────────┐
   │  ORT graph partitioner: ask each EP, in priority  │
   │  order, "which subgraphs can you run?"            │
   └────────┬─────────────────────────────────────────┘
     ┌──────┼───────┬────────┬────────┬────────┬───────┐
     ▼      ▼       ▼        ▼        ▼        ▼       ▼
   CUDA  TensorRT  CPU    OpenVINO  CoreML  NNAPI    QNN
   (fallback is ALWAYS the CPU EP — which is why an unsupported op
    doesn't fail, it just gets slow and you never notice)
```

The crucial operational point, identical in spirit to [lesson 7](07-edge-and-on-device.md)'s NPU partitioning: **silent partial fallback is the default failure mode.** Your model "runs on the GPU/NPU" while three subgraphs execute on CPU with a device transfer at every boundary. Always:

1. Dump the partition/placement report (`ORT_LOGGING_LEVEL=VERBOSE`, TensorRT's `--verbose`, Core ML's performance report, TFLite delegate logs).
2. **Count the partitions.** More than one or two is a red flag.
3. Assert it in CI — "number of CPU-EP nodes == 0" is a legitimate gate, and it catches the regression where an upgraded op version silently drops off the accelerator.

---

## The four ways conversion fails

### 1. Unsupported operator

The most common and most visible. Symptoms: export error, "op not supported by opset N", or a runtime error at load.

Fixes, in order of preference: raise/lower the opset; replace the op with a supported composition in the source model; use the runtime's custom-op API; or (last resort) split the graph and run the unsupported piece on the host. The `optimum` library and per-runtime op-support matrices exist precisely to shortcut this triage — **read the support matrix before starting a port**, not after.

### 2. Shape dynamism

Tracing records one concrete shape. Symptoms: works for batch 1, wrong or failing for batch 8; a sequence-length dimension baked to 128.

```
  torch.onnx.export(..., dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                        "logits":    {0: "batch", 1: "seq"}})
```

Then verify with two different shapes — a conversion "tested" at exactly the traced shape has tested nothing. On static-shape backends (TensorRT with fixed profiles, XLA, Neuron, NPUs) the answer is **shape buckets** plus an artifact per bucket, warmed at startup ([lesson 5](05-hardware-diversity.md)).

### 3. Numerical drift

The dangerous one, because nothing errors. Different fusion, accumulation order, dtype promotion, epsilon handling, or a fast-math flag changes the outputs slightly.

```
  Typical honest tolerances (measured on the same inputs):
    fp32 → fp32 across runtimes:   max |Δ| ~1e-5,   cosine > 0.99999
    fp32 → fp16:                   max |Δ| ~1e-2,   cosine > 0.9995
    fp32 → int8 (static PTQ):      per-op Δ large; judge by TASK METRIC only
  Classifier:  top-1 agreement ≥ 99.5%, mean logit cosine ≥ 0.999
  LLM:         greedy token agreement over 200 prompts ≥ 99%, and
               perplexity delta on a fixed corpus < 0.5%
```

Two rules:

- **Compare distributions and task metrics, never exact strings** — the same principle as [Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)'s quality gate. For generative models, tiny logit drift changes sampled tokens legitimately; greedy decoding is the only stable comparison, and even then divergence compounds along the sequence.
- **Drift that is tiny per-op can be large end-to-end.** A 1e-3 difference in an early layer, amplified through 40 layers and an argmax, flips tokens. Measure at the output you actually serve, not at layer 3.

### 4. Semantic mismatch outside the graph

The graph is correct and the product is wrong, because the *edges* changed: tokenizer version, chat template, image normalization, channel order, sample rate, label ordering, or output post-processing (a softmax the original model applied and the exported one doesn't).

**This is the most common real-world conversion bug and the least discussed.** The defense is to make the artifact complete: bundle the tokenizer/preprocessing config with the model and hash them together, or push preprocessing *into* the graph (ONNX preprocessing ops, a Triton ensemble's preprocess stage, a DALI pipeline definition).

---

## Conversion as a build step

Treat it exactly like compiling a binary ([Phase 8 lessons 7-8](../phase-8/07-model-registry-and-artifacts.md)):

```
  source checkpoint (safetensors + config + tokenizer, digest-pinned)
        │  convert.py  — pinned exporter versions, explicit opset, dynamic axes
        ▼
  portable artifact (ONNX / GGUF / ExportedProgram)   → registry, digest
        │  optimize/quantize — calibration data pinned too
        ▼
  optimized portable artifact                          → registry, digest
        │  compile (TensorRT / Neuron / CoreML / delegate cache)
        ▼
  pinned artifact, keyed by (model, runtime version, hw arch, shapes)
        │
        ▼
  GATES: (a) graph loads on the target runtime
         (b) partition report: zero unexpected CPU fallbacks
         (c) numerical tolerance vs reference on a fixed input set
         (d) task metric within threshold
         (e) latency/throughput within threshold on target hardware
  → only then promote
```

Non-negotiables that teams learn the hard way:

- **Pin every tool version.** `torch`, `onnx`, `onnxruntime`, `tensorrt`, `coremltools`, `llama.cpp` commit. A converter upgrade changes outputs; without pins your "reproducible" artifact isn't.
- **Store the conversion command with the artifact.** `metadata.json` with the exact command, versions, opset, buckets, and calibration-set digest. Otherwise nobody can rebuild it in six months — and rebuild is your rollback path for compiled artifacts.
- **Compile in CI, never at container start.** A 20-minute cold start is the alternative ([Phase 8 lesson 3](../phase-8/03-image-size-and-cold-start.md)).
- **Gate on the target hardware.** A conversion validated on a laptop CPU and deployed to an NPU has been validated for nothing.

---

## Graph optimizations you get from a runtime

Worth knowing what the optimizers actually do, so you can tell whether you need them ([Phase 2 lesson 7](../phase-2/07-fusion-and-flash-attention.md) is the GPU-side version of the same ideas):

| Optimization | Effect | Typical gain |
|---|---|---|
| Constant folding | evaluate static subgraphs once | small, free |
| Op fusion (conv+BN+ReLU, MatMul+Add+GELU) | fewer kernels, fewer memory round trips | 10-40% |
| Layout transformation (NCHW↔NHWC) | match hardware-preferred layout | 10-40% on convnets |
| Attention fusion | a single fused attention kernel | large on transformers |
| Dead-code/identity elimination | remove export cruft (`Identity`, redundant casts) | small |
| Dynamic-shape specialization | compile per bucket | large on compilers |
| Quantization (dynamic/static) | int8 weights/activations | 2-4× size, 1.5-3× speed on CPU |

Enable the runtime's highest safe optimization level first (ORT `ORT_ENABLE_ALL`, OpenVINO defaults, TensorRT builder flags), re-run the tolerance gate, and only then hand-optimize. Note that **aggressive optimization is a numerical change**: fast-math/TF32/fp16 accumulation flags belong in the same gated build step, not flipped on in production.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Export fails: "op not supported in opset N" | op/opset mismatch | change opset, rewrite the op, or add a custom op |
| Works at batch 1, breaks at batch 8 | shapes baked in by tracing | `dynamic_axes` / export with dynamic dims; test two shapes |
| Converted model is *slower* than PyTorch | partial fallback, bad layout, or no graph optimization | read the partition report; set layout; raise optimization level |
| Accuracy degraded slightly, no errors | numerical drift or quantization | tolerance + task-metric gate; per-channel quant; better calibration |
| Output totally wrong | preprocessing/tokenizer mismatch, or channel/label order | bundle preprocessing with the artifact; compare intermediate tensors |
| Same artifact, different results on two nodes | different GPU arch/library version with a portable graph | pin arch in artifact metadata; separate compiled artifacts per arch |
| Engine fails to load after an upgrade | compiled artifact pinned to old runtime version | rebuild in CI as part of the upgrade; keep the portable source artifact |
| Cold start 20 minutes | compiling on startup | compile in CI; cache the plan keyed by environment tuple |
| `torch.load` of a downloaded checkpoint is flagged in review | pickle executes code | use safetensors; verify checksums and provenance ([lesson 9](09-security-and-multi-tenancy.md)) |
| Conversion irreproducible | unpinned tool versions | pin everything; store the command in metadata |

---

## What to read and run

- **Code:** `microsoft/onnxruntime` — read `onnxruntime/core/framework/graph_partitioner.cc`-level concepts via the [EP docs](https://onnxruntime.ai/docs/execution-providers/) first, then the source; `huggingface/optimum` (`optimum.exporters.onnx`) for how a serious library handles per-architecture export configs; `ggerganov/llama.cpp` `convert_hf_to_gguf.py` and `gguf-py/` for a format whose metadata design you can read in an afternoon.
- **Docs:** ONNX operator/opset documentation (skim the versioning rules), `torch.export` docs (the current PyTorch capture story), TensorRT builder docs on optimization profiles.

---

## Do this now (60 minutes)

1. **Export and verify.** Take any small model, export to ONNX with explicit `dynamic_axes`, and run it in ONNX Runtime. Compare against PyTorch on 200 fixed inputs: max absolute difference, mean cosine similarity, and top-1 (or greedy-token) agreement. Write down the tolerances you would gate on.
2. **Break shape dynamism on purpose.** Re-export without `dynamic_axes`, then run at a different batch size. Record the exact error or wrong result. This is a five-minute experiment that inoculates you for years.
3. **Read a partition report.** Run the same ONNX model with a non-CPU EP available to you (CoreML, CUDA, OpenVINO, NNAPI) at verbose logging, and count the nodes assigned to each EP. If everything landed on one EP, deliberately insert an exotic op and watch the graph split.
4. **Write the conversion build script** with pinned versions and a `metadata.json` (source digest, tool versions, opset, buckets, calibration digest, measured tolerances). This script is the reusable artifact from this lesson and slots straight into [lesson 11](11-build-rag-and-cpu-serving.md).

---

**Next:** [Security and multi-tenancy →](09-security-and-multi-tenancy.md) — what changes the moment your model serves untrusted input on shared hardware, and why prompt injection has no complete fix.
