# 7 — Edge and On-Device Inference

> **You'll be able to say:** "On-device serving is the same engineering with three budgets I cannot negotiate: binary size in megabytes, energy in milliwatt-hours, and thermal headroom that throttles sustained throughput to a fraction of burst. The NPU takes static-shape int8 graphs or nothing, so quantization is a *compile-time contract*, not a tuning pass. The first inference is 10-100× slower than the rest because of model load and kernel compilation, and models ship out-of-band from the app binary with a server-side fallback, because I cannot log into the device and cannot roll back what a user hasn't updated."

This is the one lesson where everything you learned about GPU fleets inverts: there is no batch, no autoscaling, no observability by default, and no rollback. What replaces them is packaging discipline and measured constraint budgets.

---

## Why bother at all

| Reason | Concretely |
|---|---|
| **Latency** | 5-30 ms local versus 100-500 ms round trip; no network variance, no cold-start queue |
| **Privacy** | data never leaves the device — often a legal requirement (health, biometrics, keyboards) |
| **Offline** | works on a plane, in a basement, in a country with bad connectivity |
| **Cost** | inference is free to you: the user's battery pays. At a billion invocations/day this is the entire business case |
| **Scale of "free"** | Face ID, keyboard prediction, camera pipelines, live captions, photo search — all on-device, all ~100% of requests |

And the honest limits: models are 10-1000× smaller than server models, so quality is lower; you cannot debug a specific user's failure; and the deployment surface is thousands of device/OS/SoC combinations. The dominant production pattern is therefore **hybrid**: small model on-device for the common/latency-critical/private path, escalate to the server model when confidence is low, input is complex, or the device is plugged in.

---

## The three budgets

```
  1. SIZE       app-store download limits + user tolerance
                ├─ app binary increase: keep under ~50 MB if possible
                ├─ downloaded model assets: tens to low hundreds of MB
                └─ a 7B Q4 LLM is ~4 GB. Flagship phones can. Most cannot.

  2. ENERGY     measured in mWh per inference, or mAh per hour of use
                ├─ NPU:  1×    (baseline, most efficient)
                ├─ GPU:  2-5×  the NPU for the same work
                ├─ CPU:  5-20× the NPU  ← never ship a hot loop here
                └─ 1% battery ≈ 40-60 mWh on a phone. Budget per feature.

  3. THERMAL    sustained ≠ peak. Phones throttle in 30-120 s of load.
                ├─ burst:      full clocks, the number in every benchmark
                ├─ sustained:  40-70% of burst after throttling
                └─ a 30 fps demo that runs for 20 s and drops to 12 fps is
                   a FAILED feature, and no single-shot benchmark shows it
```

**Report sustained numbers, always.** The canonical on-device benchmarking mistake is measuring 100 inferences from a cold device and quoting the mean. Run for five minutes, plot latency over time, and report the plateau — that is what users get.

---

## The runtimes

| Runtime | Platform | Formats | Accelerators | Notes |
|---|---|---|---|---|
| **Core ML** | Apple | `.mlpackage` (from `coremltools`) | CPU / GPU / **ANE** | best-in-class when the ANE accepts your graph; opaque when it doesn't |
| **TFLite / LiteRT** | Android, embedded, MCUs | `.tflite` | CPU (XNNPACK), GPU, NNAPI, vendor delegates (QNN, Hexagon) | the broadest reach; delegate support is fragmented |
| **ExecuTorch** | iOS, Android, embedded | `.pte` (from `torch.export`) | CoreML/MPS, XNNPACK, QNN, Vulkan | PyTorch-native path; the strategic direction for PyTorch on device |
| **ONNX Runtime Mobile** | both + Windows/Linux | `.ort` / ONNX | NNAPI, CoreML, QNN, XNNPACK | one graph, many EPs ([lesson 8](08-model-formats-and-runtimes.md)); trimmed build to cut binary size |
| **MLC-LLM / llama.cpp** | both + desktop | GGUF / MLC | Metal, Vulkan, OpenCL | how on-device *LLMs* actually ship today |
| **MediaPipe** | both + web | task bundles | delegates | full pipelines (vision/audio) rather than bare models |

**Choosing:** ship what the platform prefers unless you have a reason. iOS-only feature → Core ML. Android-first → TFLite with XNNPACK, then a vendor delegate for the SoCs that matter. Cross-platform with one artifact → ONNX Runtime Mobile or ExecuTorch. On-device LLM → llama.cpp/MLC, today, without much debate.

---

## The NPU contract: static shapes, int8, supported ops

Mobile NPUs (Apple ANE, Qualcomm Hexagon, Google Tensor TPU, MediaTek APU) are fixed-function-ish accelerators. They are 5-20× more energy-efficient than the CPU and they impose conditions:

| Constraint | Detail | Consequence |
|---|---|---|
| **Static shapes** | dimensions fixed at conversion time | one artifact per input size; LLM decode needs fixed-length KV buffers and padding |
| **Quantized weights *and* activations** | usually int8 (some int4/fp16 paths) | you need **static quantization with calibration**, not dynamic ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)) |
| **Coarse quantization granularity** | often per-tensor, not per-channel | accuracy loss is larger than server INT8; per-channel where supported is a big win |
| **Limited op set** | unsupported ops fall back to CPU/GPU | a fallback *in the middle of a graph* costs a round trip per boundary — worse than never using the NPU |
| **Layout requirements** | specific tensor layouts/alignments | inserted transposes silently eat the gain |
| **No control flow** | dynamic `if`/`while` unsupported or slow | export a straight-line graph; move control flow to host code |

The failure to internalize: **partial NPU offload can be slower than pure CPU.** Each accelerator↔host boundary copies tensors and synchronizes; three boundaries in a small graph can cost more than the whole model. Always inspect the delegate/partition report (Core ML's compute-unit report, TFLite's delegate logs, QNN's partition output) and count the partitions. One partition good; five partitions is a bug.

### Quantization on device, practically

1. **Start with fp16.** Free 2× size cut, usually no accuracy loss, supported everywhere. Many features ship here and stop.
2. **Dynamic int8** (weights int8, activations float at runtime): easy, good size win, works on CPU paths, but usually *not* NPU-eligible.
3. **Static int8** (weights + activations quantized with calibration data): the NPU entry ticket. Needs 100-1000 representative samples; expect 0.5-3% task-metric loss on vision, more on generative models.
4. **QAT** (quantization-aware training) when static PTQ loses too much — a training project, not a serving change, but sometimes the only way to hit a target.
5. **int4 / mixed** for on-device LLMs: GGUF Q4_K_M and MLC's group quantization are the practical state of the art; expect visible quality loss versus the server model and design the UX to tolerate it.

**Calibration data is the thing teams get wrong**: calibrate on data drawn from real device inputs (phone camera JPEGs, real microphone audio), not on a clean academic dataset, or your activation ranges are wrong for the deployment distribution.

---

## Cold start on device: why the first inference is 10-100× slower

```
  app launch / feature first use
  ├─ read model file from flash          10-500 ms   (size-dependent; mmap helps)
  ├─ parse/deserialize graph              5-50 ms
  ├─ COMPILE for the accelerator        50-3000 ms   ← the big one; per device+OS
  ├─ allocate + warm buffers             10-100 ms
  └─ first inference (cold caches)        2-10× steady state
```

Countermeasures, all of which are standard practice in shipped apps:

- **Cache the compiled artifact on disk**, keyed by model digest × OS version × device model (Core ML does this for you; TFLite/QNN need explicit serialization of the delegate cache). Invalidate on OS upgrade — a stale compiled blob after an OS update is a real crash class.
- **`mmap` the weights** instead of reading them; on-device LLM runtimes rely on this so the OS pages in what it needs.
- **Warm at a plausible moment** — app launch, screen entry, or on idle — never inside the first user-visible interaction.
- **Keep the interpreter/session alive** for the feature's lifetime; re-creating a session per call is the most common on-device performance bug.

This is exactly [Phase 8 lesson 3](../phase-8/03-image-size-and-cold-start.md)'s five-stage cold-start decomposition, transposed to a device: measure the stages, attack the dominant one, cache what you can.

---

## Shipping and updating models you cannot log into

The deployment model inverts everything in Phase 8:

| Server (Phase 8) | On-device |
|---|---|
| Rollback in < 5 min | **users decide** when to update; old versions live for years |
| One fleet you control | thousands of device/OS/SoC combinations |
| Prometheus scrape | sampled, privacy-filtered telemetry, delayed by hours |
| Canary by traffic % | staged rollout by user cohort, days long |
| Digest-pinned artifact | same idea, plus signature verification on the device |

Practices that follow:

1. **Ship models out-of-band from the app binary.** A model registry the app downloads from (signed, versioned, digest-checked) means model fixes don't wait for app review. Keep one small model bundled as the offline/first-launch fallback.
2. **Every device reports which model version it is running**, or you cannot interpret any quality metric you collect.
3. **Staged rollout with a kill switch.** Remote config that can force a cohort back to the previous model version or to the server path. That kill switch *is* your rollback.
4. **Compatibility matrix, and enforce it.** Min OS version, min runtime version, required delegate. Devices outside it get the fallback — silently, not crashing.
5. **Device-tier routing.** Flagship → the bigger on-device model; mid-tier → the small one; low-end → server or feature disabled. One artifact for all devices means either the flagships are underserved or the cheap phones catch fire.
6. **Verify signatures before load.** A model file is executable-adjacent input; an unsigned model fetched over the network is a code-execution-shaped risk ([lesson 9](09-security-and-multi-tenancy.md) covers the supply-chain angle, including why `pickle` formats have no place here).

---

## Measuring on device (the only numbers that count)

| Metric | How | Trap |
|---|---|---|
| Steady-state latency | 1000 inferences after warmup, report p50/p99 | cold runs inflate the mean |
| **Sustained throughput** | run 5 minutes, plot over time | thermal throttling appears only here |
| Energy per inference | platform energy tools; or battery delta over N inferences, screen off, airplane mode | screen and radio dominate if you don't control them |
| Peak memory | platform profiler; watch for OS kill thresholds | a mid-tier phone kills apps well under its nominal RAM |
| Binary/asset size impact | app-size report before/after | the runtime library counts, not just the model |
| Accelerator utilization | compute-unit/delegate report | "it ran" ≠ "it ran on the NPU" |
| Accuracy on device | run the eval set **through the deployed artifact on real hardware** | never assume the converted model matches the original — [lesson 8](08-model-formats-and-runtimes.md) |

Do this on **at least three tiers** of hardware: a current flagship, a 3-4-year-old mid-range device, and (for Android) a cheap one. The mid-range device is where features actually get cut.

---

## On-device LLMs, briefly and honestly

```
  3B Q4  ≈ 1.7 GB weights → runs on most flagships, ~10-30 tok/s
  7B Q4  ≈ 4.0 GB weights → high-memory flagships only; memory pressure risks
                              OS termination while other apps are alive
  prefill is compute-bound: a 2000-token prompt can take seconds
  sustained generation heats the device and throttles within a minute
```

Consequences for product design: keep prompts short (prefill is the visible cost), keep outputs short, expect the OS to evict you, and design the hybrid escalation path first. The realistic on-device niches today are summarization of short text, rewriting, classification/routing, autocomplete, and offline assistants — not long-context reasoning.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| First use feels broken, later fine | compile + load cold start | cache compiled artifact, warm early, mmap weights |
| Fast in benchmarks, slow in the app | thermal throttling / other app load | report sustained numbers; reduce duty cycle; smaller model |
| NPU "enabled" but no speedup | graph partitioned across accelerator boundaries | read the partition report; replace unsupported ops; re-export |
| Accuracy much worse than server | per-tensor static int8 + bad calibration data | per-channel where supported, calibrate on real device inputs, consider QAT |
| Crashes after an OS update | stale compiled delegate cache | key the cache on OS version; validate on load, recompile on mismatch |
| App killed on mid-tier devices | peak memory over the OS threshold | smaller model, mmap, tile activations, device-tier routing |
| Battery complaints | model on CPU, or running too often | move to NPU/GPU, add trigger conditions, throttle frequency |
| Cannot fix a bad model for weeks | model shipped inside the app binary | out-of-band model delivery + remote kill switch |
| Quality metrics uninterpretable | no model-version reporting from devices | report version with every telemetry event |

---

## What to read and run

- **Code:** `pytorch/executorch` (export → `.pte` → delegate; read `examples/` and the backend docs), `google-ai-edge/LiteRT` (formerly TFLite) delegate docs, Apple `coremltools` conversion guide (and `ml-ane-transformers` for what the ANE actually likes), `mlc-ai/mlc-llm`, `ggerganov/llama.cpp` for Metal/Vulkan builds.
- **Read:** Apple's Core ML performance-report documentation (compute-unit attribution is the single most useful on-device debugging tool); Qualcomm AI Hub docs for the QNN constraint list; Google's on-device ML guides for staged model delivery.

---

## Do this now (60 minutes; laptop-only is fine)

1. **Convert and measure a small vision model** three ways: fp32, fp16, and static int8 with 200 calibration images. Report size, latency (steady state), and top-1 agreement with the original. If you have a phone, run the same artifact there and compare; if not, use ONNX Runtime/Core ML on your laptop's CPU and NPU where available.
2. **Produce the cold-start decomposition**: time file read, session creation/compile, and first-vs-steady inference. Then add a compiled-artifact cache and re-measure the second launch.
3. **Find the sustained number.** Run inference in a loop for 5 minutes on whatever device you have, plot latency over time, and report burst versus plateau. If the plateau is > 20% worse, you have found your real capacity.
4. **Write the update plan** for a hypothetical on-device feature: where the model is hosted, how it's signed and verified, how versions are reported, the staged-rollout cohorts, the kill switch, and the bundled fallback. One page, and it is the artifact interviewers actually probe.

---

**Next:** [Model formats and runtimes →](08-model-formats-and-runtimes.md) — the conversion machinery underneath every runtime in this lesson, and how to keep a converted model numerically honest.
