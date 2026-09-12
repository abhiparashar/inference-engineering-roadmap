# 7 — Model Registry and Artifacts

> **You'll be able to say:** "A model artifact is weights **plus** tokenizer, config, chat template, quantization scales and — if compiled — an engine plan pinned to an exact GPU arch and library version. It is content-addressed by digest, checksum-verified at load, retained at least as long as the rollback window, and carries a lineage record: what produced it, what evaluated it, and what it is approved for. A registry is not S3 with good intentions; the minimum viable version is a versioned bucket plus a metadata file plus a promotion state, and it exists so that 'roll back the model' is a lookup rather than an archaeology project."

[Lesson 6](06-deploying-and-rollouts.md) pinned a weight digest in a manifest. This lesson is what that digest points at, and why the naive `s3://models/latest/` is the most common way teams lose the ability to roll back.

---

## What is actually in a "model"

Loading only `model.safetensors` reproduces nothing. The artifact is a set:

| Component | Why it must be versioned with the weights |
|---|---|
| Weight shards (`*.safetensors`) | the obvious part |
| `config.json` | architecture, `rope_theta`, `max_position_embeddings` — a mismatch is silent garbage |
| Tokenizer (`tokenizer.json`, merges, special tokens) | a changed tokenizer shifts every token id: quality collapses, no error raised |
| **Chat template** | the highest-risk, lowest-attention file in the set; a stray newline measurably changes outputs |
| `generation_config.json` | default temperature/top-p/stop tokens — silently changes behaviour |
| Quantization metadata (scales, zero-points, `quant_config.json`) | AWQ/GPTQ artifacts are useless without them ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)) |
| LoRA adapters | version *and* the base they apply to |
| **Compiled engine plan** (TensorRT-LLM `.engine`, compiled graph) | valid only for one (GPU arch × TRT version × TP size × max batch/seq) tuple ([Phase 5 lesson 6](../phase-5/06-tensorrt-llm-and-compiled-engines.md)) |
| Eval results | the evidence that approved it for production |
| Licence and provenance | a real blocker at any company with a legal department |

**Rule: one artifact = one immutable directory = one digest.** If any file in it changes, it is a new artifact with a new version, not an update to the old one.

The compiled-engine row is the sharpest edge. A TensorRT-LLM plan built for H100 + TRT 10.x + TP2 will refuse to load — or worse, silently underperform — on an L40S or after a library bump. Engine plans must be built in CI ([lesson 8](08-ci-cd-and-benchmark-gates.md)), keyed by that full tuple, and never built at pod startup where they'd add minutes to every cold start.

---

## Naming, addressing, and the `latest` trap

```
  BAD   s3://models/llama3-8b/latest/                mutable; rollback target can change
  BAD   s3://models/llama3-8b-v2-final-FIXED/        human naming, unordered, unverifiable
  OK    s3://models/llama-3.1-8b-instruct/v7/        versioned, immutable by convention
  BEST  s3://models/llama-3.1-8b-instruct/sha256-7bd41e.../
        + a mutable *pointer* file: channels/prod → sha256-7bd41e...
```

Content addressing gives you three properties for free: deduplication (two artifacts sharing shards share storage), verification (the name *is* the checksum), and unambiguous rollback (the old digest is still a valid, still-existing address). Keep human-friendly channels (`prod`, `staging`, `candidate`) as small pointer objects that are *deployed like config* — changing a channel is a deploy, subject to the same gates.

### The minimum viable registry

You do not need MLflow to be professional. You need these five things:

```
  models/llama-3.1-8b-instruct/sha256-7bd41e.../
      ├── model-0000x-of-0000y.safetensors
      ├── config.json  tokenizer.json  chat_template.jinja  generation_config.json
      ├── SHA256SUMS                # every file, checksummed
      └── metadata.json             # the lineage record
  models/llama-3.1-8b-instruct/channels/prod        → "sha256-7bd41e..."
```

```json
{
  "name": "llama-3.1-8b-instruct",
  "digest": "sha256:7bd41e...",
  "created_at": "2026-02-14T09:12:03Z",
  "created_by": "ci-run/quantize-awq#4821",
  "lineage": {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "base_digest": "sha256:1c9f00...",
    "transform": "AWQ 4-bit, group_size=128, zero_point=true",
    "calibration_set": "s3://datasets/calib-2k-v3 (sha256:aa19...)"
  },
  "runtime_requirements": {
    "engine": "vllm>=0.7.0,<0.8.0",
    "gpu_arch": ["sm_89", "sm_90"],
    "tp_size": 1,
    "min_gpu_memory_gb": 24,
    "max_model_len": 8192
  },
  "evaluation": {
    "harness": "s3://evals/harness@sha256:44c2...",
    "mmlu": 0.681, "gsm8k": 0.742, "internal_qa_pass_rate": 0.951,
    "baseline_digest": "sha256:1c9f00...", "quality_delta_pct": -0.8
  },
  "benchmark": { "gpu": "L40S", "output_tok_s_at_slo": 812, "p99_ttft_s": 0.41 },
  "approval": { "state": "production", "by": "inference-team", "ticket": "INF-2291" },
  "license": "llama3.1", "retention_until": "2026-08-14"
}
```

That file is the entire difference between "we think this is the model that was running in February" and knowing. It also answers, in an incident, the two questions you always need: *what produced this?* and *what was it measured at?*

### When to use a real registry

| Tool | Fits when |
|---|---|
| Versioned S3/GCS + `metadata.json` (above) | small teams, one or few models. Genuinely sufficient |
| **OCI artifacts** (ORAS / Docker registry) | you already run a registry; get digest addressing, signing, replication, and lazy pull for free |
| Hugging Face Hub (private) | you want the ecosystem's tooling and revision pinning (`revision=<commit sha>`) |
| MLflow / W&B model registry | you also own training and need experiment↔model lineage |
| Vendor registries (SageMaker, Vertex) | you're deep in one cloud's deployment tooling |

**OCI artifacts are underrated for inference.** Storing weights as an OCI artifact means the same digest-pinning, signing (cosign), replication, and lazy-pull machinery ([lesson 3](03-image-size-and-cold-start.md)) applies to weights as to images — one distribution system instead of two.

---

## The promotion pipeline

Artifacts move through states; states are what "approved" means operationally.

```
  IMPORTED ──► CONVERTED/QUANTIZED ──► EVALUATED ──► CANARY ──► PRODUCTION ──► DEPRECATED ──► DELETED
     │              │                      │            │            │              │
   licence      safetensors,           quality       Phase 7      channel        retention
   + checksum   arch-pinned engine     harness       L8 gates     pointer        window ends
```

Rules that make it work:

- **Each transition is automated and recorded**, with the metadata file appended, not overwritten.
- **Nothing skips EVALUATED.** A "quick fix" checkpoint deployed straight to prod is how quality regressions ship.
- **DEPRECATED ≠ DELETED.** The gap is your rollback window.

### Retention: the rule that makes rollback real

> **Retain every artifact that has ever served production traffic for at least `rollback_window + 1` releases, and never garbage-collect an artifact referenced by a live or recent manifest.**

Concretely: if you deploy weekly and want to be able to revert four weeks, retain 5+ generations. Weight storage is cheap (16 GB × 5 = 80 GB ≈ pennies per month); the cost of a garbage-collected rollback target is an outage. Automate GC by *reading the manifests in git*, not by age alone.

The same applies to container images: an aggressive registry lifecycle policy ("delete untagged after 7 days") will happily delete the digest your rollback needs.

---

## Loading, verification, and cache keys

At startup, in this order:

1. **Resolve** the digest from config (never resolve `latest` at runtime — the pod that restarts at 3 a.m. must get the same artifact as its siblings).
2. **Check the node-local cache** keyed by digest (`/mnt/nvme/weights/sha256-7bd41e.../`).
3. **Fetch** in parallel if missing; write to a temp dir and atomically rename on completion — a partially-downloaded cache entry that looks complete is a nasty, node-sticky bug.
4. **Verify checksums** against `SHA256SUMS`. Refuse to start on mismatch, loudly. A truncated shard yields 200 OK garbage, the invisible failure class from [Phase 7 lesson 1](../phase-7/01-what-to-measure.md).
5. **Check `runtime_requirements`** against the actual environment (engine version, `sm_XX`, TP size, free HBM). Fail fast with a readable message instead of a 40-line CUDA traceback nine minutes later.
6. **Emit the digest as a metric label and in the startup log**: `inf_model_info{name,digest,quant,engine_version} 1`. Now every dashboard, alert and trace can be attributed to an exact artifact — and "which model was serving during the incident?" is a query, not a debate.

```python
info = Gauge("inf_model_info", "loaded model", ["name", "digest", "quant", "engine"])
info.labels(meta["name"], meta["digest"], meta["quant"], vllm.__version__).set(1)
```

---

## Security and supply chain

Three concerns, in descending order of how often they actually bite:

1. **Pickle execution.** `.bin`/`.pt` checkpoints are `pickle` — loading one executes arbitrary code. **Use `safetensors` for anything you did not produce**, and convert third-party checkpoints in a sandbox before they touch a serving node.
2. **Integrity.** Sign artifacts (cosign for OCI, GPG or a signed manifest otherwise) and verify at load; require checksums in the metadata. This blocks the "someone re-uploaded the bucket path" class of incident, including your own accidents.
3. **Licence and provenance.** Model licences (Llama community licence, Gemma terms, research-only checkpoints, dataset restrictions) constrain commercial use. Record the licence in metadata and check it at promotion — this is a genuine deployment blocker discovered too late by many teams.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Rollback fails: "artifact not found" | GC deleted the old weights or untagged image | retention ≥ rollback window; GC driven by manifests in git |
| Two replicas serve different outputs | `latest` resolved at pod start; a new artifact landed mid-rollout | resolve digest at deploy time, never at pod start |
| Quality drops with no code change | tokenizer or chat template updated inside a "same" artifact | immutable artifacts; chat template versioned with weights |
| Engine fails to load after a node-type change | compiled plan pinned to another arch | key engine artifacts by (arch × lib version × TP × shapes); build in CI |
| Garbage output, no error | truncated/corrupt shard | checksum verification at load, fail closed |
| Pod start takes 4 extra minutes at random | cache key collision or partial cache entry | digest-keyed cache, atomic rename |
| Nobody knows what produced the model in prod | no lineage record | `metadata.json`, appended at each transition |
| Legal blocks a launch | licence discovered post-integration | licence field checked at promotion |
| Security review blocks deployment | `.bin` pickle checkpoints | convert to `safetensors` in a sandbox |

---

## Do this now (45 minutes)

1. **Build the minimum viable registry** for one model you actually serve: content-addressed directory, `SHA256SUMS`, `metadata.json` with lineage + eval + runtime requirements, and a `channels/prod` pointer.
2. **Make your server digest-aware**: resolve from config, verify checksums, validate `runtime_requirements`, refuse to start on any mismatch, and export `inf_model_info`. Then corrupt one byte of a shard and confirm it fails fast with a clear message instead of serving garbage.
3. **Write your retention policy** as a sentence with numbers ("every artifact that served production is retained 60 days; GC reads live manifests from git"), then check whether your current bucket/registry lifecycle rules would already have deleted last month's rollback target. For most teams the honest answer is yes.

---

**Next:** [CI/CD and benchmark gates →](08-ci-cd-and-benchmark-gates.md) — the pipeline that builds these artifacts, and the four gate classes (unit, smoke, quality, performance) that decide whether one is allowed to become a candidate.
