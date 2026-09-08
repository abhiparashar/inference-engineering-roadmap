# 10 — Build: Framework Shootout + Triton Ensemble

> **You'll be able to say:** "I ran the same model on two engines with a **leveled** configuration — same context limit, same token budget, same prefix-caching and CUDA-graph settings, same dtype, same open-loop client — and reported TTFT/TPOT/throughput/memory with an explanation for every delta, including the ones that disappeared once I leveled the config. And I built a three-stage Triton ensemble with per-stage dynamic batching and a per-stage latency budget, then showed which stage was the bottleneck and what fixed it."

Two deliverables. **Part A** is the phase exit artifact and needs a GPU for a few hours. **Part B** runs fine on a laptop with CPU models and is the more transferable skill outside LLM shops.

---

# Part A — the framework shootout

## The leveling checklist (do this before any measurement)

Almost every published framework comparison is wrong because of this list. Go through it explicitly and record the values in your report.

| # | Axis | Why an unleveled setting invalidates the run |
|---|---|---|
| 1 | **Same model, same weights, same dtype** | An FP16 vs BF16 vs pre-quantized checkpoint difference is a bigger effect than the engines' |
| 2 | **Same max context** (`--max-model-len` / `--max-total-tokens` / `--max-seq-len`) | It divides the KV pool: 4× the context is ~¼ the concurrency ([lesson 3](03-vllm-in-production.md)) |
| 3 | **Same KV pool size** | vLLM sizes by memory fraction, TGI by tokens. Convert one to the other and match within ~5%: `pool_tokens × KV_bytes_per_token ≈ pool_GB` |
| 4 | **Same token budget** | `--max-num-batched-tokens` vs `--max-batch-prefill-tokens` vs `--chunked-prefill-size` |
| 5 | **Same concurrency cap** | `--max-num-seqs` vs `--max-concurrent-requests`/`--max-batch-size` vs `--max-running-requests` |
| 6 | **Prefix caching same state** | Report *both* on and off. It's the single largest defaults difference between engines |
| 7 | **Chunked prefill same state** | Changes TPOT variance dramatically |
| 8 | **CUDA graphs same state** | `--enforce-eager` on one side and graphs on the other is a ~10-30% handicap at low batch |
| 9 | **Same client, open loop** | Your `bench.py`, not each vendor's harness ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) |
| 10 | **Same workload distribution & seed** | Ragged outputs, realistic prompt lengths, identical RNG seed |
| 11 | **Same warmup and settle** | First-request compile/graph capture must not land in the measurement |
| 12 | **Same hardware, one run each way** | Same GPU, no other tenants; alternate A/B/A to catch drift |

Anything you *can't* level (e.g. an engine that refuses your exact quantization) is a finding — write it down as one instead of pretending the run was fair.

## Adapt the harness (15 minutes)

Your Phase-3 `bench.py` posts `{"prompt": ..., "max_tokens": ...}`. Every engine here speaks the OpenAI API, so add a body builder and a shared-prefix option. Save as `labs/phase5/bench_openai.py` (a thin patch over the Phase-3 file — keep the original intact):

```python
#!/usr/bin/env python3
"""Phase 5 shootout client: Phase-3 bench.py with OpenAI-compatible bodies,
a shared-prefix workload knob, and a /metrics snapshot around each run.

Usage:
  python bench_openai.py --url http://127.0.0.1:8000/v1/completions \
      --model Qwen/Qwen2.5-1.5B-Instruct --qps 1 2 4 8 --duration 60 \
      --prompt-tokens 512 --shared-prefix-tokens 1024 --max-tokens 64 128 256 \
      --metrics-url http://127.0.0.1:8000/metrics --tag vllm-baseline
"""
import argparse, asyncio, json, random, statistics, time, urllib.request

from bench import one_request        # reuse Phase 3's timing core unchanged

P = lambda ok, k, p: statistics.quantiles([r[k] for r in ok], n=100)[p - 1]

SHARED = None  # the shared system prefix, built once so it is byte-identical every request


def build_body(model, prompt_tokens, shared_prefix_tokens, max_tokens, rnd):
    global SHARED
    if SHARED is None:
        SHARED = "You are a helpful assistant. " * max(shared_prefix_tokens // 6, 0)
    # unique tail so only the PREFIX is cacheable — otherwise you measure a response cache
    tail = " ".join(f"w{rnd.randrange(10**6)}" for _ in range(max(prompt_tokens // 2, 1)))
    return {
        "model": model,
        "prompt": SHARED + tail,
        "max_tokens": max_tokens,
        "temperature": 0.0,          # deterministic: quality diffs are then comparable
        "stream": True,              # REQUIRED: TTFT/ITL are meaningless without streaming
        "ignore_eos": True,          # vLLM/SGLang: fixed output length → controlled workload
    }


def scrape(metrics_url):
    if not metrics_url:
        return {}
    with urllib.request.urlopen(metrics_url, timeout=5) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line.startswith("#") or " " not in line:
                continue
            k, v = line.rsplit(" ", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out
```

Then a runner that records a JSON line per load point, with the metrics delta attached:

```python
async def run_point(a, qps):
    rnd, results, tasks = random.Random(0), [], []
    before = scrape(a.metrics_url)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.duration:
        await asyncio.sleep(rnd.expovariate(qps))
        rec = {"t_intended": time.perf_counter(), "status": None}
        results.append(rec)
        body = build_body(a.model, a.prompt_tokens, a.shared_prefix_tokens,
                          rnd.choice(a.max_tokens), rnd)
        tasks.append(asyncio.ensure_future(one_request(a.url, body, rec)))
    await asyncio.gather(*tasks, return_exceptions=True)
    after = scrape(a.metrics_url)

    ok = [r for r in results if r["status"] == 200 and r["t_intended"] - t0 >= a.warmup]
    if len(ok) < 20:
        print(json.dumps({"tag": a.tag, "qps": qps, "error": "too few samples"})); return
    row = {
        "tag": a.tag, "qps": qps, "n": len(ok),
        "shed": sum(1 for r in results if r["status"] not in (200, None)),
        "ttft_p50": P(ok, "ttft", 50), "ttft_p99": P(ok, "ttft", 99),
        "tpot_p50": P(ok, "tpot", 50), "itl_p99": P(ok, "itl_p99", 99),
        "e2e_p99":  P(ok, "e2e", 99),
        "out_tok_s": sum(r["tokens"] for r in ok) /
                     (max(r["e2e"] + r["t_intended"] for r in ok) - min(r["t_intended"] for r in ok)),
        "engine": {k: after.get(k, 0) - before.get(k, 0)
                   for k in after if k.endswith(("_total", "_hits", "_queries", "preemptions_total"))},
    }
    print(json.dumps(row), flush=True)     # one JSON line per point → easy to table later
```

Three details that are the difference between a real result and a plausible-looking one:

- **`stream: True` is mandatory.** Without streaming there is no TTFT, and "latency" collapses into one number that hides everything ([Phase 3 lesson 1](../phase-3/01-what-a-serving-system-is.md)).
- **`ignore_eos` (or a forced length)** makes output length a controlled variable instead of a model-dependent one — otherwise engine A "wins" because its sampling stopped earlier.
- **The unique tail** ensures prefix caching is exercised but response caching is not. Measuring a full-response cache hit and calling it inference performance is a real mistake people publish.

## Launch both engines, leveled

```bash
# --- Engine A: vLLM ---------------------------------------------------------
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --enable-prefix-caching --port 8000
# record from the startup log: KV cache size (tokens), max concurrency, startup seconds

# --- Engine B: TGI ----------------------------------------------------------
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --max-input-tokens 3968 --max-total-tokens 4096 \
  --max-batch-prefill-tokens 8192 --max-batch-total-tokens <MATCH_VLLM_POOL_TOKENS> \
  --max-concurrent-requests 64
# TGI states the pool in tokens: set it to vLLM's reported KV cache size for axis 3

# --- Optional Engine C: SGLang ---------------------------------------------
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --context-length 4096 --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 --max-running-requests 64 --port 30000
```

Then the matrix. Six runs minimum, two workloads × (2-3 engines) × prefix cache on/off:

```bash
for tag in vllm tgi; do
  for pc in on off; do            # restart the engine with the flag flipped
    python bench_openai.py --url $URL --model $MODEL --tag "$tag-pc-$pc" \
      --qps 1 2 4 8 16 --duration 60 --warmup 10 --settle 15 \
      --prompt-tokens 512 --shared-prefix-tokens 1024 --max-tokens 64 128 256 \
      --metrics-url $METRICS >> results.jsonl
  done
done
```

**Workload 2** (run the whole matrix again): `--shared-prefix-tokens 0 --prompt-tokens 2048 --max-tokens 32` — long unique prompts, short outputs. This is prefill-dominated, and it inverts several of the conclusions from the chat-like workload. Reporting both is what makes the artifact credible.

## The report

`projects/09-framework-shootout/README.md`. Table shape (fill with **your** numbers — never copy anyone's):

```
workload: shared-prefix chat (1024-token shared system prompt, 512-token tail, 64-256 out)
model: Qwen2.5-1.5B-Instruct   GPU: <name>   date/commit: <engine versions>

engine  cfg      qps   ttft_p50  ttft_p99  tpot_p50  itl_p99  out_tok/s  shed  KV pool  notes
─────────────────────────────────────────────────────────────────────────────────────────────
vllm    pc=on     4        ___       ___       ___      ___        ___    ___    ___
vllm    pc=off    4        ___       ___       ___      ___        ___    ___    ___
tgi     pc=on     4        ___       ___       ___      ___        ___    ___    ___
…
```

Then write the three sections that make it an engineering document rather than a spreadsheet:

1. **Leveling log.** The 12-item checklist with the value used on each engine, and every item you could *not* level.
2. **Per-delta explanation.** For each gap above ~10%, name the mechanism and the evidence: engine metrics (`vllm:num_preemptions`, prefix-cache hit rate, `iteration_tokens_total`; TGI's `tgi_batch_current_size`, `tgi_request_queue_duration`), or a config difference you couldn't level. **"Engine A is faster" is not a finding; "Engine A ran a mean batch of 38 vs 21 because its budget admitted waiting requests without pausing decode" is.**
3. **Startup, memory, and ergonomics.** Time to first served request, peak memory, config surface, quality of `/metrics`, error behavior under overload (does it shed with 429 or queue unboundedly?). Operationally this section decides real deployments more often than the tokens/sec table does.

Predict before you measure, as always: from [Phase 4 lesson 1](../phase-4/01-what-to-optimize.md) you can compute expected tokens/sec at the batch size the pool allows. Any engine within ~30% of that ceiling is doing its job; a factor of 2+ gap between two engines almost always means a config axis you failed to level.

---

# Part B — the Triton ensemble pipeline

Goal: one endpoint, three stages, per-stage dynamic batching, a measured per-stage latency budget. Pick a real pipeline — audio → transcription → summarization, or image → preprocess → classifier → policy. Below is a CPU-runnable skeleton you can substitute real models into.

## Repository

```
model_repository/
├── preprocess/{config.pbtxt, 1/model.py}       Python backend
├── model/{config.pbtxt, 1/model.onnx}          ONNX Runtime (or python for a stub)
├── postprocess/{config.pbtxt, 1/model.py}      Python backend
└── pipeline/{config.pbtxt, 1/}                 ensemble
```

`preprocess/config.pbtxt`:

```protobuf
name: "preprocess"
backend: "python"
max_batch_size: 16
input  [ { name: "TEXT"   data_type: TYPE_STRING dims: [1]   } ]
output [ { name: "TOKENS" data_type: TYPE_INT32  dims: [-1]  } ]
instance_group [ { count: 4  kind: KIND_CPU } ]        # CPU stage: scale by instances
dynamic_batching { preferred_batch_size: [8, 16] max_queue_delay_microseconds: 2000 }
```

`preprocess/1/model.py`:

```python
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = 128

    def execute(self, requests):                       # a LIST: Triton hands you the batch
        texts = [pb_utils.get_input_tensor_by_name(r, "TEXT").as_numpy()[0].decode()
                 for r in requests]
        enc = self.tok(texts, padding="max_length", truncation=True,
                       max_length=self.max_len, return_tensors="np")
        return [pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("TOKENS", enc["input_ids"][i].astype(np.int32))])
                for i in range(len(requests))]
```

Note what `execute(requests)` proves: **the batch is visible to you**. Tokenizing 16 texts in one call is materially cheaper than 16 separate calls, and that saving is invisible in any per-request design.

`pipeline/config.pbtxt` is the ensemble from [lesson 7](07-triton-inference-server.md), wiring `TEXT → preprocess → model → postprocess → LABEL`.

## Measure it properly

```bash
# 1. correctness
curl -s localhost:8000/v2/models/pipeline/infer -d '{"inputs":[{"name":"TEXT","shape":[1,1],
     "datatype":"BYTES","data":["hello world"]}]}' | head -c 400

# 2. per-stage timing under load (SDK container)
perf_analyzer -m pipeline --request-rate-range 5:40:5 --measurement-interval 10000 \
              -i grpc --concurrency-range 1 --percentile 99

# 3. per-stage metrics, before/after each load point
curl -s localhost:8002/metrics | grep -E 'nv_inference_(count|exec_count|queue_duration_us|compute_infer_duration_us)'
```

Compute, per stage: `avg_batch = nv_inference_count / nv_inference_exec_count`, `avg_queue_ms`, `avg_compute_ms`. Those three numbers per stage *are* the report.

## Deliverable

`projects/10-triton-ensemble/README.md` with:

- The four `config.pbtxt` files and what each field does *for this pipeline* (not copied documentation).
- The **latency budget table** (planned vs measured per stage, plus network/serde).
- A **bottleneck study**: identify the limiting stage from `queue_duration`, fix it (instance count for CPU stages, batch size / preferred sizes for GPU stages, or DALI/GPU preprocessing), and show the before/after throughput at fixed p99.
- A **queue-delay sweep** on the batching-sensitive stage: `max_queue_delay_microseconds` ∈ {0, 500, 2000, 10000} vs avg batch size and p99 latency. This reproduces [Phase 3 lesson 3](../phase-3/03-dynamic-batching.md)'s tradeoff curve inside a production server — the plot is the artifact.
- **Ensemble vs client-orchestrated** comparison: implement the same three calls from the client and measure the difference. Expect a gap from round trips, serialization, and lost cross-request batching; report the actual split.
- An honest limits section: what your ensemble doesn't handle (failure of one stage, versioning across stages, per-stage autoscaling — which is where [Ray Serve](08-ray-serve-and-composition.md) would come in).

---

## Common ways both builds go wrong

| Symptom | Cause | Fix |
|---|---|---|
| Engine B "wins" by 3× | Prefix caching on for one, off for the other | Level axis 6; report both states |
| Both engines report identical throughput at every load | Client is the bottleneck (closed loop, or one Python process saturating) | Check `drift`; run the client on another machine or with multiple processes |
| TTFT excellent, throughput terrible | Offered load below capacity — you measured an idle server | Push to the knee; report the ladder, not one point |
| Numbers change 20% between runs | Thermal/clock drift, other tenants, no settle time | Alternate A/B/A, fix clocks if possible, increase `--settle` |
| Triton avg batch = 1 under load | Dynamic batching off, or `max_batch_size: 0`, or delay window ≈ 0 | Check the config actually loaded (`/v2/models/<m>/config`) |
| Ensemble slower than three curl calls | You measured one request at a time | Ensembles win under *concurrency*; measure at load |

---

**Next:** [Exercises & exit artifact →](11-exercises-and-artifacts.md) — the full exercise set, the no-notes self-check, and what "finished Phase 5" means.
