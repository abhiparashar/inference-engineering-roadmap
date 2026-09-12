# 10 — Build: Container, CI Gate, and a Deploy You Can Undo

> **What you're building:** the whole delivery path for the server you wrote in Phases 3 and 7 — a small GPU-aware image, a digest-pinned artifact, a CI pipeline with smoke/quality/performance gates, a Kubernetes deployment that drains cleanly, and a rollback you have *timed with a stopwatch*. The acceptance test is not "it deploys." It is: **a deliberately regressed build is blocked by CI, a deliberately bad model is caught by the canary gate, and a rollback completes in under five minutes without a rebuild.**

This is [Project 15](../../projects/README.md) and the Phase 8 exit artifact. Everything runs on a laptop with `kind` and a CPU model; a GPU makes the numbers real but changes none of the mechanics.

---

## Target architecture

```
   git push
      │
      ▼
  ┌────────────────────── CI (GitHub Actions) ──────────────────────┐
  │  lint+unit → build image (digest) → smoke → quality → perf gate │
  │                                     │        │         │        │
  │                                 tiny model  prompt set  bench   │
  │                                              vs baseline vs base│
  └────────────────────────────┬─────────────────────────────────────┘
                               │ all green: publish image@sha256 + weights@sha256
                               ▼
  ┌───────────────── kind / k8s cluster ─────────────────┐
  │  Deployment (probes, preStop, grace, PDB)            │
  │    ├─ stable  (v1, weight digest A)                  │
  │    └─ canary  (v2, weight digest B)  ← 10% traffic   │
  │  Prometheus  →  gate query  →  promote | rollback    │
  └──────────────────────────────────────────────────────┘
```

Repository layout:

```
  serving/            server.py, warmup, health endpoints, SIGTERM drain
  docker/             Dockerfile, .dockerignore
  bench/              run.py, compare.py, prompts_v1.jsonl
  quality/            gate.py, prompts.jsonl, baselines/quality.json
  deploy/             base/ + overlays/{staging,prod}   (kustomize) or chart/
  registry/           publish.py  (content-addressed artifact + metadata.json)
  .github/workflows/  ci.yml, perf.yml
  scripts/            rollback.sh, timed-rollback.sh
```

---

## Step 1 — the server has to be deployable first

Three changes to your Phase-3/7 server, all small, all load-bearing.

```python
# serving/lifecycle.py
import asyncio, signal, time, os, json, hashlib
from prometheus_client import Gauge

READY = False
LAST_STEP = time.monotonic()          # stamped by the engine loop every iteration
STALL_TIMEOUT_S = float(os.getenv("STALL_TIMEOUT_S", "60"))
DRAINING = False

MODEL_INFO = Gauge("inf_model_info", "loaded model", ["name", "digest", "engine"])
STARTUP = Gauge("inf_startup_stage_seconds", "startup stage duration", ["stage"])


def verify_artifact(path: str) -> dict:
    """Refuse to start on a corrupt or mismatched artifact (lesson 7)."""
    meta = json.load(open(f"{path}/metadata.json"))
    for line in open(f"{path}/SHA256SUMS"):
        want, name = line.split()
        h = hashlib.sha256()
        with open(f"{path}/{name}", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise SystemExit(f"FATAL: checksum mismatch for {name}")
    return meta


async def warmup(engine, max_batch: int):
    """Warm the shapes you actually serve, then flip readiness (lesson 3)."""
    global READY
    t0 = time.monotonic()
    for n_prompt in (128, 512, 2048):
        for batch in (1, min(8, max_batch), max_batch):
            await engine.generate(["token " * n_prompt] * batch, max_tokens=8)
    STARTUP.labels("warmup").set(time.monotonic() - t0)
    READY = True


def install_drain(app, engine, grace_s: float = 150.0):
    """SIGTERM: stop accepting, finish in-flight, then exit (lessons 2, 4)."""
    loop = asyncio.get_event_loop()

    async def drain():
        global READY, DRAINING
        READY, DRAINING = False, True             # readiness false → endpoints removed
        deadline = time.monotonic() + grace_s
        while engine.num_running and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        loop.stop()

    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(drain()))
```

```python
# serving/health.py
@app.get("/health/ready")
def ready():
    if not READY or DRAINING:
        return Response(status_code=503)
    if engine.num_waiting > ADMISSION_LIMIT:      # shed at the LB, bounded (lesson 4)
        return Response(status_code=503)
    return Response(status_code=200)

@app.get("/health/live")
def live():
    idle = engine.num_running == 0 and engine.num_waiting == 0
    stalled = (time.monotonic() - LAST_STEP) > STALL_TIMEOUT_S
    return Response(status_code=503 if (stalled and not idle) else 200)
```

**Verify before continuing** — these three tests take five minutes and each one fails on most first attempts:

```bash
# 1. graceful shutdown: must exit well under the grace period, no dropped stream
( curl -sN localhost:8000/v1/completions -d '{"prompt":"x","max_tokens":400,"stream":true}' & )
sleep 1 && time docker stop --time 180 srv          # expect a few seconds, full response

# 2. liveness catches a hang
docker exec srv bash -c 'kill -STOP $(pgrep -f engine)'   # expect 503 within STALL_TIMEOUT_S

# 3. readiness is false during warmup
docker run -d --name srv img && curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/health/ready
```

---

## Step 2 — the image

Use the multi-stage Dockerfile from [lesson 2](02-containers-for-gpu-workloads.md) verbatim, then measure:

```bash
docker build -t llm-server:dev -f docker/Dockerfile .
docker images llm-server:dev --format '{{.Size}}'
docker history --human --no-trunc llm-server:dev | head -20
```

Record a before/after table. A realistic result going from a naive single-stage build to the multi-stage one:

| Change | Size | Cold pull |
|---|---|---|
| naive single-stage `devel` + pip cache + weights baked | 21.4 GB | 6 m 40 s |
| multi-stage, `runtime` base | 9.1 GB | 2 m 55 s |
| + `.dockerignore`, no pip cache, `--no-install-recommends` | 7.6 GB | 2 m 25 s |
| + weights out of the image | **3.9 GB** | **1 m 10 s** |
| + arch-trimmed `TORCH_CUDA_ARCH_LIST` | 3.4 GB | 1 m 02 s |

Your numbers will differ; the *shape* — weights and the devel base dominating — will not.

---

## Step 3 — the artifact and the registry

```python
# registry/publish.py — content-addressed artifact with lineage (lesson 7)
import hashlib, json, pathlib, shutil, sys, datetime

def digest_dir(src: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(src.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(src).as_posix().encode())
            h.update(p.read_bytes() if p.stat().st_size < (1 << 22) else _stream(p, h) or b"")
    return h.hexdigest()

def publish(src: pathlib.Path, store: pathlib.Path, meta: dict) -> str:
    d = digest_dir(src)
    dst = store / f"sha256-{d}"
    if dst.exists():
        return d                                   # idempotent: same bytes, same address
    tmp = store / f".tmp-{d}"
    shutil.copytree(src, tmp)
    with open(tmp / "SHA256SUMS", "w") as fh:
        for p in sorted(tmp.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
                         f"{p.relative_to(tmp).as_posix()}\n")
    meta |= {"digest": f"sha256:{d}",
             "created_at": datetime.datetime.now(datetime.UTC).isoformat()}
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(dst)                                # atomic publish
    return d
```

Then a channel pointer that a deploy updates:

```bash
echo "sha256-7bd41e..." > $STORE/channels/prod     # changing this IS a deploy
```

Prove two properties: publishing the same directory twice yields the same digest and does nothing the second time; corrupting one byte and restarting the server aborts with a checksum error instead of serving.

---

## Step 4 — CI with real gates

```yaml
# .github/workflows/ci.yml
name: ci
on: { pull_request: {}, push: { branches: [main] } }
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements-dev.txt && ruff check . && pytest -q tests/unit

  build:
    needs: unit
    runs-on: ubuntu-latest
    outputs: { digest: ${{ steps.push.outputs.digest }} }
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - id: push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: ghcr.io/${{ github.repository }}/server:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true

  smoke:
    needs: build
    runs-on: ubuntu-latest        # CPU model; use [self-hosted, gpu] for the real thing
    steps:
      - uses: actions/checkout@v4
      - run: |
          IMG=ghcr.io/${{ github.repository }}/server@${{ needs.build.outputs.digest }}
          docker run -d --name smoke -p 8000:8000 -e MODEL=sshleifer/tiny-gpt2 "$IMG"
          timeout 300 bash -c 'until curl -sf localhost:8000/health/ready; do sleep 2; done'
          curl -sf localhost:8000/v1/completions \
            -d '{"prompt":"hello","max_tokens":8}' -o out.json
          python -c "import json;assert json.load(open('out.json'))['choices'][0]['text'].strip()"
          start=$SECONDS; docker stop --time 60 smoke; echo "drain took $((SECONDS-start))s"
          test $((SECONDS-start)) -lt 30       # graceful shutdown is a GATE, not a hope

  quality:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python quality/gate.py --baseline quality/baselines/quality.json \
                                    --prompts quality/prompts.jsonl
```

```python
# quality/gate.py — tier-0/1 distributional gate (lesson 8)
import json, argparse, statistics as st

def evaluate(client, prompts):
    outs = [client.generate(p["prompt"], max_tokens=256, temperature=0.0) for p in prompts]
    return {
        "n": len(outs),
        "empty_rate": sum(1 for o in outs if not o.text.strip()) / len(outs),
        "mean_len": st.mean(len(o.token_ids) for o in outs),
        "finish_stop_frac": sum(1 for o in outs if o.finish_reason == "stop") / len(outs),
        "exact_match": sum(1 for o, p in zip(outs, prompts)
                           if p.get("answer") and p["answer"].lower() in o.text.lower())
                       / max(1, sum(1 for p in prompts if p.get("answer"))),
    }

def main(res, base):
    checks = [
        ("empty outputs",      res["empty_rate"] <= 0.001),
        ("length distribution", 0.90 <= res["mean_len"] / base["mean_len"] <= 1.10),
        ("truncation rate",    res["finish_stop_frac"] >= base["finish_stop_frac"] - 0.02),
        ("task quality",       res["exact_match"] >= base["exact_match"] - 0.02),
    ]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
```

### The performance gate

```python
# bench/compare.py — relative regression against a recorded baseline (lesson 8)
import json, argparse, sys

a = argparse.ArgumentParser()
a.add_argument("--baseline"); a.add_argument("--result")
a.add_argument("--max-throughput-drop", type=float, default=0.07)   # = 3σ, measured
a.add_argument("--max-p99-rise",        type=float, default=0.13)
o = a.parse_args()
b, r = json.load(open(o.baseline)), json.load(open(o.result))

assert b["gpu"] == r["gpu"] and b["prompt_set"] == r["prompt_set"], \
    "incomparable runs: different GPU or prompt set"

tput_drop = (b["output_tok_s"] - r["output_tok_s"]) / b["output_tok_s"]
p99_rise  = (r["ttft_p99_s"] - b["ttft_p99_s"]) / b["ttft_p99_s"]

rows = [("output tok/s", b["output_tok_s"], r["output_tok_s"], -tput_drop, -o.max_throughput_drop),
        ("p99 TTFT (s)", b["ttft_p99_s"],  r["ttft_p99_s"],   p99_rise,  o.max_p99_rise)]
print(f"{'metric':<14}{'baseline':>12}{'current':>12}{'delta':>10}{'limit':>10}")
for name, bv, rv, d, lim in rows:
    print(f"{name:<14}{bv:>12.2f}{rv:>12.2f}{d:>9.1%}{lim:>10.1%}")

fail = tput_drop > o.max_throughput_drop or p99_rise > o.max_p99_rise
sys.exit(1 if fail else 0)
```

**Before you pick those thresholds, measure your noise** — 10 identical runs on `main`:

| Run | output tok/s | p99 TTFT (s) |
|---|---|---|
| 1-10 | mean 812, σ 17 (2.1%) | mean 0.41, σ 0.018 (4.3%) |
| threshold = 3σ | 7% drop | 13% rise |

Write those σ values as comments in the workflow. A threshold with no recorded noise measurement behind it is a guess, and guesses become flaky gates become deleted gates.

---

## Step 5 — deploy, canary, rollback

Use the manifests from [lesson 4](04-kubernetes-for-gpu-serving.md) and the rollout from [lesson 6](06-deploying-and-rollouts.md). Then build the one script that matters:

```bash
#!/usr/bin/env bash
# scripts/timed-rollback.sh — measures the number that defines this phase
set -euo pipefail
start=$(date +%s)
kubectl rollout undo deployment/llm-inference
kubectl rollout status deployment/llm-inference --timeout=600s
until curl -sf "http://$ENDPOINT/health/ready" >/dev/null; do sleep 1; done
curl -sf "http://$ENDPOINT/v1/completions" -d '{"prompt":"ping","max_tokens":4}' >/dev/null
echo "ROLLBACK COMPLETE in $(( $(date +%s) - start ))s"
```

Run it four times under increasingly hostile conditions and record all four numbers:

| Scenario | Expected |
|---|---|
| Old ReplicaSet present, image cached on node | 20-60 s |
| Image evicted from the node cache (`crictl rmi`) | + pull time |
| Weight cache cleared on the node | + fetch time |
| Rollback of a *model* change (channel pointer + restart) | your weight-swap path |

---

## Acceptance tests

The build is done when all seven pass. Each one corresponds to a lesson; if one fails, that lesson isn't finished.

| # | Test | Pass condition |
|---|---|---|
| 1 | Image | final image < 40% of the naive build; weights not in it; runs non-root ([L2](02-containers-for-gpu-workloads.md), [L3](03-image-size-and-cold-start.md)) |
| 2 | Cold start | five stages instrumented; warm-node start < 90 s; readiness gated on warmup ([L3](03-image-size-and-cold-start.md)) |
| 3 | Drain | load test through a rolling update: **0 5xx, 0 truncated streams** ([L4](04-kubernetes-for-gpu-serving.md), [L6](06-deploying-and-rollouts.md)) |
| 4 | Liveness | `kill -STOP` on the engine ⇒ pod restarted within the stall timeout ([L4](04-kubernetes-for-gpu-serving.md)) |
| 5 | Artifact | corrupt one weight byte ⇒ server refuses to start; `inf_model_info` carries the digest ([L7](07-model-registry-and-artifacts.md)) |
| 6 | CI gates | a deliberate 20% slowdown fails the perf gate; a deliberate chat-template break fails the quality gate and passes the perf gate ([L8](08-ci-cd-and-benchmark-gates.md)) |
| 7 | Rollback | timed, scripted, **< 5 min** in the worst of the four scenarios, with no rebuild ([L6](06-deploying-and-rollouts.md), [L7](07-model-registry-and-artifacts.md)) |

Test 6 is the one to design deliberately. Two injected regressions:

```python
# regression A — performance: caught by bench, invisible to quality
time.sleep(0.02)                       # in the request path; ~20% TTFT rise at your QPS

# regression B — quality: caught by the quality gate, invisible to bench
CHAT_TEMPLATE = CHAT_TEMPLATE.replace("<|eot_id|>", "")   # outputs never stop cleanly
```

Showing that A and B are each caught by exactly one gate is the proof that you need both — and it is the single most persuasive artifact from this phase in an interview.

---

## Stretch goals

1. **Progressive delivery for real**: install Argo Rollouts in `kind`, run the [lesson 6](06-deploying-and-rollouts.md) canary with the Prometheus analysis template, and show it auto-aborting on regression B.
2. **Lazy image pull**: enable SOCI/eStargz and measure the pull-to-start delta on your 4 GB image.
3. **Terraform the sandbox**: apply the [lesson 9](09-infrastructure-as-code.md) stack against a real cloud sandbox, with Infracost commenting the monthly delta on the PR.
4. **Model rollback path**: make weights swappable without a pod restart (load into a second slot, flip atomically) and re-measure rollback time — this is how the fastest teams get model rollback under 10 seconds.
5. **Multi-node**: deploy a 2-rank LeaderWorkerSet group and verify that killing one rank restarts the whole group rather than leaving a hung collective ([Phase 6 lesson 9](../phase-6/09-multi-node-operations.md)).

---

## What to write down

A `README.md` in `projects/15-docker-cicd-gate/` containing: the image-size and cold-start before/after tables, the measured benchmark noise (σ) and the thresholds derived from it, screenshots or logs of gates rejecting regressions A and B, the four rollback timings, and one paragraph on what your remaining slowest stage is and what you'd do next. That document is the Phase 8 exit artifact ([lesson 11](11-exercises-and-artifacts.md)).

---

**Next:** [Exercises & exit artifact →](11-exercises-and-artifacts.md) — the drills that make these mechanics permanent, and the self-check before Phase 9.
