# 8 — CI/CD and Benchmark Gates

> **You'll be able to say:** "ML CI has four gate classes — unit, smoke, **quality**, and **performance** — and the last two are what make it ML CI rather than software CI. A performance gate on shared cloud runners is noise; it needs a dedicated GPU runner, a pinned clock, a warmup, and a threshold derived from measured run-to-run variance, expressed as a *relative regression against a recorded baseline*, not an absolute number. Quality gates compare distributions, never exact strings. And every gate must be able to distinguish 'the code got slower' from 'the runner was busy', or people will disable it within a month."

[Lesson 7](07-model-registry-and-artifacts.md) defined the artifact. This lesson is the pipeline that produces it and the evidence that lets it be promoted. It is where the benchmarking discipline from [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md) and [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md) stops being a manual ritual and becomes automation that blocks merges.

---

## The pipeline shape

```
  PR opened
    ├── (1) LINT + UNIT + TYPE            30-90 s   every PR, CPU runner
    ├── (2) BUILD IMAGE                   2-8 min   cache-heavy; produces a digest
    ├── (3) SMOKE: start engine, 1 gen    2-5 min   GPU runner, tiny model
    ├── (4) QUALITY: fixed prompt set     5-20 min  GPU runner, real model
    └── (5) PERFORMANCE: benchmark        10-30 min GPU runner, exclusive, pinned clocks
                                                 │
  merge to main ─────────────────────────────────┤
    ├── (6) publish image + artifact by digest
    ├── (7) deploy to staging, replay traffic sample
    └── (8) canary in prod with Phase-7 gates (lesson 6)
```

Stages 1-2 are ordinary software CI. **Stages 3-5 are the phase's subject**, and stage 5 is the one everybody gets wrong first.

Cost discipline matters: GPU CI minutes are 20-100× CPU minutes. The standard split is stages 1-3 on every PR, stages 4-5 on PRs labelled `perf`/`model` plus a nightly run on `main`, and the full matrix pre-release. Making the expensive gates *manually triggerable on any PR* is what stops people from routing around them.

---

## Gate 3 — smoke

The cheapest gate with the highest catch rate. It answers: does this image start, allocate a GPU, load a model, and produce a token?

```yaml
- name: Smoke
  run: |
    docker run -d --gpus all --name smoke -p 8000:8000 "$IMAGE_DIGEST" \
      --model hf-internal-testing/tiny-random-gpt2 --max-model-len 512
    timeout 300 bash -c 'until curl -sf localhost:8000/health/ready; do sleep 2; done'
    curl -sf localhost:8000/v1/completions \
      -d '{"model":"m","prompt":"hello","max_tokens":8}' | tee out.json
    python -c "import json,sys; d=json.load(open('out.json')); \
               assert d['choices'][0]['text'].strip(), 'empty generation'"
    docker stop --time 30 smoke      # also asserts graceful shutdown in <30s
```

It catches: CUDA/arch mismatches, missing runtime libs, non-root permission failures, broken entrypoints, the shell-form `SIGTERM` bug from [lesson 2](02-containers-for-gpu-workloads.md), and unset env defaults. A `tiny-random-*` model keeps it under five minutes.

---

## Gate 4 — quality

The gate that does not exist in software CI, and the reason ML deployments regress silently.

**Never assert exact output strings.** They change with any kernel, batch composition, or library version — floating-point non-determinism makes exact-match gates flap, and a flapping gate gets deleted. Assert *distributional* and *task-level* properties instead:

| Tier | Check | Cost | Catches |
|---|---|---|---|
| 0 | Non-empty; finish reason distribution; output-length ratio vs baseline | seconds | truncation, template breakage, EOS-token bugs — most real regressions |
| 1 | Deterministic-mode logprob/greedy-token agreement vs baseline (allow small drift) | minutes | numerics regressions from a kernel or dtype change |
| 2 | Task metrics on a small fixed set: exact-match on 200 QA items, pass@1 on 50 code items | 5-20 min | genuine capability loss from quantization/distillation |
| 3 | Embedding similarity of outputs vs baseline, aggregated | minutes | paraphrase-level drift |
| 4 | LLM-as-judge pairwise on a sampled set | expensive, noisy | subtle style/instruction-following changes |

Tiers 0-2 belong in CI. Tiers 3-4 belong in the canary analysis ([Phase 7 lesson 8](../phase-7/08-canary-and-shadow-traffic.md)) where you have real traffic and a bigger sample.

```python
# tier 0-2 gate, sketch
res = run_prompt_set(engine, PROMPTS)             # fixed, versioned, committed
base = json.load(open("baselines/quality.json"))  # produced by a recorded main-branch run

assert res["empty_rate"] <= 0.001,                    "empty outputs"
assert 0.90 <= res["mean_len"] / base["mean_len"] <= 1.10, "length distribution moved"
assert res["finish_stop_frac"] >= base["finish_stop_frac"] - 0.02, "more length-truncations"
assert res["qa_exact_match"] >= base["qa_exact_match"] - 0.02, "task quality regression"
```

Two disciplines make this trustworthy: **the prompt set is versioned in the repo** (a moving eval set is not a gate), and **the baseline is a recorded artifact from a specific commit**, refreshed deliberately with a reviewed PR — never auto-updated on every green run, which would let a slow drift walk the baseline anywhere.

Also test the **serving path, not the model**: run through your actual HTTP API with your actual chat template and sampling defaults. A gate that calls `model.generate()` directly misses the entire class of bugs this phase exists to catch.

---

## Gate 5 — performance

The hard one, because a benchmark in CI is a measurement in the least controlled environment you own.

### Making the number trustworthy

| Requirement | Why | How |
|---|---|---|
| **Dedicated GPU runner, exclusive** | a co-tenant job destroys the measurement | self-hosted runner, concurrency group of 1, no other GPU jobs |
| **Pinned clocks** | boost/thermal drift is ±10% by itself | `nvidia-smi -lgc <clock>` and `-pl <power>`; record the settings ([Phase 2 lesson 8](../phase-2/08-profiling-in-practice.md)) |
| **Warmup excluded** | first iterations include JIT, autotune, graph capture | discard the first N seconds ([lesson 3](03-image-size-and-cold-start.md)) |
| **Fixed load, open loop** | closed-loop benchmarks self-throttle and hide latency | Phase-3 harness at fixed QPS ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) |
| **Fixed input distribution** | prompt-length mix drives everything | committed prompt set with a recorded length histogram |
| **Percentiles, not means** | means hide the regression you care about | p50/p90/p99 TTFT, TPOT, and tokens/s |
| **Repeat and report variance** | you cannot set a threshold without knowing the noise | ≥3 runs; record stddev |
| **Same GPU model every time** | comparing an L40S run to an A100 run is meaningless | pin the runner label; record GPU + driver + engine version in the result |

### Setting the threshold

The mistake is `assert p99 < 500ms`. That is an SLO assertion in the wrong place: it passes a 40% regression that still fits under the limit, and it fails when the runner is warm-but-slightly-busy. Gate on **relative regression against a recorded baseline, with a band derived from measured noise**:

```
  measure run-to-run variance first:  10 identical runs on main
     throughput  σ ≈ 2.1%      p99 TTFT σ ≈ 4.3%
  threshold = max(3σ, minimum meaningful effect)
     throughput regression > 7%   → FAIL
     p99 TTFT regression   > 13%  → FAIL
     both within band             → PASS, record the new numbers as a data point
```

Then keep the history: a per-commit trend line makes a 2%-per-week creep visible, which no single-commit gate can ever catch. Publish the table as a PR comment so the tradeoff is visible at review time rather than in a postmortem.

```yaml
name: perf-gate
on: { pull_request: { types: [labeled] }, schedule: [{ cron: "0 3 * * *" }] }
concurrency: { group: gpu-bench, cancel-in-progress: false }   # exclusivity
jobs:
  bench:
    if: github.event.label.name == 'perf' || github.event_name == 'schedule'
    runs-on: [self-hosted, gpu, l40s]
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - name: Pin clocks
        run: sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1350
      - name: Run benchmark
        run: |
          python bench/run.py --qps 8 --duration 300 --warmup 60 \
                 --prompts bench/prompts_v3.jsonl --out result.json
      - name: Compare to baseline
        run: python bench/compare.py --baseline baselines/l40s.json \
                 --result result.json --max-throughput-drop 0.07 --max-p99-rise 0.13
      - uses: actions/upload-artifact@v4
        with: { name: bench-${{ github.sha }}, path: result.json }
      - name: Comment table on PR
        if: github.event_name == 'pull_request'
        run: python bench/comment.py result.json baselines/l40s.json
```

**When the gate fails legitimately** (you knowingly traded latency for throughput), the fix is a reviewed baseline update in the same PR — visible, justified, and diffable. That is the whole reason the baseline is a committed file rather than a database row.

---

## Building artifacts in CI

Two things belong in the pipeline that people wrongly do by hand:

- **Quantization runs** ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)): AWQ/GPTQ conversion with a pinned calibration set, producing a new artifact with lineage — reproducible, not "Bob ran a notebook."
- **Compiled engine builds** ([Phase 5 lesson 6](../phase-5/06-tensorrt-llm-and-compiled-engines.md)): TensorRT-LLM plans take minutes to hours and are keyed to (GPU arch × library version × TP × shapes). Build them in a matrix job, store each as an artifact with that tuple in its metadata, and never build at pod startup.

Both are long jobs; both are perfect nightly/manual-dispatch work whose output is a registry artifact ([lesson 7](07-model-registry-and-artifacts.md)) rather than a container image.

---

## Secrets, runners, and supply chain

| Concern | Practice |
|---|---|
| Registry/cloud credentials | OIDC federation to a short-lived role, not long-lived keys in secrets |
| Self-hosted GPU runners | **never** on public-repo PRs from forks without approval — arbitrary code on your GPU |
| Runner hygiene | ephemeral runners or a cleanup step; a leftover process silently ruins the next benchmark |
| Model/dataset access tokens | scoped read-only, rotated, never echoed into logs |
| Image provenance | sign with cosign; generate an SBOM; record the git sha as an image label |
| Reproducibility | `--require-hashes` lockfiles ([lesson 2](02-containers-for-gpu-workloads.md)); pin base images by digest |

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Perf gate flaps; team disables it | shared/noisy runner, no clock pinning, threshold below noise | dedicated exclusive runner, pinned clocks, threshold = 3σ |
| Gate passes, production regresses | benchmark load unrepresentative (short prompts, closed loop) | match the real prompt-length and concurrency distribution |
| Quality gate flaps on identical code | exact-match assertions on non-deterministic output | distributional checks; greedy + fixed seed for tier-1 checks |
| Slow 2%/week regression never caught | per-commit gate only, no trend | store every result; alert on 30-day drift |
| Baseline silently walks upward | auto-updating baseline on green runs | baseline is a reviewed committed file |
| CI green, pod crashes on start | no smoke test with the real entrypoint | stage 3 exists for exactly this |
| GPU CI bill exceeds the inference bill | full matrix on every PR | tiered triggers; nightly for the expensive gates |
| "Works in CI" only | CI uses a tiny model, prod uses a 70B | run at least one gate on the real model/config combination pre-release |
| Engine plan built at deploy time | not treated as an artifact | build in a CI matrix, publish to the registry |

---

## Do this now (60 minutes)

1. **Add the smoke gate** to your repo — start the container, wait for readiness, generate, assert non-empty, `docker stop` under 30 s. Confirm it fails when you deliberately break the entrypoint.
2. **Measure your benchmark noise** before writing any threshold: run your Phase-3 harness 10 times unchanged and compute σ for throughput and p99 TTFT. Set gate thresholds at 3σ and write both numbers into the workflow as comments so the next person knows where they came from.
3. **Add a tier-0 quality gate** (empty rate, length ratio, finish-reason mix) against a committed prompt set and a recorded baseline. Prove it catches a deliberately broken chat template that the smoke and perf gates both pass.
4. **Make the perf job comment a table on the PR.** The cultural effect is larger than the technical one: performance becomes something reviewers see, and regressions get argued about before merge instead of after deploy.

---

**Next:** [Infrastructure as code →](09-infrastructure-as-code.md) — provisioning the GPU capacity, cluster and registry reproducibly with Terraform, including quota, cost guardrails, and the parts of infrastructure that must never be created by clicking.
