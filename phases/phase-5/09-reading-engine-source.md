# 9 — Reading and Modifying an Engine

> **You'll be able to say:** "I have a method for entering a 300k-line codebase: orient by size and entrypoint, find the loop with three greps, follow the *data structure* rather than the call graph, then confirm with a running process — `py-spy dump`, a debug log, a print in the scheduler. With that I can take a real GitHub issue about batching or memory, reproduce it, locate the responsible code, explain the root cause in terms of the mechanism, and open a patch that includes the test that would have caught it."

This is the phase's self-check and the skill with the longest half-life in this entire roadmap. Frameworks will change; "read unfamiliar production code and explain its behavior" will not.

---

## The method

### 1. Orient by size and entrypoint (10 minutes, never skip)

```bash
find <pkg> -name '*.py' | xargs wc -l | sort -rn | head -20   # where the mass is
cat pyproject.toml | grep -A8 'scripts\]'                      # what the CLI actually calls
git log --oneline -20                                          # what's changing now
git shortlog -sn --since='6 months' | head -5                  # who to read after
```

Big files are load-bearing. In every engine you've met this phase, the three biggest files in the core package are the scheduler, the model runner, and the config — and that's not a coincidence, it's what an inference engine *is*.

### 2. Find the loop with three greps

Every engine has one function that runs forever and does everything. Find it:

```bash
grep -rn "while True" --include=*.py <pkg> | grep -iE "step|loop|engine|sched"
grep -rn "def step\|def schedule\|def execute_model\|def batching_task" -r <pkg>
grep -rn "forward\b" --include=*.py <pkg>/…/model_runner*   # where tensors finally move
```

Read that function top to bottom before reading anything else. Everything else in the repo is either preparation for it or bookkeeping after it.

### 3. Follow the data structure, not the call graph

Call graphs in these codebases are deep and boring (config → config → config). The *state* is short and revealing. For any engine, find and read these four types:

| Type | vLLM | TGI | SGLang |
|---|---|---|---|
| Per-request state | `vllm/v1/request.py::Request` | `Entry` / protobuf `Request` | `managers/schedule_batch.py::Req` |
| The batch | `worker/gpu/input_batch.py` | `Batch` (Python server) | `ScheduleBatch` |
| Scheduler↔executor contract | `core/sched/output.py::SchedulerOutput` | gRPC protobufs | `ModelWorkerBatch` |
| KV bookkeeping | `core/kv_cache_manager.py` | `block_allocator.rs` | `mem_cache/radix_cache.py` |

Print the fields of those four and you have the engine's mental model. This takes an hour and replaces a week of aimless reading.

### 4. Confirm with a running process

Reading tells you what the code *can* do; running tells you what it *does*.

```bash
# What is the process doing right now (no restart, no instrumentation)?
py-spy dump --pid $(pgrep -f 'vllm serve' | head -1)
py-spy top  --pid <pid>                      # sampling profiler, live
py-spy record -o profile.svg --pid <pid> --duration 30

# Engine-side logging
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> …
TORCH_LOGS=recompiles,graph_breaks python …   # torch.compile churn (Phase 2 lesson 6)

# C++/CUDA side
gdb -p <pid>; thread apply all bt            # TRT-LLM / Triton backends
compute-sanitizer, nsys profile               # Phase 2 lesson 8 tooling still applies
```

`py-spy dump` against a hung or slow server is the single highest-value debugging command in this phase: no restart, no code change, and it names the exact line every thread is on.

### 5. Use the tests as documentation

`tests/v1/core/test_scheduler.py`, `tests/v1/core/test_prefix_caching.py` and friends are executable specifications of the behavior you're trying to understand, and they run on CPU in seconds. When you can't tell what a function guarantees, find its test — then *modify* the test to ask your question, which is also how you verify a patch later.

### 6. Use git as an oracle

```bash
git log -p --follow <file> | head -200        # why this code looks like this
git log -S"max_num_batched_tokens" --oneline  # every commit that touched a concept
git blame -L 120,160 <file>                   # then read that PR's discussion on GitHub
```

The PR discussion attached to a confusing line usually contains the benchmark that motivated it. That's where the *reasons* live, and reasons are what you're actually reading for.

---

## From issue to root cause: the workflow

The phase self-check is: *given a framework's issues page, find a real batching/memory bug report and explain the root cause from the source.* Here's how to do that reliably rather than luckily.

### Pick a tractable issue

```
  Good hunting grounds (vLLM as the example):
    label:bug + "preempt" | "OOM" | "prefix cache" | "throughput regression"
    label:performance
    "usage" issues where the user's config contradicts their expectation
  Prefer issues that have:
    ✓ a reproducible command line and a version
    ✓ a metric that moved (not "it feels slow")
    ✓ activity from maintainers (you can check your reasoning against theirs — AFTER
      you form your own)
  Avoid at first:
    ✗ hardware-specific hangs (NCCL, driver)   ✗ "support model X"   ✗ flaky CI
```

### The five steps

1. **Restate the claim as a number.** "TTFT rises from 200 ms to 4 s when `--max-num-seqs 256` with 8k prompts." If you can't state it numerically, the issue is under-specified — ask, or pick another.
2. **Predict, from the mechanism, before reading any code.** You have the models: KV bytes ([Phase 4 lesson 1](../phase-4/01-what-to-optimize.md)), queueing `1/(1−ρ)` ([Phase 3 lesson 5](../phase-3/05-queueing-theory.md)), preemption cost, token budget contention. Write down which one you think it is. This is what separates an engineer from a bisect script.
3. **Reproduce small.** Smallest model, shortest prompts, `--enforce-eager`, one GPU (or CPU) that still shows the *shape* of the problem. If the reproduction needs an H100 and 20 minutes, shrink it until it doesn't — most scheduler/memory bugs reproduce at 0.5B.
4. **Locate.** Metrics first (`/metrics`: waiting, kv usage, preemptions), then the loop, then the specific branch. Add a temporary print/log in the scheduler; confirm the branch you suspect is the branch that's taken.
5. **Explain in mechanism terms, with evidence.** Not "the scheduler has a bug" but: *"`allocate_slots` fails once pool usage passes X because each running sequence reserves a block for its speculative tokens; the scheduler then preempts the newest request, whose 8k prefill is recomputed on re-admission, which adds N ms per cycle — see `scheduler.py:LNNN` and the `num_preemptions` counter climbing in the attached graph."*

### The write-up template

```markdown
**Environment**: vLLM <version/commit>, GPU, model, exact command line
**Symptom (measured)**: metric, before → after, with the workload that produces it
**Expected**: what the mechanism should do and why
**Root cause**: file:line + the condition that triggers it, in mechanism language
**Evidence**: metric graph / log excerpt / py-spy dump / a 20-line repro script
**Fix sketch**: the smallest change that addresses the cause (not the symptom)
**Test that would have caught it**: the assertion, in the existing test file
```

That last line is what makes maintainers take you seriously, and it's also the honest test of whether you understood the bug: if you can't name the assertion, you have a theory, not a root cause.

---

## Making the change

Scope discipline first: **one behavior per PR.** Reformatting, renaming and "while I was in there" changes turn a reviewable 30-line diff into an unreviewable 600-line one, and they are the main reason first contributions stall.

The mechanics, common to all of these projects:

```bash
# 1. Read the contributing guide FIRST — it is short and it is enforced by CI.
#    CONTRIBUTING.md, plus pre-commit hooks (ruff/black/clang-format), plus DCO sign-off.
pre-commit install
git commit -s -m "…"            # -s = Signed-off-by, required by several of these repos

# 2. Run only the relevant tests (the full suite needs GPUs you don't have)
pytest tests/v1/core/test_scheduler.py -x -q

# 3. Prove the behavior change with a number, not an adjective
#    (before/after with your Phase-3 harness, or the repo's own benchmark script)
```

Good first contributions, in ascending order of difficulty and all genuinely welcome:

1. **Documentation that's wrong**, especially a flag whose described default doesn't match the code. You'll find these while doing [lesson 3](03-vllm-in-production.md).
2. **A failing test for an open bug.** Enormously useful, low risk, and it forces you to nail the reproduction.
3. **A metric or log line** that would have made someone's issue diagnosable.
4. **A scheduler/cache fix** with a benchmark showing the delta.

Interview-wise, one merged doc/test PR plus one written root-cause analysis is worth more than "I've read the vLLM source", because it's checkable.

---

## Where the loop lives, per engine (cheat sheet)

| Engine | Start here | Then |
|---|---|---|
| **vLLM** | `vllm/v1/core/sched/scheduler.py::schedule` | `kv_cache_manager.py`, `worker/gpu/model_runner.py`, `engine/core.py` |
| **TGI** | `backends/v3/src/backend.rs::batching_task` | `queue.rs`, `router/src/validation.rs`, `server/…/flash_causal_lm.py` |
| **SGLang** | `python/sglang/srt/managers/scheduler.py` (event loop) | `schedule_policy.py`, `mem_cache/radix_cache.py` |
| **TensorRT-LLM** | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp` | `microBatchScheduler.cpp`, `kvCacheManager.cpp`, `_torch/pyexecutor/` |
| **Triton Server** | `src/` core: `dynamic_batch_scheduler.cc` (in `triton-inference-server/core`) | backend API, `ensemble_scheduler.cc` |
| **Ray Serve** | `python/ray/serve/_private/router.py` | `replica.py`, `autoscaling_policy.py` |

---

## Do this (3-4 hours — this is the phase self-check)

1. **Timed orientation.** Pick an engine you have *not* read yet this phase. Give yourself 15 minutes to find its scheduling loop and its per-request state type. Write both down with line numbers.
2. **Trace.** Produce the 10-hop request trace for that engine (like the table in [lesson 2](02-vllm-architecture.md)), at a pinned commit.
3. **Issue hunt.** Find two open or recently-closed issues about batching/memory/throughput. For each: restate the claim numerically, write your mechanism prediction *before* reading the thread, then read the thread and the code and grade your prediction.
4. **Root-cause write-up.** Pick the better of the two and write the full template above (aim for 400-600 words). This is a deliverable of [lesson 11](11-exercises-and-artifacts.md) and a genuinely good thing to publish.
5. **Optional but recommended:** open the smallest real PR you can honestly justify — a doc fix, a metric, or a test. Getting through CI and review once removes the activation energy forever.

---

**Next:** [Build: framework shootout + Triton ensemble →](10-build-shootout-and-ensemble.md) — the two projects, with the leveling checklist that makes the comparison honest.
