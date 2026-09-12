# 6 — Deploying and Rollouts

> **You'll be able to say:** "Rolling, blue-green and canary trade capacity against blast radius, and on GPUs capacity is the expensive axis: `maxSurge: 1` means one extra GPU must exist or the rollout hangs `Pending` forever, and blue-green means paying for the whole fleet twice. I template with Helm or Kustomize, pin images and weights by digest so the manifest is the rollback target, drive progressive delivery with Argo Rollouts using the Phase-7 SLO queries as automated gates, and I have measured — not assumed — my time to rollback."

[Lesson 4](04-kubernetes-for-gpu-serving.md) made one replica behave. [Phase 7 lesson 8](../phase-7/08-canary-and-shadow-traffic.md) defined the *evidence* a new version must produce. This lesson is the delivery mechanism that carries the version from a merged commit to 100% of traffic, and back again in under five minutes when the evidence turns bad.

---

## The three strategies, priced in GPUs

For a 20-replica, 1-GPU-per-replica service:

| Strategy | Extra GPUs needed | Blast radius | Rollback speed | Good for |
|---|---|---|---|---|
| **Rolling**, `maxSurge: 1 / maxUnavailable: 0` | **+1** | 1/20 of traffic at a time | minutes (roll forward the old version) | the default |
| **Rolling**, `maxSurge: 0 / maxUnavailable: 1` | 0 | 1/20, but you run at 19/20 capacity | minutes | capacity-constrained fleets |
| **Blue-green** | **+20 (2×)** | zero during bake, 100% at the flip | **seconds** (flip the Service selector) | high-stakes, small fleets, or when you have committed capacity to spare |
| **Canary** (5% → 25% → 100%) | +1 to +2 | exactly the canary % | seconds (shift weight to 0) | anything model-affecting |
| **Shadow / mirror** | +1 to +2 | **zero** (responses discarded) | n/a | pre-canary evidence ([Phase 7 L8](../phase-7/08-canary-and-shadow-traffic.md)) |

The line most teams learn the expensive way: **`maxSurge: 1` on a GPU fleet requires a free GPU to exist.** If the cluster is full, the surge pod is `Pending`, the rollout stalls, and — with `maxUnavailable: 0` — it stalls *safely* but indefinitely. With `maxUnavailable: 25%` it instead tears down capacity first and you brown out. Decide which failure you want, and reserve the surge headroom explicitly in the capacity arithmetic from [lesson 5](05-scheduling-capacity-and-autoscaling.md).

`progressDeadlineSeconds` deserves a deliberate value: it must exceed your cold start ([lesson 3](03-image-size-and-cold-start.md)) or Kubernetes will declare a perfectly healthy slow-loading rollout failed.

```yaml
spec:
  minReadySeconds: 60            # a pod must stay Ready 60s before the next one is replaced
  progressDeadlineSeconds: 1200  # > worst-case cold start, or the rollout "fails" spuriously
  strategy:
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
```

`minReadySeconds` is underrated here: it inserts a soak between pod replacements, so a version that crashes 30 seconds after readiness takes down one replica instead of all twenty.

---

## Templating: Helm or Kustomize

You need one, and the choice matters less than the discipline.

| | Helm | Kustomize |
|---|---|---|
| Model | templates + `values.yaml`, packaged and versioned | base + overlays, patch-based, no templating language |
| Strength | one artifact per release, easy third-party charts, `helm rollback` | plain YAML stays plain YAML; no Go-template debugging |
| Weakness | template soup at scale; `helm rollback` hides the diff | no packaging/versioning story of its own |
| Use when | you ship a chart others install; you want release history | you own the cluster and prefer readable diffs |

The rule that matters regardless: **environments differ by values, never by forked YAML.** One base, overlays for dev/staging/prod that change *only* replica counts, resource sizes, model digest, and thresholds. The moment `prod/deployment.yaml` and `staging/deployment.yaml` are separate files, staging stops predicting production and your gates become theatre.

```yaml
# values-prod.yaml — everything environment-specific, nothing structural
replicaCount: 20
image:
  repository: registry.example.com/llm-server
  digest: sha256:9f3a2c...            # digest, never a tag
model:
  uri: s3://models/llama-3.1-8b-instruct
  digest: sha256:7bd41e...            # weights pinned too (lesson 7)
engine:
  maxNumSeqs: 64
  gpuMemoryUtilization: 0.90
  maxModelLen: 8192
slo:
  ttftP99Seconds: 0.5                 # the same number as the histogram bucket & the alert
```

Note the last line. The SLO threshold, the Prometheus histogram boundary ([Phase 7 lesson 2](../phase-7/02-instrumenting-with-prometheus.md)), the burn-rate alert, and the canary gate should all read from **one** declared value. When they drift, you get an SLO that is subtly not the thing you're alerting on — a bug that is nearly invisible and lasts for months.

### GitOps

Argo CD or Flux reconciling a git repo into the cluster gives you three things that matter for this phase: the running state is a *diffable artifact*, `kubectl edit` drift gets reverted automatically, and rollback is `git revert` — which means your rollback path is exercised by every normal deploy rather than being a special emergency procedure nobody has run in six months.

---

## Progressive delivery with automated gates

Argo Rollouts (or Flagger) replaces the `Deployment` strategy with a stepped plan whose steps are guarded by *queries against your own metrics*. This is where Phase 7's work becomes deployment machinery:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata: { name: llm-inference }
spec:
  replicas: 20
  strategy:
    canary:
      canaryService: llm-canary
      stableService: llm-stable
      trafficRouting:
        istio: { virtualService: { name: llm-vs } }
      steps:
        - setWeight: 5
        - pause: { duration: 30m }        # bake: long enough for sample size (Phase 7 L8)
        - analysis:
            templates: [{ templateName: inference-gates }]
        - setWeight: 25
        - pause: { duration: 30m }
        - analysis:
            templates: [{ templateName: inference-gates }]
        - setWeight: 50
        - pause: { duration: 1h }
        - setWeight: 100
      analysis:
        templates: [{ templateName: inference-gates }]   # runs continuously during the rollout
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata: { name: inference-gates }
spec:
  metrics:
    - name: error-rate
      interval: 2m
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(inf_requests_total{version="canary",outcome="error"}[5m]))
            / sum(rate(inf_requests_total{version="canary"}[5m]))
      successCondition: result[0] <= 0.005
    - name: ttft-p99
      interval: 2m
      failureLimit: 2
      provider:
        prometheus:
          query: |
            histogram_quantile(0.99,
              sum by (le) (rate(inf_ttft_seconds_bucket{version="canary"}[5m])))
      successCondition: result[0] <= 0.5
    - name: output-length-ratio          # the QUALITY gate — the one that catches truncation
      interval: 5m
      failureLimit: 1
      provider:
        prometheus:
          query: |
            avg(inf_output_tokens{version="canary"}) / avg(inf_output_tokens{version="stable"})
      successCondition: result[0] >= 0.9 && result[0] <= 1.1
```

Three properties make this real rather than decorative:

1. **The quality gate exists.** Latency and error gates pass happily while a broken chat template truncates every response ([Phase 7 lesson 8](../phase-7/08-canary-and-shadow-traffic.md)). Output-length ratio, finish-reason distribution, and empty-output rate are the cheapest three quality signals and they catch most real regressions.
2. **The bake time is derived from sample size**, not vibes. At 5% of 50 req/s you get 2.5 req/s; detecting a 1-point shift in a proportion needs thousands of samples, so a 5-minute pause proves nothing. Compute it once, write the number in the manifest.
3. **Failure auto-aborts.** `failureLimit` exceeded → Argo shifts weight back to stable automatically, without a human. That is the difference between a 3-minute and a 40-minute incident at 4 a.m.

**Sticky routing caveat:** if your router does prefix-affinity ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)), a 5% *request* weight is not 5% of *users*, and a conversation can bounce between versions mid-session. Route canary by session/user hash, not per request, so a user gets a consistent version and your quality comparison is per-user rather than per-turn.

---

## Rollback: the number that defines this phase

Measure it, don't assume it:

```
  alert fires ─► decide ─► execute ─► old version at 100% ─► verified healthy
       │           │          │              │
       t0          t1         t2             t3
       ROLLBACK TIME = t3 − t0.   Target < 5 min.  Measure it in a game day.
```

What makes it fast:

| Practice | Effect |
|---|---|
| Old ReplicaSet still around (`revisionHistoryLimit ≥ 5`) | `kubectl rollout undo` is seconds; no build, no pull |
| Image + weights pinned by **digest** in git | the rollback target is unambiguous and still exists ([lesson 7](07-model-registry-and-artifacts.md)) |
| Old image still on the nodes | no re-pull; `IfNotPresent` + no aggressive GC |
| Old weights still in the node cache and the registry | retention window ≥ rollback window |
| Traffic-weight rollback (canary) | seconds, no pod churn at all |
| A rehearsed command / one-button script | no improvisation under stress ([Phase 7 lesson 10](../phase-7/10-incident-response-and-chaos.md)) |
| Backward-compatible schema/config changes | you *can* roll back — see below |

What makes it impossible: a mutable tag that got overwritten, garbage-collected weights, a config migration that isn't backward compatible, or a database/cache schema the new version wrote that the old version can't read.

**Expand/contract is the rule for anything stateful.** Deploy N+1 able to read both formats, migrate, and only *then* remove the old path in N+2. In inference this shows up with prompt templates, KV-cache formats in a shared cache tier, request/response schemas between router and engine, and tokenizer versions. If version N+1 can't be reverted without data loss, you don't have a rollback — you have a hope.

---

## Deploying the model separately from the code

Because weights and code version independently ([lesson 1](01-why-shipping-is-the-job.md)), you want two independent rollout paths:

| Change | Needs a new image? | Rollout type |
|---|---|---|
| Server bug fix | yes | rolling, standard gates |
| Engine version bump (vLLM 0.6 → 0.7) | yes | canary with **quality** gates — defaults change |
| New model checkpoint | **no** — new weight digest, same image | canary with quality gates; this is the highest-risk change per byte |
| Quantization change (FP16 → AWQ) | maybe | shadow first, then canary; paired quality evidence mandatory |
| `max_num_seqs`, admission thresholds | no — config | fast rollout, latency gates; keep it revertible in seconds |
| Sampling defaults / prompt template | no — config | quality gates; this is a *model behaviour* change disguised as config |

The last row is a favourite production trap: a "config-only" prompt-template change ships without gates because it isn't code, and it is the single most likely thing to silently degrade output quality.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Rollout stuck, surge pod `Pending` | no spare GPU for `maxSurge` | reserve surge headroom, or `maxSurge: 0` + one extra baseline replica |
| Rollout marked failed on a healthy slow-loading pod | `progressDeadlineSeconds` < cold start | raise it above worst-case cold start |
| Brownout during every deploy | `maxUnavailable > 0` on a tight fleet | `maxUnavailable: 0` and pay for the surge |
| All replicas replaced before the crash appears | no `minReadySeconds` | soak each pod before continuing |
| Canary passes, full rollout regresses quality | canary traffic unrepresentative (no long prompts / one tenant) | segment gates by prompt-length bucket and `tenant_class` |
| Users see two different model behaviours mid-conversation | per-request canary weighting with sticky sessions | hash canary assignment by session/user |
| "Roll back" needs a 20-minute rebuild | mutable tags; no digest pinning | pin digests; keep old ReplicaSets and artifacts |
| Rollback restores code but not weights | weights deployed out-of-band | one manifest pins both; both roll back together |
| Config change bypassed all gates | config not treated as a deploy | route config through the same pipeline and gates |

---

## Do this now (60 minutes)

1. **Template your service** with Helm or Kustomize, with prod/staging differing only in values, and both pinning image *and* weight digests.
2. **Do a real rolling update under load** at fixed QPS, measuring 5xx and truncated streams. Then repeat with `maxUnavailable: 25%` and no `minReadySeconds` to see the difference your settings buy. Keep both tables.
3. **Time a rollback with a stopwatch**, starting from a simulated page. Then remove one crutch (delete the local image so it must re-pull) and time it again. Write down the two numbers and what closed the gap — this pair is your answer to "how fast can you undo?", which is the most common senior-level deployment interview question.
4. **Add one quality gate** (output-length ratio or empty-output rate) to your rollout analysis, and prove it rejects a deliberately truncating build that passes every latency gate.

---

**Next:** [Model registry and artifacts →](07-model-registry-and-artifacts.md) — weights as first-class, digest-addressed artifacts with checksums, retention windows, and a lineage record that survives the person who trained them.
