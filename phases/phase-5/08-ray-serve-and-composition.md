# 8 — Ray Serve and Multi-Model Composition

> **You'll be able to say:** "Ray Serve turns each model into a *deployment* — a group of replicas (Ray actors) with its own resources and its own autoscaling policy — and composition is just one deployment awaiting a handle call on another, in Python. That buys per-stage autoscaling and arbitrary control flow, which Triton ensembles can't do; it costs you a Ray cluster, an extra network/serialization hop between stages, and a distributed-systems debugging surface. The autoscaling signal that matters is **ongoing requests per replica**, not GPU utilization, and cold start is minutes for a GPU model, which is why `min_replicas` is a business decision."

This is the last framework lesson, and it's about the layer *above* the engine: several models, different hardware, different scaling curves, one application.

---

## The model

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment(
    num_replicas="auto",
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    autoscaling_config={
        "min_replicas": 1,            # keep 1 warm: GPU cold start is minutes, not ms
        "max_replicas": 8,
        "target_ongoing_requests": 4, # the real signal: queue+in-flight per replica
        "upscale_delay_s": 10,        # react fast to load
        "downscale_delay_s": 300,     # release slowly: re-loading weights is expensive
    },
    max_ongoing_requests=8,           # per-replica concurrency cap (backpressure)
)
class Embedder:
    def __init__(self):
        self.model = load_embedder()          # runs once per replica, in the actor
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts)

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
class Policy:
    async def __call__(self, doc, scores): ...

@serve.deployment
class Pipeline:                                # the ingress deployment
    def __init__(self, embedder: DeploymentHandle, policy: DeploymentHandle):
        self.embedder, self.policy = embedder, policy
    async def __call__(self, request):
        body = await request.json()
        vecs   = await self.embedder.remote(body["texts"])     # cross-deployment call
        return await self.policy.remote(body, vecs)

app = Pipeline.bind(Embedder.bind(), Policy.bind())
serve.run(app, route_prefix="/predict")
```

Four things this expresses that a Triton ensemble cannot:

1. **Per-stage autoscaling with different policies.** The GPU embedder scales 1→8 on queue depth; the CPU policy stage sits at 2. In Triton, stage concurrency is `instance_group` on one machine, changed by editing config and reloading.
2. **Arbitrary control flow.** Branching, retries, fan-out/fan-in, calling an external API mid-pipeline, conditional model selection — it's Python.
3. **Heterogeneous placement.** Deployments declare `num_gpus`/`num_cpus`/custom resources; Ray schedules them across a cluster, so one application can span a GPU node and CPU nodes.
4. **Composition across *frameworks and processes* without a network protocol you have to design.** Handle calls are the API.

And the costs, stated as plainly:

- **A hop per stage.** Handle calls serialize through Ray's object store / RPC. For small payloads it's sub-millisecond; for large tensors it's a real tax that Triton's in-process ensemble avoids. Keep tensors on the GPU or pass references, not megabytes of floats, and measure it.
- **You now operate Ray**: head node, workers, dashboards, object-store memory, actor failures, version skew. That's a genuine platform commitment ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)).
- **Autoscaling GPU replicas is slow.** Weight loading plus CUDA-graph capture plus warmup is 1-5 minutes for an LLM. Autoscaling that reacts in 10 s to a spike still delivers capacity minutes later; `min_replicas` and pre-warmed pools are how you survive that, and it's the same problem Phase 6 discusses for the whole fleet.

---

## Autoscaling on the right signal

The instinct is to scale on GPU utilization. Don't.

```
  GPU utilization is a LIAR for inference:
    • decode is memory-bandwidth-bound; the GPU reports high "utilization" while
      doing almost no math (Phase 2 lesson 5) — busy ≠ saturated
    • a replica can be at 100% "utilization" with a batch of 1 (badly underloaded)
      or with a batch of 64 (genuinely saturated)

  Scale on QUEUE, because that is what the user feels:
    target_ongoing_requests ≈ the number of concurrent requests per replica at which
    your SLO is still met — find it from the load ladder you already ran (Phase 3 lesson 7):
      run the ladder, find the concurrency where p99 TTFT hits the SLO, divide by replicas.
```

`max_ongoing_requests` is the per-replica backpressure cap: beyond it, Ray Serve queues at the handle rather than piling into the replica. That's admission control ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)) again — the third time this phase, in a third framework, under a third name. **Notice the pattern: every serving system converges on the same four controls — a batch/token budget, a concurrency cap, a queue policy, and a shed rule.**

For LLM deployments specifically, one subtlety: an LLM engine already batches internally, so a Ray Serve "replica" is an entire vLLM engine holding tens of concurrent sequences. `max_ongoing_requests` should therefore be *at least* the engine's `--max-num-seqs`, or Ray will starve the engine's batcher trying to protect it. Getting this backwards is the most common Ray-Serve-in-front-of-vLLM misconfiguration.

---

## Model multiplexing: many models, few GPUs

```python
@serve.deployment
class Multiplexed:
    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_id: str):
        return load_lora(model_id)                 # cold load, cached per replica

    async def __call__(self, request):
        model = await self.get_model(serve.get_multiplexed_model_id())
        return model(request)
```

Ray Serve routes a request to a replica that **already has** that model loaded when possible, and LRU-evicts models per replica otherwise. This is the long-tail-fine-tune problem: 200 customer LoRAs, 8 GPUs, most adapters idle most of the time. Same shape as prefix-aware sticky routing ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)): **routing that respects what a replica has already loaded is worth more than perfectly even load balancing.**

The alternative for LoRA specifically is a single engine serving many adapters natively (vLLM's `--enable-lora`, [lesson 3](03-vllm-in-production.md)). Multiplexing generalizes to *any* model type; native LoRA is faster when it applies. Know both.

---

## Ray Serve + LLMs

Ray Serve ships LLM-specific building blocks (`ray.serve.llm`) that wrap engines like vLLM in a deployment with an OpenAI-compatible ingress, letting you place several models — or several TP-sharded replicas — in one application. The engine still does the paging, batching and prefix caching; Ray contributes placement, replication, autoscaling and routing.

That layering is the point of this lesson, and the durable takeaway:

```
  ENGINE  (vLLM/TRT-LLM/SGLang)  →  per-GPU efficiency: batching, KV, kernels
  SERVE LAYER (Ray Serve / K8s)  →  per-fleet efficiency: replicas, routing,
                                     autoscaling, multi-model packing
```

They optimize different denominators. Confusing them produces both classic mistakes: tuning kernels when your fleet is 30% idle, and adding replicas when your batch size is 2.

---

## Choosing between the three composition options

| | **Triton ensemble** | **Ray Serve** | **Client-side orchestration** |
|---|---|---|---|
| Inter-stage transfer | In-process, GPU-resident | Ray RPC / object store | Full network round trips |
| Per-stage batching | Yes, automatic across concurrent pipelines | Only if you implement it (`@serve.batch`) | No |
| Per-stage autoscaling | No (instance_group, per-box) | Yes, first-class | Yes (separate services) |
| Control flow | Static DAG (BLS for logic) | Arbitrary Python | Arbitrary |
| Ops burden | One server + a repo dir | A Ray cluster | N services + a mesh |
| Best for | Fixed straight-line pipelines, latency-critical | Heterogeneous stages, spiky independent load | Small scale, or stages owned by different teams |

Default recommendation, stated as a rule you can defend: **fixed pipeline, tight latency budget, one box → Triton ensemble. Independently scaling stages, heavy Python logic, existing Ray/K8s platform → Ray Serve. Neither, and it's two stages at low QPS → just call two services and stop building platforms.**

`@serve.batch` deserves one line of its own: it's a decorator that accumulates concurrent calls into a list-arg batch with a max size and timeout — Phase 3's dynamic batching, in a decorator, for models that don't batch themselves. Use it on embedding/ranking stages; never in front of an LLM engine that already batches.

---

## Do this (90 minutes, CPU-only is fine)

1. `pip install "ray[serve]"`. Build a 2-deployment app: a "featurizer" (CPU, `@serve.batch`-decorated) and a "model" (any tiny model), composed through a handle. Serve it and hit it.
2. Load it with your Phase-3 harness at increasing rates. Watch the Serve dashboard's per-deployment queue and replica counts; find the concurrency at which p99 exceeds your target and set `target_ongoing_requests` from that measurement, not from a guess.
3. Set `min_replicas: 0` and measure cold-start latency for the first request after idle. That number is why serverless GPU inference is hard; write it down.
4. Add `@serve.multiplexed` with three fake "models" and confirm from logs that repeat requests for the same id land on the replica that already loaded it.
5. Write the one-paragraph decision you'd make for *your* capstone pipeline ([Phase 10](../../ROADMAP.md#phase-10--capstones-this-is-where-top-1-gets-proven)), using the table above, and name the measurement that would change your mind.

---

**Next:** [Reading and modifying an engine →](09-reading-engine-source.md) — the method for entering a 300k-line codebase, finding the bug, and explaining it from the source.
