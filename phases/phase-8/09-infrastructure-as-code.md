# 9 — Infrastructure as Code

> **You'll be able to say:** "The GPU capacity, cluster, registry, buckets and IAM are Terraform, with remote locked state, environments as separate state files sharing modules, and a plan reviewed in the PR. GPU specifics that generic IaC tutorials skip: quota is a prerequisite not a resource, capacity reservations and committed-use discounts are the real cost lever, spot pools are separate node groups with their own taints, and `terraform destroy` on a bucket holding production weights is the fastest way to lose a rollback target. I also know where IaC stops — Terraform provisions the cluster; it should not deploy my app."

[Lessons 4-6](04-kubernetes-for-gpu-serving.md) assumed a cluster with GPU nodes. This lesson creates it reproducibly, and it is the difference between "our staging cluster is subtly different from prod and nobody knows how" and an environment you can rebuild from an empty account.

---

## Why this matters more for GPUs than for CPUs

| Property | CPU infra | GPU infra |
|---|---|---|
| Cost of a mistake | a $30 instance left running | **a $30/hr instance left running** — $720/day |
| Availability | always there | quota-limited, region-limited, frequently sold out |
| Purchase options | on-demand or reserved | on-demand, spot, capacity reservations, committed use, multi-year contracts |
| Provisioning latency | seconds | minutes, sometimes "next week" (quota tickets) |
| Blast radius of `destroy` | recreate in minutes | recreate *if capacity still exists* — it may not |

Those rows are why the guardrails below (cost estimation in CI, `prevent_destroy`, hard limits) are not bureaucracy here; they're the difference between a normal Tuesday and a five-figure surprise.

---

## Layout: stacks, modules, state

Split state by **blast radius and change frequency**, not by team convenience:

```
  infra/
    modules/
      network/            vpc, subnets, nat, endpoints
      gpu-cluster/        eks/gke cluster + gpu node pools + addons
      model-storage/      buckets, lifecycle, replication
      registry/           image + artifact registry, retention policies
      observability/      prometheus/grafana stack, alert routes
    live/
      staging/  main.tf   ← its own state file
      prod/     main.tf   ← its own state file
```

Rules that prevent the classic disasters:

1. **Remote state with locking** (S3 + DynamoDB, GCS, or Terraform Cloud). Local state means two engineers can corrupt each other's infrastructure; it is the first thing to fix in any repo you inherit.
2. **One state per environment.** A single state holding staging and prod means a bad `apply` can take out both, and a slow plan on prod blocks staging work.
3. **Environments differ by variables, not by copied code.** Same modules, different `.tfvars` — the same discipline as [lesson 6](06-deploying-and-rollouts.md)'s Helm values, for the same reason: staging only predicts prod if it is structurally identical.
4. **Pin provider and module versions.** An unpinned provider upgrade rewriting your node pool is a real outage mode.
5. **Never click in the console.** Manual changes produce drift; drift makes the next `apply` do something nobody predicted. Detect it (`terraform plan -detailed-exitcode` on a schedule) and treat drift as an incident-lite.

---

## A GPU node pool, annotated

```hcl
terraform {
  required_version = "~> 1.9"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.60" } }
  backend "s3" {
    bucket         = "acme-tfstate"
    key            = "prod/gpu-cluster.tfstate"
    region         = "us-east-1"
    dynamodb_table = "tfstate-locks"      # locking: mandatory
    encrypt        = true
  }
}

variable "gpu_instance_type" { type = string  default = "g6e.2xlarge" }   # 1× L40S
variable "min_gpu_nodes"     { type = number  default = 3 }
variable "max_gpu_nodes"     { type = number  default = 24 }              # hard cost ceiling

resource "aws_eks_node_group" "gpu_ondemand" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-ondemand"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.private_subnet_ids          # one AZ per pool: see note below
  instance_types  = [var.gpu_instance_type]
  capacity_type   = "ON_DEMAND"
  ami_type        = "AL2023_x86_64_NVIDIA"          # driver preinstalled

  scaling_config { min_size = var.min_gpu_nodes  desired_size = var.min_gpu_nodes
                   max_size = var.max_gpu_nodes }
  update_config  { max_unavailable = 1 }

  labels = { workload = "inference", "nvidia.com/gpu.present" = "true" }
  taint {
    key    = "nvidia.com/gpu"
    value  = "present"
    effect = "NO_SCHEDULE"                          # keep CPU workloads off $/hr nodes
  }

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]  # the autoscaler owns this, not TF
  }
  tags = { Environment = "prod", CostCenter = "inference", ManagedBy = "terraform" }
}
```

Four GPU-specific details worth internalizing:

- **`ignore_changes` on `desired_size`.** Without it, every `terraform apply` fights your autoscaler and resets the replica count — usually downward, usually at the worst time.
- **Single-AZ node pools per zone.** Multi-AZ pools plus zonal GPU stockouts produce confusing partial failures; one pool per AZ makes capacity failures explicit and lets you drain a zone cleanly.
- **Separate spot pool.** Different node group, different taint (`spot=true:NoSchedule`), its own scaling limits, tolerated only by workloads that can lose a node in 120 s ([lesson 5](05-scheduling-capacity-and-autoscaling.md)).
- **`max_size` is a budget control.** `max_size × $/hr × 730` is your worst-case monthly bill for this pool. Compute it, write it in a comment, and alert at 80% of the node count.

### Capacity reservations and commitments

The largest cost lever in this entire phase is *purchase mode*, not engineering:

| Mode | Discount | Risk | Use for |
|---|---|---|---|
| On-demand | 0% | none | burst, canary headroom |
| Spot / preemptible | 50-80% | eviction in 30-120 s | batch, evals, surge on top of a sufficient baseline |
| Capacity reservation (ODCR) | 0% (guarantees availability) | you pay for it whether used or not | protecting against stockouts for a known peak |
| Savings plan / committed use (1-3 yr) | 30-60% | you owe the money regardless | the steady-state baseline you are certain of |

The standard shape: **commit the trough, on-demand the peak, spot the batch.** Express it in Terraform as separate node pools with explicit sizes so the financial decision is visible in code review rather than buried in a billing console.

---

## What Terraform should and should not do

| Terraform owns | Something else owns |
|---|---|
| VPC, subnets, NAT, private endpoints | pod-level networking |
| Cluster, node pools, IAM, quotas | Deployments, Services, HPAs → Helm/Argo ([lesson 6](06-deploying-and-rollouts.md)) |
| Buckets for weights, retention policies | the artifacts themselves ([lesson 7](07-model-registry-and-artifacts.md)) |
| Registries and their lifecycle rules | image contents ([lesson 8](08-ci-cd-and-benchmark-gates.md)) |
| Cluster addons (GPU operator, ingress, cert-manager) | app config, model digests |
| Alert routing, notification channels | dashboards you iterate on daily |

**Do not deploy the application with Terraform.** It is technically possible via the `kubernetes`/`helm` providers and it is a trap: application deploys happen 50× more often than infra changes, need progressive delivery and instant rollback, and shouldn't be gated behind a state lock held by an infra plan. Terraform's cadence is weeks; your deploy cadence is hours. Different tools, different loops.

The one legitimate overlap is **bootstrap addons** — the GPU operator, ingress controller, metrics stack — installed once by Terraform so a fresh cluster is usable.

---

## Guardrails

```hcl
resource "aws_s3_bucket" "model_weights" {
  bucket = "acme-model-weights-prod"
  lifecycle { prevent_destroy = true }        # rollback targets live here (lesson 7)
}

resource "aws_s3_bucket_versioning" "model_weights" {
  bucket = aws_s3_bucket.model_weights.id
  versioning_configuration { status = "Enabled" }   # accidental overwrite ≠ data loss
}

resource "aws_budgets_budget" "gpu" {
  name         = "gpu-inference-monthly"
  budget_type  = "COST"
  limit_amount = "40000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  cost_filter { name = "TagKeyValue"  values = ["user:CostCenter$inference"] }
  notification {
    comparison_operator = "GREATER_THAN"
    threshold           = 80
    threshold_type      = "PERCENTAGE"
    notification_type   = "FORECASTED"          # forecast, so it warns before it happens
    subscriber_email_addresses = ["inference-oncall@acme.com"]
  }
}
```

In CI, for every infra PR:

| Step | Tool | Blocks merge when |
|---|---|---|
| `fmt` + `validate` | terraform | syntax/format issues |
| `plan` posted as a PR comment | terraform + a bot | always visible; a human reads the diff |
| Policy check | OPA/Conftest, Sentinel, tfsec/checkov | public bucket, unencrypted volume, missing tags, instance type outside the allowlist |
| **Cost diff** | Infracost | monthly delta > threshold without an explicit approval label |
| Drift detection | scheduled `plan -detailed-exitcode` | nightly; drift opens a ticket |

The cost-diff comment is the single highest-value addition for GPU infrastructure. "This PR increases monthly cost by $18,400" in a review thread prevents an entire genre of incident, and it takes an afternoon to set up.

---

## Quota: the prerequisite that is not a resource

No IaC can provision capacity you are not allowed to have. Before any of the above matters:

- Know your **per-region, per-instance-family GPU quota** (AWS: vCPU-based quotas per family, e.g. "Running On-Demand P instances"; GCP: `NVIDIA_H100_GPUS` per region; Azure: per-family vCPU).
- Request increases **weeks** ahead of a launch; large GPU quota requests are human-reviewed and can take days and sometimes require a commitment.
- **Alert on approaching quota**, not on hitting it — `pods Pending > 5 min` ([lesson 5](05-scheduling-capacity-and-autoscaling.md)) plus a quota-utilization metric.
- Record quota per region in the same repo as the Terraform, as documentation. It is a real constraint on your capacity plan and it is invisible in the code otherwise.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| `terraform apply` scales the fleet down mid-incident | `desired_size` managed by TF and the autoscaler | `ignore_changes` on desired size |
| Two engineers corrupt state | local or unlocked state | remote backend with locking |
| Staging doesn't reproduce prod bugs | forked configs | shared modules, differing variables only |
| Provider upgrade recreates node pools | unpinned versions | pin `required_providers`; read the plan |
| `destroy` deletes production weights | no `prevent_destroy`, no versioning | both, plus separate state for storage |
| Costs explode with no code change | unbounded `max_size`, autoscaler with no ceiling | hard `max_size`, budget alerts on forecast |
| Cluster can't get GPUs at launch | quota not raised in advance | quota as a launch checklist item, tracked in the repo |
| Nightly drift, nobody notices | console changes | scheduled drift detection → ticket |
| Deploys blocked behind an infra plan | app deployed via Terraform | move app deploys to Helm/Argo |
| Spot evictions take down the service | spot in the same pool as baseline | separate pools, separate taints, spot as surge only |

---

## Do this now (60 minutes)

1. **Write a minimal but real stack**: remote locked state, one GPU node group (taint + labels + `ignore_changes`), a weights bucket with versioning and `prevent_destroy`, and a registry with a retention policy that respects your rollback window ([lesson 7](07-model-registry-and-artifacts.md)). Apply it to a sandbox — or run `plan` only if you don't want the bill; the plan output is most of the learning.
2. **Compute the worst-case monthly cost** of your node pool (`max_size × $/hr × 730`) and put it in a comment next to `max_size`. If that number surprises you, the limit is wrong.
3. **Add cost estimation to CI** (Infracost or an equivalent) so every infra PR shows a monthly delta, and one policy check (tfsec/checkov) that fails on an untagged or public resource.
4. **Write your quota table**: region × instance family × current quota × what your peak plan needs. Any row where need > quota is a launch blocker you now know about weeks early.

---

**Next:** [Build: containerize, gate, and deploy →](10-build-container-and-cicd-gate.md) — assembling lessons 2-9 into a working pipeline whose acceptance test is a measured rollback time and a benchmark gate that blocks a real regression.
