# 4 — What "Inference" Actually Is

> **You'll be able to say:** "A trained model is just a big pile of fixed numbers (weights). Inference is one forward pass — feeding input through those numbers to get an output. No learning, no gradients. That's why serving a model is a completely different performance problem than training one."

---

## Training vs inference in one picture

```
TRAINING (done once, offline, by researchers)
  data ──▶ forward pass ──▶ prediction ──▶ compare to truth (loss)
                                              │
            weights updated ◀── backward pass (gradients) ◀──┘
  repeat billions of times until the weights are "good"
  → produces: a fixed set of numbers (the weights / checkpoint)

INFERENCE (done constantly, in production, by you)
  input ──▶ forward pass ──▶ prediction    ← and that's it. Done.
  no comparing, no gradients, no weight updates. Weights are frozen.
```

**Training** is the process of *finding* good weights: run data forward, measure how wrong the output is, compute *gradients* (which direction to nudge every weight to be less wrong), and update the weights. Repeat astronomically many times. It's iterative, stateful, and hungry for memory (it must remember intermediate values to compute gradients).

**Inference** is *using* those frozen weights: run one input forward through the fixed numbers, read the output. No measuring wrongness, no gradients, no updates. Just the forward pass.

That difference — *"inference is only the forward pass"* — is the reason this whole discipline exists. It changes what's expensive, what's possible, and what you optimize.

---

## What a "model" and a "forward pass" really are

Strip away the mystique. A neural network is a **function**: numbers in, numbers out. Inside, it's mostly:

1. **Matrix multiplications** — the input vector gets multiplied by weight matrices. This is ~99% of the compute.
2. **Element-wise functions ("activations")** — small nonlinear tweaks like ReLU or GELU applied to each number, so the network can represent complex relationships (not just straight lines).
3. **A few normalizations and additions** — housekeeping to keep numbers well-behaved.

A **forward pass** is running the input through all these layers, in order, once. The "weights" are the numbers *in those matrices* — fixed after training. A 7-billion-parameter model literally means 7 billion numbers sitting in those matrices.

```
input vector ─▶ [× W1] ─▶ activation ─▶ [× W2] ─▶ activation ─▶ … ─▶ output
                 ▲ weights (frozen)      ▲ weights (frozen)
```

That's it. When someone says "the model runs," they mean "the input vector got multiplied through this stack of fixed matrices once."

### A one-line inference (no ML library, just math)

A layer is `output = activation(input @ W + b)`. Here's a tiny made-up "model" doing real inference in NumPy:

```python
import numpy as np

# a "trained" model = fixed weights (normally loaded from a file)
W1 = np.array([[0.2, -0.5], [0.1, 0.4], [-0.3, 0.8]])   # shape (3, 2)
b1 = np.array([0.0, 0.1])
W2 = np.array([[1.0], [-1.0]])                            # shape (2, 1)
b2 = np.array([0.5])

def relu(x): return np.maximum(0, x)

def forward(x):                 # x: input vector, shape (3,)
    h = relu(x @ W1 + b1)       # layer 1: matmul + bias + activation
    y = h @ W2 + b2             # layer 2: matmul + bias
    return y

print(forward(np.array([1.0, 2.0, 3.0])))   # one forward pass = inference
```

There are no gradients anywhere. Loading real weights and adding more layers is the same thing, bigger. You'll build a *real* GPT-2 forward pass exactly like this in [Phase 1](../phase-1/09-build-gpt2-from-scratch.md).

---

## Why inference is its own performance problem

Because it's *only* the forward pass, inference has a profile that's almost the opposite of training:

| | Training | Inference |
|---|---|---|
| Gradients / backward pass | Yes (doubles+ the work, huge memory) | **No** |
| Weights | Changing every step | **Frozen** — can be pre-processed, quantized, compiled |
| Memory pressure from | Storing activations for backprop | The **KV-cache** and model weights |
| Runs how often | Once (offline) | **Billions of times** (every user request) |
| What you optimize for | Throughput over days | **Latency per request AND throughput AND cost per token** |
| Batch size | Huge, you control it | Whatever users happen to send, *right now* |

Two consequences that shape the whole roadmap:

1. **Weights are frozen, so you can pre-process them.** You can convert them to fewer bits (quantization, [lesson 5](05-floating-point-and-precision.md)), rearrange them for the hardware, or compile the whole forward pass into an optimized program — because they'll never change. Training can't do this; the weights move every step.
2. **Inference is >90% of the lifetime compute cost** of any deployed AI product. A model is trained once but serves requests forever. So a 2x inference speedup is a ~2x cut in the ongoing bill for ChatGPT-scale traffic. *That's* why companies pour engineering into it, and why this skill is valuable.

---

## The two kinds of inference workload (know the difference)

Not all inference looks the same. Two broad shapes, and they behave very differently:

- **Single-shot models** (image classifier, embedding model, ranking model): one input → one forward pass → one output. Done. Predictable, easy to batch, latency ≈ one forward pass.
- **Autoregressive models** (LLMs / text generators): the output is produced **one token at a time**, and each new token is fed back in to produce the next. So generating a 200-token answer means ~200 forward passes, in sequence, one after another. This is why LLMs are slow and latency-sensitive in a way a classifier never is.

```
Classifier:   input ─▶ [forward] ─▶ answer                    (1 pass)

LLM:  prompt ─▶ [forward] ─▶ tok1 ─▶ [forward] ─▶ tok2 ─▶ [forward] ─▶ tok3 …
              each pass depends on the previous token's output  (N passes, sequential)
```

That "one token at a time, each depending on the last" property is called **autoregressive decoding**, and it is *the* defining challenge of LLM inference. All of Phase 1 unpacks it; here just lock in the shape: **an LLM answer is a loop of forward passes, not one forward pass.**

---

## Key takeaways

- A trained model = a big pile of **frozen numbers** (weights). Inference = **one forward pass** through them. No gradients, no learning.
- A forward pass is mostly **matrix multiplications** plus small activation functions and normalizations.
- Because weights are frozen, you can pre-process them (quantize, compile, rearrange) — the foundation of every Phase 4 optimization.
- Inference is a *different* performance problem than training: no backward pass, frozen weights, but it runs constantly and is judged on latency, throughput, **and** cost per token — and it's >90% of an AI product's lifetime compute.
- **LLMs generate one token at a time** (autoregressive), so an answer is a *sequence* of forward passes. That single fact drives Phase 1 and everything after.

**Next:** [Floating point & precision →](05-floating-point-and-precision.md) — the numbers inside those matrices, and why using fewer bits per number is the highest-leverage trick in the field.
