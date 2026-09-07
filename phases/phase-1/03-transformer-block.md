# 3 — The Full Transformer Block (Stacking Into a Real Model)

> **You'll be able to say:** "A transformer block = attention + a small MLP, each wrapped with a residual connection and a normalization. A model is just N of these blocks stacked, plus embeddings at the bottom and a projection to vocabulary at the top. I know what every piece is for."

Attention ([lesson 2](02-attention-mechanism.md)) is the star, but it can't work alone. This lesson assembles the complete machine.

---

## The block, at a glance

One transformer block has **two sub-layers**, each wrapped identically:

```
                x  (batch, seq, d_model)
                │
        ┌───────┴────────┐
        │  LayerNorm     │
        │  Attention     │   ← mixes information ACROSS tokens
        └───────┬────────┘
                │
           x = x + (…)      ← residual: add the input back
                │
        ┌───────┴────────┐
        │  LayerNorm     │
        │  MLP (FFN)     │   ← processes EACH token independently
        └───────┬────────┘
                │
           x = x + (…)      ← residual again
                │
                ▼  (batch, seq, d_model)  — same shape out, ready for the next block
```

Notice the block takes `(batch, seq, d_model)` and returns the *same* shape. That's the design that lets you stack them: the output of block 1 is the input of block 2, unchanged in shape, just richer in content. GPT-2 small stacks 12; a 7B model ~32; big models 80+.

Two new pieces to understand: the **MLP**, and the two wrappers — **residuals** and **normalization**.

---

## Piece 1: The MLP / feed-forward network (FFN)

After attention mixes information *between* tokens, the **MLP** processes *each token on its own*, giving the model room to "think" about what it just gathered. It's two linear layers with an activation in between:

```python
def mlp(x, W1, b1, W2, b2):
    h = gelu(x @ W1 + b1)   # expand: d_model → 4*d_model, then nonlinearity
    return h @ W2 + b2       # project back: 4*d_model → d_model
```

- It **expands** the width (usually 4×: GPT-2 goes 768 → 3072), applies a nonlinear activation (**GELU**, a smooth cousin of ReLU), then **projects back** to `d_model`.
- It runs **independently per token** — no mixing across the sequence (that was attention's job). Same weights applied to every position.

**Why it's a big deal for inference:** the MLP holds *most of the model's parameters* — typically ~2/3 of the weights (two matrices of size `d_model × 4·d_model`). So when you compute "how many bytes must I read per token" ([lesson 8](08-inference-math-and-memory.md)), the MLP weights dominate. When you quantize a model ([Phase 0 lesson 5](../phase-0/05-floating-point-and-precision.md) / Phase 4), the MLP is the biggest prize.

---

## Piece 2: Residual connections (the `x = x + …`)

A **residual connection** means: instead of replacing `x` with a sub-layer's output, you *add* the output to `x`. `x = x + Attention(x)`, then `x = x + MLP(x)`.

Why? Two reasons, both practical:

1. **Gradients flow (training reason):** stacking dozens of layers naively makes training collapse; the "+x" gives a clean highway for signal to pass through, letting very deep models train at all.
2. **Each layer edits, doesn't overwrite (intuition):** think of `x` as a shared scratchpad ("the residual stream"). Each sub-layer *reads* it, computes a small update, and *adds* the update back. Information accumulates instead of being destroyed layer to layer.

For inference you mostly need to recognize it in code — a `+` around every sub-layer — and know that this "residual stream" of shape `(batch, seq, d_model)` is the model's running state that flows top to bottom.

---

## Piece 3: Normalization (LayerNorm / RMSNorm)

Deep stacks of matrix multiplies make numbers drift — some explode, some vanish — which wrecks the math. **Normalization** rescales each token's vector to a stable distribution (roughly mean 0, variance 1) before each sub-layer, keeping everything well-behaved.

- **LayerNorm** (GPT-2): subtract the mean, divide by the standard deviation, then apply learned scale/shift. Normalizes across the `d_model` dimension, per token.
- **RMSNorm** (LLaMA and most modern models): a cheaper variant that skips the mean-subtraction. Slightly faster; nearly as effective.
- **Pre-norm vs post-norm:** modern models put the norm *before* each sub-layer ("pre-norm", shown in the diagram) because it trains more stably. GPT-2 is pre-norm too.

You don't need the exact formula memorized for inference. Know: it's a cheap per-token rescale that keeps numbers sane, there are two of them per block, and RMSNorm is the modern default.

---

## The full model: bottom to top

Stack the pieces from [lesson 1](01-tokens-and-embeddings.md) through the blocks to the output:

```
token IDs (batch, seq)
   │  embedding lookup + positional info            (lesson 1)
   ▼
x : (batch, seq, d_model)   ← the residual stream enters
   │
   │  Block 1  ─┐
   │  Block 2   │  N transformer blocks, each: norm→attn→+, norm→mlp→+
   │   …        │  (this lesson)
   │  Block N  ─┘
   ▼
final LayerNorm
   │
   │  "unembedding": multiply by (d_model × vocab_size) matrix
   ▼
logits : (batch, seq, vocab_size)   ← a raw score for EVERY possible next token, at every position
```

The last step, the **language-model head** (or "unembedding"), projects each token's `d_model` vector to a `vocab_size` vector of raw scores called **logits** — one score per possible next token. Often this matrix is the *same* as the embedding table from lesson 1, reused ("weight tying") to save parameters.

During generation you only care about the logits at the **last position** — that's the model's prediction for the *next* token. Turning those logits into an actual chosen token is [lesson 7 (sampling)](07-sampling.md).

---

## Putting numbers on it (GPT-2 small)

To make the abstract concrete, here's GPT-2 small's shape, which you'll rebuild in [lesson 9](09-build-gpt2-from-scratch.md):

| Thing | Value |
|---|---|
| Vocabulary size | 50,257 |
| `d_model` (width) | 768 |
| Number of blocks (layers) | 12 |
| Attention heads | 12 (so `d_head` = 64) |
| MLP hidden width | 3072 (= 4 × 768) |
| Max context length | 1024 tokens |
| Total parameters | ~124 million |

Trace where those 124M parameters live and you'll find the MLPs (two 768×3072 matrices per block × 12 blocks) and the embedding table dominate — exactly the pieces quantization targets.

---

## Key takeaways

- A **transformer block** = `norm → attention → +residual`, then `norm → MLP → +residual`. In and out shape is `(batch, seq, d_model)`, so blocks stack cleanly.
- **Attention** mixes info *across* tokens; the **MLP** processes *each token independently* and holds most of the parameters (the main quantization/bandwidth target).
- **Residual connections** (`x = x + …`) create a running "residual stream" each sub-layer edits rather than overwrites.
- **Normalization** (LayerNorm / RMSNorm) is a cheap per-token rescale that keeps numbers stable; two per block.
- A full model = embeddings → N blocks → final norm → project to **logits** (`vocab_size` scores per position). You use only the *last* position's logits to predict the next token.

**Next:** [Autoregressive decoding →](04-autoregressive-decoding.md) — how those next-token predictions become a generated sequence, one token at a time.
