# 2 — The Attention Mechanism (The Heart of the Transformer)

> **You'll be able to say:** "For each token, attention looks back at every other token, decides how relevant each one is (Q·K), turns those into weights (softmax), and gathers a weighted blend of their information (·V). I can write every tensor shape."

This is the most important lesson in Phase 1. Go slow. Draw shapes.

---

## The intuition first (no math)

Read this sentence: *"The animal didn't cross the street because **it** was too tired."* What does "it" refer to — the animal or the street? You resolved it instantly by letting "it" *look back* at the other words and decide "animal" is the relevant one. That looking-back-and-weighting is **exactly** what attention does, mechanically, for every token.

Attention's job: for each token, produce a new vector that is a **relevance-weighted blend of information from all the tokens it's allowed to see.** "Relevance" is computed; "blend" is a weighted sum. That's the whole idea. Now we make it precise.

---

## Query, Key, Value (the three roles)

For each token's vector, attention computes three new vectors by multiplying it with three learned weight matrices (`W_Q`, `W_K`, `W_V`):

- **Query (Q)** — "what am I looking for?" (the current token's search request)
- **Key (K)** — "what do I offer?" (each token advertises what it's about)
- **Value (V)** — "what information do I actually carry?" (the content to hand over if selected)

> **Analogy — a library search.** Your **Query** is what you type into the catalog. Each book has a **Key** (its catalog description) and a **Value** (its actual contents). You match your query against every book's key to score relevance, then you read the contents (values) of the relevant books, weighted by how relevant each was. Attention does this for every token against every other token, in parallel, with numbers.

```python
import numpy as np
# one sequence, seq_len tokens, each a d_model vector  (batch dropped for clarity)
seq_len, d_model = 4, 8
x = np.random.randn(seq_len, d_model)          # input: (4, 8)

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

Q = x @ W_Q          # (4, 8)  each token's "query"
K = x @ W_K          # (4, 8)  each token's "key"
V = x @ W_V          # (4, 8)  each token's "value"
```

---

## The formula, one piece at a time

The famous line is `Attention(Q, K, V) = softmax(QKᵀ / √d) V`. Let's build it up.

### Step 1 — Scores: how much does each token relate to each other token?

Take the dot product of every Query with every Key. A dot product is big when two vectors point the same way → "these two tokens are relevant to each other."

```python
scores = Q @ K.T        # (4, 8) @ (8, 4) = (4, 4)
```

`scores[i, j]` = how much token *i* (asking) cares about token *j* (offering). The result is a `(seq_len, seq_len)` grid — **every token scored against every token.** This N×N grid is why attention is O(N²): its size grows with the *square* of sequence length. Remember that; it's the cost that the KV-cache and FlashAttention both fight.

### Step 2 — Scale by √d (keep numbers sane)

Dot products of `d`-dimensional vectors get large as `d` grows, which makes the next step (softmax) too "spiky." Dividing by `√d` keeps them in a reasonable range. Pure numerical hygiene.

```python
d_k = Q.shape[-1]
scores = scores / np.sqrt(d_k)
```

### Step 3 — Causal mask (you can't see the future)

For text *generation*, token *i* must only attend to tokens `≤ i` — it can't peek at words that come later (they don't exist yet when generating). So we set the "future" scores to −∞ before the softmax, which makes their weight zero. This is the **causal mask**, and it's why generation works left-to-right.

```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)  # True above diagonal = future
scores[mask] = -np.inf                                          # can't attend to the future
```

```
scores grid after masking (rows=asking token, cols=offered token):
        t0    t1    t2    t3
  t0 [  s   -inf  -inf  -inf ]   ← token 0 sees only itself
  t1 [  s    s   -inf  -inf ]    ← token 1 sees 0,1
  t2 [  s    s    s   -inf ]     ← token 2 sees 0,1,2
  t3 [  s    s    s    s   ]     ← token 3 sees everything before + itself
```

(Encoder models like BERT skip the mask — they see the whole input at once. Generative/decoder LLMs always use it.)

### Step 4 — Softmax: turn scores into weights that sum to 1

**Softmax** converts a row of raw scores into positive numbers that add up to 1 — i.e. proper weights. Bigger score → bigger weight. The −∞ entries become exactly 0.

```python
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)     # (4, 4); each row sums to 1
```

Now `weights[i, j]` = "how much of token *j*'s information token *i* should absorb."

### Step 5 — Weighted sum of Values: gather the information

Multiply the weights by V. Each token's output is a blend of all the Values it attended to, weighted by relevance.

```python
output = weights @ V          # (4, 4) @ (4, 8) = (4, 8)
```

Output shape = input shape `(seq_len, d_model)`. Attention took in a matrix of token vectors and returned a matrix of token vectors — but now **each token's vector has been enriched with relevant context from the others.** That enrichment is the entire point.

### The whole thing in one function

```python
def attention(x, W_Q, W_K, W_V):
    Q, K, V = x @ W_Q, x @ W_K, x @ W_V
    scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    n = x.shape[0]
    scores[np.triu(np.ones((n, n)), k=1).astype(bool)] = -np.inf
    return softmax(scores) @ V
```

That's a working (single-head) causal self-attention. Everything else is scaling it up.

---

## Multi-head attention (do it several times in parallel)

One set of Q/K/V learns *one* kind of relationship (say, "which noun does this pronoun refer to"). We want the model to track many relationships at once (syntax, subject, tense…). So we run attention **h times in parallel** with different learned projections — each is a **head** — then concatenate the results and mix them with one more matrix (`W_O`).

The trick: instead of `h` full-width attentions, we split `d_model` into `h` pieces of size `d_head = d_model / h`. Total compute stays about the same; you just reshape.

```
d_model = 512, heads = 8  →  d_head = 64
Q,K,V each reshaped from (seq, 512) to (8 heads, seq, 64)
run attention independently per head → (8, seq, 64)
concat heads back to (seq, 512) → multiply by W_O (512,512) → output (seq, 512)
```

Each head attends over the *same* tokens but in its own 64-dim subspace, capturing a different pattern. This is why you'll hear "32 heads, head_dim 128" describing a model's shape.

---

## The shapes, for real (the self-check answer)

Here are the exact tensor shapes for the Phase-1 self-check: **`batch=2, seq_len=10, heads=8, d_model=512`** (so `d_head = 512/8 = 64`):

```
input x                : (2, 10, 512)              (batch, seq, d_model)
Q = x @ W_Q            : (2, 10, 512)
K = x @ W_K            : (2, 10, 512)
V = x @ W_V            : (2, 10, 512)
reshape to heads       : (2, 8, 10, 64)            (batch, heads, seq, d_head)
scores = Q @ Kᵀ        : (2, 8, 10, 10)            ← the N×N grid, PER head. Note the 10×10.
                                                     memory here ∝ seq² → the O(N²) cost
softmax(scores)        : (2, 8, 10, 10)
out = scores @ V       : (2, 8, 10, 64)
concat heads           : (2, 10, 512)
project with W_O       : (2, 10, 512)              ← same shape as input x
```

If you can reproduce this table from memory, you have cleared the single hardest Phase-1 self-check. The two numbers to *understand* (not just memorize): the `(…, 10, 10)` scores grid is the **O(N²)** term, and the output is back to `(batch, seq, d_model)` so blocks can stack ([lesson 3](03-transformer-block.md)).

---

## Why attention dominates the inference story

- The N×N scores grid means attention compute and memory grow with **seq_len²**. Long contexts (32k, 128k tokens) make this brutal → motivates **FlashAttention** (Phase 2: compute it without ever storing the full grid) and context-length tricks.
- Q, K, V are computed from the token vectors — and during generation, the K and V of past tokens **don't change**. Recomputing them every step is wasteful. Caching them is the **KV-cache** ([lesson 5](05-kv-cache.md)) — the most important optimization in LLM serving, and it falls directly out of this lesson.

---

## Key takeaways

- Attention = for each token, **score** it against every other token (Q·K), **normalize** to weights (softmax), and **gather** a weighted blend of their information (·V).
- **Q** = what I'm looking for, **K** = what I offer, **V** = what I carry.
- The **causal mask** stops tokens attending to the future — that's what makes left-to-right generation valid.
- The scores form an **N×N grid** → attention is **O(seq²)** in compute and memory. This single fact drives FlashAttention and long-context engineering.
- **Multi-head** = run attention in several subspaces in parallel to capture different relationships.
- Output shape equals input shape `(batch, seq, d_model)`, so blocks stack. Know the shape table cold.

**Next:** [The full transformer block →](03-transformer-block.md) — wrap attention with an MLP, residuals, and norms to build the real thing.
