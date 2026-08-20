# Concepts: rank and SVD

**Rank** = the number of linearly independent columns in a matrix, equivalently the
dimension of the space its columns span. It's the answer to "how many genuinely
distinct directions does this matrix actually contain?"

`F_aug` is `(n_samples, n_fac+1)`. Its rank is at most `n_fac+1`. It would be less
if, say, factor 3 were exactly the sum of factors 1 and 2 — then that column adds no
new direction, and the rank drops to `n_fac`. The matrix still has `n_fac+1` columns;
it just doesn't have `n_fac+1` columns' worth of information.

## How SVD relates

SVD decomposes any matrix into a weighted sum of rank-1 pieces:

```
A = s₁·u₁v₁ᵀ + s₂·u₂v₂ᵀ + ... + sᵣ·uᵣvᵣᵀ
```

Each `uᵢvᵢᵀ` is one elementary direction-pair, and the singular value `sᵢ` is exactly
the weight on it — how much of `A` that piece accounts for. They come sorted,
`s₁ ≥ s₂ ≥ ...`.

So: **rank = the number of nonzero singular values.** If a direction contributes
nothing, its weight is zero and the term vanishes. This is why rank and SVD weights
are the same question asked two ways — SVD doesn't just tell you the rank, it tells
you *how strongly* each of the r directions contributes, which is strictly more
information.

## Where `rcond` comes in

In exact arithmetic a singular value is zero or it isn't. In floating point you get
things like `s = 3e-15` — mathematically zero, numerically noise. So you need a
threshold, and "rank" becomes *numerical* rank: the count of `sᵢ > rcond · s₁`.
That's the rank determination `rcond` controls in `np.linalg.lstsq`.

## One thing to keep separate

The `W` from `lstsq` are regression weights — factor→region coefficients. The `sᵢ`
are singular values of `F_aug`, a property of the design matrix alone, with no
reference to `O`. Different objects.

But they're coupled through the solve, which is `W = V·diag(1/sᵢ)·Uᵀ·O`. Note the
`1/sᵢ`: a small singular value gets *inverted* into a large multiplier. So a
near-degenerate direction in the factors (small `s`) becomes an enormous, unstable
entry in the regression weights.

```python
s = np.linalg.svd(F_aug, compute_uv=False)
print(s[0] / s[-1])  # condition number
```

Below ~1e6, the `rcond` choice is irrelevant. Above that, prefer `rcond=None` over
`rcond=0`.
