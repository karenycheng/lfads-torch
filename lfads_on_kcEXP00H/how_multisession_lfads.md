# How multi-session LFADS works — from single-session to multi-fly

## Data shape

Your data for one trial is a matrix: **75 regions × T timepoints**.

At any single moment in time `t`, you have one measurement per region — that snapshot is `x_t`, a vector of 75 numbers:

```
x_1 = [0.12, -0.03, 0.45, ..., 0.07]   ← all 75 regions at timepoint 1
x_2 = [0.14, -0.01, 0.48, ..., 0.09]   ← all 75 regions at timepoint 2
...
x_T = [0.09,  0.02, 0.31, ..., 0.05]   ← all 75 regions at timepoint T
```

---

## Single-session LFADS — step by step

```
ONE TRIAL  (75 regions × T timepoints)

Step 1 — Encoder (BiGRU)
  reads x_1, x_2, ..., x_T one at a time
  output: ic_mean (64,) and ic_std (64,)   ← one number per latent dim, not per timepoint

Step 2 — Sample
  g0 = ic_mean + noise × ic_std            ← one vector (64,) = "brain state at t=0"

Step 3 — Generator GRU
  starts from g0, steps forward T times
  at each step t: produces h_t (128,)      ← internal state, one per timepoint

Step 4 — Factor layer  (one linear layer, no activation)
  h_t (128,) → f_t (50,)                  ← compress to fewer dims, one per timepoint

Step 5 — Reconstruction  (one linear layer)
  f_t (50,) → x̂_t (75,)                  ← predicted value per region
```

Steps 3–5 repeat T times (once per timepoint). The encoder summarizes the full trial into `g0`; the generator then replays a smooth trajectory from that single seed without ever seeing the original data again.

### Why BiGRU?

A GRU reads a sequence one vector at a time and updates a running memory state. After seeing x_1…x_T, its final state summarizes the whole trial. **Bidirectional** means it does this twice — forward (x_1 → x_T) and backward (x_T → x_1) — then concatenates the two final states. This ensures the summary at any point reflects both past and future context.

### Why does the arrow stop at g0 (ic_mean, ic_std)?

This is the VAE bottleneck. The encoder outputs a **distribution** (mean and std), not a point. You then sample from it:

```
g0 = ic_mean + (random noise) × ic_std
```

The model is forced to commit to a single starting state `g0` and explain the whole trial from there. This is the "variational" part — it regularizes the latent space and prevents the model from trivially memorizing the data. The stochastic sampling is also why the encoder and generator are separate: you cannot backpropagate through randomness directly, so the reparameterization trick (`g0 = mean + noise × std`) makes the gradient flow through `mean` and `std` instead.

---

## The multi-session problem

Across 5 flies, valid region counts differ: fly 1 has 69 usable regions, fly 2 has 71, etc. The shared encoder expects a **fixed-size input**. You cannot feed it vectors of different lengths.

The fix: add one linear layer per fly **before** the encoder (the **readin**) to translate each fly into a common 50-dim PC space, and one linear layer per fly **after** the factor layer (the **readout**) to reconstruct each fly's specific neural activity.

---

## Multi-session LFADS — step by step

```
ONE TRIAL FROM FLY i  (n_fly_regions × T timepoints)

Step 0 — Readin  (per-fly, one linear layer)
  x_t (n_fly_regions,) → z_t (50,)
  z_t = W_readin_i @ (x_t - bias_i)
  weights: frozen after PCR init, never updated during training

↓ z_t is now 50-dim for every fly ↓

Step 1 — Encoder (BiGRU)           ← SHARED across all flies, same weights
  reads z_1, z_2, ..., z_T
  output: ic_mean (64,) and ic_std (64,)

Step 2 — Sample
  g0 = ic_mean + noise × ic_std

Step 3 — Generator GRU             ← SHARED across all flies
  g0 → h_1, h_2, ..., h_T    each h_t is (128,)

Step 4 — Factor layer              ← SHARED across all flies
  h_t (128,) → f_t (50,)

Step 5 — Readout  (per-fly, one linear layer)
  f_t (50,) → x̂_t (n_fly_regions,)
  x̂_t = W_readout_i @ f_t + b_i
  weights: trained during training
```

Steps 1–4 are identical to single-session. Steps 0 and 5 are the only additions.

---

## What is shared vs. per-fly

| Component | Shared or per-fly | Trained? |
|---|---|---|
| Readin layer | per-fly (one per fly) | No — frozen |
| Encoder BiGRU | shared | Yes |
| Generator GRU | shared | Yes |
| Factor layer | shared | Yes |
| Readout layer | per-fly (one per fly) | Yes |

---

## Why the readin is frozen and the readout is trained

**Readin is frozen** because PCR already found the geometrically correct transformation — the Ridge regression mapped each fly's regions into the shared PC space derived from the pooled population response. Letting it train would allow it to drift away from that alignment.

**Readout is trained** because reconstructing each fly's specific neural activity from shared factors requires learning fly-specific details (which regions respond strongly, baseline offsets, etc.) that PCR did not capture.

The readout is initialized as the pseudo-inverse of the readin (`pinv(W_readin)`), so at the first training step it approximately inverts the readin transformation. Training then refines it from there.

---

## What PCR does and why it matters here

PCR (Principal Components Regression) computes the readin weights in two steps:

**Step 7 — Global PCA**: Fit PCA on condition-averaged ΔF/F concatenated across all flies **along the region axis**. The matrix shape is `(n_conds × n_window, total_regions_across_flies)`. Each row is one (condition, timepoint) pair, with all flies' regions as columns — so the same condition occupies the same row for every fly. PCA finds a 50-dim space that captures the shared condition structure across all flies.

**Step 8 — Per-fly Ridge regression**: For each fly, fit a regression from that fly's mean-centred ΔF/F to the global PC scores. The regression weights (shape `n_fly_regions × 50`) become `W_readin_i`.

This pre-aligns all five readins so that "odour response" in fly 1 and "odour response" in fly 2 both project to the same region of the 50-dim space before a single gradient step is taken. Without this, the shared encoder would receive incoherent inputs from different flies and training would spend most of its time learning the alignment.

---

## Concretely, what "shared" means during training

On each training step, a batch contains trials from multiple flies. For a fly 1 trial and a fly 2 trial in the same batch:

```
fly1:  x_t (69) → [W_readin_1] → z_t (50) → encoder → g0 → generator → f_t (50) → [W_readout_1] → x̂_t (69)
fly2:  x_t (71) → [W_readin_2] → z_t (50) → encoder → g0 → generator → f_t (50) → [W_readout_2] → x̂_t (71)
```

Both trials pass through the **exact same encoder, generator, and factor layer weights**. Gradients from both trials accumulate into those shared weights simultaneously, forcing the model to learn representations that generalize across flies.

---

## Where these terms appear in the original literature

- **LFADS architecture** (Steps 1–5): Pandarinath et al. 2018, *Nature Methods*
- **Readin / readout layers and multi-session extension**: Keshtkaran et al. 2022, *Nature Methods*; Sedler & Pandarinath 2023, arXiv
- **Code implementation**: `lfads_torch/modules/readin_readout.py` — `MultisessionReadin` and `MultisessionReadout`
