# Evaluation notebook explained — `3_evaluation_KYC.ipynb`

A plain-language, cell-by-cell breakdown of the multisession evaluation notebook
(`tutorials/multisession/3_evaluation_KYC.ipynb`), which evaluates an LFADS model trained on the
Rouse reaching dataset. This is the template the kcEXP00H decoding notebook
(`4_decode_stimulus_kcEXP00H.ipynb`) adapts. See also `how_multisession_lfads.md` and
`what_to_do_with_lfads.md`.

---

## Cell 1 — Load the real experimental data

```python
DATA_PATHS = sorted(glob("AnimalP/P_Spikes_*-data_PSTH_prep_bin_20.mat"))
bin_width_sec = 0.02
```

Loads the ground-truth recordings for Animal P, one `.mat` file per session (12 days). Data is
binned in **20 ms** windows. For each session it extracts three things:

- `spikes[sess]` — neural data, reshaped to **(trials, time, neurons)**. Multiplied by
  `bin_width_sec` to convert spike *counts* to a per-bin scale. Each session has a different neuron
  count (87–133) because they are separate recordings — which is exactly why you need a
  multisession model.
- `conds[sess]` — the condition / reach-direction ID per trial (`- 1` shifts MATLAB's 1-based
  labels to 0-based).
- `velocity[sess]` — hand/joystick velocity, `np.gradient` of position over time. This is the
  behavior to decode.

**Drop bad trials:** any trial containing a `NaN` is removed across all three dicts, keeping them
aligned.

**Smoothing:** builds a Gaussian kernel (`std = 1 bin = 20 ms`) and convolves the spikes
(`lfilter`); `np.roll` re-centers it so the smoothing is non-causal (symmetric, not lagged).
`smth_spikes` is the **classical baseline** — the traditional way to estimate firing rates without
LFADS. The first `len(window)` timepoints are set to `NaN` (filter warm-up).

Output dict like `'20170630': (1575, 30, 113)` = 1575 trials, 30 time-bins, 113 neurons.

## Cells 3–7 — Exploratory / sanity checks

Scratch cells: print the LFADS output paths, grab one path, extract the session date via regex,
and list the keys inside one h5 file. Not part of the pipeline. (Cell 7 has a stray leading-space
indentation that would raise `IndentationError` if run — harmless, it is just an inspection cell.)

## Cell 8 — Load LFADS outputs

For each of the 12 LFADS output h5 files it extracts:

- `factors[session]` — the **LFADS latent factors**, shape (trials, 30, **50**). The 50-D shared
  latent dynamics the model inferred.
- `rates[session]` — `output_params` (inferred firing rates), divided by `bin_width_sec` to convert
  back to **Hz**.

The `merge_train_valid` helper is needed because LFADS stores train and valid trials separately,
each with an index array recording its original trial positions. The function allocates a full
`(n_train + n_valid, ...)` array and scatters each subset back into its original trial order — so
`factors`/`rates` line up trial-for-trial with `spikes`/`conds`/`velocity` from cell 1.

> **Bug fixed in this notebook:** `train_inds` / `valid_inds` are stored as **float32**, and numpy
> can only fancy-index with integer/boolean arrays. They must be read with `[()]` and cast to
> `int` (`f["train_inds"][()].astype(int)`), otherwise you get
> `IndexError: only integers, slices ... are valid indices`.

## Cell 9 — Visual comparison

For one trial of one session, plots four heatmaps side by side: **raw spikes → smoothed spikes →
LFADS rates → LFADS factors**. The point is to eyeball the denoising: raw spikes are sparse/binary,
smoothed spikes are blurry, LFADS rates are clean structured estimates, and the 50 factors are the
compact latent representation driving those rates.

## Cell 11 — PSTH comparison

`plot_psths` averages activity across trials *within each condition* (color = reach direction,
`conds % 8` collapses to 8 directions) for 20 neurons. It draws this twice: once for **smoothed
spikes** and once for **LFADS rates**. If LFADS is working, its rate PSTHs reproduce the
condition-dependent structure of the smoothed-spike PSTHs — but cleaner, because LFADS pools
statistical strength across the population.

## Cell 12 — Mask invalid timepoints and align dimensions

```python
mask = ~np.isnan(smth_spikes[s]).any((0, 2))
```

The cell-1 smoothing left `NaN`s at the start of each trial. This finds the timepoints valid across
*all* trials and neurons, then trims `smth_spikes`, `factors` (→ `factors_masked`), and `velocity`
to just those bins (30 → 24). This guarantees the decoders get clean, equal-length inputs.

## Cell 14 — Decode velocity (the core evaluation)

A `Ridge` linear regression. For each session it builds:

- **X (features)** = `factors_masked[sess]` → `(trials, time, 50)` flattened to `(trials*time, 50)`.
  Every timepoint of every trial is one sample, described by 50 factor values.
- **y (target)** = `velocity[sess]` → `(trials*time, 2)` (x- and y-velocity).

Two mechanics:

- **Lag** (`n_lag = 5`): neural activity is shifted ~100 ms *earlier* than the behavior it predicts
  (`data[:, :-5]` vs `vel[:, 5:]`), because neural activity precedes movement.
- **Grouping**: `GroupKFold` with one group per trial, so cross-validation never splits a single
  trial's timepoints across train/test (avoids leakage).

It evaluates two regimes:

- **within-day** — train and test on the same session (upper bound on decodability).
- **across-day** — train on the other 11 sessions, test on the held-out one. Only possible with
  LFADS factors (smoothed spikes can't, since neuron identity differs across sessions).

## Cells 16, 18 — Results and state-space plots

Cell 16 bar-plots decoding R² by session/data-type. Cell 18 fits a single PCA across all sessions'
factors and plots per-session 3D factor trajectories colored by condition — if the factors
generalized, all sessions look alike.

Typical numbers: LFADS factors ≈ **0.85** within-day, ≈ **0.83** across-day; smoothed spikes ≈
**0.64** within-day (and across-day is impossible).

---

## Why use factors if they "just look like the PSTHs"?

The PSTH match is a **sanity check, not the deliverable.**

A PSTH is a **trial average** — you pool many trials of one condition and average. Averaging is what
makes it clean. So of course smoothed spikes and LFADS rates agree at the PSTH level: you've washed
out the trial-to-trial noise. That plot only confirms LFADS didn't break the condition structure.
The value of LFADS lives in what the PSTH throws away: **single trials.**

1. **Clean single-trial estimates.** One trial of smoothed spikes is a blurry mess (a handful of
   spikes); one trial of LFADS rate is a clean trajectory, because LFADS denoises per trial by
   borrowing strength across the whole population and the learned dynamics. Gaussian smoothing only
   sees one neuron's own history.
2. **A shared, fixed-dimensional latent space.** Sessions have 87–133 *different* neurons with no
   correspondence; LFADS maps every session into the **same 50-D factor space**. That is what makes
   across-day decoding possible at all.
3. **Trial-to-trial variability is the science, not noise.** Why one trial was faster or corrective
   is a *deviation from the PSTH* — which averaging defines out of existence. Factors keep it, which
   is why they decode behavior far better than smoothed spikes.

The decoding table is the proof: if factors were "just PSTHs," within- and across-day numbers would
be identical and across-day would be impossible. They're not.

**PSTH agreement = the model is faithful. Single-trial decoding = the model is useful.** The first
is necessary; the second is why you ran LFADS.

---

## How this maps to kcEXP00H

`4_decode_stimulus_kcEXP00H.ipynb` reuses this exact structure with three changes:

| | Rouse (this notebook) | kcEXP00H decode notebook |
|---|---|---|
| X | factors `(trials, time, 50)` | factors `(trials, time, 50)` |
| y | velocity (continuous) | odor / walk / vision (**binary 0/1**, per timepoint) |
| model | `Ridge` (regression) | `LogisticRegression` (classification) |
| metric | R² | **ROC-AUC** (classes imbalanced) |
| lag | neural *leads* behavior | sensory neural *lags* stimulus → lag 0 |

The within-day / across-day split becomes **within-fly / across-fly**, testing whether LFADS found
a shared, fly-invariant latent code for each stimulus.
