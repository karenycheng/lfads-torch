# Session Log — kcEXP00H LFADS data preparation

## Session: 2026-06-25

**Goal:** Adapt the lfads-torch multisession PCR tutorial (`tutorials/multisession/1_data_prep.ipynb`) for kcEXP00H LFM whole-brain calcium imaging data and prepare HDF5 files for training.

---

### Files created

| File | Description |
|---|---|
| `1_data_prep_kcEXP00H.ipynb` | Full pipeline: load → NaN handling → window → psths → PCA → PCR → HDF5 |
| `1-1_data_prep_w_zscore_kcEXP00H.ipynb` | Same pipeline with per-region z-score normalisation before PCA (recommended) |
| `what_to_do_with_lfads.md` | Reference: LFADS outputs table, 7 analysis use cases, caveats, reconstruction type |

---

### Key decisions and findings

**Data format**
- ΔF/F is already continuous — no Gaussian smoothing needed (unlike spike-based LFADS)
- Condition-mean ΔF/F directly replaces PSTHs
- Use `Gaussian` (not `Poisson`) reconstruction loss

**NaN handling (Step 3)**
- Each fly has 75 brain regions; 4–6 are always NaN (dead channels) → dropped
- Remaining ~69–71 regions have 2 entirely-NaN trials per region (recording sessions missing from those trials) → filled by trial-mean imputation per region
- No trials dropped; remaining NaNs = 0 after imputation

**Imaging rate and window (Step 4)**
- Imaging rate is **30 Hz** 
- Window: `PRE_FRAMES=25`, `POST_FRAMES=150` → 175 frames total (~0.83 s pre / ~5 s post odour onset)

**Global PCA (Step 7)**
- With 8 conditions: **2 PCs explain 90% of variance** (unusually low — reflects limited condition diversity)
- n_components set to 50 for PCR; only first 2 are meaningful

**Per-fly PC trajectories (Step 8 visualization)**
- Two flies (H34006-007, H37003-004) showed compact trajectories relative to the other 3
- Cause: weaker overall ΔF/F amplitude → those flies' variance underrepresented in global PCA

**Z-scoring (Step 5.5 — implemented in `1-1_data_prep_w_zscore_kcEXP00H.ipynb`)**
- Z-score per region per fly, using mean/std computed from the flattened `(n_conds × n_window)` condition-mean matrix
- Same parameters applied to raw trial traces before HDF5 saving
- `zscore_mean` and `zscore_std` stored in each HDF5 file for inverse transform post-LFADS
- Inverse: `output_params * zscore_std + zscore_mean` recovers original ΔF/F units

**HDF5 output**
- Unscaled: `../../datasets/kcEXP00H_multisession/`
- Z-scored: `../../datasets/kcEXP00H_multisession_zscored/`
- 14 train / 4 valid trials per fly (every 5th trial → validation)

---

### Annotated tutorial

`tutorials/multisession/1_data_prep_KYC.ipynb` — the original spike-based tutorial annotated with 16 markdown cells explaining each step.

---

## Session: 2026-06-26

**Goal:** Run multisession autoLFADS (PBT) on the AnimalP tutorial data; document repo architecture.

### Files created

| File | Description |
|---|---|
| `how_multisession_lfads.md` | Architecture walkthrough (readin/readout, PCR, shared vs. per-fly) + autoLFADS vs. RADICaL section appended |
| `2_region_contributions_kcEXP00H.ipynb` | Per-region contribution analysis notebook (readout weights, factor–region correlation, variance decomposition) |

### Training run: AnimalP / rouse_multisession

- HDF5 files confirmed in `datasets/rouse_multisession/` (12 sessions from AnimalP `.mat` files)
- Run script: `tutorials/multisession/2_run_pbt.py`
- `RUN_DIR` set to `runs/rouse_multisession_20260626/pbt/rouse_multisession/<date>`
- 20 PBT trials, `resources_per_trial=dict(cpu=3, gpu=0.5)`

### Error encountered: wandb tag validation failure

All 20 trials failed immediately:
```
pydantic.error_wrappers.ValidationError: 2 validation errors for Settings
run_tags -> 1  none is not an allowed value
run_tags -> 2  none is not an allowed value
```

**Cause:** `configs/pbt.yaml` `wandb_logger` block has `null` tag placeholders for `DATASET_STR` and `RUN_TAG`. These are meant to be overwritten by `mandatory_overrides` in the run script, but those three lines had been commented out → wandb received `tags: [pbt, None, None]` and its pydantic validator rejected `None`.

**Fix 2026-06-26 16:00:** Removed the `wandb_logger` block from `configs/pbt.yaml` entirely; kept the three `logger.wandb_logger.*` lines commented out in `tutorials/multisession/2_run_pbt.py`. Training now uses only `csv_logger` and `tensorboard_logger`. To re-enable wandb later: restore the block in `pbt.yaml` and uncomment those three lines in the run script.

---

## Session: 2026-06-27

**Goal:** Add the kcEXP00H training script + configs, a stimulus-decoding notebook, and an evaluation breakdown doc; renumber the region-contributions notebook.

### Files created

| File | Description |
|---|---|
| `configs/datamodule/kcEXP00H_multisession.yaml` | Datamodule pointing at `datasets/kcEXP00H_multisession/lfads_*.h5`, `batch_size: 200` |
| `configs/model/kcEXP00H_multisession_PCR.yaml` | Clone of `rouse_multisession_PCR.yaml` with `encod_seq_len: 175` and **MSE** reconstruction |
| `lfads_on_kcEXP00H/2_run_pbt_kcEXP00H.py` | PBT training script for kcEXP00H (fresh timestamped `RUN_DIR`, no `resume=True`) |
| `lfads_on_kcEXP00H/4_decode_stimulus_kcEXP00H.ipynb` | Binary, time-resolved decoding of odor/walk/vision from factors, within- and across-fly |
| `lfads_on_kcEXP00H/3_evaluation_explained.md` | Cell-by-cell breakdown of `tutorials/multisession/3_evaluation_KYC.ipynb` + "why factors vs PSTHs" |

### Renamed

- `2_region_contributions_kcEXP00H.ipynb` → `3_region_contributions_kcEXP00H.ipynb` (content unchanged). Folder order is now `1_`/`1-1_` prep → `2_run_pbt_kcEXP00H.py` → `3_region_contributions` → `4_decode_stimulus`.

### Key decisions and findings

**Reconstruction correction — use MSE, not Gaussian.** The earlier note (Session 2026-06-25) said
"use Gaussian (not Poisson)." With the PCR-initialized readout this is wrong: `pcr_init` builds the
readout from `pinv(readin_weight)`, giving **75 outputs = mean only**. `Gaussian` has `n_params=2`
(needs 150 outputs: mean + logvar) and is incompatible with that init. `MSE` has `n_params=1` and
matches exactly (`lfads_torch/modules/recons.py`). The model config therefore uses
`recons.MSE`. (To use Gaussian you'd need a non-PCR readout sized for 2 params.)

**Config dims.** With `pcr_init=True`, readin/readout shapes come entirely from `readin_weight` in
the h5 files (`readin_readout.py`), so `encod_data_dim` and `fac_dim` stay **50** (the shared PC
space = n_components), exactly like Rouse. Only `encod_seq_len` changes to **175** (= PRE 25 +
POST 150).

**`config_path` resolution.** `hydra.initialize` in `run_model.py` resolves `config_path` relative
to `lfads_torch/run_model.py`, not the calling script — so `2_run_pbt_kcEXP00H.py` keeps
`config_path="../configs/pbt.yaml"` despite living in `lfads_on_kcEXP00H/`, and the new configs go
in the top-level `configs/`.

**Decode targets.** Stimulus regressors come from the xarray coords `odor_state_frame` /
`fw_state_frame` / `vis_state_frame` (per-timepoint 0/1) gated by `*_trial_bool` (per-trial
presence), windowed on the same odor onset as the factors. Data prep keeps all 18 trials in
original order, so targets align with merged factors 1:1. Experiment-start baseline trials
(`is_baseline==True`, trials 1 & 10) are dropped via `EXCLUDE_BASELINE=True`; randomized `xxx`
trials are kept as all-off negatives. Decoder = `LogisticRegression(class_weight="balanced")`,
metric = ROC-AUC.

**Resume deadlock avoided.** `2_run_pbt_kcEXP00H.py` deliberately uses a fresh timestamped
`RUN_DIR` each launch and does **not** pass `resume=True` — resuming an all-PAUSED run previously
caused a multi-hour deadlock (no RUNNING trial ever unpauses the others).

**Train on z-scored data.** `2_run_pbt_kcEXP00H.py` uses `DATASET_STR =
"kcEXP00H_multisession_zscored"` (the recommended per-region z-scored ΔF/F). The model name is
decoupled (`MODEL_STR = "kcEXP00H_multisession_PCR"`): the model config is dataset-agnostic because
its readin/readout/reconstruction read `${datamodule.datafile_pattern}`, so one model config serves
both the scaled and unscaled datamodules. New config:
`configs/datamodule/kcEXP00H_multisession_zscored.yaml`. MSE recon suits z-scored (zero-mean,
unit-variance) data especially well.

### Pending

- `datasets/kcEXP00H_multisession_zscored/lfads_*.h5` not yet generated — run
  `1-1_data_prep_w_zscore_kcEXP00H.ipynb` (the z-scored prep, not the plain `1_`) before training.
- After training, set `H5_DIR` in `4_decode_stimulus_kcEXP00H.ipynb` to the run's `best_model` dir.

---

## Session: 2026-06-29

**Goal:** Train multisession autoLFADS on the z-scored kcEXP00H data and stand up the evaluation notebook.

### Training run

- Run dir: `runs/kcEXP00H_multisession_202606291751/pbt/kcEXP00H_multisession_zscored/260629/best_model`
- Produced `lfads_output_lfads_*.h5` for all 5 flies (`run_posterior_sampling` at end of `2_run_pbt_kcEXP00H.py`).
- **Whole-trial training (change from 2026-06-27):** factors/recon span the **full 1050-frame trial**
  (`factors` are `(18, 1050, 50)`), not the 175-frame odor-onset window assumed earlier. Downstream
  windowing in the eval/decode notebooks was updated to match (see 2026-06-30).
- Per-fly region counts in the h5 are **69–71** (each fly keeps only its non-dead regions of the
  shared 75), and each h5 stores `readin_weight (n_reg,50)` + `readout_bias (n_reg,)`.

### Files created

| File | Description |
|---|---|
| `lfads_on_kcEXP00H/3_evaluate_kcEXP00H.ipynb` | Evaluation: reconstruction R², single-trial denoising, condition PSTHs, latent state-space |

---

## Session: 2026-06-30

**Goal:** Evaluate the trained model end-to-end — reconstruction quality, latent geometry, stimulus
decoding, and per-region contributions — and make the eval/decode/region notebooks run.

### Infrastructure / environment

- `utils.Xarray_UtilFns` (provides `load_group_xarrays`) was relocated to the **lfads-torch repo
  root** (`utils/`). Notebooks run from `lfads_on_kcEXP00H/`, so the import path is
  `sys.path.insert(0, '..')` (was a stale absolute path into `kcEXP00H_PythonNotebooks`).
- Installed `nbconvert` into the `lfads-torch` conda env to execute notebooks in place.

### `3_evaluate_kcEXP00H.ipynb` — reconstruction & latent geometry

- **Reconstruction R²** (z-scored ΔF/F): mean **train 0.90 / valid 0.70** → good fit, mild
  (expected) overfit given 14 train / 4 valid trials. Weakest fly H37003-004 (valid 0.54).
- **Latent state-space:** pooled PCA over factors — **PC1 = 53%** of variance and is a
  stimulus-locked ramp **shared by every condition** (grand-mean trajectory PC1 range ≫ PC2/PC3).
- Added cells **4b/4c** to answer "the 3D trajectories overlap → are conditions indistinct?":
  cross-fly condition-mean trajectories in PC1-3 vs PC2-4, plus a between/within-condition variance
  ratio (~1.0–1.5). Conclusion: the overlap is the shared PC1 ramp + ~2 trials/condition noise, **not**
  collapsed conditions.

### `4_decode_stimulus_kcEXP00H.ipynb` — stimulus decoding

- **Whole-trial windowing fix:** the old notebook cropped targets to a 175-frame window while
  factors are 1050 frames → silent flatten-and-decode misalignment. Targets now built over the
  **full 1050-frame axis** so they align frame-for-frame with the factors.
- Time-resolved cell reworked for the whole trial (strided per-frame decoders, per-fly onset marker).
- Added a **windowed-control cell**: re-decodes on a balanced ±[25,150] onset window (cropping both
  factors and targets) to guard against the PC1 trial-phase confound.
- **Results** (mean over 5 flies, ROC-AUC, whole-trial / onset-windowed):
  - walk: within 0.74 / 0.83, across 0.73 / 0.78 — **strongest, clearly real**
  - odor: within 0.52 / 0.63, across 0.63 / 0.66 — modest but real
  - vision: within 0.43 / 0.62, across 0.63 / 0.54 — weak, least consistent across flies
  - Windowed ≥ whole-trial for walk/odor → genuine stimulus encoding, **not** the trial-phase ramp.
  - Across-fly ≈ / > within-fly → **shared cross-fly latent code** (multisession payoff); within-fly
    is undertrained (~9 trials).

### Renamed: `3_region_contributions` → `5_region_contributions_kcEXP00H.ipynb`

Forward-references in `3_evaluate` and `4_decode` updated; historical rename entries left intact.

### `5_region_contributions_kcEXP00H.ipynb` — region anatomy

- **Correctness fix:** notebook assumed 75 regions but each fly's h5 has **69–71**; every figure was
  mislabeled. Now recomputes each fly's keep-mask (`~np.isnan(deltaF_rg).all(axis=(0,1))`), maps h5
  regions → canonical names, and **lifts all per-fly quantities onto the canonical 75-region frame**
  (absent regions grey, ignored in cross-fly nanmeans). Readin now read from the h5 (no checkpoint);
  readout recovered by OLS.
- Added **Fig 8:** rotate the latents to the **odor/walk/vision decoder directions** (from
  `4_decode`) and project into region space via `loading = W_fly @ w_mod`, averaged across flies.
- **Results (anatomically sensible):** vision → optic lobe (**AOTU, MED, LO, BU**); walk → central
  complex (**EB, NO, BU**); odor → mushroom body (**MB.PED**). Cross-fly consistency (Fig 7) flags
  LO-R, PB, AOTU-R, BU, MB.VL. Caveat: **PRW** loads on every axis (outlier — discount).
- Added **Fig 9 — multimodal regions:** thresholds each axis's |loading| at its top tertile and
  counts how many axes each region contributes to (24 unimodal / 21 bimodal / 3 trimodal). Trimodal
  hubs = **GA-R, CRE-R, MB.PED-R** (crepine/gall/MB-peduncle — integrative neuropils). Prints
  axis cosine + region-loading correlations so overlaps can be judged: **odor–walk independent
  (r=0.06, trustworthy)**, odor–vision correlated (r=0.49, partly trivial + PRW outlier).
- Added **Fig 10 — per-fly reliability:** instead of averaging `W_f @ w_mod` over flies, keeps the
  per-fly values — strip plots + across-fly mean, labelled with **sign-consistency / #flies present**,
  and **axis reproducibility** = cosine of each fly's own decoder axis vs the pooled axis
  (odor 0.36, walk 0.49, vision 0.39 ± big spread → the shared axis is a population-average direction,
  a consequence of 18 trials/fly). Reliable encodings (all 5 flies, 100% sign-consistent):
  **AOTU-R→vision, CRE-R→vision, SLP-R→vision, GA-R→odor**. Exposes that headline region **MED-L**
  is present in only 1 fly and **PRW** in 4 with ~random sign → discount both. Definitions: an "axis"
  is the unit-normalised pooled logistic-regression weight vector in 50-D factor space separating
  stimulus on/off; region loading = `W_fly @ w_mod`.

### Overall narrative

Good reconstruction (nb3) → walk/odor/vision are decodable and shared across flies (nb4) → and map
onto the expected brain systems (nb5). Main caveats: 18 trials/fly, the PC1 ramp, and the PRW outlier.

### `5_region_contributions` — figure-review additions (later on 2026-06-30)

- **Artifact exclusion.** Added `ARTIFACT_BASES = ['MED','AME','LO','LOP']` (optic-lobe neuropils
  flagged as likely stimulus-light/high-variance artifacts) → `ARTIFACT_REGIONS`/`ARTIFACT_MASK`.
  These are excluded from the region **rankings/hubs** in Figs 2, 7, 8, 9, 10 (shown grey in the
  heatmaps, not dropped from the model). Effect: Fig 2 factor-0 top regions become MB/SMP instead of
  optic lobe; Fig 7 loses LO-R; Fig 8 walk loses AME-L (→ PVLP-R/ATL-R/EB); trimodal hubs
  (GA-R, CRE-R, MB.PED-R) unchanged. Set `ARTIFACT_BASES = []` to restore. **PRW** is a separate
  outlier (present in only 4 flies, ~random sign) not yet excluded.
- **Fig 3b — factor global-ness:** mean |r| vs peak |r| across regions per factor. Most factors are
  global (top ~25 have mean |r| > 0.6; regions are globally coupled, ~2 PCs = 90% var); only ~5 are
  selective (factors 5, 36, 18, 15, 1). Global mode = factor 0 ≈ PC1 (the shared stimulus-locked ramp).
- **Fig 3c — global-signal regression:** regress the brain-wide mean signal out of both factors and
  regions before correlating; the broad common-mode stripes collapse, exposing region-specific
  structure. Decoder-axis figures (8-10) are less affected because stimulus axes are ~orthogonal to
  the global ramp.
- **Factors are strongly collinear — Fig 4 rewritten.** Measured: mean |off-diag factor corr| ≈ 0.49,
  **participation ratio ≈ 3.7 / 50** (the "50-D" factor space is effectively ~4-D; top mode = 45%,
  top-3 = 75% of factor variance). Naive per-factor variance split mis-estimates region variance by
  ~70% (cross-terms dominate). Fig 4 was therefore replaced: instead of `W²·Var(f)` per factor, it now
  decomposes each region's reconstructed variance over the **orthogonal eigen-modes** of each fly's
  factor covariance — `Var(x̂ᵢ)=Σₖ(wᵢ·vₖ)²λₖ`, exact. Result: mode 0 (global) dominates nearly every
  region; region-specific structure is confined to m1-m3; m4+ negligible. Per-fly eff. dim 3.2-3.8.
- **Fig 4.1 — latent trajectories of the orthogonal modes** (added): project factors onto the pooled
  factor-covariance eigenmodes and plot (a) condition-averaged mode time courses, (b) 3D condition-mean
  trajectories (modes 0-1-2 and 1-2-3). Mode 0 (45% var) = brain-wide **odor-locked** response (sharp
  dip at frame ~450 = odor onset, deepest for odor-ON conditions; odor-OFF stay near origin); modes
  1-3 carry condition-discriminating structure (steps at stimulus onsets ~frame 150 & 450).
  Eigenvector signs are arbitrary, so Fig 4.1 fixes a convention: each mode oriented so its dominant
  **non-artifact** region loads positive (avoids anchoring to MED/LOP/PRW, which dominate modes 1-3's
  raw readout — i.e. sub-global variance is heavily artifact/outlier-driven), plus a `MODE_FLIP` manual
  override (currently `[1]`) to match expected ΔF/F polarity. After fixing, modes 0 & 1 both show
  positive odor-evoked peaks at onset.

- **Fig 5 fixed to be a real failsafe.** As written it computed R²(factors → output_params) ≈ 1.0 by
  construction (readout is linear → tautology). Changed to R²(`output_params` vs `recon_data`, the true
  z-scored ΔF/F): per-fly mean 0.85-0.89, matches nb3's 0.87; regions below 0.8 (AOTU-L, LO-L, AME,
  MED-R, EB, LOP-R) are the poorly-reconstructed ones. Required loading `recon_data` into `fly_data`.
- **Fig 6 gained a deviation-from-init row.** `corr(readout, readin) ≈ 0.10` but
  `corr(readout, pinv(readin)) ≈ 0.86-0.93` — readout inits at pinv(readin), so raw Out−In is NOT ~0.
  Added a 3rd row = `readout − pinv(readin)` (deviation from PCR init): pale (~0) for factors ~20-49,
  structured in the first ~20 (high-variance) factors; mean|Δ| 0.15-0.19. Training refined the readout
  along the dominant modes and left low-variance factors near init.

- **Fig 7.1 — region contribution per stimulus condition** (added, in response to wanting the 8-way
  condition breakdown rather than Fig 7's condition-agnostic |W|). Per-region **evoked response** of
  the model reconstruction (mean over response window [150,750) minus baseline [0,140)), per condition,
  cross-fly; plus a condition-specific panel (− mean across conditions). Findings: overall drive scales
  with salience (xOF/VOF strong, Vxx/xxx ≈ 0). Modality anatomy is clean: **xOx (odor) → whole mushroom
  body (MB.CA/ML/VL/PED)**; walk conditions → FLA/VES/CAN/PRW (motor/mechanosensory); Vxx (vision) weak
  (AOTU/IPS); xxx (off) ≈ baseline. Confirms Fig 8 decoder-axis anatomy from raw condition means.
  Caveat: conditions differ mostly in shared-response *magnitude*; region reallocation is second-order.
- **Fig 7.2 — region contributions on the JFRC brain volume** (added; Python port of the MATLAB
  `kcRIDGE_projectBetasToVolume`). Loads `JFRCMask_separate_75regions.nii` (1024×512×44×75, 6.9 GB) via
  nibabel (installed into lfads-torch env), builds a voxel→region lookup (cached ds×2 to
  `/home/kyc_hpz8/Documents/regions-analysis/voxelRegion_ds2_75.npy`), paints each region's evoked
  scalar (Fig 7.1 `C`) into its voxels, and max-|·| projects dorsal/lateral/coronal (black bg, RdBu).
  7.2 = odor/walk/vision (xOx/xxF/Vxx) × 3 views; 7.2b = all 8 conditions dorsal. Anatomy validates the
  region alignment: **vision → optic lobes**, odor → central brain. Not ported: time-evolving lag
  snapshots, per-region vector patches (Illustrator), PDF export. Assumes mask 4th-dim order ==
  REGION_NAMES.
- **Fig 7.2 → dorsal-only, rotated 90° left** (proper fly-brain landscape orientation; lateral/coronal
  dropped per request). **Fig 7.3 — time-evolving dorsal projection** (added): condition-mean response
  at snapshot frames [100,250,400,550,700,900] (±0.5 s windows, baseline-subtracted), for odor/walk/
  vision, per-row colour scale. Reveals modality **latency differences** (walk peaks ~8 s, odor ~18 s)
  and a **lateralized vision response** (left optic lobe) that the static evoked window averaged out.
- **Clarified: Fig 7 (cross-fly importance) plots WEIGHTS** (mean |W| across factors, condition-agnostic),
  vs Fig 7.1-7.3/8.1 which plot **responses** (evoked ΔF/F). Conditions come only from the per-trial
  `condition` coord in the .nc; the readout weights have no condition axis. **Fig 7.4** (added) projects
  the weight-importance onto the brain (per fly + cross-fly mean, hot cmap) — fairly uniform, confirming
  it carries little spatial/stimulus structure (the meaningful anatomy is in the response figures).

- **Fig 8 flagged as decoding-geometry, not anatomy.** The decoder-axis region loadings `W_fly @ w_mod`
  correlate ~0 with the actual evoked responses (odor +0.04, walk −0.01, vision −0.18) and are dominated
  by the #1 readout-weight region PRW — so the odor top regions (PRW/GA/NO) are meaningless. Correct
  anatomy = Fig 7.1 (evoked). Caveat added to Fig 8 markdown.
- **Fig 8.1 — modality interactions** (added; the encoding/interaction question the user actually wanted).
  Treats the per-region evoked responses as a 2×2×2 factorial (V,O,F): (a) additivity scatters (effect
  of B with vs without A) — generalises `xOF−xOx` vs `xxF−xxx`; (b) factorial contrast heatmap (main +
  interaction terms). Findings: **odor main effect → MB.CA/AL (correct olfactory anatomy)**; **walk×odor
  is superadditive** (points above diagonal, r≈0.96; top regions IPS-L, AMMC-L, SAD, CAN-L, VES-L =
  mechanosensory/motor); vision combines additively (odor×vision r=0.82, walk×vision r=0.99). O×F
  interaction (0.13) ≈ vision main effect. **Figs 9-10 still built on the unreliable decoder loading —
  pending reframe onto the factorial/response basis.**

### Notebook execution

All three notebooks (`3_evaluate`, `4_decode`, `5_region_contributions`) were run end-to-end with
the `lfads-torch` kernel and have embedded outputs.

---

## Session: 2026-07-01 — figure-by-figure walkthrough of `5_region_contributions` (compact recap)

Reviewed every figure with the user and rebuilt the notebook to be correct and interpretable. Key
structural fixes: per-fly region counts are **69-71, not 75** (lift each fly's weights to the canonical
75-region frame, absent = grey); **factors are effectively ~4-D** (participation ratio 3.7, mean |corr|
0.49); **optic-lobe regions MED/AME/LO/LOP flagged as artifacts** (`ARTIFACT_BASES`, excluded from all
rankings); **PRW** is a separate outlier (#1 readout weight, present in 4/5 flies).

Figure inventory (all executed, outputs embedded):

| Fig | Plots | Note |
|---|---|---|
| 1 | Readout weight heatmap (regions×factors) | factor 0 = global mode (PCR/PC1) |
| 2 | Top regions per factor | artifacts excluded → MB/SMP not optic lobe |
| 3 / 3b / 3c | Factor–region corr / global-ness / after global-signal regression | most factors are global; GSR exposes region-specific structure |
| 4 | Region variance over **orthogonal** factor modes | replaced invalid per-factor pie (70% of var was cross-terms); exact |
| 4.1 | Latent trajectories of orthogonal modes | mode 0 = odor-locked; sign convention + `MODE_FLIP` |
| 5 | Per-region R² (model vs **recon_data**) | fixed from tautology (~1) → real fit, mean 0.85-0.89 |
| 6 | Readin vs readout + **readout − pinv(readin)** | corr(readout,readin)=0.1 vs pinv=0.87; training moved only top factors |
| 7 / 7.4 | Weight importance (|W|) heatmap / on brain | WEIGHTS, condition-agnostic, ~uniform (not very meaningful) |
| 7.1 / 7.2 / 7.3 | Per-condition **response**: table / brain volume / time-evolving | odor→MB, vision→optic lobe; walk peaks ~8 s, odor ~18 s; vision lateralized (L optic lobe) |
| 8 | Decoder-axis region loading | **flagged: decoding geometry, NOT anatomy** (corr~0 w/ response, PRW-dominated) |
| 8.1 | 2×2×2 factorial main effects + **interactions** | odor→MB/AL; **walk×odor superadditive** (IPS/AMMC/SAD/VES/CAN); vision additive |
| 9 / 10 | Multimodal hubs / per-fly reliability | **still on decoder loading — pending reframe onto factorial/response basis** |

Infra: installed `nbconvert` + `nibabel` in `lfads-torch` env; JFRC mask projection cached to
`/home/kyc_hpz8/Documents/regions-analysis/voxelRegion_ds2_75.npy`.

**Next:** rebuild Figs 9 & 10 on the factorial/response basis (Fig 9 = regions with ≥2 main effects or
interactions; Fig 10 = per-fly reliability of the O/F/O×F contrasts). Optional: vector/PDF export of
brain projections; split Fig 7.1 window into vision vs odor epochs.
