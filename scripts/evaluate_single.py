#!/usr/bin/env python
"""Evaluate and visualize LFADS single-session run outputs.

Usage:
    python scripts/evaluate_single.py                        # auto-detect most recent run
    python scripts/evaluate_single.py path/to/lfads_output.h5
"""

import argparse
import sys
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_output_file():
    files = sorted(glob("runs/**/**/lfads_output*.h5", recursive=True))
    if not files:
        sys.exit(
            "No lfads_output*.h5 found under runs/. "
            "Run scripts/run_single.py first, or pass the path as an argument."
        )
    return files[-1]


def load_data(path):
    with h5py.File(path) as f:
        return {k: f[k][()] for k in f}


def cond_labels(cond_idx, n_trials):
    """Map ragged cond_idx (one array of trial indices per condition) to a
    flat per-trial condition label array."""
    labels = np.full(n_trials, -1, dtype=int)
    for c, idxs in enumerate(cond_idx):
        labels[idxs] = c
    return labels


# ---------------------------------------------------------------------------
# Figure 1 — example trial rasters
# ---------------------------------------------------------------------------

def fig1_example_trials(d, out_dir, n_trials=3, seed=0):
    print(
        "\n── Fig 1: Input spikes vs. LFADS inferred rates ──────────────────────────\n"
        "Each row is one trial.  The rates should look like a smoothed, denoised\n"
        "version of the spikes — sharp edges and noisy bins should give way to\n"
        "smooth bumps.  If the rates look identical to the spikes or completely\n"
        "flat, the model may be under- or over-regularised."
    )
    rng = np.random.default_rng(seed)
    encod = d["valid_encod_data"].astype(float)          # (N, T_enc, C_enc)
    rates = d["valid_output_params"].astype(float)       # (N, T_recon, C_all)
    n_obs = encod.shape[1]
    rates_obs = rates[:, :n_obs, :]                      # align to encoder window

    idx = rng.choice(len(encod), size=n_trials, replace=False)
    vmax = max(encod[idx].max(), rates_obs[idx].max())

    fig, axes = plt.subplots(n_trials, 2, figsize=(10, 2.5 * n_trials))
    for row, i in enumerate(idx):
        for col, (arr, title) in enumerate(
            [(encod[i], "Input spikes"), (rates_obs[i], "LFADS rates")]
        ):
            ax = axes[row, col]
            ax.imshow(arr.T, aspect="auto", vmin=0, vmax=vmax, origin="lower")
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.set_ylabel(f"Trial {i}\nNeuron" if col == 0 else "")
    for ax in axes[-1]:
        ax.set_xlabel("Time bin")
    fig.suptitle("Fig 1 — Input spikes vs. LFADS inferred rates", fontsize=11)
    plt.tight_layout()
    save = out_dir / "fig1_example_trials.png"
    fig.savefig(save, dpi=150)
    print(f"  Saved → {save}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — PSTH comparison
# ---------------------------------------------------------------------------

def psth_r2(gt, pred):
    """R² between ground-truth and predicted PSTHs (over all valid conditions)."""
    mask = ~np.isnan(pred).any(axis=(1, 2))
    g, p = gt[mask].ravel(), pred[mask].ravel()
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def fig2_psth(d, out_dir, n_neurons=20, n_conds=8):
    print(
        "\n── Fig 2: PSTH comparison ────────────────────────────────────────────────\n"
        "Each panel is one neuron for one condition, showing the condition-averaged\n"
        "firing rate over time.  Blue = empirical PSTH; Red = LFADS predicted.\n"
        "Good fit: curves overlap.  PSTH R² near 1 means the model captured the\n"
        "trial-averaged dynamics accurately."
    )
    psth_gt = d["psth"].astype(float)                     # (C, T_enc, N_all)
    rates   = d["valid_output_params"].astype(float)
    n_obs   = psth_gt.shape[1]
    rates_obs = rates[:, :n_obs, :]

    cond_idx = d["valid_cond_idx"]
    n_total_conds = len(cond_idx)
    pred_psth = np.full_like(psth_gt, np.nan)
    for c, idxs in enumerate(cond_idx):
        if len(idxs) > 0:
            pred_psth[c] = rates_obs[idxs].mean(axis=0)

    r2 = psth_r2(psth_gt, pred_psth)
    print(f"  PSTH R² = {r2:.4f}")

    sel_conds   = np.linspace(0, n_total_conds - 1, n_conds, dtype=int)
    sel_neurons = np.linspace(0, psth_gt.shape[2] - 1, n_neurons, dtype=int)
    t = np.arange(n_obs)

    fig, axes = plt.subplots(
        n_conds, n_neurons,
        figsize=(n_neurons * 1.4, n_conds * 1.4),
        squeeze=False,
    )
    for ri, c in enumerate(sel_conds):
        for ci, n in enumerate(sel_neurons):
            ax = axes[ri, ci]
            ax.plot(t, psth_gt[c, :, n],    color="steelblue", lw=0.9, label="GT")
            ax.plot(t, pred_psth[c, :, n],  color="tomato",    lw=0.9, label="LFADS")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"N{n}", fontsize=6)
        axes[ri, 0].set_ylabel(f"C{c}", fontsize=6, rotation=0, labelpad=18)

    # legend on one axes
    axes[0, -1].legend(fontsize=6, loc="upper right", framealpha=0.8)
    fig.suptitle(f"Fig 2 — PSTH comparison  (R² = {r2:.3f})", fontsize=11)
    plt.tight_layout()
    save = out_dir / "fig2_psth.png"
    fig.savefig(save, dpi=150)
    print(f"  Saved → {save}")
    plt.close(fig)
    return r2


# ---------------------------------------------------------------------------
# Figure 3 — velocity decoding
# ---------------------------------------------------------------------------

def _decode_velocity(x_tr, y_tr, x_va, y_va):
    """Ridge regression with 5-fold group CV on trials; returns R² per axis."""
    n_trials, n_time, n_feat = x_tr.shape
    # groups: each time-step within a trial belongs to the same group
    groups = np.repeat(np.arange(n_trials), n_time)
    x_tr_f = x_tr.reshape(-1, n_feat)
    y_tr_f = y_tr.reshape(-1, y_tr.shape[-1])
    x_va_f = x_va.reshape(-1, n_feat)
    y_va_f = y_va.reshape(-1, y_va.shape[-1])

    model = GridSearchCV(
        Ridge(),
        param_grid={"alpha": np.logspace(-1, 2, 4)},
        cv=GroupKFold(n_splits=5),
    )
    model.fit(x_tr_f, y_tr_f, groups=groups)
    y_pred = model.predict(x_va_f)
    # R² per axis
    ss_res = np.sum((y_va_f - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_va_f - y_va_f.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / ss_tot


def fig3_decoding(d, out_dir):
    print(
        "\n── Fig 3: Velocity decoding ──────────────────────────────────────────────\n"
        "A linear decoder (Ridge) predicts 2D hand velocity from neural features,\n"
        "trained on the training set and evaluated on the validation set.\n"
        "High R² = features contain movement-relevant information in a linearly\n"
        "accessible form.  LFADS factors R² >> raw spikes R² means the model\n"
        "amplified the signal.  Similar values may indicate over-regularisation."
    )
    n_obs = d["valid_encod_data"].shape[1]

    # Training features / targets (decode_mask selects designated trials)
    tr_mask = d["train_decode_mask"].squeeze().astype(bool)
    va_mask = d["valid_decode_mask"].squeeze().astype(bool)

    tr_fac  = d["train_factors"][tr_mask, :n_obs, :].astype(float)
    va_fac  = d["valid_factors"][va_mask, :n_obs, :].astype(float)
    tr_spk  = d["train_encod_data"][tr_mask].astype(float)
    va_spk  = d["valid_encod_data"][va_mask].astype(float)
    tr_beh  = d["train_behavior"][tr_mask, :n_obs, :].astype(float)
    va_beh  = d["valid_behavior"][va_mask, :n_obs, :].astype(float)

    r2_fac = _decode_velocity(tr_fac, tr_beh, va_fac, va_beh)
    r2_spk = _decode_velocity(tr_spk, tr_beh, va_spk, va_beh)

    axis_labels = ["x-velocity", "y-velocity"]
    x = np.arange(len(axis_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - width / 2, r2_fac, width, label="LFADS factors", color="tomato")
    ax.bar(x + width / 2, r2_spk, width, label="Raw spikes",    color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(axis_labels)
    ax.set_ylabel("Validation R²")
    ax.set_ylim(0, 1)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    fig.suptitle("Fig 3 — Velocity decoding (train → valid)", fontsize=11)
    plt.tight_layout()
    save = out_dir / "fig3_decoding.png"
    fig.savefig(save, dpi=150)
    print(f"  Factors  R² — x: {r2_fac[0]:.3f}, y: {r2_fac[1]:.3f}, mean: {r2_fac.mean():.3f}")
    print(f"  Spikes   R² — x: {r2_spk[0]:.3f}, y: {r2_spk[1]:.3f}, mean: {r2_spk.mean():.3f}")
    print(f"  Saved → {save}")
    plt.close(fig)
    return r2_fac, r2_spk


# ---------------------------------------------------------------------------
# Figure 4 — factor state-space trajectories
# ---------------------------------------------------------------------------

def fig4_statespace(d, out_dir, n_conds=8, n_individual=20, seed=0):
    print(
        "\n── Fig 4: Factor state-space trajectories ────────────────────────────────\n"
        "PCA projects 100-D latent factors to 3D.  Each coloured curve is the\n"
        "mean trajectory for one reach condition over time.  Well-trained models\n"
        "show smooth, condition-separated orbits ('rotational' dynamics) — different\n"
        "conditions trace distinct but similarly shaped paths."
    )
    # Combine train + valid for a richer PCA basis
    fac_tr = d["train_factors"].astype(float)
    fac_va = d["valid_factors"].astype(float)
    fac_all = np.concatenate([fac_tr, fac_va], axis=0)   # (N_all, T, F)
    n_tr = len(fac_tr)

    # Fit PCA on all data
    ss  = StandardScaler()
    pca = PCA(n_components=3)
    flat = fac_all.reshape(-1, fac_all.shape[-1])
    pca.fit(ss.fit_transform(flat))

    # We'll visualise validation trials only (labeled by valid_cond_idx)
    cond_idx = d["valid_cond_idx"]
    n_va = len(fac_va)
    labels_va = cond_labels(cond_idx, n_va)
    unique_conds = np.array([c for c in np.unique(labels_va) if c >= 0])
    sel_conds = unique_conds[np.linspace(0, len(unique_conds) - 1, n_conds, dtype=int)]

    # Project validation factors
    fac_va_lowd = pca.transform(ss.transform(fac_va.reshape(-1, fac_va.shape[-1])))
    fac_va_lowd = fac_va_lowd.reshape(n_va, -1, 3)      # (N_va, T, 3)
    var_exp = pca.explained_variance_ratio_[:3] * 100

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(seed)

    for ci, c in enumerate(sel_conds):
        color = cm.hsv(ci / n_conds)
        trial_idx = np.where(labels_va == c)[0]
        # individual trials (thin, transparent)
        sample_idx = rng.choice(trial_idx, size=min(n_individual, len(trial_idx)), replace=False)
        for t in sample_idx:
            ax.plot(*fac_va_lowd[t].T, color=color, alpha=0.15, linewidth=0.6)
        # condition mean
        mean_traj = fac_va_lowd[trial_idx].mean(axis=0)
        ax.plot(*mean_traj.T, color=color, linewidth=2.0, label=f"Cond {c}")
        ax.scatter(*mean_traj[0], color=color, s=30, zorder=5)  # start marker

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=8)
    ax.set_zlabel(f"PC3 ({var_exp[2]:.1f}%)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
    ax.view_init(elev=20, azim=40)
    ax.axis("off")
    fig.suptitle("Fig 4 — Factor state-space trajectories (PCA)", fontsize=11)
    plt.tight_layout()
    save = out_dir / "fig4_statespace.png"
    fig.savefig(save, dpi=150)
    print(f"  Saved → {save}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — IC posterior distribution
# ---------------------------------------------------------------------------

def fig5_ic_dist(d, out_dir):
    print(
        "\n── Fig 5: Initial condition (IC) posterior distribution ──────────────────\n"
        "LFADS encodes each trial's starting state into an IC.  PCA projects the\n"
        "IC posterior means to 2D.  Clear clustering by condition means the encoder\n"
        "distinguished trial types from their initial dynamics.  A blurry scatter\n"
        "may indicate low IC dimensionality or weak condition-specificity."
    )
    ic_tr = d["train_ic_mean"].astype(float)
    ic_va = d["valid_ic_mean"].astype(float)
    ic_all = np.concatenate([ic_tr, ic_va], axis=0)

    labels_tr = cond_labels(d["train_cond_idx"], len(ic_tr))
    labels_va = cond_labels(d["valid_cond_idx"],  len(ic_va))
    labels_all = np.concatenate([labels_tr, labels_va])

    ss  = StandardScaler()
    pca = PCA(n_components=2)
    ic_2d = pca.fit_transform(ss.fit_transform(ic_all))
    var_exp = pca.explained_variance_ratio_ * 100

    unique_labels = np.array([l for l in np.unique(labels_all) if l >= 0])
    n_unique = len(unique_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in unique_labels:
        mask = labels_all == c
        ax.scatter(
            ic_2d[mask, 0], ic_2d[mask, 1],
            color=cm.hsv(c / n_unique),
            s=4, alpha=0.4, linewidths=0,
        )
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    fig.suptitle("Fig 5 — IC posterior mean distribution (PCA)", fontsize=11)
    plt.tight_layout()
    save = out_dir / "fig5_ic_dist.png"
    fig.savefig(save, dpi=150)
    print(f"  Saved → {save}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "h5_path", nargs="?", default=None,
        help="Path to lfads_output*.h5 (default: auto-detect under runs/)",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path) if args.h5_path else Path(find_output_file())
    print(f"Loading {h5_path}")
    d = load_data(h5_path)
    out_dir = h5_path.parent

    fig1_example_trials(d, out_dir)
    r2_psth       = fig2_psth(d, out_dir)
    r2_fac, r2_spk = fig3_decoding(d, out_dir)
    fig4_statespace(d, out_dir)
    fig5_ic_dist(d, out_dir)

    print(
        "\n══ Summary ══════════════════════════════════════════════════════════════\n"
        f"  PSTH R²              : {r2_psth:.3f}  (1.0 = perfect condition-averaged fit)\n"
        f"  Decoding R² (factors): {r2_fac.mean():.3f}  (mean over x/y velocity)\n"
        f"  Decoding R² (spikes) : {r2_spk.mean():.3f}  (mean over x/y velocity)\n"
        f"  LFADS gain           : {r2_fac.mean() - r2_spk.mean():+.3f}  "
        f"({'factors better' if r2_fac.mean() > r2_spk.mean() else 'spikes better'})\n"
        "════════════════════════════════════════════════════════════════════════"
    )


if __name__ == "__main__":
    main()
