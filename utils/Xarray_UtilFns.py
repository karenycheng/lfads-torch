"""
Xarray_UtilFns.py
=================
Shared utility functions for kcEXP00H xarray-based analyses.

Sections
--------
  DATA LOADING        — load .nc xarray datasets, compute within-trial frame attrs
  COLOURS             — fly palette, condition/window colour dicts
  SIGNAL PROCESSING   — Butterworth lowpass filter, exponential decay fit
  METRICS: AUC        — trial epoch AUC, peak, tau (per fly → group)
  METRICS: SLOPE      — pre-odor onset slope (per fly → group)
  METRICS: DERIVATIVE — first-derivative max (per fly → group)
  METRICS: SUMMATION  — fixed-coeff summation R², Pearson r, normalised R²
  PLOT UTILS          — dark theme, A4 PDF fit
  PLOT: TIMESERIES    — per-fly overlay, grand-mean ± SEM, interactive explorer
  PLOT: OVERLAY       — condition grand-mean ± SEM, stimulus bars
  PLOT: HEATMAP       — per-fly ΔF heatmap, all trials concatenated
  PLOT: SCATTER-BOX   — group AUC / slope scatter + boxplot
  PLOT: SUMMATION     — scatter-box and violin for summation metrics, z-scored traces
  PLOT: REGION PANEL  — per-region N×3 grid (timeseries overlay | AUC | deriv max)
"""

import os
import colorsys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, leaves_list


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def add_within_trial_frame_starts(ds, dsfactor=400, frames_per_trial=1050):
    """
    Compute within-trial frame indices for every stimulus key and store in ds.attrs.

    Converts scan counts (DAQ clock) to imaging frames:
        frame = floor(scan / dsfactor)
        within_trial_frame = frame - (trial_index - 1) * frames_per_trial
    """
    scan_keys = {
        'odor': ['odor_scanstarts', 'odor_scanends'],
        'fw':   ['fw_scanstarts',   'fw_scanends'],
        'vis':  ['vis_scanstarts',  'vis_patt_scanon', 'vis_patt_scanoff'],
    }
    for stim, keys in scan_keys.items():
        trial_indices = ds.attrs[f'{stim}_trial_indices']
        for key in keys:
            frame_key  = key.replace('scan', 'frame')
            abs_frames = np.floor(ds.attrs[key] / dsfactor).astype(int)
            ds.attrs[frame_key] = abs_frames
            ds.attrs[f'{frame_key}_within_trial'] = (
                abs_frames - (trial_indices - 1) * frames_per_trial
            )
    return ds


def load_group_xarrays(xarray_dir, pattern='.nc', dsfactor=400,
                        frames_per_trial=1050, verbose=True):
    """
    Load every *pattern* file in xarray_dir and return a list of xr.Datasets.

    Each dataset gets:
    - within-trial frame-start attrs (via add_within_trial_frame_starts)
    - ds.attrs['fly_id'] set to the filename stem

    Returns
    -------
    ds_list : list[xr.Dataset]
    """
    files = sorted(
        os.path.join(xarray_dir, f)
        for f in os.listdir(xarray_dir)
        if f.endswith(pattern)
    )
    if not files:
        raise FileNotFoundError(f'No {pattern} files found in {xarray_dir}')

    ds_list = []
    for path in files:
        ds = xr.open_dataset(path)
        ds = add_within_trial_frame_starts(ds, dsfactor=dsfactor,
                                           frames_per_trial=frames_per_trial)
        fly_id = os.path.splitext(os.path.basename(path))[0]
        ds.attrs['fly_id'] = fly_id
        ds_list.append(ds)
        if verbose:
            conds = list(np.unique(ds.condition.values))
            print(f'  {fly_id:40s}  {ds.sizes["trial"]} trials  conds: {conds}')

    print(f'\nLoaded {len(ds_list)} datasets from {xarray_dir}')
    return ds_list


def get_condition_mean(ds, condition, data_var='deltaF_rg'):
    """
    Return the trial-averaged ΔF/F trace for one condition in one dataset.

    Returns
    -------
    mean_da : xr.DataArray (time, region), or None if condition absent
    n_trials : int
    """
    mask = ds.condition == condition
    if not bool(mask.any()):
        return None, 0
    da = ds[data_var].sel(trial=mask)
    return da.mean('trial'), int(mask.sum())


# ══════════════════════════════════════════════════════════════════════════════
# COLOURS
# ══════════════════════════════════════════════════════════════════════════════

def _make_fly_colours(n=9, lightness=0.65, saturation=0.90, hue_step_deg=30):
    """
    Generate n perceptually-uniform colours spaced hue_step_deg apart (HLS space).
    Blue/violet hues (h ≈ 0.55–0.75) get a small lightness boost so they read
    clearly on a black background.
    Default: 9 flies × 30° = 270° of hue range (no wrap-around).
    """
    step = hue_step_deg / 360.0
    colours = []
    for i in range(n):
        h = (i * step) % 1.0
        l_adj = lightness + 0.08 if 0.55 < h < 0.75 else lightness
        r, g, b = colorsys.hls_to_rgb(h, l_adj, saturation)
        colours.append('#{:02x}{:02x}{:02x}'.format(
            int(r * 255), int(g * 255), int(b * 255)))
    return colours


#: One colour per fly; cycles if n_flies > 20.
FLY_COLOURS = _make_fly_colours(20)

#: Perceptually-uniform qualitative palette (Paul Tol "bright", reordered).
#: Blue-family hues (blue #0, cyan #5) are 5 steps apart — never sequential.
_QUAL_PALETTE = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#44AA44',  # green  (lightened from Tol #228833 for dark-bg legibility)
    '#AA3377',  # reddish-purple
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan / sky-blue
    '#BBBBBB',  # grey
]

#: Canonical color per base condition — single source of truth.
#: Epoch suffixes (e.g. _OEpoch) are stripped before lookup.
_COND_COLOR_MAP = {
    'xOF': _QUAL_PALETTE[0],  # blue
    'xOx': _QUAL_PALETTE[2],  # green
    'xxF': _QUAL_PALETTE[4],  # yellow
    'VOF': _QUAL_PALETTE[1],  # red
    'VOx': _QUAL_PALETTE[3],  # reddish-purple
    'VxF': _QUAL_PALETTE[5],  # cyan
    'Vxx': _QUAL_PALETTE[6],  # grey
    'xxx': '#555555',          # dark grey
}


def condition_color(condition):
    """Canonical _QUAL_PALETTE color for a condition, epoch suffix ignored."""
    return _COND_COLOR_MAP.get(condition.split('_')[0], _QUAL_PALETTE[0])


#: Human-readable base labels (walk always last in compound labels).
_COND_LABEL_MAP = {
    'xOx': 'odor',
    'xxF': 'walk',
    'xOF': 'odor + walk',
    'Vxx': 'vis',
    'VOx': 'vis + odor',
    'VxF': 'vis + walk',
    'VOF': 'vis + odor + walk',
    'xxx': 'baseline',
}
_EPOCH_LABEL_MAP = {
    'OEpoch':      'odor epoch',
    'FEpoch':      'walk epoch',
    'VisMotEpoch': 'vis-mot epoch',
    'VisLumEpoch': 'vis-lum epoch',
}


def condition_label(condition):
    """
    Human-readable label for a condition name.
    Epoch suffixes (e.g. _OEpoch) are appended on a new line in parentheses.
    Walk always appears last in compound labels.
    """
    parts = condition.split('_', 1)
    label = _COND_LABEL_MAP.get(parts[0], parts[0])
    if len(parts) > 1:
        epoch = _EPOCH_LABEL_MAP.get(parts[1], parts[1])
        label = f'{label}\n({epoch})'
    return label


def condition_legend_handles(conditions, kind='line', lw=2.0, ls='-'):
    """
    Legend handles using canonical condition colors and human-readable labels.

    Parameters
    ----------
    conditions : list[str]
    kind       : 'line' (Line2D) or 'patch' (Patch)
    """
    if kind == 'patch':
        return [mpatches.Patch(color=condition_color(c), label=condition_label(c))
                for c in conditions]
    return [Line2D([0], [0], color=condition_color(c), lw=lw, ls=ls,
                   label=condition_label(c)) for c in conditions]


#: Stimulus bar colours for per-fly heatmap plots.
STIM_COLORS = {
    'odor': '#C8900A',  # amber
    'vis':  '#1A4FBF',  # blue
    'fw':   '#A855F7',  # purple
}

#: Condition colours for grand-mean overlay plots — derived from _COND_COLOR_MAP.
OVERLAY_COLOURS = {c: _COND_COLOR_MAP[c] for c in ('xOx', 'xxF', 'xOF', 'VOF')}

#: Summation-metric window labels and colours.
WINDOW_LABELS  = ['full', 'fw_window']
WINDOW_DISPLAY = {'full': 'Full window (4–32 s)', 'fw_window': 'FW on→off'}
WINDOW_COLORS  = {k: _QUAL_PALETTE[i] for i, k in enumerate(WINDOW_LABELS)}

#: Condition colours for z-scored timeseries plots.
COND_COLORS = {
    'xOF':  _COND_COLOR_MAP['xOF'],
    'xxF':  _COND_COLOR_MAP['xxF'],
    'xOx':  _COND_COLOR_MAP['xOx'],
    'pred': _QUAL_PALETTE[1],  # red — only used in summation plots (no VOF conflict)
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE STYLE
# ══════════════════════════════════════════════════════════════════════════════

def KCpreferred_FigSettings(axes=None):
    """Apply KC's preferred figure style as rcParams.

    Always updates rcParams globally (affects all subsequent figures).
    If *axes* is provided (single Axes or array/list of Axes), also hides
    the top and right spines on each axis.

    Parameters
    ----------
    axes : matplotlib.axes.Axes or array-like of Axes, optional
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.major.size':   5.25,
        'ytick.major.size':   5.25,
        'xtick.minor.size':   3.0,
        'ytick.minor.size':   3.0,
        'xtick.major.width':  1.5,
        'ytick.major.width':  1.5,
        'xtick.labelsize':    12,
        'ytick.labelsize':    12,
        'axes.labelsize':     16,
        'axes.titlesize':     14,
        'axes.titleweight':   'bold',
        'axes.linewidth':     2.0,
        'legend.fontsize':    12,
        'legend.frameon':     False,
    })
    if axes is not None:
        import numpy as np
        ax_list = list(np.array(axes).flat) if hasattr(axes, '__iter__') else [axes]
        for ax in ax_list:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_linewidth(2.0)
                ax.spines[sp].set_position(('outward', 8))
            ax.tick_params(which='major', direction='in', labelsize=12, length=5.25)
            ax.tick_params(which='minor', direction='in', length=3.0)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def lowpass_filter(data, cutoff=2.0, lfmrate=30, order=4):
    """
    Zero-phase Butterworth lowpass filter along the first (time) axis.

    Parameters
    ----------
    data    : np.ndarray  shape (time, ...) or (time,)
    cutoff  : float       cut-off frequency in Hz
    lfmrate : int         imaging frame rate in fps
    order   : int         filter order

    Returns
    -------
    filtered : np.ndarray  same shape as data
    """
    nyq = lfmrate / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)


def fit_exp_decay(y, t):
    """
    Fit y = A * exp(-t / tau) + C.

    Returns tau in seconds (frames converted using lfmrate=30 implicitly via
    the caller), or np.nan if the fit fails.
    """
    def _exp(t, A, tau, C):
        return A * np.exp(-t / tau) + C

    try:
        A0, tau0, C0 = y[0] - y[-1], len(t) / 3, y[-1]
        popt, _ = curve_fit(_exp, t, y, p0=[A0, tau0, C0],
                            bounds=([0, 0.1, -np.inf],
                                    [np.inf, len(t) * 10, np.inf]),
                            maxfev=5000)
        return popt[1] / 30   # frames → seconds
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# METRICS: AUC / Peak / Tau
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(ds, duration_sec=5, lfmrate=30,
                         baseline_sec=2, cutoff_hz=2.0,
                         data_var='deltaF_rg'):
    """
    Compute AUC, peak (baseline-corrected), and decay tau per epoch per trial.

    Returns three dicts (auc, peak, tau), each mapping
    ``'<condition>_<EpochSuffix>'`` → xr.DataArray(trial, region).

    Epoch definitions
    -----------------
    O → odor epoch     (OEpoch)
    F → walking epoch  (FEpoch)
    V → visual motion  (VisMotEpoch) + visual luminance (VisLumEpoch)
    """
    duration_frames = int(duration_sec  * lfmrate)
    baseline_frames = int(baseline_sec  * lfmrate)

    epoch_defs = {
        'O': [('odor', 'odor_framestarts_within_trial',  'OEpoch')],
        'F': [('fw',   'fw_framestarts_within_trial',    'FEpoch')],
        'V': [('vis',  'vis_framestarts_within_trial',   'VisMotEpoch'),
              ('vis',  'vis_patt_frameon_within_trial',  'VisLumEpoch')],
    }

    dff_raw      = ds[data_var].values                           # (trials, time, regions)
    dff_filtered = np.stack(
        [lowpass_filter(dff_raw[i], cutoff=cutoff_hz, lfmrate=lfmrate)
         for i in range(dff_raw.shape[0])], axis=0)

    auc_dict, peak_dict, tau_dict = {}, {}, {}

    for trial_num in ds.trial.values:
        cond      = str(ds.condition.sel(trial=trial_num).values)
        trial_idx = int(np.where(ds.trial.values == trial_num)[0])
        trace     = dff_filtered[trial_idx]                      # (time, regions)

        for char, defs in epoch_defs.items():
            if char not in cond:
                continue

            for stim_key, start_attr, suffix in defs:
                label         = f'{cond}_{suffix}'
                trial_indices = ds.attrs[f'{stim_key}_trial_indices']
                framestarts   = ds.attrs[start_attr]

                elem = np.where(trial_indices == trial_num)[0]
                if len(elem) == 0:
                    continue
                elem = elem[0]

                t_start      = int(framestarts[elem])
                t_end        = t_start + duration_frames
                t_base_start = max(0, t_start - baseline_frames)
                baseline_mean = trace[t_base_start:t_start, :].mean(axis=0)

                epoch    = trace[t_start:t_end, :]
                epoch_bc = epoch - baseline_mean

                auc  = np.trapezoid(epoch_bc, axis=0)
                peak = epoch_bc.max(axis=0)

                n_regions = epoch_bc.shape[1]
                tau = np.zeros(n_regions)
                for r in range(n_regions):
                    y         = epoch_bc[:, r]
                    pf        = int(np.argmax(y))
                    y_decay   = y[pf:]
                    t_decay   = np.arange(len(y_decay)).astype(float)
                    tau[r]    = (fit_exp_decay(y_decay, t_decay)
                                 if len(y_decay) >= 5 and y_decay[0] > 0
                                 else np.nan)

                def _da(values):
                    return (xr.DataArray(values, dims=['region'],
                                         coords={'region': ds.region.values})
                            .expand_dims(trial=[trial_num]))

                for d, val in [(auc_dict, auc), (peak_dict, peak), (tau_dict, tau)]:
                    d.setdefault(label, []).append(_da(val))

    def _concat(d):
        return {k: xr.concat(v, dim='trial') for k, v in d.items()}

    return _concat(auc_dict), _concat(peak_dict), _concat(tau_dict)


def compute_group_auc(ds_list, duration_sec=5, lfmrate=30, data_var='deltaF_rg'):
    """
    Run compute_all_metrics on every fly and collect trial-averaged AUCs.

    Returns
    -------
    group_auc : dict  { label: xr.DataArray(fly, region) }
    fly_ids   : list[str]
    """
    fly_ids, per_fly_auc = [], []

    for ds in ds_list:
        auc_all, _, _ = compute_all_metrics(ds, duration_sec=duration_sec,
                                             lfmrate=lfmrate, data_var=data_var)
        per_fly_auc.append(auc_all)
        fly_ids.append(ds.attrs.get('fly_id', 'unknown'))

    all_labels = sorted({label for auc in per_fly_auc for label in auc})

    group_auc = {}
    for label in all_labels:
        fly_means, present = [], []
        for fly_id, auc in zip(fly_ids, per_fly_auc):
            if label not in auc:
                continue
            fly_means.append(auc[label].mean('trial'))
            present.append(fly_id)
        if fly_means:
            group_auc[label] = xr.concat(
                fly_means, dim=pd.Index(present, name='fly'))

    return group_auc, fly_ids


def compute_group_peak(ds_list, baseline_sec=2, cutoff_hz=2.0, lfmrate=30,
                       data_var='deltaF_rg'):
    """
    Compute peak odor response within the actual odor window
    (odor_framestarts_within_trial → odor_frameends_within_trial), baseline-corrected.

    Returns
    -------
    group_peak : dict { '<condition>_OEpoch': xr.DataArray(fly, region) }
    fly_ids    : list[str]
    """
    baseline_frames = int(baseline_sec * lfmrate)
    fly_ids, all_peak_dicts = [], []

    for ds in ds_list:
        dff_raw = ds[data_var].values
        dff_lp  = np.stack(
            [lowpass_filter(dff_raw[i], cutoff=cutoff_hz, lfmrate=lfmrate)
             for i in range(dff_raw.shape[0])], axis=0)

        trial_indices = ds.attrs['odor_trial_indices']
        framestarts   = ds.attrs['odor_framestarts_within_trial']
        frameends     = ds.attrs['odor_frameends_within_trial']

        peak_dict = {}
        for trial_num in ds.trial.values:
            cond = str(ds.condition.sel(trial=trial_num).values)
            if 'O' not in cond:
                continue
            elem = np.where(trial_indices == trial_num)[0]
            if len(elem) == 0:
                continue
            elem      = elem[0]
            trial_idx = int(np.where(ds.trial.values == trial_num)[0])
            trace     = dff_lp[trial_idx]

            t_start      = int(framestarts[elem])
            t_end        = int(frameends[elem])
            t_base_start = max(0, t_start - baseline_frames)
            baseline_mean = trace[t_base_start:t_start, :].mean(axis=0)

            epoch_bc = trace[t_start:t_end, :] - baseline_mean
            peak     = epoch_bc.max(axis=0)

            label = f'{cond}_OEpoch'
            da = (xr.DataArray(peak, dims=['region'],
                               coords={'region': ds.region.values})
                  .expand_dims(trial=[trial_num]))
            peak_dict.setdefault(label, []).append(da)

        peak_dict = {k: xr.concat(v, dim='trial') for k, v in peak_dict.items()}
        all_peak_dicts.append(peak_dict)
        fly_ids.append(ds.attrs.get('fly_id', 'unknown'))

    all_labels = sorted({label for d in all_peak_dicts for label in d})
    group_peak = {}
    for label in all_labels:
        fly_means, present = [], []
        for fly_id, d in zip(fly_ids, all_peak_dicts):
            if label not in d:
                continue
            fly_means.append(d[label].mean('trial'))
            present.append(fly_id)
        if fly_means:
            group_peak[label] = xr.concat(
                fly_means, dim=pd.Index(present, name='fly'))

    return group_peak, fly_ids


# ══════════════════════════════════════════════════════════════════════════════
# METRICS: Pre-odor onset slope
# ══════════════════════════════════════════════════════════════════════════════

def compute_pre_odor_slope(ds, n_frames=1, cutoff_hz=2.0, lfmrate=30,
                            data_var='deltaF_rg'):
    """
    Compute ΔF/F slope at odor onset for each odor trial (lowpass-filtered).

    n_frames=1 : slope = ΔF[t_odor] − ΔF[t_odor − 1]
    n_frames>1 : slope = (mean(ΔF[t:t+n]) − mean(ΔF[t−n:t])) / n

    Returns
    -------
    dict { condition : xr.DataArray(trial, region) }
    """
    dff_raw = ds[data_var].values
    dff_lp  = np.stack(
        [lowpass_filter(dff_raw[i], cutoff=cutoff_hz, lfmrate=lfmrate)
         for i in range(dff_raw.shape[0])], axis=0)

    trial_indices = ds.attrs['odor_trial_indices']
    framestarts   = ds.attrs['odor_framestarts_within_trial']
    slope_dict    = {}

    for trial_num in ds.trial.values:
        cond = str(ds.condition.sel(trial=trial_num).values)
        elem = np.where(trial_indices == trial_num)[0]
        if len(elem) == 0:
            continue
        trial_idx = int(np.where(ds.trial.values == trial_num)[0])
        trace     = dff_lp[trial_idx]
        t_odor    = int(framestarts[elem[0]])

        if n_frames == 1:
            slope = trace[t_odor] - trace[max(0, t_odor - 1)]
        else:
            before = trace[max(0, t_odor - n_frames):t_odor]
            after  = trace[t_odor:t_odor + n_frames]
            slope  = (after.mean(axis=0) - before.mean(axis=0)) / n_frames

        da = xr.DataArray(
            slope[np.newaxis, :],
            dims=['trial', 'region'],
            coords={'trial': [trial_num], 'region': ds.region.values},
        )
        slope_dict.setdefault(cond, []).append(da)

    return {k: xr.concat(v, dim='trial') for k, v in slope_dict.items()}


def compute_group_pre_odor_slope(ds_list, n_frames=1, cutoff_hz=2.0,
                                  lfmrate=30, data_var='deltaF_rg'):
    """
    Run compute_pre_odor_slope on every fly and collect trial-mean slopes.

    Returns
    -------
    group_slope : dict { condition : xr.DataArray(fly, region) }
    fly_ids     : list[str]
    """
    fly_ids, per_fly = [], []
    for ds in ds_list:
        per_fly.append(compute_pre_odor_slope(
            ds, n_frames=n_frames, cutoff_hz=cutoff_hz,
            lfmrate=lfmrate, data_var=data_var))
        fly_ids.append(ds.attrs.get('fly_id', 'unknown'))

    all_conds = sorted({c for s in per_fly for c in s})
    group_slope = {}
    for cond in all_conds:
        fly_means, present = [], []
        for fly_id, s in zip(fly_ids, per_fly):
            if cond not in s:
                continue
            fly_means.append(s[cond].mean('trial'))
            present.append(fly_id)
        if fly_means:
            group_slope[cond] = xr.concat(
                fly_means, dim=pd.Index(present, name='fly'))

    return group_slope, fly_ids


# ══════════════════════════════════════════════════════════════════════════════
# METRICS: First-derivative max
# ══════════════════════════════════════════════════════════════════════════════

def compute_derivative_metrics(ds, cutoff_hz=2.0, lfmrate=30,
                                data_var='deltaF_rg'):
    """
    Compute the first derivative (np.gradient) of the lowpass-filtered ΔF/F.

    Returns
    -------
    dict { condition : {'deriv'     : DataArray(trial, time, region),
                        'max_val'   : DataArray(trial, region),
                        'max_frame' : DataArray(trial, region)} }
    """
    dff_raw   = ds[data_var].values
    dff_lp    = np.stack(
        [lowpass_filter(dff_raw[i], cutoff=cutoff_hz, lfmrate=lfmrate)
         for i in range(dff_raw.shape[0])], axis=0)
    dff_deriv = np.gradient(dff_lp, axis=1)

    result = {}
    coords = {'region': ds.region.values}

    for trial_num in ds.trial.values:
        cond      = str(ds.condition.sel(trial=trial_num).values)
        trial_idx = int(np.where(ds.trial.values == trial_num)[0])
        deriv     = dff_deriv[trial_idx]

        result.setdefault(cond, {'deriv': [], 'max_val': [], 'max_frame': []})
        result[cond]['deriv'].append(
            xr.DataArray(deriv[np.newaxis],
                         dims=['trial', 'time', 'region'],
                         coords={'trial': [trial_num], **coords}))
        result[cond]['max_val'].append(
            xr.DataArray(deriv.max(axis=0)[np.newaxis],
                         dims=['trial', 'region'],
                         coords={'trial': [trial_num], **coords}))
        result[cond]['max_frame'].append(
            xr.DataArray(deriv.argmax(axis=0).astype(float)[np.newaxis],
                         dims=['trial', 'region'],
                         coords={'trial': [trial_num], **coords}))

    return {c: {k: xr.concat(v, dim='trial') for k, v in arrays.items()}
            for c, arrays in result.items()}


def compute_group_derivative_max(ds_list, cutoff_hz=2.0, lfmrate=30,
                                  data_var='deltaF_rg'):
    """
    Run compute_derivative_metrics on every fly and collect trial-mean max values.

    Returns
    -------
    group_max_val : dict { condition : xr.DataArray(fly, region) }
    fly_ids       : list[str]
    per_fly       : list of per-fly metric dicts (pass to plot_derivative_traces)
    """
    fly_ids, per_fly = [], []
    for ds in ds_list:
        per_fly.append(compute_derivative_metrics(
            ds, cutoff_hz=cutoff_hz, lfmrate=lfmrate, data_var=data_var))
        fly_ids.append(ds.attrs.get('fly_id', 'unknown'))

    all_conds = sorted({c for m in per_fly for c in m})
    group_max_val = {}
    for cond in all_conds:
        fly_means, present = [], []
        for fly_id, m in zip(fly_ids, per_fly):
            if cond not in m:
                continue
            fly_means.append(m[cond]['max_val'].mean('trial'))
            present.append(fly_id)
        if fly_means:
            group_max_val[cond] = xr.concat(
                fly_means, dim=pd.Index(present, name='fly'))

    return group_max_val, fly_ids, per_fly


# ══════════════════════════════════════════════════════════════════════════════
# METRICS: Summation R² / Pearson r / normalised R²
# ══════════════════════════════════════════════════════════════════════════════

def compute_summation_metrics_per_fly(
        ds_list,
        cond_stim1='xOx', cond_stim2='xxF', cond_combo='xOF',
        frame_start=120, frame_end=960,
        cutoff_hz=2.0, lfmrate=30,
        data_var='deltaF_rg'):
    """
    Evaluate the fixed-coefficient summation  pred = stim1 + stim2  (coeff=1,
    no intercept) against the observed combo response.

    The trial-mean timeseries for each condition is **lowpass-filtered** before
    computing metrics (cutoff_hz, default 2 Hz).

    Three metrics computed for two time windows:

    +-----------+--------------------------------------------------------------+
    | r2        | Standard R² — penalises amplitude mismatch                  |
    | pearson_r | Pearson correlation — shape only, amplitude-invariant        |
    | r2_norm   | R² after z-scoring — amplitude-invariant version of R²       |
    +-----------+--------------------------------------------------------------+

    Returns
    -------
    dict  { '<metric>_<window>': DataArray(fly, region) }
    Keys: r2_full, r2_fw, pearson_r_full, pearson_r_fw, r2_norm_full, r2_norm_fw
    """
    region_names = ds_list[0].region.values
    n_regions    = len(region_names)

    fly_ids = []
    accum   = {k: [] for k in ('r2_full', 'r2_fw',
                                'pearson_r_full', 'pearson_r_fw',
                                'r2_norm_full',   'r2_norm_fw')}

    for ds in ds_list:
        conds_in_fly = set(ds.condition.values)
        if not {cond_stim1, cond_stim2, cond_combo}.issubset(conds_in_fly):
            print(f'  Skipping {ds.attrs.get("fly_id", "?")}: missing condition(s)')
            continue

        fly_ids.append(ds.attrs.get('fly_id', f'fly{len(fly_ids)}'))

        def _fly_mean_lp(cond):
            mask = ds.condition == cond
            raw  = ds[data_var].sel(trial=mask).mean('trial').values  # (time, region)
            return lowpass_filter(raw, cutoff=cutoff_hz, lfmrate=lfmrate)

        s1   = _fly_mean_lp(cond_stim1)
        s2   = _fly_mean_lp(cond_stim2)
        obs  = _fly_mean_lp(cond_combo)
        pred = s1 + s2

        def _metrics_window(fs, fe):
            r2_arr  = np.full(n_regions, np.nan)
            pr_arr  = np.full(n_regions, np.nan)
            r2n_arr = np.full(n_regions, np.nan)

            for r in range(n_regions):
                valid = (np.isfinite(s1[fs:fe, r]) &
                         np.isfinite(s2[fs:fe, r]) &
                         np.isfinite(obs[fs:fe, r]))
                if valid.sum() < 2:
                    continue

                o = obs[fs:fe, r][valid]
                p = pred[fs:fe, r][valid]

                ss_res = np.sum((o - p) ** 2)
                ss_tot = np.sum((o - np.mean(o)) ** 2)
                r2_arr[r] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                if o.std() > 0 and p.std() > 0:
                    pr_arr[r] = float(np.corrcoef(o, p)[0, 1])

                o_z = (o - o.mean()) / o.std() if o.std() > 0 else o * 0
                p_z = (p - p.mean()) / p.std() if p.std() > 0 else p * 0
                ss_res_n  = np.sum((o_z - p_z) ** 2)
                ss_tot_n  = np.sum((o_z - np.mean(o_z)) ** 2)
                r2n_arr[r] = 1.0 - ss_res_n / ss_tot_n if ss_tot_n > 0 else np.nan

            return r2_arr, pr_arr, r2n_arr

        n_time  = s1.shape[0]
        fs_full = frame_start if frame_start is not None else 0
        fe_full = frame_end   if frame_end   is not None else n_time
        r2f, prf, r2nf = _metrics_window(fs_full, fe_full)
        accum['r2_full'].append(r2f)
        accum['pearson_r_full'].append(prf)
        accum['r2_norm_full'].append(r2nf)

        fw_starts = ds.attrs.get('fw_framestarts_within_trial', None)
        fw_ends   = ds.attrs.get('fw_frameends_within_trial',   None)
        if fw_starts is not None and len(fw_starts) > 0:
            r2w, prw, r2nw = _metrics_window(
                int(np.round(np.mean(fw_starts))),
                int(np.round(np.mean(fw_ends))))
        else:
            r2w = prw = r2nw = np.full(n_regions, np.nan)

        accum['r2_fw'].append(r2w)
        accum['pearson_r_fw'].append(prw)
        accum['r2_norm_fw'].append(r2nw)

    if not fly_ids:
        return None

    dims   = ['fly', 'region']
    coords = {'fly': fly_ids, 'region': region_names}
    return {k: xr.DataArray(np.stack(v), dims=dims, coords=coords)
            for k, v in accum.items()}


def build_window_dict(results, metric):
    """Extract {'full': DataArray, 'fw_window': DataArray} for a given metric key."""
    return {
        'full':      results[f'{metric}_full'],
        'fw_window': results[f'{metric}_fw'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT UTILS
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams['pdf.fonttype'] = 42   # TrueType → text editable in Illustrator
plt.rcParams['ps.fonttype']  = 42

A4_W, A4_H = 8.27, 11.69   # inches


def fit_to_a4(fig):
    """Scale fig in-place to fit within A4, preserving aspect ratio (never upscales)."""
    w, h  = fig.get_size_inches()
    scale = min(A4_W / w, A4_H / h, 1.0)
    fig.set_size_inches(w * scale, h * scale)


def apply_dark_theme(fig):
    """
    Post-process a figure for a black background.

    - Sets figure + axes facecolor to black.
    - Inverts any hardcoded-black elements (spines, tick labels, lines, patches,
      legend text) to white.
    - Leaves all other colours (fly colours, condition colours) untouched.
    """
    BLACK = np.array([0., 0., 0.])

    def _is_black(c, atol=0.08):
        try:
            return np.allclose(mcolors.to_rgb(c), BLACK, atol=atol)
        except Exception:
            return False

    fig.patch.set_facecolor('black')

    for ax in fig.axes:
        ax.set_facecolor('black')

        for spine in ax.spines.values():
            if _is_black(spine.get_edgecolor()):
                spine.set_edgecolor('white')

        ax.tick_params(colors='white', which='both')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        for line in ax.get_lines():
            if _is_black(line.get_color()):
                line.set_color('white')

        for patch in ax.patches:
            if _is_black(patch.get_facecolor()):
                patch.set_facecolor('white')
            if _is_black(patch.get_edgecolor()):
                patch.set_edgecolor('white')

        for coll in ax.collections:
            try:
                fc = np.array(coll.get_facecolor(), dtype=float)
                for i in range(len(fc)):
                    if np.allclose(fc[i, :3], BLACK, atol=0.08):
                        fc[i, :3] = 1.0
                coll.set_facecolor(fc)
            except Exception:
                pass
            try:
                ec = np.array(coll.get_edgecolor(), dtype=float)
                for i in range(len(ec)):
                    if np.allclose(ec[i, :3], BLACK, atol=0.08):
                        ec[i, :3] = 1.0
                coll.set_edgecolor(ec)
            except Exception:
                pass

        for txt in ax.texts:
            if _is_black(txt.get_color()):
                txt.set_color('white')

        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#1a1a1a')
            legend.get_frame().set_edgecolor('#555555')
            for text in legend.get_texts():
                text.set_color('white')
            for handle in legend.legend_handles:
                try:
                    if _is_black(handle.get_facecolor()):
                        handle.set_facecolor('white')
                except Exception:
                    pass

    if fig._suptitle is not None:
        fig._suptitle.set_color('white')


def save_dark_pdf(figures, path):
    """
    Apply dark theme to each figure, fit to A4, and save to a multi-page PDF.

    Parameters
    ----------
    figures : list of (name_str, fig) tuples
    path    : str  output PDF path
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with PdfPages(path) as pdf:
        for name, fig in figures:
            apply_dark_theme(fig)
            fit_to_a4(fig)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f'  saved: {name}')
    print(f'\nDone → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: TIMESERIES — per-fly overlay, grand-mean ± SEM, interactive explorer
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_timeseries(ds_list, condition, region_idx, lfmrate=30,
                           alpha_individual=0.45, show_sem=True,
                           show_legend=True, ax=None,
                           data_var='deltaF_rg'):
    """
    Overlay trial-averaged ΔF/F traces for every fly that has *condition*.
    Grand mean ± SEM drawn in black (→ white after apply_dark_theme).

    Returns
    -------
    fig, ax  (or None, None if no fly has the condition)
    """
    fly_traces, fly_labels, fly_ntrial = [], [], []

    for ds in ds_list:
        mean_da, n = get_condition_mean(ds, condition, data_var=data_var)
        if mean_da is None:
            continue
        fly_traces.append(mean_da.isel(region=region_idx).values)
        fly_labels.append(ds.attrs.get('fly_id', '?'))
        fly_ntrial.append(n)

    if not fly_traces:
        print(f"No flies have condition '{condition}'")
        return None, None

    n_flies   = len(fly_traces)
    time_axis = np.arange(fly_traces[0].shape[0]) / lfmrate

    region_name = str(region_idx)
    for ds in ds_list:
        if 'region' in ds.coords:
            region_name = str(ds.region.values[region_idx])
            break

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    for i, (trace, label, n) in enumerate(zip(fly_traces, fly_labels, fly_ntrial)):
        ax.plot(time_axis, trace,
                color=FLY_COLOURS[i % len(FLY_COLOURS)], lw=1.0,
                alpha=alpha_individual, label=f'{label}  (n={n})')

    stack      = np.stack(fly_traces, axis=0)
    grand_mean = stack.mean(axis=0)
    grand_sem  = stack.std(axis=0) / np.sqrt(n_flies)

    ax.plot(time_axis, grand_mean, color='black', lw=2.2,
            label=f'Grand mean  (n={n_flies} flies)', zorder=5)
    if show_sem:
        ax.fill_between(time_axis,
                        grand_mean - grand_sem,
                        grand_mean + grand_sem,
                        color='black', alpha=0.15, zorder=4)

    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('ΔF/F', fontsize=10)
    ax.set_title(f'Condition: {condition}  |  Region: {region_name}', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5)

    if show_legend:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.7,
                  ncol=max(1, n_flies // 6))

    return fig, ax


def plot_group_overview(ds_list, region_idx, conditions=None, lfmrate=30,
                         ncols=4, data_var='deltaF_rg'):
    """
    One subplot per condition, all flies overlaid, for a fixed region.

    Returns
    -------
    fig
    """
    if conditions is None:
        conditions = sorted({str(c) for ds in ds_list
                             for c in np.unique(ds.condition.values)
                             if str(c) != 'xxx'})

    nrows = int(np.ceil(len(conditions) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.2 * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for i, (ax, cond) in enumerate(zip(axes, conditions)):
        plot_group_timeseries(ds_list, cond, region_idx, lfmrate=lfmrate,
                              show_sem=True, show_legend=(i == len(conditions) - 1),
                              ax=ax, data_var=data_var)
        ax.set_title(cond, fontsize=10)
        ax.set_xlabel('')

    for ax in axes[len(conditions):]:
        ax.set_visible(False)

    region_name = str(region_idx)
    for ds in ds_list:
        if 'region' in ds.coords:
            region_name = str(ds.region.values[region_idx])
            break

    fig.suptitle(f'All conditions — Region: {region_name}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def launch_group_ts_explorer(ds_list, lfmrate=30, data_var='deltaF_rg'):
    """
    Interactive ipywidgets GUI: pick condition and region with sliders/toggles.
    Requires ipywidgets and a running Jupyter kernel.
    """
    import ipywidgets as widgets
    from IPython.display import display

    all_conds = sorted({str(c) for ds in ds_list
                        for c in np.unique(ds.condition.values)
                        if str(c) != 'xxx'})
    n_regions = ds_list[0].sizes['region']

    cond_toggle   = widgets.ToggleButtons(options=all_conds,
                                          description='Condition:',
                                          style={'button_width': '70px'})
    region_slider = widgets.IntSlider(value=0, min=0, max=n_regions - 1,
                                      description='Region:',
                                      continuous_update=False,
                                      layout=widgets.Layout(width='480px'))
    sem_checkbox  = widgets.Checkbox(value=True, description='Show SEM',
                                     indent=False)
    out = widgets.Output()

    def update(cond, region_idx, show_sem):
        with out:
            out.clear_output(wait=True)
            fig, ax = plot_group_timeseries(ds_list, cond, region_idx,
                                             lfmrate=lfmrate, show_sem=show_sem,
                                             data_var=data_var)
            if fig is not None:
                n_with = sum(1 for ds in ds_list
                             if bool((ds.condition == cond).any()))
                fig.suptitle(f'Group time series — {n_with}/{len(ds_list)} flies',
                             fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.show()

    interactive = widgets.interactive_output(
        update, {'cond': cond_toggle, 'region_idx': region_slider,
                 'show_sem': sem_checkbox})
    display(widgets.VBox([cond_toggle,
                          widgets.HBox([region_slider, sem_checkbox]),
                          out]),
            interactive)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: OVERLAY — grand-mean ± SEM with optional stimulus bars
# ══════════════════════════════════════════════════════════════════════════════

#: Stimulus bar definitions: (letter_in_cond_name, start_attr, end_attr, label, colour)
STIM_DEFS = [
    ('O', 'odor_framestarts_within_trial', 'odor_frameends_within_trial',    'Odor',   '#C8900A'),
    ('V', 'vis_framestarts_within_trial',  'vis_patt_frameoff_within_trial', 'Visual', '#1A4FBF'),
    ('F', 'fw_framestarts_within_trial',   'fw_frameends_within_trial',      'Walk',   '#A855F7'),
]


def _get_stim_window(ds_list, attr_start, attr_end, lfmrate=30):
    starts, ends = [], []
    for ds in ds_list:
        if attr_start in ds.attrs and attr_end in ds.attrs:
            starts.extend(np.asarray(ds.attrs[attr_start]).ravel().tolist())
            ends.extend(  np.asarray(ds.attrs[attr_end  ]).ravel().tolist())
    if not starts:
        return None
    return float(np.mean(starts)) / lfmrate, float(np.mean(ends)) / lfmrate


def _add_stim_bars(ax, ds_list, conditions, lfmrate=30, dark_bg=True):
    _tc   = 'white' if dark_bg else 'black'
    ymin, ymax = ax.get_ylim()
    span  = ymax - ymin
    bar_h = span * 0.06
    gap   = span * 0.02
    active = {l for cond in conditions for l in ('V', 'O', 'F') if l in cond}
    y_cursor = ymin - gap

    for letter, attr_start, attr_end, label, color in STIM_DEFS:
        if letter not in active:
            continue
        win = _get_stim_window(ds_list, attr_start, attr_end, lfmrate)
        if win is None:
            continue
        onset, offset = win
        y_top, y_bot  = y_cursor, y_cursor - bar_h
        ax.fill_between([onset, offset], [y_bot, y_bot], [y_top, y_top],
                        color=color, alpha=1.0, zorder=6, linewidth=0)
        ax.text(onset - 0.15, (y_top + y_bot) / 2, label,
                ha='right', va='center', fontsize=10, color=_tc)
        y_cursor = y_bot - gap

    ax.set_ylim(y_cursor, ymax)


def _plot_one_region(ds_list, conditions, region_idx, lfmrate,
                     colours, alpha_sem, ax, dark_bg, data_var='deltaF_rg'):
    _tc = 'white' if dark_bg else 'black'
    if dark_bg:
        ax.get_figure().patch.set_facecolor('black')
        ax.set_facecolor('black')

    region_name = str(region_idx)
    for ds in ds_list:
        if 'region' in ds.coords:
            region_name = str(ds.region.values[region_idx])
            break

    for cond in conditions:
        fly_traces = []
        for ds in ds_list:
            mean_da, n = get_condition_mean(ds, cond, data_var=data_var)
            if mean_da is None:
                continue
            fly_traces.append(mean_da.isel(region=region_idx).values)
        if not fly_traces:
            continue

        n_flies    = len(fly_traces)
        t          = np.arange(fly_traces[0].shape[0]) / lfmrate
        stack      = np.stack(fly_traces, axis=0)
        grand_mean = stack.mean(axis=0)
        grand_sem  = stack.std(axis=0) / np.sqrt(n_flies)
        col        = colours.get(cond, 'white')

        ax.plot(t, grand_mean, color=col, lw=2.2,
                label=f'{cond}  (n={n_flies})', zorder=5)
        ax.fill_between(t, grand_mean - grand_sem, grand_mean + grand_sem,
                        color=col, alpha=alpha_sem, zorder=4)

    ax.axhline(0, color='#888888', lw=0.6, ls='--')
    ax.set_xlabel('Time (s)', fontsize=11, color=_tc)
    ax.set_ylabel('ΔF/F', fontsize=11, color=_tc)
    ax.set_title(f'{region_name}  [{region_idx + 1}]', fontsize=12, color=_tc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color(_tc)
    ax.tick_params(width=1.5, colors=_tc, labelsize=11)
    ax.xaxis.label.set_color(_tc)
    ax.yaxis.label.set_color(_tc)
    legend = ax.legend(fontsize=11, loc='upper right', framealpha=0.3)
    for text in legend.get_texts():
        text.set_color(_tc)
    legend.get_frame().set_edgecolor(_tc)


def plot_conditions_overlay(ds_list, conditions, region_idx=None,
                             lfmrate=30, colours=None, alpha_sem=0.35,
                             ncols=4, ax=None, dark_bg=True,
                             data_var='deltaF_rg'):
    """
    Grand mean ± SEM per condition overlaid on a single axes (or a grid).

    region_idx : None (all), list of ints, or single int.
    """
    if colours is None:
        colours = {c: FLY_COLOURS[i % len(FLY_COLOURS)]
                   for i, c in enumerate(conditions)}
    if region_idx is None:
        region_idx = list(range(ds_list[0].sizes['region']))

    if isinstance(region_idx, (list, tuple)):
        n     = len(region_idx)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 4 * nrows),
                                 facecolor='black' if dark_bg else 'white')
        for ax_i, ridx in zip(np.array(axes).flatten(), region_idx):
            _plot_one_region(ds_list, conditions, ridx, lfmrate,
                             colours, alpha_sem, ax_i, dark_bg, data_var)
        for ax_i in np.array(axes).flatten()[n:]:
            ax_i.set_visible(False)
        return fig, axes

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()
    _plot_one_region(ds_list, conditions, region_idx, lfmrate,
                     colours, alpha_sem, ax, dark_bg, data_var)
    return fig, ax


def plot_conditions_overlay_stim(ds_list, conditions, region_idx=None,
                                  lfmrate=30, colours=None, alpha_sem=0.35,
                                  ncols=4, dark_bg=True, data_var='deltaF_rg'):
    """Like plot_conditions_overlay but with stimulus-onset bars below each subplot."""
    if colours is None:
        colours = {c: FLY_COLOURS[i % len(FLY_COLOURS)]
                   for i, c in enumerate(conditions)}
    if region_idx is None:
        region_idx = list(range(ds_list[0].sizes['region']))

    if isinstance(region_idx, (list, tuple)):
        n     = len(region_idx)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 5 * nrows),
                                 facecolor='black' if dark_bg else 'white')
        for ax_i, ridx in zip(np.array(axes).flatten(), region_idx):
            _plot_one_region(ds_list, conditions, ridx, lfmrate,
                             colours, alpha_sem, ax_i, dark_bg, data_var)
            _add_stim_bars(ax_i, ds_list, conditions, lfmrate, dark_bg)
        for ax_i in np.array(axes).flatten()[n:]:
            ax_i.set_visible(False)
        return fig, axes

    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_one_region(ds_list, conditions, region_idx, lfmrate,
                     colours, alpha_sem, ax, dark_bg, data_var)
    _add_stim_bars(ax, ds_list, conditions, lfmrate, dark_bg)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: HEATMAP — per-fly ΔF heatmap, all trials concatenated
# ══════════════════════════════════════════════════════════════════════════════

#: Stimulus bar spec: (stim_key, trial_idx_attr, start_attr, end_attr, label, color)
_HMAP_STIM = [
    ('odor', 'odor_trial_indices', 'odor_framestarts_within_trial',  'odor_frameends_within_trial',    'Odor',   STIM_COLORS['odor']),
    ('vis',  'vis_trial_indices',  'vis_framestarts_within_trial',   'vis_patt_frameoff_within_trial', 'Visual', STIM_COLORS['vis']),
    ('fw',   'fw_trial_indices',   'fw_framestarts_within_trial',    'fw_frameends_within_trial',      'Walk',   STIM_COLORS['fw']),
]


def plot_fly_heatmap(ds, data_var='trialTS_rg', lfmrate=30, cmap='magma', clim_pct=99):
    """
    Per-fly ΔF heatmap: all trials concatenated, regions on y-axis.
    Three stimulus-bar rows below (Odor / Visual / Walk) with STIM_COLORS.

    Parameters
    ----------
    ds       : xr.Dataset with data_var of shape (n_trials, frames_per_trial, n_regions)
               and stimulus timing attrs added by add_within_trial_frame_starts.
    data_var : variable name in ds (default 'trialTS_rg')
    """
    fly_id = str(ds.attrs.get('fly_id', ds.attrs.get('identifier', 'fly')))

    dff_sorted = ds[data_var].sortby('trial').values   # (n_trials, fpt, n_regions)
    n_trials, fpt, n_regions = dff_sorted.shape
    dff_cat     = np.nan_to_num(dff_sorted.reshape(n_trials * fpt, n_regions).T, nan=0.0)
    total_frames = n_trials * fpt

    trial_nums = sorted(int(t) for t in ds.trial.values.tolist())
    trial_pos  = {t: i for i, t in enumerate(trial_nums)}
    region_labels = ds.region.values if 'region' in ds.coords else np.arange(n_regions)

    vmin = np.nanpercentile(dff_cat,  1)
    vmax = np.nanpercentile(dff_cat, clim_pct)

    hmap_h = n_regions * 0.12
    fig_w  = 12
    fig_h  = max(5, hmap_h + 2.0)
    fig    = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('black')

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           width_ratios=[30, 0.6],
                           height_ratios=[hmap_h, 0.5, 0.5, 0.5],
                           hspace=0.06, wspace=0.03,
                           left=0.10, right=0.95, top=0.94, bottom=0.08)

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_odor = fig.add_subplot(gs[1, 0], sharex=ax_heat)
    ax_vis  = fig.add_subplot(gs[2, 0], sharex=ax_heat)
    ax_walk = fig.add_subplot(gs[3, 0], sharex=ax_heat)

    # heatmap
    im = ax_heat.imshow(dff_cat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[0, total_frames / lfmrate, n_regions - 0.5, -0.5],
                        interpolation='nearest')
    for i in range(1, n_trials):
        ax_heat.axvline(i * fpt / lfmrate, color='white', lw=0.3, alpha=0.25, zorder=5)

    ax_heat.set_ylabel('Region', fontsize=11, color='white')
    ax_heat.set_title(fly_id, fontsize=12, color='white', pad=6)
    ax_heat.tick_params(colors='white', labelsize=8, width=1)
    ax_heat.yaxis.set_ticks(np.arange(n_regions))
    ax_heat.yaxis.set_ticklabels(region_labels, fontsize=7)
    plt.setp(ax_heat.get_xticklabels(), visible=False)
    for sp in ax_heat.spines.values():
        sp.set_visible(False)

    # colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(colors='white', labelsize=8, width=1)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.label.set_color('white')
    cbar.set_label('ΔF', fontsize=9, color='white')

    # stimulus bars
    for (_, tidx_attr, start_attr, end_attr, label, color), ax in zip(
            _HMAP_STIM, [ax_odor, ax_vis, ax_walk]):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_facecolor('black')
        for sp in ax.spines.values():
            sp.set_visible(False)
        if tidx_attr in ds.attrs and start_attr in ds.attrs and end_attr in ds.attrs:
            t_indices = np.asarray(ds.attrs[tidx_attr]).ravel().astype(int)
            starts    = np.asarray(ds.attrs[start_attr]).ravel()
            ends      = np.asarray(ds.attrs[end_attr]).ravel()
            for t_num, s, e in zip(t_indices, starts, ends):
                pos = trial_pos.get(t_num)
                if pos is None:
                    continue
                t_on  = (pos * fpt + s) / lfmrate
                t_off = (pos * fpt + e) / lfmrate
                ax.fill_between([t_on, t_off], [0, 0], [1, 1],
                                color=color, alpha=1.0, linewidth=0, zorder=3)
        ax.text(-0.005, 0.5, label, transform=ax.transAxes,
                ha='right', va='center', fontsize=10, color='white')

    ax_walk.spines['bottom'].set_visible(True)
    ax_walk.spines['bottom'].set_color('white')
    ax_walk.spines['bottom'].set_linewidth(0.8)
    ax_walk.tick_params(axis='x', colors='white', labelsize=9, width=1, length=4)
    ax_walk.set_xlabel('Time (s)', fontsize=11, color='white')
    plt.setp(ax_odor.get_xticklabels(), visible=False)
    plt.setp(ax_vis.get_xticklabels(),  visible=False)

    for row in [1, 2, 3]:
        ax_blank = fig.add_subplot(gs[row, 1])
        ax_blank.set_visible(False)

    return fig


def plot_fly_heatmap_sorted(ds, data_var='trialTS_rg', lfmrate=30, cmap='magma',
                            clim_pct=99, sort_by='hierarchical'):
    """
    Like plot_fly_heatmap but rows (regions) are reordered by activity clustering
    and all-NaN regions are dropped before plotting.

    Parameters
    ----------
    ds       : xr.Dataset — same format as plot_fly_heatmap
    sort_by  : 'hierarchical' (default) — Ward linkage on mean ΔF per frame
               'mean'                   — sort rows by their grand mean (ascending)
    """
    fly_id = str(ds.attrs.get('fly_id', ds.attrs.get('identifier', 'fly')))

    dff_raw  = ds[data_var].sortby('trial').values          # (n_trials, fpt, n_regions)
    n_trials, fpt, n_regions = dff_raw.shape
    dff_2d   = dff_raw.reshape(n_trials * fpt, n_regions).T  # (n_regions, total_frames)
    total_frames  = n_trials * fpt
    region_labels = ds.region.values if 'region' in ds.coords else np.arange(n_regions)

    # ── drop all-NaN regions ─────────────────────────────────────────────────
    valid_mask    = ~np.all(np.isnan(dff_2d), axis=1)
    dff_2d        = dff_2d[valid_mask]
    region_labels = np.asarray(region_labels)[valid_mask]
    n_regions     = dff_2d.shape[0]

    dff_cat = np.nan_to_num(dff_2d, nan=0.0)

    # ── cluster / sort rows ──────────────────────────────────────────────────
    if sort_by == 'hierarchical':
        if n_regions > 1:
            Z     = linkage(dff_cat, method='ward', metric='euclidean')
            order = leaves_list(Z)
        else:
            order = np.array([0])
    elif sort_by == 'mean':
        order = np.argsort(dff_cat.mean(axis=1))
    else:
        raise ValueError(f"sort_by must be 'hierarchical' or 'mean', got {sort_by!r}")

    dff_sorted    = dff_cat[order]
    sorted_labels = region_labels[order]

    vmin = np.nanpercentile(dff_sorted,  1)
    vmax = np.nanpercentile(dff_sorted, clim_pct)

    trial_nums = sorted(int(t) for t in ds.trial.values.tolist())
    trial_pos  = {t: i for i, t in enumerate(trial_nums)}

    hmap_h = n_regions * 0.12
    fig_w  = 12
    fig_h  = max(5, hmap_h + 2.0)
    fig    = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('black')

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           width_ratios=[30, 0.6],
                           height_ratios=[hmap_h, 0.18, 0.18, 0.18],
                           hspace=0.06, wspace=0.03,
                           left=0.10, right=0.95, top=0.94, bottom=0.08)

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_odor = fig.add_subplot(gs[1, 0], sharex=ax_heat)
    ax_vis  = fig.add_subplot(gs[2, 0], sharex=ax_heat)
    ax_walk = fig.add_subplot(gs[3, 0], sharex=ax_heat)

    # heatmap
    im = ax_heat.imshow(dff_sorted, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[0, total_frames / lfmrate, n_regions - 0.5, -0.5],
                        interpolation='nearest')
    for i in range(1, n_trials):
        ax_heat.axvline(i * fpt / lfmrate, color='white', lw=0.3, alpha=0.25, zorder=5)

    ax_heat.set_ylabel('Region (clustered)', fontsize=11, color='white')
    ax_heat.set_title(f'{fly_id}  [rows sorted by {sort_by} activity]',
                      fontsize=12, color='white', pad=6)
    ax_heat.tick_params(colors='white', labelsize=8, width=1)
    ax_heat.yaxis.set_ticks(np.arange(n_regions))
    ax_heat.yaxis.set_ticklabels(sorted_labels, fontsize=7)
    plt.setp(ax_heat.get_xticklabels(), visible=False)
    for sp in ax_heat.spines.values():
        sp.set_visible(False)

    # colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(colors='white', labelsize=8, width=1)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.label.set_color('white')
    cbar.set_label('ΔF', fontsize=9, color='white')

    # stimulus bars
    for (_, tidx_attr, start_attr, end_attr, label, color), ax in zip(
            _HMAP_STIM, [ax_odor, ax_vis, ax_walk]):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_facecolor('black')
        for sp in ax.spines.values():
            sp.set_visible(False)
        if tidx_attr in ds.attrs and start_attr in ds.attrs and end_attr in ds.attrs:
            t_indices = np.asarray(ds.attrs[tidx_attr]).ravel().astype(int)
            starts    = np.asarray(ds.attrs[start_attr]).ravel()
            ends      = np.asarray(ds.attrs[end_attr]).ravel()
            for t_num, s, e in zip(t_indices, starts, ends):
                pos = trial_pos.get(t_num)
                if pos is None:
                    continue
                t_on  = (pos * fpt + s) / lfmrate
                t_off = (pos * fpt + e) / lfmrate
                ax.fill_between([t_on, t_off], [0, 0], [1, 1],
                                color=color, alpha=1.0, linewidth=0, zorder=3)
        ax.text(-0.005, 0.5, label, transform=ax.transAxes,
                ha='right', va='center', fontsize=9, color='white')

    ax_walk.spines['bottom'].set_visible(True)
    ax_walk.spines['bottom'].set_color('white')
    ax_walk.spines['bottom'].set_linewidth(0.8)
    ax_walk.tick_params(axis='x', colors='white', labelsize=9, width=1, length=8)
    ax_walk.set_xlabel('Time (s)', fontsize=11, color='white')
    plt.setp(ax_odor.get_xticklabels(), visible=False)
    plt.setp(ax_vis.get_xticklabels(),  visible=False)

    for row in [1, 2, 3]:
        ax_blank = fig.add_subplot(gs[row, 1])
        ax_blank.set_visible(False)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SCATTER-BOX — group AUC and pairwise slope/derivative
# ══════════════════════════════════════════════════════════════════════════════

def _scatter_box_axes(ax, group_dict, labels, region_indices, region_coords,
                      colors, show_scatter, box_width, violin=False):
    """Internal: draw scatter + box/violin for one row of regions."""
    n_conds   = len(labels)
    spacing   = 1.0
    gap       = 2.0
    group_w   = n_conds * spacing
    x_offsets = np.arange(n_conds) * spacing - (group_w - spacing) / 2
    n_flies   = max(group_dict[l].sizes['fly'] for l in labels
                    if l in group_dict)

    xtick_pos, xtick_labels = [], []

    for r_idx, reg_i in enumerate(region_indices):
        region_center = r_idx * (group_w + gap)

        for c_idx, label in enumerate(labels):
            if label not in group_dict:
                continue
            x_center  = region_center + x_offsets[c_idx]
            vals      = group_dict[label].isel(region=reg_i).values.astype(float)
            box_color = colors[label]

            if show_scatter:
                for fly_i, v in enumerate(vals):
                    if np.isnan(v):
                        continue
                    jitter = (np.random.rand() - 0.5) * 0.25
                    ax.scatter(x_center + jitter, v,
                               color=FLY_COLOURS[fly_i % len(FLY_COLOURS)],
                               s=50, alpha=1.0, zorder=2, linewidths=0)

            clean = vals[~np.isnan(vals)]
            if violin:
                if len(clean) >= 2:
                    parts = ax.violinplot(clean, positions=[x_center],
                                         widths=box_width, showmeans=False,
                                         showmedians=True, showextrema=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor(box_color)
                        pc.set_edgecolor(box_color)
                        pc.set_alpha(1.0)
                        pc.set_linewidth(1.0)
                    for partname in ('cbars', 'cmins', 'cmaxes'):
                        if partname in parts:
                            parts[partname].set_edgecolor('black')
                            parts[partname].set_linewidth(0.8)
                    if 'cmedians' in parts:
                        parts['cmedians'].set_edgecolor('black')
                        parts['cmedians'].set_linewidth(2.0)
                elif len(clean) == 1:
                    ax.scatter([x_center], clean, color=box_color, s=40, zorder=3)
            else:
                if len(clean) > 0:
                    rgb = mcolors.to_rgb(box_color)
                    ax.boxplot(clean, positions=[x_center], widths=box_width,
                               patch_artist=True, zorder=3,
                               medianprops=dict(color='black', lw=1.8),
                               boxprops=dict(facecolor=(*rgb, 0.3),
                                             edgecolor=box_color, lw=1.2),
                               whiskerprops=dict(color=box_color, lw=1.0),
                               capprops=dict(color=box_color, lw=1.0),
                               showfliers=False)

        if show_scatter:
            for fly_i in range(n_flies):
                fly_xs, fly_vs = [], []
                for c_idx, label in enumerate(labels):
                    if label not in group_dict:
                        continue
                    da = group_dict[label]
                    if fly_i >= da.sizes['fly']:
                        continue
                    v = float(da.isel(region=reg_i, fly=fly_i).values)
                    if np.isnan(v):
                        continue
                    fly_xs.append(region_center + x_offsets[c_idx])
                    fly_vs.append(v)
                if len(fly_xs) > 1:
                    ax.plot(fly_xs, fly_vs,
                            color=FLY_COLOURS[fly_i % len(FLY_COLOURS)],
                            lw=0.9, zorder=1)

        xtick_pos.append(region_center)
        xtick_labels.append(region_coords[reg_i])

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=13)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=10)
    return n_flies


def plot_group_auc(group_auc, conditions=None, regions=None,
                   show_scatter=True, box_width=0.5, title=None, violin=False):
    """
    Scatter + box/violin plot of per-fly AUC values per region.

    Parameters
    ----------
    group_auc   : dict { label : DataArray(fly, region) }
    conditions  : list[str] or None  (None = sorted keys)
    regions     : list[int] or None
    show_scatter: if False, omit individual fly dots and connecting lines
    violin      : if True, draw violins instead of boxes
    """
    labels = conditions or sorted(group_auc.keys())
    colors = {l: _QUAL_PALETTE[i % len(_QUAL_PALETTE)] for i, l in enumerate(labels)}

    region_coords  = next(iter(group_auc.values())).coords['region'].values
    region_indices = list(range(len(region_coords))) if regions is None else regions

    chunks = [region_indices[i:i + 20] for i in range(0, len(region_indices), 20)]
    fig, axes = plt.subplots(len(chunks), 1,
                              figsize=(11.7, 4 * len(chunks)), sharey=True)
    if len(chunks) == 1:
        axes = [axes]

    n_flies = 0
    for ax, chunk in zip(axes, chunks):
        n_flies = _scatter_box_axes(ax, group_auc, labels, chunk,
                                    region_coords, colors, show_scatter, box_width,
                                    violin=violin)
        ax.set_ylabel('AUC (∫ΔF)', fontsize=14)
        ax.set_xlabel('Regions', fontsize=14)

    cond_handles = [mpatches.Patch(color=colors[l], label=l) for l in labels]
    legend_handles = cond_handles
    if show_scatter:
        fly_handles   = [mpatches.Patch(color=FLY_COLOURS[i % len(FLY_COLOURS)],
                                        label=f'Fly {i}') for i in range(n_flies)]
        legend_handles = cond_handles + fly_handles
    axes[0].legend(handles=legend_handles,
                   fontsize=9, loc='upper right', framealpha=0.8)
    axes[0].set_title(title or 'Group AUC', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_pairwise_slope(group_slope, conditions, regions=None,
                        title_suffix='', show_scatter=True, box_width=0.5,
                        violin=False):
    """
    Scatter + box/violin plot of per-fly slope (or derivative-max) per region.
    Same visual style as plot_group_auc.

    Parameters
    ----------
    show_scatter : if False, omit individual fly dots and connecting lines
    violin       : if True, draw violins instead of boxes
    """
    labels = [c for c in conditions if c in group_slope]
    if not labels:
        print('No matching conditions in group_slope.')
        return None

    colors = {l: _QUAL_PALETTE[i % len(_QUAL_PALETTE)] for i, l in enumerate(labels)}

    region_coords  = next(iter(group_slope.values())).coords['region'].values
    region_indices = list(range(len(region_coords))) if regions is None else regions

    chunks = [region_indices[i:i + 20] for i in range(0, len(region_indices), 20)]
    fig, axes = plt.subplots(len(chunks), 1,
                              figsize=(11.7, 4 * len(chunks)), sharey=True)
    if len(chunks) == 1:
        axes = [axes]

    n_flies = 0
    for ax, chunk in zip(axes, chunks):
        n_flies = _scatter_box_axes(ax, group_slope, labels, chunk,
                                    region_coords, colors, show_scatter, box_width,
                                    violin=violin)
        ax.set_ylabel('Slope (ΔF/frame)', fontsize=14)
        ax.set_xlabel('Region', fontsize=14)

    cond_handles   = [mpatches.Patch(color=colors[l], label=l) for l in labels]
    legend_handles = cond_handles
    if show_scatter:
        fly_handles    = [mpatches.Patch(color=FLY_COLOURS[i % len(FLY_COLOURS)],
                                         label=f'Fly {i}') for i in range(n_flies)]
        legend_handles = cond_handles + fly_handles
    axes[0].legend(handles=legend_handles,
                   fontsize=9, loc='upper right', framealpha=0.8)
    axes[0].set_title(f'Pre-odor onset slope — {", ".join(labels)}{title_suffix}',
                      fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: DERIVATIVE TRACES
# ══════════════════════════════════════════════════════════════════════════════

def plot_derivative_traces(ds_list, per_fly_metrics, conditions,
                            region_idx, lfmrate=30):
    """
    Per-condition plot of trial-averaged first-derivative traces (one line per fly).
    Vertical lines mark odor onset and the grand-mean derivative argmax.

    Returns
    -------
    fig, axes
    """
    n_conds = len(conditions)
    fig, axes = plt.subplots(1, n_conds, figsize=(6 * n_conds, 4), sharey=True)
    if n_conds == 1:
        axes = [axes]

    region_name = str(region_idx)
    for ds in ds_list:
        if 'region' in ds.coords:
            region_name = str(ds.region.values[region_idx])
            break

    for ax, condition in zip(axes, conditions):
        fly_traces, odor_frames = [], []

        for ds_idx, (ds, metrics) in enumerate(zip(ds_list, per_fly_metrics)):
            if condition not in metrics:
                continue
            color  = FLY_COLOURS[ds_idx % len(FLY_COLOURS)]
            deriv_da   = metrics[condition]['deriv'].isel(region=region_idx)
            mean_deriv = deriv_da.mean('trial').values
            time_axis  = np.arange(len(mean_deriv)) / lfmrate

            ax.plot(time_axis, mean_deriv, color=color, lw=1.0, alpha=0.55,
                    label=ds.attrs.get('fly_id', f'fly_{ds_idx}'))
            fly_traces.append(mean_deriv)

            trial_indices = ds.attrs['odor_trial_indices']
            cond_trials   = ds.trial.values[ds.condition.values == condition]
            if len(cond_trials) > 0:
                odor_start = ds.attrs['odor_framestarts_within_trial']
                elem = np.where(np.isin(trial_indices, cond_trials))[0]
                if len(elem) > 0:
                    odor_frames.append(float(np.mean(odor_start[elem])))

        if fly_traces:
            grand_mean = np.stack(fly_traces, axis=0).mean(axis=0)
            time_axis  = np.arange(len(grand_mean)) / lfmrate
            ax.plot(time_axis, grand_mean, color='black', lw=2.0, zorder=5,
                    label='Grand mean')
            peak_t = float(np.argmax(grand_mean)) / lfmrate
            ax.axvline(peak_t, color='white', lw=0.8, ls=':', zorder=6)

        if odor_frames:
            ax.axvline(float(np.mean(odor_frames)) / lfmrate,
                       color='#888888', lw=0.8, ls='--', zorder=6)

        ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.set_title(f'{condition}  |  {region_name}', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('dΔF/dt (ΔF/frame)', fontsize=10)
    axes[-1].legend(fontsize=6, loc='upper right')
    fig.suptitle(f'First derivative — region {region_name}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SUMMATION METRICS — scatter-box, violin, z-scored traces
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_scatter_box(window_dict, ylabel='metric', regions=None,
                             title=None, box_width=0.5, show_scatter=True,
                             violin=False, ref_lines=None):
    """
    Scatter + box/violin plot of per-fly values for each region.
    One group per window type (full / fw_window) per region.

    Parameters
    ----------
    window_dict  : {'full': DataArray(fly,region), 'fw_window': DataArray(fly,region)}
    ylabel       : y-axis label
    show_scatter : if False, omit individual fly dots and connecting lines
    violin       : if True, draw violins instead of boxes
    ref_lines    : list of (y, colour, linestyle) or None
                   default: [(0, 'gray', '--'), (1, 'steelblue', ':')]
    """
    windows   = WINDOW_LABELS
    n_wins    = len(windows)
    spacing   = 1.0
    gap       = 2.0
    x_offsets = np.arange(n_wins) * spacing - (n_wins * spacing - spacing) / 2

    region_coords  = window_dict['full'].coords['region'].values
    region_indices = list(range(len(region_coords))) if regions is None else regions
    n_flies        = window_dict['full'].sizes['fly']

    chunks = [region_indices[i:i + 20] for i in range(0, len(region_indices), 20)]
    fig, axes = plt.subplots(len(chunks), 1,
                              figsize=(11.7, 4 * len(chunks)), sharey=True)
    if len(chunks) == 1:
        axes = [axes]

    for ax, chunk in zip(axes, chunks):
        xtick_pos, xtick_labels = [], []

        for r_idx, reg_i in enumerate(chunk):
            region_center = r_idx * (n_wins * spacing + gap)

            for w_idx, win in enumerate(windows):
                x_center  = region_center + x_offsets[w_idx]
                vals      = window_dict[win].isel(region=reg_i).values.astype(float)
                box_color = WINDOW_COLORS[win]

                if show_scatter:
                    for fly_i, v in enumerate(vals):
                        if np.isnan(v):
                            continue
                        jitter = (np.random.rand() - 0.5) * 0.25
                        ax.scatter(x_center + jitter, v,
                                   color=FLY_COLOURS[fly_i % len(FLY_COLOURS)],
                                   s=50, alpha=1.0, zorder=2, linewidths=0)

                clean = vals[~np.isnan(vals)]
                if violin:
                    if len(clean) >= 2:
                        parts = ax.violinplot(clean, positions=[x_center],
                                              widths=box_width, showmeans=False,
                                              showmedians=True, showextrema=True)
                        for pc in parts['bodies']:
                            pc.set_facecolor(box_color)
                            pc.set_edgecolor(box_color)
                            pc.set_alpha(1.0)
                            pc.set_linewidth(1.0)
                        for partname in ('cbars', 'cmins', 'cmaxes'):
                            if partname in parts:
                                parts[partname].set_edgecolor('black')
                                parts[partname].set_linewidth(0.8)
                        if 'cmedians' in parts:
                            parts['cmedians'].set_edgecolor('black')
                            parts['cmedians'].set_linewidth(2.0)
                    elif len(clean) == 1:
                        ax.scatter([x_center], clean, color=box_color, s=40, zorder=3)
                else:
                    if len(clean) > 0:
                        rgb = mcolors.to_rgb(box_color)
                        ax.boxplot(clean, positions=[x_center], widths=box_width,
                                   patch_artist=True, zorder=3,
                                   medianprops=dict(color='black', lw=1.8),
                                   boxprops=dict(facecolor=(*rgb, 0.3),
                                                 edgecolor=box_color, lw=1.2),
                                   whiskerprops=dict(color=box_color, lw=1.0),
                                   capprops=dict(color=box_color, lw=1.0),
                                   showfliers=False)

            if show_scatter:
                for fly_i in range(n_flies):
                    fly_xs, fly_vs = [], []
                    for w_idx, win in enumerate(windows):
                        v = float(window_dict[win].isel(region=reg_i, fly=fly_i).values)
                        if np.isnan(v):
                            continue
                        fly_xs.append(region_center + x_offsets[w_idx])
                        fly_vs.append(v)
                    if len(fly_xs) > 1:
                        ax.plot(fly_xs, fly_vs,
                                color=FLY_COLOURS[fly_i % len(FLY_COLOURS)],
                                lw=0.9, zorder=1)

            xtick_pos.append(region_center)
            xtick_labels.append(region_coords[reg_i])

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=13)
        for yval, col, ls in (ref_lines or [(0, 'gray', '--'), (1, 'steelblue', ':')]):
            ax.axhline(yval, color=col, lw=0.6, ls=ls)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel('Regions', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.5)
        ax.tick_params(width=1.5, labelsize=10)

    win_handles    = [mpatches.Patch(color=WINDOW_COLORS[w], label=WINDOW_DISPLAY[w])
                      for w in windows]
    legend_handles = win_handles
    if show_scatter:
        fly_handles    = [mpatches.Patch(color=FLY_COLOURS[i % len(FLY_COLOURS)],
                                         label=f'Fly {i}') for i in range(n_flies)]
        legend_handles = win_handles + fly_handles
    axes[0].legend(handles=legend_handles,
                   fontsize=9, loc='upper right', framealpha=0.8)
    axes[0].set_title(title or ylabel, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_metric_violin(window_dict, ylabel='metric', regions=None,
                        title=None, ref_lines=None):
    """
    Violin plot (no individual scatter) of per-fly values.
    Convenience wrapper around plot_metric_scatter_box(violin=True, show_scatter=False).
    """
    return plot_metric_scatter_box(window_dict, ylabel=ylabel, regions=regions,
                                   title=title, show_scatter=False, violin=True,
                                   ref_lines=ref_lines)


def plot_zscored_timeseries(ds_list, lfmrate=30, ncols=5,
                             cond_stim1='xOx', cond_stim2='xxF',
                             cond_combo='xOF', data_var='deltaF_rg'):
    """
    One subplot per region: grand-mean z-scored timeseries for stim1, stim2,
    combo, and their sum (stim1+stim2, dashed).

    Each condition's grand-mean is z-scored independently across the full
    trial axis (shape-preserving, amplitude-normalised).
    """
    def _grand_mean(cond):
        fly_means = []
        for ds in ds_list:
            mask = ds.condition == cond
            if not bool(mask.any()):
                continue
            fly_means.append(ds[data_var].sel(trial=mask).mean('trial').values)
        if not fly_means:
            return None
        return np.nanmean(np.stack(fly_means, axis=0), axis=0)

    gm = {c: _grand_mean(c) for c in (cond_stim1, cond_stim2, cond_combo)}
    if any(v is None for v in gm.values()):
        print('Could not compute grand mean for one or more conditions.')
        return None

    def _zscore(arr):
        mu  = np.nanmean(arr, axis=0, keepdims=True)
        sig = np.nanstd( arr, axis=0, keepdims=True)
        sig[sig == 0] = 1.0
        return (arr - mu) / sig

    zs = {c: _zscore(gm[c]) for c in gm}
    zs['pred'] = _zscore(gm[cond_stim1] + gm[cond_stim2])

    region_coords = ds_list[0].region.values
    n_regions     = len(region_coords)
    time_axis     = np.arange(zs[cond_combo].shape[0]) / lfmrate

    nrows = int(np.ceil(n_regions / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    plot_order = [cond_combo, cond_stim1, cond_stim2, 'pred']
    linestyles = {cond_combo: '-', cond_stim1: '-', cond_stim2: '-', 'pred': '--'}
    linewidths = {cond_combo: 1.8, cond_stim1: 1.2, cond_stim2: 1.2, 'pred': 1.2}
    labels     = {cond_stim1: cond_stim1, cond_stim2: cond_stim2,
                  cond_combo: cond_combo, 'pred': f'{cond_stim1}+{cond_stim2} (pred)'}

    for ax, r in zip(axes, range(n_regions)):
        for key in plot_order:
            trace = zs[key][:, r].copy().astype(float)
            trace[~np.isfinite(trace)] = np.nan
            ax.plot(time_axis, trace,
                    color=COND_COLORS.get(key, 'white'),
                    ls=linestyles[key], lw=linewidths[key],
                    zorder=3 if key == cond_combo else 2)
        ax.axhline(0, color='gray', lw=0.4, ls='--', zorder=1)
        ax.set_title(region_coords[r], fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=5)
        ax.set_xticks([0, 10, 20, 30])

    for ax in axes[n_regions:]:
        ax.set_visible(False)

    axes[0].legend(
        handles=[Line2D([0], [0], color=COND_COLORS.get(k, 'white'),
                         lw=linewidths[k], ls=linestyles[k], label=labels[k])
                 for k in plot_order],
        fontsize=6, loc='upper right')
    fig.suptitle(
        f'Z-scored grand-mean timeseries\n'
        f'{cond_stim1}, {cond_stim2}, {cond_combo}, and {cond_stim1}+{cond_stim2}',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: REGION PANEL — per-region N×3 grid (timeseries | AUC | deriv max)
# ══════════════════════════════════════════════════════════════════════════════

def plot_region_overlay(ax, ds_list, region_idx, conditions,
                        lfmrate=30, data_var='deltaF_rg', skip_frames=0):
    """
    Grand mean ± SEM timeseries for selected conditions at one region.
    Colors and labels are derived from condition_color / condition_label.
    Draws a dashed vertical line at the mean odour onset if available.

    Parameters
    ----------
    ax          : matplotlib Axes
    region_idx  : int, 0-based
    conditions  : list[str]
    skip_frames : int  number of leading frames to omit from the plot (default 0)
    """
    sl = slice(skip_frames, None)
    for cond in conditions:
        traces = []
        for ds in ds_list:
            mean_da, n = get_condition_mean(ds, cond, data_var=data_var)
            if mean_da is None or n == 0:
                continue
            traces.append(mean_da.isel(region=region_idx).values)
        if not traces:
            continue
        mat        = np.stack(traces, axis=0)
        grand_mean = mat.mean(axis=0)
        sem        = mat.std(axis=0) / np.sqrt(mat.shape[0])
        t          = np.arange(len(grand_mean)) / lfmrate
        color      = condition_color(cond)
        ax.plot(t[sl], grand_mean[sl], color=color, lw=2.0, label=condition_label(cond), zorder=4)
        ax.fill_between(t[sl], grand_mean[sl] - sem[sl], grand_mean[sl] + sem[sl],
                        color=color, alpha=0.35, zorder=3)

    # stimulus window rectangles — walk most back, then vis, then odor
    _stim_spans = [
        ('fw_framestarts_within_trial',   'fw_frameends_within_trial',         STIM_COLORS['fw'],   1),
        ('vis_framestarts_within_trial',  'vis_patt_frameoff_within_trial',    STIM_COLORS['vis'],  2),
        ('odor_framestarts_within_trial', 'odor_frameends_within_trial',       STIM_COLORS['odor'], 3),  # noqa: E501
    ]
    for ds in ds_list:
        for start_key, end_key, color, zord in _stim_spans:
            if start_key in ds.attrs and end_key in ds.attrs:
                t0 = float(np.nanmean(ds.attrs[start_key])) / lfmrate
                t1 = float(np.nanmean(ds.attrs[end_key]))   / lfmrate
                ax.axvspan(t0, t1, color=color, alpha=0.2, zorder=zord, linewidth=0)
        break

    ax.axhline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(2.0)
    ax.tick_params(width=1.5, labelsize=10)
    ax.legend(fontsize=9, loc='upper right', frameon=False)


def plot_selected_regions(ds_list, region_indices,
                           ts_conditions, group_auc, auc_conditions,
                           group_deriv, deriv_conditions,
                           group_peak=None, peak_conditions=None,
                           cond_stim1='xOx', cond_stim2='xxF', cond_combo='xOF',
                           violin=True, show_scatter=False,
                           box_width=0.8, lfmrate=30, data_var='deltaF_rg',
                           sum_skip_frames=0):
    """
    N×6 grid — one row per region.
    Columns: timeseries overlay | AUC | peak response | rise amplitude | R² | summation

    Parameters
    ----------
    region_indices         : list[int]  0-based region indices
    ts_conditions          : list[str]  conditions for timeseries column
    group_auc              : dict {label: DataArray(fly, region)}
    auc_conditions         : list[str]
    group_deriv            : dict {label: DataArray(fly, region)}
    deriv_conditions       : list[str]
    group_peak             : dict {label: DataArray(fly, region)}  or None to skip column
    peak_conditions        : list[str]  or None
    cond_stim1/stim2/combo : conditions for the z-scored summation column
    violin                 : bool  draw violins instead of boxes
    show_scatter           : bool  overlay individual fly dots and lines
    """
    n             = len(region_indices)
    region_coords = next(iter(group_auc.values())).coords['region'].values

    auc_labels   = [c for c in auc_conditions   if c in group_auc]
    deriv_labels = [c for c in deriv_conditions if c in group_deriv]
    auc_colors   = {l: condition_color(l) for l in auc_labels}
    deriv_colors = {l: condition_color(l) for l in deriv_labels}

    _has_peak   = group_peak is not None and peak_conditions is not None
    peak_labels = ([c for c in peak_conditions if c in group_peak]
                   if _has_peak else [])
    peak_colors = {l: condition_color(l) for l in peak_labels}

    def _cond_xpos(labels):
        n_c = len(labels)
        return np.arange(n_c) * 1.0 - (n_c * 1.0 - 1.0) / 2

    def _grand_mean(cond, reg_i):
        traces = []
        for ds in ds_list:
            mean_da, n_t = get_condition_mean(ds, cond, data_var=data_var)
            if mean_da is None or n_t == 0:
                continue
            traces.append(mean_da.isel(region=reg_i).values)
        return np.nanmean(np.stack(traces, axis=0), axis=0) if traces else None

    def _style_ax(ax):
        KCpreferred_FigSettings(axes=ax)

    def _snap(ax, which='both'):
        """Extend limits one tick step beyond the outermost visible tick.

        Only call once per shared-axis group (e.g. row 0 for sharey columns)
        so the extension does not compound across rows.
        """
        pairs = []
        if which in ('x', 'both'):
            pairs.append((ax.get_xlim, ax.get_xticks, ax.set_xlim))
        if which in ('y', 'both'):
            pairs.append((ax.get_ylim, ax.get_yticks, ax.set_ylim))
        for get_lim, get_ticks, set_lim in pairs:
            lo, hi = get_lim()
            ticks = get_ticks()
            vis = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
            if len(vis) < 2:
                continue
            step = vis[1] - vis[0]
            set_lim(vis[0] - step, vis[-1] + step)

    _r2_xtick_lbl = (f'{condition_label(cond_stim1)} +\n'
                     f'{condition_label(cond_stim2)}')

    _ts_order  = [cond_combo, cond_stim1, cond_stim2, 'pred']
    _ts_colors = {cond_combo: condition_color(cond_combo),
                  cond_stim1: condition_color(cond_stim1),
                  cond_stim2: condition_color(cond_stim2),
                  'pred':     COND_COLORS['pred']}
    _ts_ls     = {cond_combo: '-',  cond_stim1: '-',  cond_stim2: '-',  'pred': '--'}
    _ts_lw     = {cond_combo: 2.0, cond_stim1: 1.5, cond_stim2: 1.5, 'pred': 1.5}
    _combo_lbl = condition_label(cond_combo)
    _ts_lbl    = {cond_combo: f'obs ({_combo_lbl})',
                  cond_stim1: condition_label(cond_stim1),
                  cond_stim2: condition_label(cond_stim2),
                  'pred':     f'pred ({_combo_lbl})'}

    fig, axes = plt.subplots(n, 6, figsize=(34, 4.5 * n),
                              sharey=False,
                              gridspec_kw={'wspace': 0.4, 'hspace': 0.55,
                                           'width_ratios': [3, 2, 2, 2, 2, 3]})
    if n == 1:
        axes = axes[np.newaxis, :]

    # Share y-axes within each of columns 0–4 (col 5 is independent per row)
    for col in range(5):
        for row in range(1, n):
            axes[row, col].sharey(axes[0, col])

    is_last_row = lambda row: row == n - 1

    all_r2_vals = []   # accumulate per-fly R² across all rows for col-4 ylim

    for row, reg_i in enumerate(region_indices):
        region_name                            = str(region_coords[reg_i])
        ax_ts, ax_peak, ax_auc, ax_deriv, ax_r2, ax_sum = axes[row]

        # col 1 — timeseries overlay
        plot_region_overlay(ax_ts, ds_list, reg_i, ts_conditions,
                            lfmrate=lfmrate, data_var=data_var,
                            skip_frames=sum_skip_frames)
        ax_ts.set_title(region_name, fontsize=14, fontweight='bold')
        ax_ts.set_ylabel('ΔF/F', fontsize=16)
        if row == 0:
            ax_ts.legend(fontsize=12, loc='upper right', frameon=False)
        else:
            leg = ax_ts.get_legend()
            if leg:
                leg.remove()
        if not is_last_row(row):
            ax_ts.set_xlabel('')
            ax_ts.tick_params(labelbottom=False)
        else:
            ax_ts.set_xlabel('Time (s)', fontsize=16)

        # col 2 — AUC
        _scatter_box_axes(ax_auc, group_auc, auc_labels, [reg_i],
                          region_coords, auc_colors, False, box_width,
                          violin=violin)
        ax_auc.set_xticks(_cond_xpos(auc_labels))
        ax_auc.set_xticklabels([condition_label(l).split('\n')[0] for l in auc_labels],
                                rotation=0, ha='center', fontsize=11)
        ax_auc.set_ylabel('AUC', fontsize=16)
        ax_auc.set_xlabel('')
        if row == 0:
            ax_auc.set_title('AUC', fontsize=14, fontweight='bold')
        if not is_last_row(row):
            ax_auc.tick_params(labelbottom=False)

        # col 3 — peak odor response
        if _has_peak and peak_labels:
            _scatter_box_axes(ax_peak, group_peak, peak_labels, [reg_i],
                              region_coords, peak_colors, False, box_width,
                              violin=violin)
            ax_peak.set_xticks(_cond_xpos(peak_labels))
            ax_peak.set_xticklabels([condition_label(l).split('\n')[0] for l in peak_labels],
                                    rotation=0, ha='center', fontsize=11)
        ax_peak.set_ylabel('peak ΔF/F', fontsize=16)
        ax_peak.set_xlabel('')
        if row == 0:
            ax_peak.set_title('Peak Odor Response', fontsize=14, fontweight='bold')
        if not is_last_row(row):
            ax_peak.tick_params(labelbottom=False)

        # col 4 — rise to peak
        _scatter_box_axes(ax_deriv, group_deriv, deriv_labels, [reg_i],
                          region_coords, deriv_colors, False, box_width,
                          violin=violin)
        ax_deriv.set_xticks(_cond_xpos(deriv_labels))
        ax_deriv.set_xticklabels([condition_label(l).split('\n')[0] for l in deriv_labels],
                                  rotation=0, ha='center', fontsize=11)
        ax_deriv.set_ylabel('rise amplitude', fontsize=16)
        ax_deriv.set_xlabel('')
        if row == 0:
            ax_deriv.set_title('Rise to Peak Odor Response', fontsize=14, fontweight='bold')
        if not is_last_row(row):
            ax_deriv.tick_params(labelbottom=False)

        # col 4 — per-fly R² violin (lowpass, coeff = 1, no fitting)
        r2_vals = []
        for ds in ds_list:
            m1, n1 = get_condition_mean(ds, cond_stim1, data_var=data_var)
            m2, n2 = get_condition_mean(ds, cond_stim2, data_var=data_var)
            mc, nc = get_condition_mean(ds, cond_combo,  data_var=data_var)
            if m1 is None or m2 is None or mc is None or n1 == 0 or n2 == 0 or nc == 0:
                continue
            lp1  = lowpass_filter(m1.isel(region=reg_i).values, lfmrate=lfmrate)
            lp2  = lowpass_filter(m2.isel(region=reg_i).values, lfmrate=lfmrate)
            lpc  = lowpass_filter(mc.isel(region=reg_i).values, lfmrate=lfmrate)
            pred = lp1 + lp2
            ss_res = np.nansum((lpc - pred) ** 2)
            ss_tot = np.nansum((lpc - np.nanmean(lpc)) ** 2)
            r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        all_r2_vals.extend(r2_vals)
        if r2_vals:
            r2_arr = np.array(r2_vals)
            parts  = ax_r2.violinplot([r2_arr], positions=[0], widths=box_width,
                                       showmedians=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(condition_color(cond_combo))
                pc.set_alpha(1.0)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_color('black')
                    parts[key].set_linewidth(1.5)
            if show_scatter:
                jitter = np.random.uniform(-0.08, 0.08, len(r2_arr))
                for i, r2v in enumerate(r2_arr):
                    ax_r2.scatter(jitter[i], r2v,
                                  color=FLY_COLOURS[i % len(FLY_COLOURS)],
                                  zorder=3, s=30)
                if row == 0:
                    fly_handles = [mpatches.Patch(color=FLY_COLOURS[i % len(FLY_COLOURS)],
                                                  label=ds.attrs.get('fly_id', f'ds_list[{i}]').split('_')[0])
                                   for i, ds in enumerate(ds_list)]
                    ax_r2.legend(handles=fly_handles, fontsize=12, loc='lower right',
                                 frameon=False)
            median_r2 = float(np.nanmedian(r2_arr))
            ax_r2.text(0.03, 0.20, f'R² = {median_r2:.2f}',
                       transform=ax_r2.transAxes,
                       ha='left', va='bottom', fontsize=13, fontweight='bold')
            ax_r2.set_xlim(-0.6, 0.6)
        ax_r2.set_xticks([])
        ax_r2.set_xlabel('')
        ax_r2.set_ylabel('R²', fontsize=16)
        if row == 0:
            ax_r2.set_title('summation R²', fontsize=14, fontweight='bold')
        _style_ax(ax_ts)
        _style_ax(ax_auc)
        _style_ax(ax_peak)
        _style_ax(ax_deriv)
        _style_ax(ax_r2)

        # col 6 — lowpass grand mean timeseries (not z-scored, coeff = 1)
        gm1 = _grand_mean(cond_stim1, reg_i)
        gm2 = _grand_mean(cond_stim2, reg_i)
        gmc = _grand_mean(cond_combo,  reg_i)
        if gm1 is not None and gm2 is not None and gmc is not None:
            lp1  = lowpass_filter(gm1, lfmrate=lfmrate)
            lp2  = lowpass_filter(gm2, lfmrate=lfmrate)
            lpc  = lowpass_filter(gmc, lfmrate=lfmrate)
            pred = lp1 + lp2
            ss_res = np.nansum((lpc - pred) ** 2)
            ss_tot = np.nansum((lpc - np.nanmean(lpc)) ** 2)
            r2_gm  = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            traces = {cond_stim1: lp1, cond_stim2: lp2, cond_combo: lpc, 'pred': pred}
            t = np.arange(len(lpc)) / lfmrate
            sl = slice(sum_skip_frames, None)
            for key in _ts_order:
                ax_sum.plot(t[sl], traces[key][sl], color=_ts_colors[key],
                            ls=_ts_ls[key], lw=_ts_lw[key], label=_ts_lbl[key])
            ax_sum.text(0.03, 0.97, f'R² = {r2_gm:.2f}', transform=ax_sum.transAxes,
                        ha='left', va='top', fontsize=13, fontweight='bold')
            if row == 0:
                ax_sum.legend(fontsize=12, loc='upper right', frameon=False)
        ax_sum.set_ylabel('ΔF/F', fontsize=16)
        if row == 0:
            ax_sum.set_title('summation', fontsize=14, fontweight='bold')
        if not is_last_row(row):
            ax_sum.set_xlabel('')
            ax_sum.tick_params(labelbottom=False)
        else:
            ax_sum.set_xlabel('Time (s)', fontsize=16)
        _style_ax(ax_sum)
        for _ax in (ax_ts, ax_auc, ax_peak, ax_deriv, ax_r2, ax_sum):
            _ax.tick_params(which='major', direction='in', labelsize=12, length=5.25)
            _ax.tick_params(which='minor', direction='in', length=3.0)

    # y-axes are shared per column — snap once on row 0, propagates to all rows
    for col in range(6):
        _snap(axes[0, col], 'y')
    # x-axes are independent per row — snap per row for timeseries columns only
    for row in range(n):
        _snap(axes[row, 0], 'x')
        _snap(axes[row, 5], 'x')

    # col 5 (index 4) — force ymax = 1 (R² ceiling); ymin follows the data
    lo, _ = axes[0, 4].get_ylim()
    axes[0, 4].set_ylim(lo, 1.0)

    fig.tight_layout()
    return fig
