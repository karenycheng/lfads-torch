# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda create --name lfads-torch python=3.9
conda activate lfads-torch
pip install -e .
pre-commit install
```

## Commands

**Run a single model:**
```bash
python scripts/run_single.py
```

**Run a hyperparameter sweep (random search):**
```bash
python scripts/run_multi.py
```

**Run population-based training (PBT):**
```bash
python scripts/run_pbt.py
```

**Linting / formatting (enforced by pre-commit):**
```bash
pre-commit run --all-files
# Individual tools:
black .
isort --profile black .
flake8 --max-line-length 88 --extend-ignore E203 .
```

## Architecture Overview

`lfads-torch` is a PyTorch Lightning + Hydra implementation of LFADS (Latent Factor Analysis via Dynamical Systems) — a variational sequential autoencoder for denoising high-dimensional neural spiking data.

### Data flow

HDF5 file → `BasicDataModule` → batches of `SessionBatch` namedtuples → `LFADS.forward()` → `SessionOutput` namedtuples → saved to HDF5 via posterior sampling.

- **`SessionBatch`** (`tuples.py`): `(encod_data, recon_data, ext_input, truth, sv_mask)`. The encod/recon split enables held-out neuron prediction.
- **`SessionOutput`** (`tuples.py`): `(output_params, factors, ic_mean, ic_std, co_means, co_stds, gen_states, gen_init, gen_inputs, con_states)`.

### Model structure (`model.py`, `modules/`)

`LFADS` (a `pl.LightningModule`) wires together:

1. **`readin`** — `nn.ModuleList` of per-session linear projections into `encod_data_dim`.
2. **`Encoder`** (`modules/encoder.py`) — bidirectional GRU producing `ic_mean/ic_std` (initial condition) and `ci` (controller inputs).
3. **`Decoder`** (`modules/decoder.py`) — unrolls a generator GRU, optionally driven by a controller that produces `co` (controller outputs). Outputs `factors`.
4. **`readout`** — `nn.ModuleList` of per-session linear projections from `fac_dim` to output channels.
5. **`reconstruction`** — `nn.ModuleList` of loss modules (one per session). Available implementations in `modules/recons.py`: `Poisson`, `PoissonBPS`, `MSE`, `Gaussian`, `Gamma`, `ZeroInflatedGamma`.
6. **`ic_prior` / `co_prior`** (`modules/priors.py`): `MultivariateNormal`, `AutoregressiveMultivariateNormal`, `MultivariateStudentT`, or `Null` (for non-variational models).

### Training pipeline (`run_model.py`)

`run_model()` is the single entry point for all training modes. It uses Hydra to compose configs, instantiates all PyTorch Lightning objects, calls `trainer.fit()`, then runs posterior sampling via `post_run/analysis.py:run_posterior_sampling()`. The run scripts (`run_single.py`, `run_pbt.py`) set a `RUN_DIR`, `os.chdir` into it, then call `run_model()` — this is why relative paths in configs resolve correctly.

### Config system

Configs are composed with Hydra. The main entry configs (`configs/single.yaml`, `configs/pbt.yaml`, `configs/multi.yaml`) use Hydra defaults to compose a `model` + `datamodule` config. Override them via the `overrides` dict passed to `run_model()`:

```python
run_model(overrides={"datamodule": "my_datamodule", "model": "my_model"})
```

Custom Hydra resolvers registered in `run_model.py`: `relpath`, `max`, `sum`.

### Augmentations (`modules/augmentations.py`)

`AugmentationStack` holds two ordered lists of transforms: `batch_transforms` (called in `process_batch`) and `loss_transforms` (called in `process_losses`). Each transform must implement `process_batch` and/or `process_losses`. Key implementations:

- `CoordinatedDropout` — masks encoder input, blocks gradients on masked outputs to fight identity overfitting.
- `SampleValidation` — holds out a random fraction of input spikes for validation.
- `SelectiveBackpropThruTime` — masks NaN time points in calcium/sub-frame data.
- `SpikeJitter`, `TemporalShift` — augmentations for temporal robustness.

Separate `train_aug_stack` and `infer_aug_stack` are applied automatically in `_shared_step` and `predict_step`.

### Multi-session support

Multi-session runs use `readin`/`readout`/`reconstruction` as `nn.ModuleList` with one entry per session. The `BasicDataModule` loads all sessions matching `datafile_pattern` (a glob) and the `LFADS.forward()` concatenates sessions, runs them through the shared encoder/decoder, then splits back by session index. Multi-session models work best with PCR-based initialization — see `tutorials/multisession/`.

### Hyperparameter search (Ray Tune)

`run_pbt.py` uses `BinaryTournamentPBT` (custom scheduler in `extensions/tune.py`) and `ImprovementRatioStopper`. The `HyperParam` dataclass defines search bounds and exploration weights. The best model is identified by `valid/recon_smth` (exponentially-smoothed validation reconstruction loss).

### Posterior sampling outputs

After training, `run_posterior_sampling()` runs `num_samples` (default 50) stochastic forward passes and averages them. Results are saved to HDF5 alongside the original data file. Each session produces one file `lfads_output_{session_name}.h5` containing all `SessionOutput` fields prefixed by split (`train_`, `valid_`, `test_`).

## Key hyperparameters to tune

Regularization is ramped in over `l2_increase_epoch` / `kl_increase_epoch` epochs starting at `l2_start_epoch` / `kl_start_epoch`. Early stopping is blocked until the ramp completes (`EarlyStoppingWithBurnInPeriod`). The primary PBT search targets: `lr_init`, `dropout_rate`, `cd_rate`, `kl_co_scale`, `kl_ic_scale`, `l2_gen_scale`, `l2_con_scale`.
