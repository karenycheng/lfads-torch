import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "pbt"
# Train on the z-scored ΔF/F data (recommended; per-region z-score before PCA).
# To train on the unscaled data instead, set DATASET_STR = "kcEXP00H_multisession".
DATASET_STR = "kcEXP00H_multisession_zscored"
# The model config is dataset-agnostic: its readin/readout/reconstruction read
# ${datamodule.datafile_pattern}, so the same PCR model config works for either
# datamodule. Keep it decoupled from DATASET_STR.
MODEL_STR = "kcEXP00H_multisession_PCR"
RUN_TAG = datetime.now().strftime("%y%m%d")
# Fresh, timestamped run directory every launch. We deliberately do NOT use
# resume=True: if a previous run was killed while every trial was PAUSED,
# resuming restores them all as PAUSED and nothing ever unpauses (deadlock).
RUN_DIR = (
    Path("/home/kyc_hpz8/Documents/lfads-torch/runs")
    / f"kcEXP00H_multisession_{datetime.now().strftime('%Y%m%d%H%M')}"
    / PROJECT_STR
    / DATASET_STR
    / RUN_TAG
)
HYPERPARAM_SPACE = {
    "model.lr_init": HyperParam(
        1e-4, 1e-3, explore_wt=0.3, enforce_limits=True, init=1e-3
    ),
    "model.dropout_rate": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "model.train_aug_stack.transforms.0.cd_rate": HyperParam(
        0.01, 0.99, explore_wt=0.3, enforce_limits=True, init=0.5, sample_fn="uniform"
    ),
    "model.kl_co_scale": HyperParam(1e-5, 1e-3, explore_wt=0.8),
    "model.kl_ic_scale": HyperParam(1e-5, 1e-3, explore_wt=0.8),
    "model.l2_gen_scale": HyperParam(1e-5, 1e-0, explore_wt=0.8),
    "model.l2_con_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
}
# ------------------------------


# Function to keep dropout and CD rates in-bounds
def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}
# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": MODEL_STR,
    # "logger.wandb_logger.project": PROJECT_STR,
    # "logger.wandb_logger.tags.1": DATASET_STR,
    # "logger.wandb_logger.tags.2": RUN_TAG,
}
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
tic = time.perf_counter()
# Run the hyperparameter search
metric = "valid/recon_smth"
num_trials = 20
perturbation_interval = 15
burn_in_period = 50 + 15
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/pbt.yaml",
        do_posterior_sample=False,
    ),
    metric=metric,
    mode="min",
    name=RUN_DIR.name,
    stop=ImprovementRatioStopper(
        num_trials=num_trials,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=6, gpu=0.25),
    num_samples=num_trials,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
print(f"PBT training complete. Elapsed: {time.perf_counter() - tic:.1f}s")
# Copy the best model to a new folder so it is easy to identify
best_model_dir = RUN_DIR / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory to this folder (usually handled by tune)
os.chdir(best_model_dir)
# Load the best model and run posterior sampling (skip training)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint._local_path).name
run_model(
    overrides=mandatory_overrides,
    checkpoint_dir=best_ckpt_dir,
    config_path="../configs/pbt.yaml",
    do_train=False,
)
