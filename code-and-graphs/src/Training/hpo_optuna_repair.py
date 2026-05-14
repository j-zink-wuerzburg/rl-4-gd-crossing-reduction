"""
Optuna-based hyperparameter optimization for repair PPO training.

This script starts from a repair config file, applies sampled PPO hyperparameters
on top of that config, runs `run_repair_ppo`, and minimizes the repair evaluation
metric `mean_best_local_ratio`.

Example:
  python src/Training/hpo_optuna_repair.py \
    --study_name repair_local_ratio_v1 \
    --config configs/config_repair_ppo_best64k.json \
    --n_trials 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from Training.train_repair_ppo import (  # noqa: E402
    _deep_copy_dict,
    load_graphs_from_list_file,
    load_graphs_from_split_count,
    load_mixed_graphs_from_counts,
    load_repair_config_bundle,
    run_repair_ppo,
)


def define_search_space(trial: optuna.trial.Trial, base_config: dict) -> dict:
    """Sample PPO hyperparameters around the loaded repair config."""
    ppo_cfg = base_config["ppo"]
    run_cfg = base_config["run"]

    base_lr = float(ppo_cfg["learning_rate"])
    base_n_steps = int(ppo_cfg["n_steps"])
    base_batch_size = int(ppo_cfg["batch_size"])
    base_ent_coef = float(ppo_cfg["ent_coef"])
    base_vf_coef = float(ppo_cfg["vf_coef"])
    base_gae_lambda = float(ppo_cfg["gae_lambda"])
    base_clip_range = float(ppo_cfg["clip_range"])

    learning_rate = trial.suggest_float(
        "learning_rate",
        max(1e-6, base_lr / 10.0),
        base_lr * 10.0,
        log=True,
    )

    n_steps = trial.suggest_int(
        "n_steps",
        max(64, base_n_steps // 2),
        max(256, base_n_steps * 4),
        step=max(32, base_n_steps // 4),
    )

    max_batch = max(64, min(8192, n_steps * int(run_cfg["n_envs"])))
    batch_size = trial.suggest_int(
        "batch_size",
        max(32, base_batch_size // 2),
        max_batch,
        step=max(32, base_batch_size // 4),
    )
    batch_size = min(batch_size, max_batch)

    ent_coef = trial.suggest_float(
        "ent_coef",
        max(1e-6, base_ent_coef / 10.0),
        max(base_ent_coef * 10.0, 1e-4),
        log=True,
    )

    vf_coef = trial.suggest_float(
        "vf_coef",
        max(1e-3, base_vf_coef / 4.0),
        max(base_vf_coef * 4.0, 1e-2),
        log=True,
    )

    gae_low = max(0.80, min(base_gae_lambda - 0.05, 0.99))
    gae_high = min(0.99, max(base_gae_lambda + 0.05, gae_low + 0.01))
    gae_lambda = trial.suggest_float(
        "gae_lambda",
        gae_low,
        gae_high,
        step=0.01,
    )

    clip_low = max(0.05, min(base_clip_range - 0.1, 0.3))
    clip_high = min(0.5, max(base_clip_range + 0.1, clip_low + 0.05))
    clip_range = trial.suggest_float(
        "clip_range",
        clip_low,
        clip_high,
        step=0.05,
    )

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
    }


def _build_trial_args(base_flat_config: dict, hparams: dict, trial_steps: int, trial_seed: int, trial_log_dir: Path):
    """Build an argparse.Namespace that satisfies run_repair_ppo's expected inputs."""
    namespace = argparse.Namespace()

    for key, value in base_flat_config.items():
        setattr(namespace, key.replace("-", "_"), value)

    namespace.seeds = [int(trial_seed)]
    namespace.ppo_steps = int(trial_steps)
    namespace.checkpoint_root = str(trial_log_dir / "checkpoints")
    namespace.output_json = str(trial_log_dir / "summary.json")
    namespace.output_jsonl = str(trial_log_dir / "events.jsonl")
    namespace.exp_name = None
    namespace.resume_from = None

    for key, value in hparams.items():
        setattr(namespace, key, value)

    return namespace


def _load_graphs_from_config(base_config: dict, flat_config: dict):
    """Load train/eval graphs according to the loaded repair config."""
    sampling = flat_config
    use_mixed_counts = any(
        int(value) > 0
        for value in (
            sampling["train_rome_count"],
            sampling["train_ba_count"],
            sampling["eval_rome_count"],
            sampling["eval_ba_count"],
        )
    )

    if use_mixed_counts:
        train_graphs, eval_graphs, mix_config = load_mixed_graphs_from_counts(
            argparse.Namespace(
                train_rome_count=sampling["train_rome_count"],
                train_ba_count=sampling["train_ba_count"],
                eval_rome_count=sampling["eval_rome_count"],
                eval_ba_count=sampling["eval_ba_count"],
            )
        )
        return train_graphs, eval_graphs, mix_config

    dataset = sampling["dataset"]
    train_graphs = []
    eval_graphs = []

    if sampling.get("train_list_file"):
        _, train_graphs, _ = load_graphs_from_list_file(sampling["train_list_file"], dataset)
    elif int(sampling["train_count"]) > 0:
        train_graphs = load_graphs_from_split_count("train", dataset, sampling["train_count"])

    if sampling.get("eval_list_file"):
        _, eval_graphs, _ = load_graphs_from_list_file(sampling["eval_list_file"], dataset)
    elif int(sampling["eval_count"]) > 0:
        eval_graphs = load_graphs_from_split_count("test", dataset, sampling["eval_count"])

    return train_graphs, eval_graphs, None


def _objective_factory(base_flat_config: dict, base_raw_config: dict, train_graphs, eval_graphs, args):
    metric_name = "mean_best_local_ratio"

    def objective(trial: optuna.trial.Trial) -> float:
        print(f"\n{'=' * 80}")
        print(f"[REPAIR HPO] Trial {trial.number} starting...")
        print(f"{'=' * 80}\n")

        hparams = define_search_space(trial, base_raw_config)
        print(f"[REPAIR HPO Trial {trial.number}] Hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")

        trial_seed = int(base_raw_config["run"]["seeds"][0]) + int(trial.number)
        trial_dir = Path(args.log_dir) / args.study_name / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_config = _deep_copy_dict(base_raw_config)
        trial_config["ppo"].update(hparams)
        trial_config["run"]["seeds"] = [trial_seed]
        trial_config["run"]["ppo_steps"] = int(args.trial_steps)
        trial_config["training"]["checkpoint_root"] = str(trial_dir / "checkpoints")
        trial_config["io"]["output_json"] = str(trial_dir / "summary.json")
        trial_config["io"]["output_jsonl"] = str(trial_dir / "events.jsonl")

        with (trial_dir / "trial_config.json").open("w", encoding="utf-8") as handle:
            json.dump(trial_config, handle, indent=2)

        trial_args = _build_trial_args(
            base_flat_config={
                **base_flat_config,
                "dataset": base_flat_config["dataset"],
                "train_count": base_flat_config["train_count"],
                "eval_count": base_flat_config["eval_count"],
                "train_rome_count": base_flat_config["train_rome_count"],
                "train_ba_count": base_flat_config["train_ba_count"],
                "eval_rome_count": base_flat_config["eval_rome_count"],
                "eval_ba_count": base_flat_config["eval_ba_count"],
                "train_list_file": base_flat_config["train_list_file"],
                "eval_list_file": base_flat_config["eval_list_file"],
            },
            hparams=hparams,
            trial_steps=int(args.trial_steps),
            trial_seed=trial_seed,
            trial_log_dir=trial_dir,
        )

        try:
            payload = run_repair_ppo(
                train_graphs,
                eval_graphs,
                trial_seed,
                trial_args,
                base_config=base_raw_config,
            )
            metric_value = payload["best_checkpoint_metric"]
            print(
                f"[REPAIR HPO Trial {trial.number}] {metric_name}: {metric_value:.6f} "
                f"(seed={trial_seed})"
            )
            return metric_value
        except optuna.TrialPruned:
            print(f"\n[REPAIR HPO Trial {trial.number}] Pruned early")
            raise
        except Exception as exc:
            print(f"\n[REPAIR HPO Trial {trial.number}] FAILED with error:")
            print(f"  {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            return 1e9

    return objective


def _setup_study(storage: str, study_name: str, n_startup_trials: int) -> optuna.study.Study:
    sampler = TPESampler(n_startup_trials=int(n_startup_trials), seed=42)
    pruner = MedianPruner(
        n_startup_trials=int(n_startup_trials),
        n_warmup_steps=max(1, int(n_startup_trials / 2)),
        interval_steps=5,
    )
    return optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for repair PPO training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config_repair_ppo_best64k.json"),
        help="Base repair config file",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: sqlite:///hpo_studies/{study_name}.db)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=None,
        help="Timesteps per trial (default: ppo_steps from the config)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="hpo_repair_results",
        help="Directory for trial outputs and summary files",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of random startup trials before TPE",
    )

    args = parser.parse_args()

    flat_config, raw_config = load_repair_config_bundle(args.config)
    trial_steps = int(args.trial_steps or flat_config["ppo_steps"])

    if args.storage is None:
        os.makedirs("hpo_studies", exist_ok=True)
        args.storage = f"sqlite:///hpo_studies/{args.study_name}.db"

    train_graphs, eval_graphs, mix_config = _load_graphs_from_config(raw_config, flat_config)
    if not train_graphs:
        raise ValueError("No training graphs were loaded from the selected config")
    if not eval_graphs:
        raise ValueError("No evaluation graphs were loaded from the selected config")

    print(f"\n{'=' * 80}")
    print("[REPAIR HPO] Starting Optuna Hyperparameter Optimization")
    print(f"{'=' * 80}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials to run: {args.n_trials}")
    print(f"Trial timesteps: {trial_steps:,}")
    print(f"Base config: {args.config}")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Eval graphs: {len(eval_graphs)}")
    if mix_config is not None:
        print(f"Mixed counts: {mix_config}")
    print(f"Log directory: {args.log_dir}")
    print(f"{'=' * 80}\n")

    study = _setup_study(args.storage, args.study_name, args.n_startup_trials)

    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]

    print("[REPAIR HPO] Study status:")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    if completed_trials:
        best_trial = study.best_trial
        print(f"  Best value so far: {best_trial.value:.6f} (trial {best_trial.number})")
    print()

    objective = _objective_factory(
        base_flat_config=flat_config,
        base_raw_config=raw_config,
        train_graphs=train_graphs,
        eval_graphs=eval_graphs,
        args=args,
    )

    try:
        study.optimize(
            objective,
            n_trials=int(args.n_trials),
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("\n\n[REPAIR HPO] Optimization interrupted by user")

    print(f"\n{'=' * 80}")
    print("[REPAIR HPO] Optimization Complete")
    print(f"{'=' * 80}\n")

    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (lower is better): {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    study_dir = Path(args.log_dir) / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    best_hparams_path = study_dir / "best_hyperparams.json"
    best_hparams_path.write_text(json.dumps(study.best_trial.params, indent=2), encoding="utf-8")
    print(f"\nBest hyperparams saved to: {best_hparams_path}")

    summary_path = study_dir / "study_summary.json"
    summary = {
        "study_name": args.study_name,
        "direction": "minimize",
        "objective_metric": "mean_best_local_ratio",
        "n_trials_requested": int(args.n_trials),
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "n_trials_failed": len(failed_trials),
        "best_value": float(study.best_trial.value),
        "best_params": study.best_trial.params,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Study summary saved to: {summary_path}")

    trials_csv = study_dir / "all_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)
    print(f"All trials saved to: {trials_csv}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()