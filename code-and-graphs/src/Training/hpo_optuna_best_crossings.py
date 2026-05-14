"""
Optuna-based hyperparameter optimization for PPO graph layout training.

This script optimizes all PPO hyperparameters plus the environment step limit
and reset threshold. The objective is the final best crossing count, selected
according to the environment optimization_goal:

- optimization_goal == "local"  -> minimize best_local_crossings
- optimization_goal == "global" -> minimize best_global_crossings

Usage:
  python hpo_optuna_best_crossings.py \
    --study_name ppo_best_crossings_v1 \
    --config configs/config_ppo_pixel_move_over_edge.json \
    --n_trials 50 \
    --trial_steps 500000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Training.dataloader import load_split_dataset
from Training.train_ppo_pixel import (  # noqa: E402
    CurriculumSampler,
    CustomGraphLayoutExtractor,
    LinearCurriculumCallback,
    make_env,
)
from env import create_graph_layout_env


DEFAULT_SPACE = {
    "learning_rate": (1e-5, 3e-3),
    "n_steps": [128, 256, 512, 1024, 2048],
    "batch_size": [64, 128, 256, 384, 448, 512, 1024],
    "n_epochs": (3, 20),
    "gamma": (0.95, 0.9999),
    "gae_lambda": (0.80, 0.99),
    "clip_range": (0.10, 0.30),
    "ent_coef": (1e-6, 0.05),
    "vf_coef": (0.10, 1.0),
    "max_grad_norm": (0.3, 2.0),
    "step_limit": [256, 512, 768, 1024, 1536, 2048, 3072, 4096],
    "reset_unsuccessful_moves_threshold": [None, 16, 32, 64, 128, 256, 512],
}


class CrossingBestMetricCallback(BaseCallback):
    """Collect the best crossing metric from finished episodes and report it to Optuna."""

    def __init__(self, trial: optuna.trial.Trial, optimization_goal: str, report_interval: int = 1000):
        super().__init__()
        self.trial = trial
        self.optimization_goal = str(optimization_goal).lower().strip()
        self.report_interval = max(1, int(report_interval))
        self.episode_best_crossings: list[float] = []
        self._printed_info_keys = False

    def _metric_key(self) -> str:
        return "best_local_crossings" if self.optimization_goal == "local" else "best_global_crossings"

    def _extract_metric(self, info: dict) -> float | None:
        key = self._metric_key()
        value = info.get(key)
        if value is None and "episode" in info:
            value = info["episode"].get(key)
        if value is None:
            return None
        return float(np.asarray(value).squeeze())

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue

            if not self._printed_info_keys:
                print(f"[HPO] Terminal info keys: {sorted(info.keys())}")
                self._printed_info_keys = True

            metric = self._extract_metric(info)
            if metric is not None:
                self.episode_best_crossings.append(metric)

        if self.n_calls % self.report_interval == 0 and self.episode_best_crossings:
            recent = self.episode_best_crossings[-max(1, len(self.episode_best_crossings) // 10):]
            current_value = float(np.median(recent))
            self.trial.report(current_value, step=self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True


def _suggest_space(trial: optuna.trial.Trial, config: dict) -> dict:
    """Suggest PPO and env hyperparameters for one Optuna trial."""
    hpo_cfg = config.get("hpo", {})

    def cfg_range(name, default_low, default_high, log=False):
        cfg = hpo_cfg.get(name, {})
        return (
            cfg.get("low", default_low),
            cfg.get("high", default_high),
            cfg.get("log", log),
        )

    learning_rate_low, learning_rate_high, learning_rate_log = cfg_range("learning_rate", *DEFAULT_SPACE["learning_rate"], log=True)
    ent_low, ent_high, ent_log = cfg_range("ent_coef", *DEFAULT_SPACE["ent_coef"], log=True)
    vf_low, vf_high, vf_log = cfg_range("vf_coef", *DEFAULT_SPACE["vf_coef"], log=True)
    gamma_low, gamma_high, _ = cfg_range("gamma", *DEFAULT_SPACE["gamma"])
    gae_low, gae_high, _ = cfg_range("gae_lambda", *DEFAULT_SPACE["gae_lambda"])
    clip_low, clip_high, _ = cfg_range("clip_range", *DEFAULT_SPACE["clip_range"])
    max_grad_low, max_grad_high, max_grad_log = cfg_range("max_grad_norm", *DEFAULT_SPACE["max_grad_norm"], log=True)

    learning_rate = trial.suggest_float("learning_rate", learning_rate_low, learning_rate_high, log=learning_rate_log)
    n_steps_choices = hpo_cfg.get("n_steps_choices", DEFAULT_SPACE["n_steps"])
    batch_size_choices = hpo_cfg.get("batch_size_choices", DEFAULT_SPACE["batch_size"])
    step_limit_choices = hpo_cfg.get("step_limit_choices", DEFAULT_SPACE["step_limit"])
    reset_choices = hpo_cfg.get("reset_unsuccessful_moves_threshold_choices", DEFAULT_SPACE["reset_unsuccessful_moves_threshold"])

    n_steps = trial.suggest_categorical("n_steps", list(n_steps_choices))
    batch_size = trial.suggest_categorical("batch_size", list(batch_size_choices))
    n_epochs = trial.suggest_int("n_epochs", int(hpo_cfg.get("n_epochs_low", DEFAULT_SPACE["n_epochs"][0])), int(hpo_cfg.get("n_epochs_high", DEFAULT_SPACE["n_epochs"][1])))
    gamma = trial.suggest_float("gamma", gamma_low, gamma_high)
    gae_lambda = trial.suggest_float("gae_lambda", gae_low, gae_high)
    clip_range = trial.suggest_float("clip_range", clip_low, clip_high)
    ent_coef = trial.suggest_float("ent_coef", ent_low, ent_high, log=ent_log)
    vf_coef = trial.suggest_float("vf_coef", vf_low, vf_high, log=vf_log)
    max_grad_norm = trial.suggest_float("max_grad_norm", max_grad_low, max_grad_high, log=max_grad_log)
    step_limit = trial.suggest_categorical("step_limit", list(step_limit_choices))
    reset_unsuccessful_moves_threshold = trial.suggest_categorical(
        "reset_unsuccessful_moves_threshold",
        list(reset_choices),
    )

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "step_limit": step_limit,
        "reset_unsuccessful_moves_threshold": reset_unsuccessful_moves_threshold,
    }


def _objective_metric_key(optimization_goal: str) -> str:
    return "best_local_crossings" if str(optimization_goal).lower().strip() == "local" else "best_global_crossings"


def _select_fixed_eval_graphs(base_config: dict, split: str, graph_count: int) -> list:
    """Load a fixed evaluation subset, prioritizing harder graphs (more nodes)."""
    dataset_type = base_config.get("dataset", {}).get("name", "rome")
    graphs = list(load_split_dataset(split, dataset_type=dataset_type))
    if not graphs:
        raise RuntimeError(f"No graphs found for eval split '{split}' and dataset '{dataset_type}'")

    # Hard-first subset by number of nodes, deterministic order.
    graphs = sorted(graphs, key=lambda g: g.number_of_nodes(), reverse=True)
    return graphs[: max(1, min(int(graph_count), len(graphs)))]


def _evaluate_model_on_fixed_graphs(model, trial_config: dict, eval_graphs: list, optimization_goal: str) -> dict:
    """Evaluate trained model on fixed graphs and return aggregate objective stats."""
    best_global = []
    best_local = []

    for idx, graph in enumerate(eval_graphs):
        env = create_graph_layout_env(graph, config=trial_config)
        obs, _ = env.reset(seed=10_000 + idx)

        done = False
        truncated = False
        info = {}

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)

        best_global.append(float(info.get("best_global_crossings", float("inf"))))
        best_local.append(float(info.get("best_local_crossings", float("inf"))))

    best_global_arr = np.asarray(best_global, dtype=float)
    best_local_arr = np.asarray(best_local, dtype=float)

    if str(optimization_goal).lower().strip() == "local":
        # Similar prioritization to is_layout_better: local first, global as tie-breaker.
        # We encode lexicographic ordering in one scalar objective.
        objective_values = best_local_arr * 1_000_000.0 + best_global_arr
        objective_value = float(np.median(objective_values))
    else:
        objective_value = float(np.median(best_global_arr))

    return {
        "objective_value": objective_value,
        "median_best_global": float(np.median(best_global_arr)),
        "median_best_local": float(np.median(best_local_arr)),
        "min_best_global": float(np.min(best_global_arr)),
        "min_best_local": float(np.min(best_local_arr)),
        "max_best_global": float(np.max(best_global_arr)),
        "max_best_local": float(np.max(best_local_arr)),
        "n_eval_graphs": int(len(eval_graphs)),
    }


def create_objective(
    base_config: dict,
    trial_steps: int,
    log_dir_base: str,
    eval_graphs: list,
) -> callable:
    """Create the Optuna objective for minimizing the final best crossing count."""

    def objective(trial: optuna.trial.Trial) -> float:
        print(f"\n{'=' * 80}")
        print(f"[HPO] Trial {trial.number} starting...")
        print(f"{'=' * 80}\n")

        trial_config = json.loads(json.dumps(base_config))
        hparams = _suggest_space(trial, trial_config)
        trial_config.setdefault("ppo", {}).update({
            "learning_rate": hparams["learning_rate"],
            "n_steps": hparams["n_steps"],
            "batch_size": hparams["batch_size"],
            "n_epochs": hparams["n_epochs"],
            "gamma": hparams["gamma"],
            "gae_lambda": hparams["gae_lambda"],
            "clip_range": hparams["clip_range"],
            "ent_coef": hparams["ent_coef"],
            "vf_coef": hparams["vf_coef"],
            "max_grad_norm": hparams["max_grad_norm"],
        })
        trial_config.setdefault("env", {})["step_limit"] = hparams["step_limit"]
        trial_config["env"]["reset_unsuccessful_moves_threshold"] = hparams["reset_unsuccessful_moves_threshold"]

        optimization_goal = trial_config["env"].get("optimization_goal", "global")
        metric_key = _objective_metric_key(optimization_goal)

        print(f"[HPO Trial {trial.number}] Hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")
        print(f"[HPO Trial {trial.number}] Objective metric: {metric_key}")

        trial_log_dir = os.path.join(log_dir_base, f"trial_{trial.number}")
        os.makedirs(trial_log_dir, exist_ok=True)
        with open(os.path.join(trial_log_dir, "trial_config.json"), "w", encoding="utf-8") as f:
            json.dump(trial_config, f, indent=2)

        vec_env = None
        model = None
        try:
            graphs = load_split_dataset("train", dataset_type="rome")
            sampler = CurriculumSampler(graphs)

            n_envs = int(trial_config["training"]["n_envs"])
            vec_env = SubprocVecEnv([
                make_env(i, trial_config, sampler, seed=trial.number)
                for i in range(n_envs)
            ])
            vec_env = VecMonitor(vec_env)

            extractor_cfg = trial_config["policy"]["features_extractor"]
            policy_kwargs = dict(
                features_extractor_class=CustomGraphLayoutExtractor,
                features_extractor_kwargs=dict(
                    features_dim=extractor_cfg.get("features_dim", 512),
                    cnn_channels=extractor_cfg.get("cnn_channels", [32, 64, 128]),
                    cnn_res_blocks=extractor_cfg.get("cnn_res_blocks", [0, 1, 1]),
                    cnn_out_dim=extractor_cfg.get("cnn_out_dim", 256),
                    tab_hidden_dim=extractor_cfg.get("tab_hidden_dim", 128),
                    tab_out_dim=extractor_cfg.get("tab_out_dim", 128),
                    fusion_hidden_dim=extractor_cfg.get("fusion_hidden_dim", 512),
                    pixel_map_key=extractor_cfg.get("pixel_map_key", "pixel_map"),
                ),
            )

            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                vec_env,
                learning_rate=trial_config["ppo"]["learning_rate"],
                n_steps=int(trial_config["ppo"]["n_steps"]),
                batch_size=int(trial_config["ppo"]["batch_size"]),
                n_epochs=int(trial_config["ppo"]["n_epochs"]),
                gamma=float(trial_config["ppo"]["gamma"]),
                gae_lambda=float(trial_config["ppo"]["gae_lambda"]),
                clip_range=float(trial_config["ppo"]["clip_range"]),
                ent_coef=float(trial_config["ppo"]["ent_coef"]),
                vf_coef=float(trial_config["ppo"]["vf_coef"]),
                max_grad_norm=float(trial_config["ppo"]["max_grad_norm"]),
                policy_kwargs=policy_kwargs,
                tensorboard_log=trial_log_dir,
                verbose=0,
                seed=trial.number,
            )

            callbacks = [
                LinearCurriculumCallback(
                    sampler=sampler,
                    total_timesteps=trial_steps,
                    config=trial_config,
                    verbose=0,
                ),
                CrossingBestMetricCallback(
                    trial=trial,
                    optimization_goal=optimization_goal,
                    report_interval=max(1, trial_steps // 20),
                ),
            ]

            print(f"\n[HPO Trial {trial.number}] Training for {trial_steps} timesteps...")
            model.learn(
                total_timesteps=trial_steps,
                callback=CallbackList(callbacks),
                log_interval=10,
            )

            # Final objective is measured on a fixed hard evaluation set,
            # not on curriculum training episodes.
            eval_stats = _evaluate_model_on_fixed_graphs(
                model=model,
                trial_config=trial_config,
                eval_graphs=eval_graphs,
                optimization_goal=optimization_goal,
            )
            print(
                f"[HPO Trial {trial.number}] Eval({eval_stats['n_eval_graphs']} graphs): "
                f"median_best_global={eval_stats['median_best_global']:.0f}, "
                f"median_best_local={eval_stats['median_best_local']:.0f}"
            )
            print(f"[HPO Trial {trial.number}] Objective value: {eval_stats['objective_value']:.4f}")

            return float(eval_stats["objective_value"])

        except optuna.TrialPruned:
            print(f"\n[HPO Trial {trial.number}] Pruned early")
            raise
        except Exception as exc:
            print(f"\n[HPO Trial {trial.number}] FAILED with error:")
            print(f"  {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            return float("inf")
        finally:
            if vec_env is not None:
                vec_env.close()
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def setup_optuna_study(
    storage: str,
    study_name: str,
    direction: str = "minimize",
    n_startup_trials: int = 5,
) -> optuna.study.Study:
    """Create or load an Optuna study."""
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=42)
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=max(1, int(n_startup_trials / 2)),
        interval_steps=5,
    )
    return optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for PPO graph layout training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_ppo_pixel_move_over_edge.json",
        help="Base config file",
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
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=500000,
        help="Timesteps per trial",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="hpo_results_best_crossings",
        help="Base directory for trial logs",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of random startup trials before TPE",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for fixed objective evaluation",
    )
    parser.add_argument(
        "--eval_graph_count",
        type=int,
        default=64,
        help="Number of hardest graphs (by node count) used for fixed objective evaluation",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    if args.storage is None:
        os.makedirs("hpo_studies", exist_ok=True)
        args.storage = f"sqlite:///hpo_studies/{args.study_name}.db"

    print(f"\n{'=' * 80}")
    print("[HPO] Starting Optuna Hyperparameter Optimization")
    print(f"{'=' * 80}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials to run: {args.n_trials}")
    print(f"Trial timesteps: {args.trial_steps:,}")
    print(f"Base config: {args.config}")
    print(f"Log directory: {args.log_dir}")
    print(f"Optimization goal: {base_config.get('env', {}).get('optimization_goal', 'global')}")
    print(f"Objective: minimize best crossing count")
    print(f"Fixed eval split: {args.eval_split}")
    print(f"Fixed eval graph count: {args.eval_graph_count}")
    print(f"{'=' * 80}\n")

    eval_graphs = _select_fixed_eval_graphs(
        base_config=base_config,
        split=args.eval_split,
        graph_count=args.eval_graph_count,
    )
    print(
        f"[HPO] Loaded fixed eval set with {len(eval_graphs)} graphs "
        f"(hardest first, max_n={max(g.number_of_nodes() for g in eval_graphs)}, "
        f"min_n={min(g.number_of_nodes() for g in eval_graphs)})"
    )

    study = setup_optuna_study(
        storage=args.storage,
        study_name=args.study_name,
        direction="minimize",
        n_startup_trials=args.n_startup_trials,
    )

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

    print("[HPO] Study status:")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    if completed_trials:
        best_trial = study.best_trial
        print(f"  Best value so far: {best_trial.value:.6f} (trial {best_trial.number})")
        print(f"  Best params so far: {best_trial.params}")
    print()

    objective = create_objective(
        base_config=base_config,
        trial_steps=args.trial_steps,
        log_dir_base=os.path.join(args.log_dir, args.study_name),
        eval_graphs=eval_graphs,
    )

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("\n\n[HPO] Optimization interrupted by user")

    print(f"\n{'=' * 80}")
    print("[HPO] Optimization Complete")
    print(f"{'=' * 80}\n")

    best_trial = study.best_trial
    print(f"Best trial: #{best_trial.number}")
    print(f"Best value (best crossings): {best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    os.makedirs(args.log_dir, exist_ok=True)
    best_hparams_path = os.path.join(args.log_dir, "best_hyperparams.json")
    with open(best_hparams_path, "w", encoding="utf-8") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"\nBest hyperparams saved to: {best_hparams_path}")

    best_config_path = os.path.join(args.log_dir, "best_trial_config.json")
    best_config = copy.deepcopy(base_config)
    best_config.setdefault("ppo", {}).update({
        "learning_rate": best_trial.params.get("learning_rate", best_config.get("ppo", {}).get("learning_rate")),
        "n_steps": best_trial.params.get("n_steps", best_config.get("ppo", {}).get("n_steps")),
        "batch_size": best_trial.params.get("batch_size", best_config.get("ppo", {}).get("batch_size")),
        "n_epochs": best_trial.params.get("n_epochs", best_config.get("ppo", {}).get("n_epochs")),
        "gamma": best_trial.params.get("gamma", best_config.get("ppo", {}).get("gamma")),
        "gae_lambda": best_trial.params.get("gae_lambda", best_config.get("ppo", {}).get("gae_lambda")),
        "clip_range": best_trial.params.get("clip_range", best_config.get("ppo", {}).get("clip_range")),
        "ent_coef": best_trial.params.get("ent_coef", best_config.get("ppo", {}).get("ent_coef")),
        "vf_coef": best_trial.params.get("vf_coef", best_config.get("ppo", {}).get("vf_coef")),
        "max_grad_norm": best_trial.params.get("max_grad_norm", best_config.get("ppo", {}).get("max_grad_norm")),
    })
    best_config.setdefault("env", {})["step_limit"] = best_trial.params.get("step_limit", best_config.get("env", {}).get("step_limit"))
    best_config["env"]["reset_unsuccessful_moves_threshold"] = best_trial.params.get(
        "reset_unsuccessful_moves_threshold",
        best_config.get("env", {}).get("reset_unsuccessful_moves_threshold"),
    )
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best trial config saved to: {best_config_path}")

    summary_path = os.path.join(args.log_dir, "study_summary.json")
    summary = {
        "study_name": args.study_name,
        "config": args.config,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "n_trials_failed": len(failed_trials),
        "best_value": float(best_trial.value),
        "best_params": best_trial.params,
        "optimization_goal": base_config.get("env", {}).get("optimization_goal", "global"),
        "eval_split": args.eval_split,
        "eval_graph_count": args.eval_graph_count,
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Study summary saved to: {summary_path}")

    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(args.log_dir, "all_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Trials table saved to: {trials_csv}")


if __name__ == "__main__":
    main()
