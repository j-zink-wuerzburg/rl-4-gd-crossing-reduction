"""
Optuna-based reward scale optimization for PPO graph layout training.

This script freezes all PPO hyperparameters and optimizes ONLY the reward_scale,
measuring success by FINAL GLOBAL CROSSING COUNT (not episode reward).

This allows us to find the reward scale that produces the best actual layouts,
independent of reward magnitude issues.

Usage:
  python hpo_optuna_reward_scale_only.py \
    --study_name reward_scale_opt_v1 \
    --base_hparams hpo_results/pixel_hpo_v1_best_hparams.json \
    --n_trials 30 \
    --trial_steps 500000
"""

import os
import sys
import json
import argparse
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, PatientPruner

import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Training.train_ppo_pixel import (
    make_env,
    CurriculumSampler,
    CustomGraphLayoutExtractor,
    LinearCurriculumCallback,
    NodeSelectionStatsCallback,
)
from Training.dataloader import load_split_dataset
from env.GraphLayoutEnvPixel import GraphLayoutEnvPixel
from env.GraphLayoutEnvPixelMoveDistance import GraphLayoutEnvPixelMoveDistance


class CrossingMetricsCallback(BaseCallback):
    """Callback to collect final crossing counts from episodes."""
    
    def __init__(self):
        super().__init__()
        self.episode_final_crossings = []
        self.episode_best_crossings = []
        self._printed_info_keys = False
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue

            if not self._printed_info_keys:
                print(f"[CrossingMetricsCallback] Terminal info keys: {sorted(info.keys())}")
                self._printed_info_keys = True

            final_crossings = info.get("global_crossings")
            if final_crossings is None and "episode" in info:
                episode_info = info["episode"]
                final_crossings = episode_info.get("final_global_crossings")

            if final_crossings is not None:
                self.episode_final_crossings.append(float(np.asarray(final_crossings).squeeze()))

            best_crossings = info.get("best_global_crossings")
            if best_crossings is None and "episode" in info:
                episode_info = info["episode"]
                best_crossings = episode_info.get("best_global_crossings")

            if best_crossings is not None:
                self.episode_best_crossings.append(float(np.asarray(best_crossings).squeeze()))
        
        return True


def define_search_space_reward_scale_only(trial: optuna.trial.Trial, config: dict) -> dict:
    """
    Define search space for reward scale ONLY.
    Returns dict with single key: reward_scale.
    """
    
    hpo_cfg = config.get("hpo", {})
    
    # Reward scale (multiplicative factor on final reward)
    # Use config defaults or hardcoded sensible range
    reward_scale_min = hpo_cfg.get("reward_scale_min", 0.1)
    reward_scale_max = hpo_cfg.get("reward_scale_max", 10.0)
    reward_scale_log = hpo_cfg.get("reward_scale_log", True)
    
    reward_scale = trial.suggest_float(
        "reward_scale",
        reward_scale_min,
        reward_scale_max,
        log=reward_scale_log
    )
    
    return {"reward_scale": reward_scale}


def create_objective_reward_scale(
    base_config: dict,
    trial_steps: int,
    log_dir_base: str = "hpo_results_reward_scale",
) -> callable:
    """
    Create an objective function that optimizes for CROSSING COUNT.
    
    Args:
        base_config: Base configuration (with best PPO hyperparams already set)
        trial_steps: Number of timesteps to train each trial for
        log_dir_base: Directory to save trial results
    
    Returns:
        Callable objective function that minimizes crossing count
    """
    
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Objective: sample reward_scale, train, return NEGATIVE crossing count.
        (Optuna maximizes, so we return -crossings to minimize crossings)
        """
        print(f"\n{'='*80}")
        print(f"[REWARD_SCALE_OPT] Trial {trial.number} starting...")
        print(f"{'='*80}\n")
        
        # Sample ONLY reward_scale
        hparams = define_search_space_reward_scale_only(trial, base_config)
        
        print(f"[Trial {trial.number}] Hyperparameters:")
        for key, val in hparams.items():
            print(f"  {key}: {val}")
        
        # Create a modified config for this trial
        trial_config = json.loads(json.dumps(base_config))  # Deep copy
        
        # Inject reward_scale into env config
        trial_config.setdefault("env", {})["reward_scale"] = hparams["reward_scale"]
        
        # Create trial-specific log directory
        trial_log_dir = os.path.join(log_dir_base, f"trial_{trial.number}")
        os.makedirs(trial_log_dir, exist_ok=True)
        
        # Save trial config
        with open(os.path.join(trial_log_dir, "trial_config.json"), 'w') as f:
            json.dump(trial_config, f, indent=2)
        
        try:
            # Load training dataset
            rome = load_split_dataset("train", dataset_type="rome")
            sampler = CurriculumSampler(rome)
            
            # Create vectorized environment
            n_envs = trial_config["training"]["n_envs"]
            vec_env = SubprocVecEnv(
                [make_env(i, trial_config, sampler, seed=trial.number) 
                 for i in range(n_envs)]
            )
            vec_env = VecMonitor(vec_env)
            
            # Setup policy kwargs
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
            
            # Initialize model with FROZEN PPO hyperparams
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                vec_env,
                learning_rate=trial_config["ppo"]["learning_rate"],
                n_steps=trial_config["ppo"]["n_steps"],
                batch_size=trial_config["ppo"]["batch_size"],
                n_epochs=trial_config["ppo"]["n_epochs"],
                gamma=trial_config["ppo"]["gamma"],
                gae_lambda=trial_config["ppo"]["gae_lambda"],
                clip_range=trial_config["ppo"]["clip_range"],
                ent_coef=trial_config["ppo"]["ent_coef"],
                vf_coef=trial_config["ppo"]["vf_coef"],
                max_grad_norm=trial_config["ppo"]["max_grad_norm"],
                policy_kwargs=policy_kwargs,
                tensorboard_log=trial_log_dir,
                verbose=0,
                seed=trial.number,
            )
            
            # Setup callbacks
            callbacks = []
            
            # Curriculum callback for difficulty scheduling
            curriculum_callback = LinearCurriculumCallback(
                sampler=sampler,
                total_timesteps=trial_steps,
                config=trial_config,
                verbose=0
            )
            callbacks.append(curriculum_callback)
            
            # Crossing metrics callback
            crossing_callback = CrossingMetricsCallback()
            callbacks.append(crossing_callback)
            
            # Train for trial_steps timesteps
            print(f"\n[Trial {trial.number}] Training for {trial_steps} timesteps...")
            model.learn(
                total_timesteps=trial_steps,
                callback=CallbackList(callbacks) if callbacks else None,
                log_interval=10,
            )
            
            # Extract crossing count metric
            metric_value = None
            
            # Use final crossings from terminal episodes (lower is better)
            if len(crossing_callback.episode_final_crossings) > 0:
                final_crossings = np.asarray(crossing_callback.episode_final_crossings, dtype=float)
                median_final_crossings = float(np.median(final_crossings))
                print(f"[Trial {trial.number}] Episodes: {len(final_crossings)}")
                print(
                    f"[Trial {trial.number}] Final crossings (per episode): "
                    f"min={np.min(final_crossings):.0f}, median={median_final_crossings:.0f}, max={np.max(final_crossings):.0f}"
                )

                # Return NEGATIVE because Optuna maximizes but we want to minimize crossings
                metric_value = -median_final_crossings
                print(f"[Trial {trial.number}] Metric for Optuna (negative median final crossings): {metric_value:.4f}")
            elif len(crossing_callback.episode_best_crossings) > 0:
                best_crossings = np.asarray(crossing_callback.episode_best_crossings, dtype=float)
                median_best_crossings = float(np.median(best_crossings))
                print(f"[Trial {trial.number}] Episodes: {len(best_crossings)}")
                print(
                    f"[Trial {trial.number}] Best crossings fallback (per episode): "
                    f"min={np.min(best_crossings):.0f}, median={median_best_crossings:.0f}, max={np.max(best_crossings):.0f}"
                )

                metric_value = -median_best_crossings
                print(f"[Trial {trial.number}] Metric for Optuna (negative median best crossings fallback): {metric_value:.4f}")
            else:
                print(f"[Trial {trial.number}] Warning: No crossing metrics collected")
                metric_value = -1000.0  # Very bad value to discourage this config
            
            # Cleanup
            vec_env.close()
            del model
            torch.cuda.empty_cache()
            
            return metric_value
        
        except Exception as e:
            print(f"\n[Trial {trial.number}] FAILED with error:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a very negative score
            return -10000.0
    
    return objective


def setup_optuna_study(
    storage: str,
    study_name: str,
    direction: str = "maximize",
    n_startup_trials: int = 3,
) -> optuna.study.Study:
    """
    Create or load an Optuna study.
    
    Args:
        storage: SQLite storage path
        study_name: Name of the study
        direction: "maximize" or "minimize"
        n_startup_trials: Number of random trials before TPE
    
    Returns:
        Optuna Study object
    """
    
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
    )
    
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=max(1, int(n_startup_trials / 2)),
        interval_steps=5,
    )
    
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Reward scale optimization for PPO graph layout training"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/config_ppo_pixel_move_distance.json",
        help="Base config file"
    )
    parser.add_argument(
        "--base_hparams",
        type=str,
        default=None,
        help="Best hyperparams JSON from a previous HPO run (will be injected into ppo section)"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: sqlite:///hpo_studies/{study_name}.db)"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of reward scale trials to run"
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=500000,
        help="Timesteps per trial"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="hpo_results_reward_scale",
        help="Base directory for trial logs"
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=3,
        help="Number of random startup trials before TPE"
    )
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.base_config, 'r') as f:
        base_config = json.load(f)
    
    # If best hyperparams provided, inject them into ppo section
    if args.base_hparams:
        print(f"[REWARD_SCALE_OPT] Loading best hyperparams from: {args.base_hparams}")
        with open(args.base_hparams, 'r') as f:
            best_hparams = json.load(f)
        
        # Inject all PPO hyperparams
        ppo_keys = [
            "learning_rate", "n_steps", "batch_size", "n_epochs", 
            "gamma", "gae_lambda", "clip_range", "ent_coef", 
            "vf_coef", "max_grad_norm"
        ]
        for key in ppo_keys:
            if key in best_hparams:
                base_config["ppo"][key] = best_hparams[key]
                print(f"  Injected {key}: {best_hparams[key]}")
    
    # Setup storage
    if args.storage is None:
        os.makedirs("hpo_studies", exist_ok=True)
        args.storage = f"sqlite:///hpo_studies/{args.study_name}.db"
    
    print(f"\n{'='*80}")
    print(f"[REWARD_SCALE_OPT] Reward Scale Optimization via Optuna")
    print(f"{'='*80}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials to run: {args.n_trials}")
    print(f"Trial timesteps: {args.trial_steps:,}")
    print(f"Base config: {args.base_config}")
    print(f"Log directory: {args.log_dir}")
    print(f"Metric: Minimize final global crossing count")
    print(f"{'='*80}\n")
    
    # Create study
    study = setup_optuna_study(
        storage=args.storage,
        study_name=args.study_name,
        direction="maximize",  # We maximize negative crossings = minimize crossings
        n_startup_trials=args.n_startup_trials,
    )
    
    # Print existing trials info
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    
    print(f"[REWARD_SCALE_OPT] Study status:")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    if len(completed_trials) > 0:
        best_trial = study.best_trial
        best_crossings = -best_trial.value  # Convert back from negative
        print(f"  Best value so far: {best_trial.value:.6f} (≈ {best_crossings:.0f} final crossings, trial {best_trial.number})")
        print(f"  Best reward_scale: {best_trial.params.get('reward_scale', 'N/A')}")
    print()
    
    # Create objective
    objective = create_objective_reward_scale(
        base_config=base_config,
        trial_steps=args.trial_steps,
        log_dir_base=args.log_dir,
    )
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("\n\n[REWARD_SCALE_OPT] Optimization interrupted by user")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"[REWARD_SCALE_OPT] Optimization Complete")
    print(f"{'='*80}\n")
    
    best_trial = study.best_trial
    best_crossings = -best_trial.value
    print(f"Best trial: #{best_trial.number}")
    print(f"Best value (negative crossings): {best_trial.value:.6f}")
    print(f"Best final crossing count: {best_crossings:.0f}")
    print(f"\nBest hyperparameters:")
    for key, val in best_trial.params.items():
        print(f"  {key}: {val}")
    
    # Save best hyperparams to JSON
    best_hparams_path = os.path.join(args.log_dir, "best_reward_scale.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(best_hparams_path, 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"\nBest hyperparams saved to: {best_hparams_path}")
    
    # Also save a summary
    summary_path = os.path.join(args.log_dir, "optimization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Reward Scale Optimization Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Study name: {args.study_name}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Completed trials: {len(completed_trials)}\n")
        f.write(f"\nBest trial: {best_trial.number}\n")
        f.write(f"Best final crossing count: {best_crossings:.0f}\n")
        f.write(f"Best reward_scale: {best_trial.params['reward_scale']}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
