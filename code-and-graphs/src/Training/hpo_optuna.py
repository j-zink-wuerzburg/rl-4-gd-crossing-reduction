"""
Optuna-based hyperparameter optimization for PPO graph layout training.

Usage:
  python hpo_optuna.py --study_name my_ppo_study --n_trials 100 --config configs/config_ppo_pixel.json
  
For distributed SLURM execution, submit multiple jobs that reference the same study:
  sbatch submit_hpo_optuna_h200.sh --study_name my_ppo_study --n_trials 100
  sbatch submit_hpo_optuna_h200.sh --study_name my_ppo_study --n_trials 100
  (Each job will work on independent trials from the same study)
"""

import os
import sys
import json
import argparse
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
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
from env import create_graph_layout_env


class OptunaCheckpointCallback(BaseCallback):
    """Callback to report metrics to Optuna during training for pruning."""
    
    def __init__(self, trial: optuna.trial.Trial, report_interval: int = 1000):
        super().__init__()
        self.trial = trial
        self.report_interval = report_interval
        self.step_count = 0
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Collect episode returns from info dicts
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_info = info["episode"]
                if "r" in episode_info:
                    self.episode_rewards.append(float(episode_info["r"]))
        
        # Report metric every N steps for pruning
        if self.step_count % self.report_interval == 0:
            if len(self.episode_rewards) > 0:
                # Use mean of recent episodes
                recent_episodes = self.episode_rewards[-max(1, len(self.episode_rewards)//10):]
                avg_reward = np.mean(recent_episodes)
                self.trial.report(float(avg_reward), step=self.step_count)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    print(f"[Optuna] Trial {self.trial.number} pruned at step {self.step_count}")
                    raise optuna.TrialPruned()
        
        return True


def define_search_space(trial: optuna.trial.Trial, config: dict) -> dict:
    """
    Define the hyperparameter search space for Optuna.
    Returns a dict of hyperparameters to override in config.
    """
    
    # Learning rate (log scale)
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True
    )
    
    # N steps (number of steps per update)
    n_steps = trial.suggest_int(
        "n_steps",
        512,
        8192,
        step=512
    )
    
    # Batch size (must be multiple of n_steps for PPO)
    batch_size = trial.suggest_int(
        "batch_size",
        64,
        min(8192, n_steps * 4),
        step=64
    )
    # Ensure batch_size <= n_steps * n_envs for valid PPO configuration
    batch_size = min(batch_size, n_steps * config["training"]["n_envs"])
    
    # Entropy coefficient (exploration)
    ent_coef = trial.suggest_float(
        "ent_coef",
        0.001,
        0.1,
        log=True
    )
    
    # Value function coefficient (critic loss weight)
    vf_coef = trial.suggest_float(
        "vf_coef",
        0.01,
        1.0,
        log=True
    )
    
    # GAE lambda (generalized advantage estimation)
    gae_lambda = trial.suggest_float(
        "gae_lambda",
        0.90,
        0.99,
        step=0.01
    )
    
    # Clip range (PPO clip range)
    clip_range = trial.suggest_float(
        "clip_range",
        0.1,
        0.3,
        step=0.05
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


def create_objective(
    base_config: dict,
    trial_steps: int,
    use_callbacks: bool = True,
    log_dir_base: str = "hpo_results",
) -> callable:
    """
    Create an objective function for Optuna optimization.
    
    Args:
        base_config: Base configuration dict (will be modified by trial)
        trial_steps: Number of timesteps to train each trial for
        use_callbacks: Whether to attach callbacks and enable pruning
        log_dir_base: Directory to save trial results
    
    Returns:
        Callable objective function that takes an Optuna trial
    """
    
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Objective function: sample hyperparams, train, return metric.
        """
        print(f"\n{'='*80}")
        print(f"[HPO] Trial {trial.number} starting...")
        print(f"{'='*80}\n")
        
        # Sample hyperparameters
        hparams = define_search_space(trial, base_config)
        
        print(f"[HPO Trial {trial.number}] Hyperparameters:")
        for key, val in hparams.items():
            print(f"  {key}: {val}")
        
        # Create a modified config for this trial
        trial_config = json.loads(json.dumps(base_config))  # Deep copy
        trial_config["ppo"].update(hparams)
        
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
            
            # Initialize model
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
            optuna_callback = None
            
            if use_callbacks:
                # Curriculum callback for difficulty scheduling
                curriculum_callback = LinearCurriculumCallback(
                    sampler=sampler,
                    total_timesteps=trial_steps,
                    config=trial_config,
                    verbose=0
                )
                callbacks.append(curriculum_callback)
                
                # Optuna pruning callback - will track episode rewards
                optuna_callback = OptunaCheckpointCallback(
                    trial=trial,
                    report_interval=max(1, trial_steps // 20)  # Report 20 times during training
                )
                callbacks.append(optuna_callback)
            
            # Train for trial_steps timesteps
            print(f"\n[HPO Trial {trial.number}] Training for {trial_steps} timesteps...")
            model.learn(
                total_timesteps=trial_steps,
                callback=CallbackList(callbacks) if callbacks else None,
                log_interval=10,
            )
            
            # Extract metric from training - use episode rewards if available
            metric_value = 0.0
            
            # First try: get metric from the Optuna callback (actual episode rewards)
            if optuna_callback is not None and len(optuna_callback.episode_rewards) > 0:
                metric_value = float(np.mean(optuna_callback.episode_rewards))
                print(f"[HPO Trial {trial.number}] Mean episode reward: {metric_value:.4f} ({len(optuna_callback.episode_rewards)} episodes)")
            else:
                # Fallback: try to extract from model logger
                if hasattr(model, "logger") and hasattr(model.logger, "name_to_value"):
                    logger_data = model.logger.name_to_value
                    if "rollout/ep_rew_mean" in logger_data:
                        metric_value = float(logger_data["rollout/ep_rew_mean"])
                        print(f"[HPO Trial {trial.number}] Metric from logger (ep_rew_mean): {metric_value:.4f}")
                    elif "rollout/ep_len_mean" in logger_data:
                        metric_value = float(logger_data["rollout/ep_len_mean"]) / 1000.0
                        print(f"[HPO Trial {trial.number}] Metric from logger (ep_len_mean/1000): {metric_value:.4f}")
                
                if metric_value == 0.0:
                    print(f"[HPO Trial {trial.number}] Warning: Could not extract metric from any source")
                    metric_value = 0.5  # Neutral value
            
            print(f"[HPO Trial {trial.number}] Final metric value: {metric_value:.4f}")
            
            # Cleanup
            vec_env.close()
            del model
            torch.cuda.empty_cache()
            
            return metric_value
        
        except optuna.TrialPruned:
            print(f"\n[HPO Trial {trial.number}] Pruned early")
            raise
        
        except Exception as e:
            print(f"\n[HPO Trial {trial.number}] FAILED with error:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a very negative score to discourage this config
            return -1000.0
    
    return objective


def setup_optuna_study(
    storage: str,
    study_name: str,
    direction: str = "maximize",
    n_startup_trials: int = 5,
) -> optuna.study.Study:
    """
    Create or load an Optuna study with TPE sampler.
    
    Args:
        storage: SQLite storage path (e.g., "sqlite:///hpo_study.db")
        study_name: Name of the study
        direction: "maximize" or "minimize"
        n_startup_trials: Number of random trials before TPE kicks in
    
    Returns:
        Optuna Study object
    """
    
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
    )
    
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=int(n_startup_trials / 2),
        interval_steps=5,
    )
    
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # Load existing study if it exists
    )
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for PPO graph layout training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_ppo_pixel.json",
        help="Base config file"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Name of the Optuna study (for multi-job parallel optimization)"
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
        default=100,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=500000,
        help="Timesteps per trial (default 500k for reasonable compute)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="hpo_results",
        help="Base directory for trial logs"
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of random startup trials before TPE"
    )
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = json.load(f)
    
    # Setup storage
    if args.storage is None:
        os.makedirs("hpo_studies", exist_ok=True)
        args.storage = f"sqlite:///hpo_studies/{args.study_name}.db"
    
    print(f"\n{'='*80}")
    print(f"[HPO] Starting Optuna Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials to run: {args.n_trials}")
    print(f"Trial timesteps: {args.trial_steps:,}")
    print(f"Base config: {args.config}")
    print(f"Log directory: {args.log_dir}")
    print(f"{'='*80}\n")
    
    # Create study
    study = setup_optuna_study(
        storage=args.storage,
        study_name=args.study_name,
        direction="maximize",
        n_startup_trials=args.n_startup_trials,
    )
    
    # Print existing trials info
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    
    print(f"[HPO] Study status:")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    if len(completed_trials) > 0:
        best_trial = study.best_trial
        print(f"  Best value so far: {best_trial.value:.6f} (trial {best_trial.number})")
    print()
    
    # Create objective
    objective = create_objective(
        base_config=base_config,
        trial_steps=args.trial_steps,
        use_callbacks=True,
        log_dir_base=args.log_dir+f"/{args.study_name}",
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
        print("\n\n[HPO] Optimization interrupted by user")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"[HPO] Optimization Complete")
    print(f"{'='*80}\n")
    
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, val in study.best_trial.params.items():
        print(f"  {key}: {val}")
    
    # Save best hyperparams to JSON
    best_hparams_path = os.path.join(args.log_dir, "best_hyperparams.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(best_hparams_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\nBest hyperparams saved to: {best_hparams_path}")
    
    # Save study summary
    summary_path = os.path.join(args.log_dir, "study_summary.json")
    summary = {
        "study_name": args.study_name,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "n_trials_failed": len(failed_trials),
        "best_value": float(study.best_trial.value),
        "best_params": study.best_trial.params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Study summary saved to: {summary_path}")
    
    # Print all trials
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(args.log_dir, "all_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"All trials saved to: {trials_csv}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
