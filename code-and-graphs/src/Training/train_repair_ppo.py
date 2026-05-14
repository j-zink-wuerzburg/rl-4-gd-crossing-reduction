import argparse
import csv
import json
import copy
from math import gcd
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from Training.dataloader import load_split_dataset
from env import create_graph_layout_env


def _deep_copy_dict(value):
    return copy.deepcopy(value)


def _deep_merge_dict(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_repair_config_bundle(config_path: str | None) -> tuple[dict, dict]:
    """
    Load repair config from nested JSON structure.
    Config must have: env, policy, ppo, training sections.
    Returns: (flat_config_dict, nested_raw_config)
    
    NO DEFAULT VALUES - all values must come from the config file.
    Config file is required and must be complete.
    
    The flat config combines values from:
    - dataset section (dataset name and split)
    - sampling section (train_count, eval_count, train_rome_count, etc.)
    - run section (seeds, ppo_steps, n_envs, device, exp_name)
    - experiment section (shortlist_size, optimization_goal, horizons, etc.)
    - training section (eval_freq_steps, normalize_reward, etc.)
    - io section (checkpoint_root, output paths)
    
    Nested sections (env, policy, ppo, training) are preserved in the raw config.
    """
    if not config_path:
        raise ValueError(
            "Config file is REQUIRED. No default config is provided. "
            "Please specify --config with path to config.json"
        )
    
    path = Path(config_path)
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    
    raw_config = _deep_copy_dict(loaded)
    
    # Validate required sections exist
    required_sections = ["dataset", "sampling", "run", "experiment", "env", "policy", "ppo", "training", "io"]
    missing = [s for s in required_sections if s not in loaded]
    if missing:
        raise ValueError(
            f"Config missing required sections: {missing}. "
            f"Config must have: dataset, sampling, run, experiment, env, policy, ppo, training, io sections"
        )
    
    # Flatten the nested config into flat keys for argparse
    # NO DEFAULTS - values must exist in config or KeyError is raised
    config = {}
    
    # Extract from dataset section (REQUIRED)
    dataset = loaded["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("Config section 'dataset' must be an object with 'name' and 'split'")
    if "name" not in dataset or "split" not in dataset:
        raise ValueError("Config section 'dataset' must define both 'name' and 'split'")
    config.update({
        "dataset": dataset["name"],
        "dataset_split": dataset["split"],
    })

    # Extract from sampling section (REQUIRED)
    sampling = loaded["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("Config section 'sampling' must be an object with graph counts")
    config.update({
        "train_count": sampling["train_count"],
        "eval_count": sampling["eval_count"],
        "train_rome_count": sampling["train_rome_count"],
        "train_ba_count": sampling["train_ba_count"],
        "eval_rome_count": sampling["eval_rome_count"],
        "eval_ba_count": sampling["eval_ba_count"],
        "train_list_file": sampling.get("train_list_file"),
        "eval_list_file": sampling.get("eval_list_file"),
    })
    
    # Extract from run section (REQUIRED)
    run = loaded["run"]
    if not isinstance(run, dict):
        raise ValueError("Config section 'run' must be an object with training settings")
    config.update({
        "seeds": run["seeds"],
        "ppo_steps": run["ppo_steps"],
        "n_envs": run["n_envs"],
        "device": run["device"],
        "exp_name": run.get("exp_name"),
    })
    
    # Extract from experiment section (REQUIRED)
    experiment = loaded["experiment"]
    if not isinstance(experiment, dict):
        raise ValueError("Config section 'experiment' must be an object with experiment settings")
    config.update({
        "shortlist_size": experiment["shortlist_size"],
        "optimization_goal": experiment["optimization_goal"],
        "best_local_bonus": experiment["best_local_bonus"],
        "local_weight": experiment["local_weight"],
        "sizemax_weight": experiment["sizemax_weight"],
        "global_weight": experiment["global_weight"],
        "standard_horizon": experiment["standard_horizon"],
        "repair_horizon": experiment["repair_horizon"],
        "repair_perturb_steps": experiment["repair_perturb_steps"],
        "repair_perturb_attempts": experiment["repair_perturb_attempts"],
        "plateau_patience": experiment["plateau_patience"],
        "plateau_bank_size": experiment["plateau_bank_size"],
        "broadened_repair": experiment["broadened_repair"],
        "plateau_only_replay": experiment["plateau_only_replay"],
        "eval_outer_restarts": experiment["eval_outer_restarts"],
        "eval_restart_perturb_steps": experiment["eval_restart_perturb_steps"],
        "eval_batch_size": experiment["eval_batch_size"],
    })
    
    # Extract from training section (REQUIRED)
    training = loaded["training"]
    if not isinstance(training, dict):
        raise ValueError("Config section 'training' must be an object with training hyperparameters")
    config.update({
        "eval_freq_steps": training["eval_freq_steps"],
        "normalize_reward": training["normalize_reward"],
        "reward_clip": training["reward_clip"],
        "use_lr_ent_schedule": training["use_lr_ent_schedule"],
        "final_learning_rate": training.get("final_learning_rate"),
        "final_ent_coef": training.get("final_ent_coef"),
        "use_self_imitation": training["use_self_imitation"],
        "sil_collect_episodes": training["sil_collect_episodes"],
        "sil_buffer_size": training["sil_buffer_size"],
        "sil_batch_size": training["sil_batch_size"],
        "sil_epochs": training["sil_epochs"],
        "sil_weight": training["sil_weight"],
        "sil_stochastic_collect": training["sil_stochastic_collect"],
    })

    env = loaded["env"]
    if not isinstance(env, dict):
        raise ValueError("Config section 'env' must be an object with environment settings")
    config.update({
        "node_selection_strategy": env.get("node_selection_strategy", "heuristic_new"),
    })
    
    # Extract io section (REQUIRED)
    io = loaded["io"]
    if not isinstance(io, dict):
        raise ValueError("Config section 'io' must be an object with checkpoint and output paths")
    config.update({
        "checkpoint_root": io["checkpoint_root"],
        "output_json": io.get("output_json"),
        "output_jsonl": io.get("output_jsonl"),
    })
    
    return config, raw_config


def load_repair_config(config_path: str | None) -> dict:
    """Load and return flat repair config."""
    return load_repair_config_bundle(config_path)[0]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class RepairGraphLayoutExtractor(BaseFeaturesExtractor):
    """
    Dict-observation extractor for shortlist repair PPO.
    Keeps the repair trainer independent from the flattened-observation pixel trainer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        cnn_channels: list | None = None,
        cnn_res_blocks: list | None = None,
        cnn_out_dim: int = 256,
        tab_hidden_dim: int = 128,
        tab_out_dim: int = 128,
        fusion_hidden_dim: int = 512,
        pixel_map_key: str = "pixel_map",
    ):
        super().__init__(observation_space, features_dim)
        self.pixel_map_key = pixel_map_key
        self.tab_keys = [key for key in observation_space.spaces.keys() if key != pixel_map_key]

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        if cnn_res_blocks is None:
            cnn_res_blocks = [0, 1, 1]
        if len(cnn_channels) != len(cnn_res_blocks):
            raise ValueError("cnn_channels and cnn_res_blocks must have the same length")

        pixel_space = observation_space.spaces[pixel_map_key]
        slot_count = int(pixel_space.shape[0])
        for key in self.tab_keys:
            slot_count = gcd(slot_count, int(observation_space.spaces[key].shape[0]))
        if slot_count <= 0:
            raise ValueError("Could not infer shortlist size from observation space")
        if int(pixel_space.shape[0]) % slot_count != 0:
            raise ValueError("pixel_map channels must be divisible by the inferred shortlist size")

        self.shortlist_size = int(slot_count)
        self.pixel_channels = int(pixel_space.shape[0]) // self.shortlist_size
        self.tab_out_dim = int(tab_out_dim)
        self.node_feature_dim = int(cnn_out_dim) + self.tab_out_dim

        self.tab_dims: dict[str, int] = {}
        flat_dim = 0
        for key in self.tab_keys:
            total_dim = int(observation_space.spaces[key].shape[0])
            if total_dim % self.shortlist_size != 0:
                raise ValueError(f"Observation '{key}' must be divisible by the inferred shortlist size")
            per_slot_dim = total_dim // self.shortlist_size
            self.tab_dims[key] = per_slot_dim
            flat_dim += per_slot_dim

        in_ch = self.pixel_channels

        cnn_layers: list[nn.Module] = []
        for out_ch, n_res in zip(cnn_channels, cnn_res_blocks):
            cnn_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            for _ in range(int(n_res)):
                cnn_layers.append(ResidualBlock(out_ch))
            in_ch = int(out_ch)

        cnn_layers.extend(
            [
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(in_ch * 4 * 4, cnn_out_dim, bias=True),
                nn.ReLU(inplace=True),
            ]
        )
        self.cnn = nn.Sequential(*cnn_layers)

        self.tab_mlp = nn.Sequential(
            nn.Linear(flat_dim, tab_hidden_dim, bias=True),
            nn.LayerNorm(tab_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(tab_hidden_dim, tab_out_dim, bias=True),
            nn.ReLU(inplace=True),
        )

        fusion_in = self.shortlist_size * self.node_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, features_dim, bias=True),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_tensor = observations[self.pixel_map_key]
        if pixel_tensor.ndim == 3:
            pixel_tensor = pixel_tensor.unsqueeze(0)
        batch_size = int(pixel_tensor.shape[0])
        pixel_tensor = pixel_tensor.reshape(
            batch_size * self.shortlist_size,
            self.pixel_channels,
            int(pixel_tensor.shape[-2]),
            int(pixel_tensor.shape[-1]),
        )

        cnn_out = self.cnn(pixel_tensor).reshape(batch_size, self.shortlist_size, -1)

        tab_tensors = []
        for key in self.tab_keys:
            tensor = observations[key]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            tab_tensors.append(tensor.reshape(batch_size, self.shortlist_size, self.tab_dims[key]))

        if tab_tensors:
            tab_tensor = torch.cat(tab_tensors, dim=-1)
            tab_out = self.tab_mlp(tab_tensor.reshape(batch_size * self.shortlist_size, -1))
            tab_out = tab_out.reshape(batch_size, self.shortlist_size, -1)
        else:
            tab_out = torch.zeros(
                (batch_size, self.shortlist_size, self.tab_out_dim),
                device=pixel_tensor.device,
                dtype=pixel_tensor.dtype,
            )

        node_features = torch.cat([cnn_out, tab_out], dim=-1).reshape(batch_size, -1)
        return self.fusion(node_features)


class RoundRobinSampler:
    def __init__(self, graphs, start_offset=0):
        self.graphs = list(graphs)
        self.idx = int(start_offset) % max(1, len(self.graphs))

    def sample(self):
        graph = self.graphs[self.idx % len(self.graphs)]
        self.idx += 1
        return graph

    def set_graphs(self, graphs, start_offset=None):
        self.graphs = list(graphs)
        if start_offset is not None:
            self.idx = int(start_offset) % max(1, len(self.graphs))
        else:
            self.idx = self.idx % max(1, len(self.graphs))


def load_base_cfg(
    step_limit,
    reset_threshold,
    local_weight=10.0,
    sizemax_weight=0.1,
    global_weight=0.05,
    optimization_goal="local",
    node_selection_strategy=None,
    base_config=None,
):
    # Load default config as base
    with (ROOT / "configs" / "config_repair_ppo.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # If base_config provided (from checkpoint), merge it in
    # This ensures checkpoint config values take precedence over defaults
    if base_config is not None:
        _deep_merge_dict(cfg, base_config)

    cfg.setdefault("env", {})
    cfg.setdefault("policy", {})
    cfg.setdefault("ppo", {})

    env_cfg = cfg["env"]
    env_cfg["type"] = "move_distance"
    env_cfg["patch_size"] = int(env_cfg.get("patch_size", 31))
    env_cfg["step_limit"] = int(step_limit)
    if node_selection_strategy is None:
        strategy = str(env_cfg.get("node_selection_strategy", "heuristic_new"))
    else:
        strategy = str(node_selection_strategy)
    valid_node_selection_strategies = {"random", "heuristic", "heuristic_new", "heuristic_global"}
    if strategy not in valid_node_selection_strategies:
        raise ValueError(f"Unsupported node_selection_strategy: {strategy}")
    env_cfg["node_selection_strategy"] = strategy
    env_cfg["node_visit_repeat_count"] = int(env_cfg.get("node_visit_repeat_count", 1))
    env_cfg["optimization_goal"] = str(optimization_goal)
    env_cfg["heuristic_new_visit_penalty_coef"] = float(env_cfg.get("heuristic_new_visit_penalty_coef", 0.5))
    env_cfg["n_distances"] = int(env_cfg.get("n_distances", 6))
    if reset_threshold is None:
        env_cfg["reset_unsuccessful_moves_threshold"] = int(env_cfg.get("reset_unsuccessful_moves_threshold", 64))
    else:
        env_cfg["reset_unsuccessful_moves_threshold"] = int(reset_threshold)
    env_cfg.setdefault("reward", {})
    env_cfg["reward"].setdefault("global_optimization", {})
    env_cfg["reward"].setdefault("local_optimization", {})
    env_cfg["reward"]["global_optimization"]["global_weight"] = float(global_weight)
    env_cfg["reward"]["global_optimization"]["sparse_penalty"] = float(env_cfg["reward"]["global_optimization"].get("sparse_penalty", -0.01))
    env_cfg["reward"]["local_optimization"]["local_weight"] = float(local_weight)
    env_cfg["reward"]["local_optimization"]["sizemax_weight"] = float(sizemax_weight)
    env_cfg["reward"]["local_optimization"]["global_weight"] = float(global_weight)
    env_cfg["reward"]["local_optimization"]["sparse_penalty"] = float(env_cfg["reward"]["local_optimization"].get("sparse_penalty", -0.01))

    policy_cfg = cfg["policy"]
    policy_cfg.setdefault("features_extractor", {})
    extractor_cfg = policy_cfg["features_extractor"]
    extractor_cfg.setdefault("features_dim", 256)
    extractor_cfg.setdefault("cnn_channels", [16, 32, 64])
    extractor_cfg.setdefault("cnn_res_blocks", [0, 0, 1])
    extractor_cfg.setdefault("cnn_out_dim", 128)
    extractor_cfg.setdefault("tab_hidden_dim", 64)
    extractor_cfg.setdefault("tab_out_dim", 64)
    extractor_cfg.setdefault("fusion_hidden_dim", 256)
    extractor_cfg.setdefault("pixel_map_key", "pixel_map")

    ppo_cfg = cfg["ppo"]
    ppo_cfg["learning_rate"] = float(ppo_cfg.get("learning_rate", 1e-4))
    ppo_cfg["n_steps"] = int(ppo_cfg.get("n_steps", 128))
    ppo_cfg["batch_size"] = int(ppo_cfg.get("batch_size", 128))
    ppo_cfg["n_epochs"] = int(ppo_cfg.get("n_epochs", 8))
    ppo_cfg["clip_range"] = float(ppo_cfg.get("clip_range", 0.2))
    ppo_cfg["clip_range_vf"] = float(ppo_cfg.get("clip_range_vf", 0.2))
    ppo_cfg["ent_coef"] = float(ppo_cfg.get("ent_coef", 0.01))
    ppo_cfg["vf_coef"] = float(ppo_cfg.get("vf_coef", 0.5))
    ppo_cfg["target_kl"] = float(ppo_cfg.get("target_kl", 0.02))
    ppo_cfg.setdefault("policy_kwargs", {})
    ppo_cfg["policy_kwargs"].setdefault("net_arch", [128, 128])
    return cfg


def build_policy_kwargs(cfg):
    extractor_cfg = cfg["policy"]["features_extractor"]
    return dict(
        features_extractor_class=RepairGraphLayoutExtractor,
        features_extractor_kwargs=dict(
            features_dim=extractor_cfg["features_dim"],
            cnn_channels=extractor_cfg["cnn_channels"],
            cnn_res_blocks=extractor_cfg["cnn_res_blocks"],
            cnn_out_dim=extractor_cfg["cnn_out_dim"],
            tab_hidden_dim=extractor_cfg["tab_hidden_dim"],
            tab_out_dim=extractor_cfg["tab_out_dim"],
            fusion_hidden_dim=extractor_cfg["fusion_hidden_dim"],
            pixel_map_key=extractor_cfg["pixel_map_key"],
        ),
        net_arch=dict(
            pi=cfg["ppo"]["policy_kwargs"]["net_arch"],
            vf=cfg["ppo"]["policy_kwargs"]["net_arch"],
        ),
    )


def make_cfg_from_args(args, step_limit, reset_threshold, base_config=None):
    return load_base_cfg(
        step_limit=step_limit,
        reset_threshold=reset_threshold,
        local_weight=args.local_weight,
        sizemax_weight=args.sizemax_weight,
        global_weight=args.global_weight,
        optimization_goal=args.optimization_goal,
        node_selection_strategy=args.node_selection_strategy,
        base_config=base_config,
    )


class ResampleWrapper(gym.Wrapper):
    def __init__(self, env_obj, sampler):
        super().__init__(env_obj)
        self.sampler = sampler

    def reset(self, **kwargs):
        graph = self.sampler.sample()
        return self.env.reset(Graph=graph, **kwargs)

    def action_masks(self):
        return self.env.action_masks()

    def set_graphs(self, graphs, start_offset=None):
        self.sampler.set_graphs(graphs, start_offset=start_offset)


class ShortlistNodeMoveWrapper(gym.Wrapper):
    def __init__(
        self,
        env_obj,
        shortlist_size=4,
        best_local_bonus=5.0,
        optimization_goal="local",
        perturb_steps=8,
        attempts=4,
        reset_mode_probs=None,
        plateau_patience=12,
        plateau_bank_size=8,
        seed=0,
    ):
        super().__init__(env_obj)
        self.shortlist_size = int(shortlist_size)
        self.best_local_bonus = float(best_local_bonus)
        self.optimization_goal = str(optimization_goal)
        self.perturb_steps = int(perturb_steps)
        self.attempts = int(attempts)
        self.reset_mode_probs = self._normalize_reset_mode_probs(reset_mode_probs)
        self.plateau_patience = int(plateau_patience)
        self.plateau_bank_size = int(plateau_bank_size)
        self.rng = np.random.default_rng(seed)
        self.shortlist_nodes = []
        self.shortlist_scores = []
        self.shortlist_rotations = []
        self._shortlist_valid = False
        self._graph_id = None
        self._no_best_improve_steps = 0
        self.plateau_bank = {}

        base_space = self.base.observation_space
        pixel_shape = base_space.spaces["pixel_map"].shape
        pixel_channels = int(pixel_shape[0])
        self.pixel_channels = pixel_channels
        self.pixel_height = int(pixel_shape[1])
        self.pixel_width = int(pixel_shape[2])

        self.action_space = gym.spaces.MultiDiscrete([self.shortlist_size, 8, self.base.n_distances])
        self.observation_space = gym.spaces.Dict(
            {
                "pixel_map": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.shortlist_size * pixel_channels, self.pixel_height, self.pixel_width),
                    dtype=np.float32,
                ),
                "cross_map": gym.spaces.Box(0.0, 1.0, shape=(self.shortlist_size * 8,), dtype=np.float32),
                "cross_map_local": gym.spaces.Box(0.0, 1.0, shape=(self.shortlist_size * 8,), dtype=np.float32),
                "local_view": gym.spaces.Box(0.0, np.inf, shape=(self.shortlist_size * 40,), dtype=np.float32),
                "local_crossings": gym.spaces.Box(0.0, np.inf, shape=(self.shortlist_size,), dtype=np.float32),
                "global_crossings": gym.spaces.Box(0.0, np.inf, shape=(self.shortlist_size,), dtype=np.float32),
                "shortlist_meta": gym.spaces.Box(0.0, np.inf, shape=(self.shortlist_size * 5,), dtype=np.float32),
                "shortlist_mask": gym.spaces.Box(0.0, 1.0, shape=(self.shortlist_size,), dtype=np.float32),
            }
        )
        self.base.defer_next_node_selection = True
        self.base.defer_step_observation = True

    @property
    def base(self):
        return self.unwrapped

    def action_masks(self):
        self._ensure_shortlist()
        mask = np.zeros(self.shortlist_size + 8 + self.base.n_distances, dtype=bool)
        valid = min(self.shortlist_size, len(self.shortlist_nodes))
        if valid <= 0:
            valid = 1
        mask[:valid] = True
        mask[self.shortlist_size:self.shortlist_size + 8] = True
        mask[self.shortlist_size + 8:] = True
        return mask

    @staticmethod
    def _normalize_reset_mode_probs(probs):
        base = {"fresh": 0.0, "repair": 1.0, "plateau": 0.0}
        if probs:
            for key, value in probs.items():
                if key in base:
                    base[key] = max(0.0, float(value))
        total = sum(base.values())
        if total <= 0.0:
            return {"fresh": 1.0, "repair": 0.0, "plateau": 0.0}
        return {key: value / total for key, value in base.items()}

    def _current_graph_key(self):
        return id(self.base.graph)

    def _snapshot(self):
        env_obj = self.base
        return {
            "pos": {n: env_obj.pos[n].copy() for n in env_obj.graph.nodes()},
            "crossings": {edge: set(s) for edge, s in env_obj.crossings.items()},
            "c_e": dict(env_obj.c_e),
            "E_star": set(env_obj.E_star),
            "global_crossings": int(env_obj.global_crossings),
            "local_crossings": int(env_obj.local_crossings),
            "best_crossings": int(env_obj.best_crossings),
            "best_local_crossings": int(env_obj.best_local_crossings),
            "best_sizemax": int(env_obj.best_sizemax),
            "best_pos": {n: env_obj.best_pos[n].copy() for n in env_obj.graph.nodes()},
            "initial_crossings": int(env_obj.initial_crossings),
            "initial_local_crossings": int(env_obj.initial_local_crossings),
            "last_crossings": int(env_obj.last_crossings),
            "current_node": env_obj.current_node,
            "step_count": int(env_obj.step_count),
            "idle_streak": int(env_obj.idle_streak),
            "unsuccessful_moves": int(env_obj.unsuccessful_moves),
            "node_visit_counts": env_obj.node_visit_counts.copy(),
            "best_improvement_steps": list(env_obj.best_improvement_steps),
            "last_improvement_step": env_obj.last_improvement_step,
            "_last_max_idx": int(env_obj._last_max_idx),
            "_node_visit_repeat_remaining": int(env_obj._node_visit_repeat_remaining),
        }

    def _restore(self, state):
        env_obj = self.base
        env_obj.pos = {n: state["pos"][n].copy() for n in env_obj.graph.nodes()}
        env_obj._rebuild_spatial_indices()
        env_obj.crossings = {edge: set(s) for edge, s in state["crossings"].items()}
        env_obj.c_e = dict(state["c_e"])
        env_obj.E_star = set(state["E_star"])
        env_obj.global_crossings = int(state["global_crossings"])
        env_obj.local_crossings = int(state["local_crossings"])
        env_obj.best_crossings = int(state["best_crossings"])
        env_obj.best_local_crossings = int(state["best_local_crossings"])
        env_obj.best_sizemax = int(state["best_sizemax"])
        env_obj.best_pos = {n: state["best_pos"][n].copy() for n in env_obj.graph.nodes()}
        env_obj.initial_crossings = int(state["initial_crossings"])
        env_obj.initial_local_crossings = int(state["initial_local_crossings"])
        env_obj.last_crossings = int(state["last_crossings"])
        env_obj.current_node = state["current_node"]
        env_obj.step_count = int(state["step_count"])
        env_obj.idle_streak = int(state["idle_streak"])
        env_obj.unsuccessful_moves = int(state["unsuccessful_moves"])
        env_obj.node_visit_counts = state["node_visit_counts"].copy()
        env_obj.best_improvement_steps = list(state["best_improvement_steps"])
        env_obj.last_improvement_step = state["last_improvement_step"]
        env_obj._last_max_idx = int(state["_last_max_idx"])
        env_obj._node_visit_repeat_remaining = int(state["_node_visit_repeat_remaining"])
        self._shortlist_valid = False
        self._no_best_improve_steps = 0

    def _restart_trackers(self):
        env_obj = self.base
        env_obj.node_visit_counts = np.zeros(env_obj.graph.number_of_nodes(), dtype=np.float64)
        env_obj.step_count = 0
        env_obj.idle_streak = 0
        env_obj.unsuccessful_moves = 0
        env_obj.best_improvement_steps = []
        env_obj.last_improvement_step = None
        env_obj.initial_crossings = int(env_obj.global_crossings)
        env_obj.initial_local_crossings = int(env_obj.local_crossings)
        env_obj.last_crossings = int(env_obj.global_crossings)
        env_obj.best_crossings = int(env_obj.global_crossings)
        env_obj.best_local_crossings = int(env_obj.local_crossings)
        env_obj.best_sizemax = len(env_obj.E_star)
        env_obj.best_pos = {n: env_obj.pos[n].copy() for n in env_obj.graph.nodes()}
        env_obj.current_node = env_obj._choose_next_node()
        env_obj._reset_node_visit_repeat_state()
        env_obj.history = None
        self._shortlist_valid = False
        self._no_best_improve_steps = 0

    def set_phase_config(
        self,
        step_limit=None,
        perturb_steps=None,
        attempts=None,
        reset_mode_probs=None,
        plateau_patience=None,
        plateau_bank_size=None,
    ):
        if step_limit is not None:
            self.base.step_limit = int(step_limit)
        if perturb_steps is not None:
            self.perturb_steps = int(perturb_steps)
        if attempts is not None:
            self.attempts = int(attempts)
        if reset_mode_probs is not None:
            self.reset_mode_probs = self._normalize_reset_mode_probs(reset_mode_probs)
        if plateau_patience is not None:
            self.plateau_patience = int(plateau_patience)
        if plateau_bank_size is not None:
            self.plateau_bank_size = int(plateau_bank_size)

    def set_training_graphs(self, graphs):
        if hasattr(self.env, "set_graphs"):
            self.env.set_graphs(graphs)

    def _store_plateau_state(self):
        graph_key = self._current_graph_key()
        score = self._current_objective_key()
        snapshot = self._snapshot()
        bank = self.plateau_bank.setdefault(graph_key, [])
        bank.append((score, snapshot))
        bank.sort(key=lambda item: item[0], reverse=True)
        if len(bank) > self.plateau_bank_size:
            del bank[self.plateau_bank_size:]

    def _current_objective_key(self):
        if self.optimization_goal == "global":
            return (
                int(self.base.global_crossings),
                int(self.base.local_crossings),
            )
        return (
            int(self.base.local_crossings),
            len(self.base.E_star),
            int(self.base.global_crossings),
        )

    def _current_best_objective_value(self):
        if self.optimization_goal == "global":
            return int(self.base.best_crossings)
        return int(self.base.best_local_crossings)

    def _choose_reset_mode(self):
        probs = self.reset_mode_probs
        graph_key = self._current_graph_key()
        plateau_available = len(self.plateau_bank.get(graph_key, [])) > 0
        modes = []
        weights = []
        for mode in ("fresh", "repair", "plateau"):
            if mode == "plateau" and not plateau_available:
                continue
            if mode == "repair" and (self.perturb_steps <= 0 or self.attempts <= 0):
                continue
            weight = float(probs.get(mode, 0.0))
            if weight > 0.0:
                modes.append(mode)
                weights.append(weight)
        if not modes:
            return "fresh"
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.sum(weights)
        return str(self.rng.choice(modes, p=weights))

    def _apply_random_repair_steps(self, num_steps):
        self._refresh_shortlist(force=True)
        for _ in range(int(num_steps)):
            idx = int(self.rng.integers(0, max(1, len(self.shortlist_nodes))))
            self.base.current_node = self.shortlist_nodes[idx]
            self.base._reset_node_visit_repeat_state()
            self.env.step(
                np.array(
                    [
                        self.rng.integers(0, 8),
                        self.rng.integers(0, self.base.n_distances),
                    ],
                    dtype=np.int64,
                )
            )
            self._refresh_shortlist(force=True)

    def _apply_repair_start(self):
        base_state = self._snapshot()
        best_state = None
        best_key = None
        for _ in range(self.attempts):
            self._restore(base_state)
            self._apply_random_repair_steps(self.perturb_steps)
            key = self._current_objective_key()
            if best_state is None or key > best_key:
                best_state = self._snapshot()
                best_key = key
        if best_state is not None:
            self._restore(best_state)
            self._restart_trackers()

    def _apply_plateau_start(self):
        graph_key = self._current_graph_key()
        bank = self.plateau_bank.get(graph_key, [])
        if not bank:
            return False
        choice_idx = int(self.rng.integers(0, len(bank)))
        _, snapshot = bank[choice_idx]
        self._restore(snapshot)
        self._restart_trackers()
        return True

    def prepare_outer_restart(self, jitter_steps=0):
        self.base.reset_to_best_position()
        if jitter_steps > 0:
            self._apply_random_repair_steps(int(jitter_steps))
        self._restart_trackers()
        self._refresh_shortlist(force=True)

    def _refresh_shortlist(self, force=False):
        graph_id = id(self.base.graph)
        if not force and self._shortlist_valid and self._graph_id == graph_id:
            return
        shortlist = self.base.get_node_shortlist(self.shortlist_size, strategy=self.base.node_selection_strategy)
        if not shortlist:
            fallback_nodes = list(self.base.graph.nodes())
            shortlist = [(node, 0.0) for node in fallback_nodes[:self.shortlist_size]]
        self.shortlist_nodes = [node for node, _ in shortlist[:self.shortlist_size]]
        self.shortlist_scores = [float(score) for _, score in shortlist[:self.shortlist_size]]
        self.shortlist_rotations = [0 for _ in self.shortlist_nodes]
        self._graph_id = graph_id
        self._shortlist_valid = True

    def _ensure_shortlist(self):
        self._refresh_shortlist(force=False)

    def _stack_observation(self):
        self._ensure_shortlist()
        pixel_map = np.zeros(
            (self.shortlist_size * self.pixel_channels, self.pixel_height, self.pixel_width),
            dtype=np.float32,
        )
        cross_map = np.zeros((self.shortlist_size * 8,), dtype=np.float32)
        cross_map_local = np.zeros((self.shortlist_size * 8,), dtype=np.float32)
        local_view = np.zeros((self.shortlist_size * 40,), dtype=np.float32)
        local_crossings = np.zeros((self.shortlist_size,), dtype=np.float32)
        global_crossings = np.zeros((self.shortlist_size,), dtype=np.float32)
        shortlist_meta = np.zeros((self.shortlist_size, 5), dtype=np.float32)
        shortlist_mask = np.zeros((self.shortlist_size,), dtype=np.float32)
        shortlist_rotations = [0 for _ in range(self.shortlist_size)]

        saved_node = self.base.current_node
        saved_last_max_idx = int(self.base._last_max_idx)
        max_score = max(1.0, max(self.shortlist_scores, default=0.0))
        max_degree = max(1, int(getattr(self.base, "max_node_degree", 1)))
        local_cross_scalar = float(self.base.local_crossings)
        global_cross_scalar = float(self.base.global_crossings)
        local_cross_denom = max(1.0, local_cross_scalar + 1.0)
        active_nodes = self.shortlist_nodes[:self.shortlist_size]
        batch_local_view, batch_cross, batch_cross_local, batch_rotations = self.base._get_observation_Octant_batch(active_nodes)
        batch_pixel_maps = self.base._get_pixel_map_batch(active_nodes, batch_rotations)

        for slot, node in enumerate(active_nodes):
            score = float(self.shortlist_scores[slot])
            shortlist_rotations[slot] = int(batch_rotations[slot])
            c0 = slot * self.pixel_channels
            c1 = c0 + self.pixel_channels
            pixel_map[c0:c1] = batch_pixel_maps[slot]
            cross_map[slot * 8:(slot + 1) * 8] = batch_cross[slot]
            cross_map_local[slot * 8:(slot + 1) * 8] = batch_cross_local[slot]
            local_view[slot * 40:(slot + 1) * 40] = batch_local_view[slot]
            visit_count = 0.0
            if self.base.node_visit_counts is not None:
                visit_count = float(self.base.node_visit_counts[self.base.node_index[node]])
            incident_cross = 0.0
            for edge in self.base.incident_by_node.get(node, ()):
                incident_cross += float(self.base.c_e.get(edge, len(self.base.crossings.get(edge, ()))))
            shortlist_meta[slot] = np.array(
                [
                    score / max_score,
                    float(self.base.node_degree.get(node, self.base.graph.degree(node))) / float(max_degree),
                    visit_count / max(1.0, float(self.base.step_count) + 1.0),
                    incident_cross / local_cross_denom,
                    1.0,
                ],
                dtype=np.float32,
            )
            shortlist_mask[slot] = 1.0

        local_crossings.fill(local_cross_scalar)
        global_crossings.fill(global_cross_scalar)

        self.base.current_node = saved_node
        self.base._last_max_idx = saved_last_max_idx
        self.shortlist_rotations = shortlist_rotations
        return {
            "pixel_map": pixel_map,
            "cross_map": cross_map,
            "cross_map_local": cross_map_local,
            "local_view": local_view,
            "local_crossings": local_crossings,
            "global_crossings": global_crossings,
            "shortlist_meta": shortlist_meta.reshape(-1),
            "shortlist_mask": shortlist_mask,
        }

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._shortlist_valid = False
        mode = self._choose_reset_mode()
        if mode == "repair":
            self._apply_repair_start()
        elif mode == "plateau":
            restored = self._apply_plateau_start()
            if not restored and self.perturb_steps > 0 and self.attempts > 0:
                self._apply_repair_start()

        self._refresh_shortlist(force=True)
        info["reset_mode"] = mode
        return self._stack_observation(), info

    def step(self, action):
        self._ensure_shortlist()
        valid = max(1, len(self.shortlist_nodes))
        slot = int(np.clip(int(action[0]), 0, valid - 1))
        dir_idx = int(action[1])
        dist_idx = int(action[2])
        chosen_node = self.shortlist_nodes[slot]
        before_best_objective = self._current_best_objective_value()
        self.base.current_node = chosen_node
        if slot < len(self.shortlist_rotations):
            self.base._last_max_idx = int(self.shortlist_rotations[slot])
        self.base._reset_node_visit_repeat_state()
        _, reward, done, truncated, info = self.env.step(np.array([dir_idx, dist_idx], dtype=np.int64))
        best_objective_gain = max(0, before_best_objective - self._current_best_objective_value())
        best_objective_bonus = self.best_local_bonus * float(best_objective_gain)
        reward = float(reward) + best_objective_bonus
        if best_objective_gain > 0:
            self._no_best_improve_steps = 0
        else:
            self._no_best_improve_steps += 1
            if (
                self.plateau_patience > 0
                and self._no_best_improve_steps >= self.plateau_patience
                and self._current_best_objective_value() > 0
            ):
                self._store_plateau_state()
                self._no_best_improve_steps = 0
        self._refresh_shortlist(force=True)
        obs = self._stack_observation()
        info["best_objective_bonus"] = best_objective_bonus
        info["best_objective_gain"] = float(best_objective_gain)
        info["best_local_bonus"] = best_objective_bonus
        info["best_local_gain"] = float(best_objective_gain)
        info["selected_shortlist_slot"] = float(slot)
        info["shortlist_valid_count"] = float(valid)
        info["plateau_streak"] = float(self._no_best_improve_steps)
        return obs, reward, done, truncated, info


def make_repair_env_fn(
    graphs,
    cfg,
    offset,
    seed,
    shortlist_size,
    best_local_bonus,
    optimization_goal,
    perturb_steps,
    attempts,
    reset_mode_probs=None,
    plateau_patience=12,
    plateau_bank_size=8,
):
    def _init():
        random.seed(seed + offset)
        np.random.seed(seed + offset)
        sampler = RoundRobinSampler(graphs, start_offset=offset)
        env_obj = create_graph_layout_env(sampler.sample(), config=cfg)
        wrapped = ResampleWrapper(env_obj, sampler)
        repair_env = ShortlistNodeMoveWrapper(
            wrapped,
            shortlist_size=shortlist_size,
            best_local_bonus=best_local_bonus,
            optimization_goal=optimization_goal,
            perturb_steps=perturb_steps,
            attempts=attempts,
            reset_mode_probs=reset_mode_probs,
            plateau_patience=plateau_patience,
            plateau_bank_size=plateau_bank_size,
            seed=seed + offset,
        )
        repair_env.reset(seed=seed + offset)
        return repair_env

    return _init


def build_model(vec_env, cfg, seed, checkpoint_path):
    device = str(cfg.get("device", "auto"))
    tensorboard_log = checkpoint_path / "tb"

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        vec_env,
        learning_rate=cfg["ppo"]["learning_rate"],
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        n_epochs=cfg["ppo"]["n_epochs"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        clip_range_vf=cfg["ppo"].get("clip_range_vf"),
        ent_coef=cfg["ppo"]["ent_coef"],
        vf_coef=cfg["ppo"]["vf_coef"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        target_kl=cfg["ppo"].get("target_kl"),
        policy_kwargs=build_policy_kwargs(cfg),
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
        device=device,
    )

    logger = _get_model_logger(model)
    if logger is not None:
        for output_format in getattr(logger, "output_formats", []):
            if hasattr(output_format, "max_length"):
                output_format.max_length = max(int(getattr(output_format, "max_length", 0)), 128)

    return model


def load_model_for_resume(vec_env, cfg, checkpoint_path, resume_from):
    device = str(cfg.get("device", "auto"))
    tensorboard_log = checkpoint_path / "tb"
    resume_path = Path(resume_from)
    if not resume_path.is_absolute():
        resume_path = ROOT / resume_path
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    model = MaskablePPO.load(str(resume_path), env=vec_env, device=device)
    model.tensorboard_log = str(tensorboard_log)

    logger = _get_model_logger(model)
    if logger is not None:
        for output_format in getattr(logger, "output_formats", []):
            if hasattr(output_format, "max_length"):
                output_format.max_length = max(int(getattr(output_format, "max_length", 0)), 128)

    return model


def train_stats(model):
    out = {}
    if getattr(model, "ep_info_buffer", None) and len(model.ep_info_buffer) > 0:
        rewards = [ep["r"] for ep in model.ep_info_buffer if "r" in ep]
        lens = [ep["l"] for ep in model.ep_info_buffer if "l" in ep]
        if rewards:
            out["ep_rew_mean"] = float(np.mean(rewards))
        if lens:
            out["ep_len_mean"] = float(np.mean(lens))
    logger_vals = dict(getattr(model.logger, "name_to_value", {}))
    for key in [
        "train/explained_variance",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/approx_kl",
        "train/entropy_loss",
        "train/loss",
    ]:
        if key in logger_vals:
            out[key.split("/", 1)[1]] = float(logger_vals[key])
    return out

def set_optimizer_lr(optimizer, lr_value):
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr_value)


def set_model_learning_rate(model, lr_value):
    lr_value = float(lr_value)
    model.lr_schedule = lambda _: lr_value
    set_optimizer_lr(model.policy.optimizer, lr_value)


def linear_schedule_value(start, end, progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(start + (end - start) * progress)


class EliteReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def add(self, obs, action, action_mask, priority):
        sample = (
            {key: np.array(value, copy=True) for key, value in obs.items()},
            np.array(action, dtype=np.int64, copy=True),
            np.array(action_mask, dtype=bool, copy=True),
            float(priority),
        )
        self.samples.append(sample)
        self.samples.sort(key=lambda item: item[3], reverse=True)
        if len(self.samples) > self.capacity:
            del self.samples[self.capacity:]

    def sample_indices(self, batch_size, rng):
        batch_size = min(int(batch_size), len(self.samples))
        if batch_size <= 0:
            return np.array([], dtype=np.int64)
        return rng.choice(len(self.samples), size=batch_size, replace=False)

    def build_batch(self, indices):
        obs_batch = {}
        first_obs = self.samples[int(indices[0])][0]
        for key in first_obs.keys():
            obs_batch[key] = np.stack([self.samples[int(i)][0][key] for i in indices], axis=0)
        actions = np.stack([self.samples[int(i)][1] for i in indices], axis=0)
        masks = np.stack([self.samples[int(i)][2] for i in indices], axis=0)
        priorities = np.asarray([self.samples[int(i)][3] for i in indices], dtype=np.float32)
        return obs_batch, actions, masks, priorities


def collect_elite_repair_samples(model, graphs, cfg, args, seed, max_episodes):
    if max_episodes <= 0 or not graphs:
        return EliteReplayBuffer(capacity=max(1, int(args.sil_buffer_size)))

    rng = np.random.default_rng(seed)
    buffer = EliteReplayBuffer(capacity=max(1, int(args.sil_buffer_size)))
    was_training = bool(getattr(model.policy, "training", False))
    model.policy.set_training_mode(False)
    try:
        for episode_idx in range(int(max_episodes)):
            graph = graphs[(seed + episode_idx) % len(graphs)]
            env_obj = create_graph_layout_env(graph, config=cfg)
            wrapper = ShortlistNodeMoveWrapper(
                env_obj,
                shortlist_size=args.shortlist_size,
                best_local_bonus=args.best_local_bonus,
                perturb_steps=args.repair_perturb_steps,
                attempts=args.repair_perturb_attempts,
                reset_mode_probs={"fresh": 0.0, "repair": 1.0, "plateau": 0.0},
                plateau_patience=args.plateau_patience,
                plateau_bank_size=args.plateau_bank_size,
                seed=seed + episode_idx,
            )
            obs, _ = wrapper.reset(seed=seed + episode_idx)
            done = False
            truncated = False
            steps = 0
            while steps < args.repair_horizon and not (done or truncated):
                mask = wrapper.action_masks()
                act_obs = {key: np.array(value, copy=True) for key, value in obs.items()}
                action, _ = model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=not bool(args.sil_stochastic_collect),
                )
                next_obs, _, done, truncated, info = wrapper.step(np.asarray(action))
                best_local_gain = float(info.get("best_local_gain", 0.0))
                if best_local_gain > 0.0:
                    buffer.add(
                        act_obs,
                        np.asarray(action, dtype=np.int64),
                        np.asarray(mask, dtype=bool),
                        priority=best_local_gain,
                    )
                obs = next_obs
                steps += 1
            wrapper.close()
    finally:
        model.policy.set_training_mode(was_training)
    return buffer


def run_self_imitation_updates(model, elite_buffer, args, rng):
    if len(elite_buffer) == 0 or int(args.sil_epochs) <= 0 or int(args.sil_batch_size) <= 0:
        return {"elite_samples": len(elite_buffer), "sil_updates": 0, "sil_loss_mean": 0.0}

    policy = model.policy
    optimizer = policy.optimizer
    was_training = bool(getattr(policy, "training", False))
    policy.set_training_mode(True)
    losses = []
    updates = 0
    try:
        for _ in range(int(args.sil_epochs)):
            perm = rng.permutation(len(elite_buffer))
            for start in range(0, len(perm), int(args.sil_batch_size)):
                idx = perm[start:start + int(args.sil_batch_size)]
                if len(idx) == 0:
                    continue
                obs_batch, actions_batch, masks_batch, priorities = elite_buffer.build_batch(idx)
                obs_tensor, _ = policy.obs_to_tensor(obs_batch)
                actions_tensor = torch.as_tensor(actions_batch, device=policy.device).long()
                masks_tensor = torch.as_tensor(masks_batch, device=policy.device)
                _, log_prob, _ = policy.evaluate_actions(obs_tensor, actions_tensor, action_masks=masks_tensor)
                weights = torch.as_tensor(priorities, device=policy.device)
                weights = weights / torch.clamp(weights.mean(), min=1e-6)
                loss = -float(args.sil_weight) * torch.mean(log_prob * weights)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(policy.parameters(), float(model.max_grad_norm))
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                updates += 1
    finally:
        policy.set_training_mode(was_training)
    return {
        "elite_samples": len(elite_buffer),
        "sil_updates": updates,
        "sil_loss_mean": float(np.mean(losses)) if losses else 0.0,
    }


def summarize_rows(rows):
    initial_global = np.array([r[0] for r in rows], dtype=np.float64)
    best_global = np.array([r[1] for r in rows], dtype=np.float64)
    initial_local = np.array([r[2] for r in rows], dtype=np.float64)
    best_local = np.array([r[3] for r in rows], dtype=np.float64)
    return {
        "mean_best_global_ratio": float(np.mean(best_global / np.maximum(initial_global, 1.0))),
        "mean_best_global_removed": float(np.mean(initial_global - best_global)),
        "mean_best_local_ratio": float(np.mean(best_local / np.maximum(initial_local, 1.0))),
        "mean_best_local_removed": float(np.mean(initial_local - best_local)),
        "best_global_solved_frac": float(np.mean(best_global == 0)),
        "best_local_solved_frac": float(np.mean(best_local == 0)),
    }


def _stack_obs_batch(obs_list):
    keys = obs_list[0].keys()
    return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in keys}


def _run_eval_batch(model, wrappers, reset_seeds, horizon):
    obs_list = []
    rows = []
    done = np.zeros(len(wrappers), dtype=bool)
    truncated = np.zeros(len(wrappers), dtype=bool)
    steps = np.zeros(len(wrappers), dtype=np.int32)
    initial_global = np.zeros(len(wrappers), dtype=np.int32)
    initial_local = np.zeros(len(wrappers), dtype=np.int32)

    for idx, wrapper in enumerate(wrappers):
        obs, _ = wrapper.reset(seed=reset_seeds[idx])
        obs_list.append(obs)
        initial_global[idx] = int(wrapper.base.initial_crossings)
        initial_local[idx] = int(wrapper.base.initial_local_crossings)

    while True:
        active = [idx for idx in range(len(wrappers)) if steps[idx] < horizon and not (done[idx] or truncated[idx])]
        if not active:
            break

        batch_obs = _stack_obs_batch([obs_list[idx] for idx in active])
        batch_masks = np.stack([wrappers[idx].action_masks() for idx in active], axis=0)
        actions, _ = model.predict(batch_obs, action_masks=batch_masks, deterministic=True)
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        for local_idx, wrapper_idx in enumerate(active):
            obs, _, done_flag, truncated_flag, _ = wrappers[wrapper_idx].step(actions[local_idx])
            obs_list[wrapper_idx] = obs
            done[wrapper_idx] = bool(done_flag)
            truncated[wrapper_idx] = bool(truncated_flag)
            steps[wrapper_idx] += 1

    for idx, wrapper in enumerate(wrappers):
        rows.append(
            (
                int(initial_global[idx]),
                int(wrapper.base.best_crossings),
                int(initial_local[idx]),
                int(wrapper.base.best_local_crossings),
            )
        )
        wrapper.close()
    return rows


def _run_rollout_batch(model, wrappers, obs_list, horizon, deterministic=True):
    done = np.zeros(len(wrappers), dtype=bool)
    truncated = np.zeros(len(wrappers), dtype=bool)
    steps = np.zeros(len(wrappers), dtype=np.int32)

    while True:
        active = [idx for idx in range(len(wrappers)) if steps[idx] < horizon and not (done[idx] or truncated[idx])]
        if not active:
            break

        batch_obs = _stack_obs_batch([obs_list[idx] for idx in active])
        batch_masks = np.stack([wrappers[idx].action_masks() for idx in active], axis=0)
        actions, _ = model.predict(batch_obs, action_masks=batch_masks, deterministic=deterministic)
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        for local_idx, wrapper_idx in enumerate(active):
            obs, _, done_flag, truncated_flag, _ = wrappers[wrapper_idx].step(actions[local_idx])
            obs_list[wrapper_idx] = obs
            done[wrapper_idx] = bool(done_flag)
            truncated[wrapper_idx] = bool(truncated_flag)
            steps[wrapper_idx] += 1

    return obs_list


def eval_standard(model, eval_graphs, cfg, seed, horizon):
    rows = []
    outer_restarts = int(cfg.get("experiment", {}).get("eval_outer_restarts", 1))
    restart_jitter_steps = int(cfg.get("experiment", {}).get("eval_restart_perturb_steps", 0))
    batch_size = int(cfg.get("experiment", {}).get("eval_batch_size", 8))
    was_training = bool(getattr(model.policy, "training", False))
    model.policy.set_training_mode(False)
    try:
        for start in range(0, len(eval_graphs), batch_size):
            batch_graphs = eval_graphs[start:start + batch_size]
            wrappers = []
            obs_list = []
            batch_rows = []

            for offset, graph in enumerate(batch_graphs):
                idx = start + offset
                env_obj = create_graph_layout_env(graph, config=cfg)
                wrapper = ShortlistNodeMoveWrapper(
                    env_obj,
                    shortlist_size=cfg["experiment"]["shortlist_size"],
                    best_local_bonus=cfg["experiment"]["best_local_bonus"],
                    optimization_goal=cfg["experiment"].get("optimization_goal", "local"),
                    perturb_steps=0,
                    attempts=0,
                    seed=seed + idx,
                )
                obs, _ = wrapper.reset(seed=seed + idx)
                wrappers.append(wrapper)
                obs_list.append(obs)
                batch_rows.append(
                    [
                        int(wrapper.base.initial_crossings),
                        int(wrapper.base.best_crossings),
                        int(wrapper.base.initial_local_crossings),
                        int(wrapper.base.best_local_crossings),
                    ]
                )

            for restart_idx in range(max(1, outer_restarts)):
                obs_list = _run_rollout_batch(model, wrappers, obs_list, horizon, deterministic=True)
                for row_idx, wrapper in enumerate(wrappers):
                    batch_rows[row_idx][1] = min(batch_rows[row_idx][1], int(wrapper.base.best_crossings))
                    batch_rows[row_idx][3] = min(batch_rows[row_idx][3], int(wrapper.base.best_local_crossings))
                if restart_idx + 1 < outer_restarts:
                    for row_idx, wrapper in enumerate(wrappers):
                        if batch_rows[row_idx][1] > 0:
                            wrapper.prepare_outer_restart(jitter_steps=restart_jitter_steps)
                            obs_list[row_idx] = wrapper._stack_observation()

            rows.extend(tuple(row) for row in batch_rows)
            for wrapper in wrappers:
                wrapper.close()
    finally:
        model.policy.set_training_mode(was_training)
    return summarize_rows(rows)


def eval_repair(model, eval_graphs, cfg, seed, perturb_steps, attempts, horizon):
    rows = []
    batch_size = int(cfg.get("experiment", {}).get("eval_batch_size", 4))
    was_training = bool(getattr(model.policy, "training", False))
    model.policy.set_training_mode(False)
    try:
        for start in range(0, len(eval_graphs), batch_size):
            batch_graphs = eval_graphs[start:start + batch_size]
            wrappers = []
            reset_seeds = []
            for offset, graph in enumerate(batch_graphs):
                idx = start + offset
                env_obj = create_graph_layout_env(graph, config=cfg)
                wrappers.append(
                    ShortlistNodeMoveWrapper(
                        env_obj,
                        shortlist_size=cfg["experiment"]["shortlist_size"],
                        best_local_bonus=cfg["experiment"]["best_local_bonus"],
                        optimization_goal=cfg["experiment"].get("optimization_goal", "local"),
                        perturb_steps=perturb_steps,
                        attempts=attempts,
                        seed=seed + idx,
                    )
                )
                reset_seeds.append(seed + idx)
            rows.extend(_run_eval_batch(model, wrappers, reset_seeds, horizon))
    finally:
        model.policy.set_training_mode(was_training)
    return summarize_rows(rows)


def sort_graphs_by_difficulty(graphs):
    return sorted(
        graphs,
        key=lambda g: (
            int(g.number_of_nodes()),
            int(g.number_of_edges()),
            float(nx.density(g)) if g.number_of_nodes() > 1 else 0.0,
        ),
    )


def build_training_phases(train_graphs, args):
    if getattr(args, "plateau_only_replay", False):
        step_total = int(args.ppo_steps)
        warmup_steps = max(1, int(round(step_total * 0.35)))
        replay_steps = max(1, step_total - warmup_steps)
        return [
            {
                "name": "warmup",
                "steps": warmup_steps,
                "graphs": list(train_graphs),
                "step_limit": int(args.repair_horizon),
                "perturb_steps": int(args.repair_perturb_steps),
                "attempts": int(args.repair_perturb_attempts),
                "reset_mode_probs": {"fresh": 0.0, "repair": 1.0, "plateau": 0.0},
                "plateau_patience": int(args.plateau_patience),
            },
            {
                "name": "plateau_replay",
                "steps": replay_steps,
                "graphs": list(train_graphs),
                "step_limit": int(args.repair_horizon),
                "perturb_steps": int(args.repair_perturb_steps),
                "attempts": int(args.repair_perturb_attempts),
                "reset_mode_probs": {"fresh": 0.0, "repair": 0.8, "plateau": 0.2},
                "plateau_patience": int(args.plateau_patience),
            },
        ]

    if not getattr(args, "broadened_repair", False):
        return [
            {
                "name": "baseline",
                "steps": int(args.ppo_steps),
                "graphs": list(train_graphs),
                "step_limit": int(args.repair_horizon),
                "perturb_steps": int(args.repair_perturb_steps),
                "attempts": int(args.repair_perturb_attempts),
                "reset_mode_probs": {"fresh": 0.0, "repair": 1.0, "plateau": 0.0},
                "plateau_patience": int(args.plateau_patience),
            }
        ]

    step_total = int(args.ppo_steps)
    phase1_steps = max(1, int(round(step_total * 0.40)))
    phase2_steps = max(1, step_total - phase1_steps)

    return [
        {
            "name": "warmup",
            "steps": phase1_steps,
            "graphs": list(train_graphs),
            "step_limit": 32,
            "perturb_steps": max(4, min(args.repair_perturb_steps, 6)),
            "attempts": max(2, min(args.repair_perturb_attempts, 3)),
            "reset_mode_probs": {"fresh": 0.30, "repair": 0.70, "plateau": 0.00},
            "plateau_patience": max(8, args.plateau_patience + 2),
        },
        {
            "name": "full",
            "steps": phase2_steps,
            "graphs": list(train_graphs),
            "step_limit": 64,
            "perturb_steps": max(6, args.repair_perturb_steps),
            "attempts": max(3, args.repair_perturb_attempts),
            "reset_mode_probs": {"fresh": 0.20, "repair": 0.60, "plateau": 0.20},
            "plateau_patience": max(6, args.plateau_patience),
        },
    ]


def mean_dict(dicts):
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def append_jsonl(path_str, payload):
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def build_training_vec_env(env_fns, gamma, normalize_reward=True, reward_clip=5.0):
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    if normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=False,
            norm_reward=True,
            clip_reward=float(reward_clip),
            gamma=float(gamma),
        )
    return vec_env


def save_checkpoint(model, checkpoint_path: Path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_path))
    vec_env = model.get_env()
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(checkpoint_path.with_suffix(".vecnormalize.pkl")))


def _logger_record_dict(logger, prefix, values):
    if logger is None or not values:
        return
    for key, value in values.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            _logger_record_dict(logger, name, value)
        elif value is None:
            continue
        elif isinstance(value, bool):
            logger.record(name, float(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            logger.record(name, float(value))
        else:
            try:
                logger.record(name, float(value))
            except (TypeError, ValueError):
                pass


def _logger_record_phase(logger, phase_idx, phase, trained_steps, seed):
    if logger is None:
        return
    logger.record("phase/index", float(phase_idx))
    logger.record("phase/seed", float(seed))
    logger.record("phase/trained_steps_at_start", float(trained_steps))
    logger.record("phase/phase_steps", float(phase["steps"]))
    logger.record("phase/graph_count", float(len(phase["graphs"])))
    logger.record("phase/step_limit", float(phase["step_limit"]))
    logger.record("phase/perturb_steps", float(phase["perturb_steps"]))
    logger.record("phase/attempts", float(phase["attempts"]))
    logger.record("phase/plateau_patience", float(phase["plateau_patience"]))

    reset_mode_probs = phase.get("reset_mode_probs", {})
    for mode_name, prob in reset_mode_probs.items():
        logger.record(f"phase/reset_mode_prob_{mode_name}", float(prob))


def _get_model_logger(model):
    return getattr(model, "_logger", None)


def _safe_record(model, key, value):
    logger = _get_model_logger(model)
    if logger is None:
        return
    if value is None:
        return
    if isinstance(value, bool):
        logger.record(key, float(value))
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        logger.record(key, float(value))
        return
    try:
        logger.record(key, float(value))
    except (TypeError, ValueError):
        pass


def _safe_record_dict(model, prefix, values):
    logger = _get_model_logger(model)
    if logger is None:
        return
    _logger_record_dict(logger, prefix, values)


def _safe_record_summary_dict(model, prefix, values):
    alias_map = {
        "mean_best_global_ratio": "best_global_ratio",
        "mean_best_global_removed": "best_global_removed",
        "mean_best_local_ratio": "best_local_ratio",
        "mean_best_local_removed": "best_local_removed",
        "best_global_solved_frac": "global_solved_frac",
        "best_local_solved_frac": "local_solved_frac",
    }
    for key, value in values.items():
        _safe_record(model, f"{prefix}/{alias_map.get(key, key)}", value)


def _ensure_logger_max_length(logger, min_length=128):
    if logger is None:
        return
    for output_format in getattr(logger, "output_formats", []):
        if hasattr(output_format, "max_length"):
            output_format.max_length = max(int(getattr(output_format, "max_length", 0)), int(min_length))


def _safe_dump(model, step):
    logger = _get_model_logger(model)
    if logger is None:
        return
    _ensure_logger_max_length(logger)
    logger.dump(step=int(step))


def run_repair_ppo(train_graphs, eval_graphs, seed, args, base_config=None):
    checkpoint_metric_name = (
        "mean_best_global_ratio"
        if str(args.optimization_goal) == "global"
        else "mean_best_local_ratio"
    )

    mix_config = None
    if any(int(value) > 0 for value in (args.train_rome_count, args.train_ba_count, args.eval_rome_count, args.eval_ba_count)):
        mix_config = {
            "train_rome_count": int(args.train_rome_count),
            "train_ba_count": int(args.train_ba_count),
            "eval_rome_count": int(args.eval_rome_count),
            "eval_ba_count": int(args.eval_ba_count),
        }
    missing_graphs = {"train": [], "eval": []}

    phases = build_training_phases(train_graphs, args)

    repair_cfg = make_cfg_from_args(
        args,
        step_limit=phases[0]["step_limit"],
        reset_threshold=None,
        base_config=base_config,
    )

    experiment_config = {
        "shortlist_size": args.shortlist_size,
        "best_local_bonus": args.best_local_bonus,
        "optimization_goal": args.optimization_goal,
        "node_selection_strategy": args.node_selection_strategy,
        "local_weight": args.local_weight,
        "sizemax_weight": args.sizemax_weight,
        "global_weight": args.global_weight,
        "standard_horizon": args.standard_horizon,
        "repair_horizon": args.repair_horizon,
        "repair_perturb_steps": args.repair_perturb_steps,
        "repair_perturb_attempts": args.repair_perturb_attempts,
        "plateau_patience": args.plateau_patience,
        "plateau_bank_size": args.plateau_bank_size,
        "broadened_repair": bool(args.broadened_repair),
        "plateau_only_replay": bool(args.plateau_only_replay),
        "eval_outer_restarts": args.eval_outer_restarts,
        "eval_restart_perturb_steps": args.eval_restart_perturb_steps,
        "eval_batch_size": args.eval_batch_size,
    }
    repair_cfg["run"] = {
        "seed": int(seed),
        "seeds": [int(seed)],
        "n_envs": int(args.n_envs),
        "ppo_steps": int(args.ppo_steps),
        "device": str(args.device),
        "exp_name": getattr(args, "exp_name", None),
    }

    env_fns = [
        make_repair_env_fn(
            phases[0]["graphs"],
            repair_cfg,
            offset=i,
            seed=seed,
            shortlist_size=args.shortlist_size,
            best_local_bonus=args.best_local_bonus,
            optimization_goal=args.optimization_goal,
            perturb_steps=phases[0]["perturb_steps"],
            attempts=phases[0]["attempts"],
            reset_mode_probs=phases[0]["reset_mode_probs"],
            plateau_patience=phases[0]["plateau_patience"],
            plateau_bank_size=args.plateau_bank_size,
        )
        for i in range(args.n_envs)
    ]

    vec_env = build_training_vec_env(
        env_fns,
        gamma=repair_cfg["ppo"]["gamma"],
        normalize_reward=args.normalize_reward,
        reward_clip=args.reward_clip,
    )

    checkpoint_root = Path(args.checkpoint_root)
    if getattr(args, "exp_name", None):
        checkpoint_root = ROOT / "results" / args.exp_name / "checkpoints"

    repair_cfg["device"] = str(args.device)
    repair_cfg["exp_name"] = getattr(args, "exp_name", None)
    resume_from = getattr(args, "resume_from", None)
    if resume_from and args.normalize_reward:
        resume_norm_path = Path(resume_from)
        if not resume_norm_path.is_absolute():
            resume_norm_path = ROOT / resume_norm_path
        resume_norm_path = resume_norm_path.with_suffix(".vecnormalize.pkl")
        if resume_norm_path.exists():
            base_venv = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
            vec_env = VecNormalize.load(str(resume_norm_path), base_venv)
            vec_env.training = True
        else:
            raise FileNotFoundError(
                f"VecNormalize state for resume checkpoint not found: {resume_norm_path}"
            )

    if resume_from:
        model = load_model_for_resume(vec_env, repair_cfg, checkpoint_root, resume_from)
    else:
        model = build_model(vec_env, repair_cfg, seed, checkpoint_root)

    standard_cfg = make_cfg_from_args(
        args,
        step_limit=args.standard_horizon,
        reset_threshold=64,
        base_config=base_config,
    )
    standard_cfg["device"] = str(args.device)
    standard_cfg["experiment"] = _deep_copy_dict(experiment_config)

    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_config_path = checkpoint_dir / "config.json"
    # Start with the complete loaded config - it's the authoritative source
    checkpoint_config = _deep_copy_dict(base_config)
    
    # Ensure all required sections are present (they should be from validation)
    checkpoint_config.setdefault("env", {})
    checkpoint_config.setdefault("policy", {})
    checkpoint_config.setdefault("ppo", {})
    checkpoint_config.setdefault("training", {})
    checkpoint_config.setdefault("io", {})
    
    # Update with computed values from training
    checkpoint_config["env"] = _deep_copy_dict(repair_cfg["env"])
    checkpoint_config["policy"] = _deep_copy_dict(repair_cfg["policy"])
    checkpoint_config["ppo"] = _deep_copy_dict(repair_cfg["ppo"])
    checkpoint_config["experiment"].update(repair_cfg["experiment"])
    checkpoint_config["run"].update(repair_cfg["run"])
    checkpoint_config["training"].update({
        "eval_freq_steps": args.eval_freq_steps,
        "normalize_reward": args.normalize_reward,
        "reward_clip": args.reward_clip,
        "use_lr_ent_schedule": args.use_lr_ent_schedule,
        "final_learning_rate": args.final_learning_rate,
        "final_ent_coef": args.final_ent_coef,
        "use_self_imitation": args.use_self_imitation,
        "sil_collect_episodes": args.sil_collect_episodes,
        "sil_buffer_size": args.sil_buffer_size,
        "sil_batch_size": args.sil_batch_size,
        "sil_epochs": args.sil_epochs,
        "sil_weight": args.sil_weight,
        "sil_stochastic_collect": args.sil_stochastic_collect,
    })
    checkpoint_config["io"].update({
        "checkpoint_root": str(checkpoint_root),
        "output_json": args.output_json,
        "output_jsonl": args.output_jsonl,
    })
    
    checkpoint_config_path.write_text(
        json.dumps(checkpoint_config, indent=2),
        encoding="utf-8",
    )

    latest_path = checkpoint_dir / "latest_model.zip"
    best_path = checkpoint_dir / "best_model.zip"

    runtime_start = time.perf_counter()
    total_steps = int(args.ppo_steps)
    eval_freq_steps = max(1, int(args.eval_freq_steps))
    trained_steps = int(getattr(model, "num_timesteps", 0))
    best_metric = float("inf")
    best_standard_eval = None
    last_standard_eval = None
    best_step = trained_steps
    phase_start = 0
    sil_rng = np.random.default_rng(seed + 4242)
    sil_stats_history = []

    init_lr = float(repair_cfg["ppo"]["learning_rate"])
    final_lr = float(
        args.final_learning_rate
        if args.final_learning_rate is not None
        else init_lr
    )
    init_ent = float(repair_cfg["ppo"]["ent_coef"])
    final_ent = float(
        args.final_ent_coef
        if args.final_ent_coef is not None
        else init_ent
    )

    _safe_record(model, "run/seed", seed)
    _safe_record(model, "run/total_steps", total_steps)
    _safe_record(model, "run/eval_freq_steps", eval_freq_steps)
    _safe_record(model, "run/n_envs", args.n_envs)
    _safe_record(model, "run/normalize_reward", bool(args.normalize_reward))
    _safe_record(model, "run/reward_clip", float(args.reward_clip))
    _safe_record(model, "run/use_lr_ent_schedule", bool(args.use_lr_ent_schedule))
    _safe_record(model, "run/use_self_imitation", bool(args.use_self_imitation))
    _safe_record(model, "run/checkpoint_metric_is_global", float(str(args.optimization_goal) == "global"))
    _safe_record(model, "run/resume_from", 1.0 if resume_from else 0.0)
    _safe_dump(model, trained_steps)

    if resume_from:
        best_standard_eval = eval_standard(
            model,
            eval_graphs,
            standard_cfg,
            seed + 5000 + trained_steps,
            args.standard_horizon,
        )
        best_metric = float(best_standard_eval[checkpoint_metric_name])
        best_step = trained_steps
        save_checkpoint(model, latest_path)
        save_checkpoint(model, best_path)

    for phase_idx, phase in enumerate(phases):
        vec_env.env_method("set_training_graphs", phase["graphs"])
        vec_env.env_method(
            "set_phase_config",
            phase["step_limit"],
            phase["perturb_steps"],
            phase["attempts"],
            phase["reset_mode_probs"],
            phase["plateau_patience"],
            args.plateau_bank_size,
        )

        logger = _get_model_logger(model)
        _logger_record_phase(logger, phase_idx, phase, trained_steps, seed)
        _safe_dump(model, trained_steps)

        phase_target = phase_start + int(phase["steps"])

        while trained_steps < min(phase_target, total_steps):
            chunk = min(
                eval_freq_steps,
                min(phase_target, total_steps) - trained_steps,
            )

            current_lr = float(model.policy.optimizer.param_groups[0]["lr"])
            current_ent = float(model.ent_coef)

            if args.use_lr_ent_schedule:
                progress = float(trained_steps) / float(max(1, total_steps))
                current_lr = linear_schedule_value(init_lr, final_lr, progress)
                current_ent = linear_schedule_value(init_ent, final_ent, progress)
                set_model_learning_rate(model, current_lr)
                model.ent_coef = float(current_ent)

                _safe_record(model, "schedule/enabled", True)
                _safe_record(model, "schedule/progress", progress)
                _safe_record(model, "schedule/learning_rate", current_lr)
                _safe_record(model, "schedule/ent_coef", current_ent)

            _safe_record(model, "train/chunk_steps", chunk)
            _safe_record(model, "train/trained_steps_before_chunk", trained_steps)
            _safe_record(model, "phase/progress_in_phase", trained_steps - phase_start)
            _safe_record(
                model,
                "phase/fraction_complete",
                float(trained_steps - phase_start) / float(max(1, phase_target - phase_start)),
            )

            model.learn(
                total_timesteps=chunk,
                progress_bar=False,
                reset_num_timesteps=(trained_steps == 0),
            )
            trained_steps += chunk

            train_metrics = train_stats(model)
            _safe_record_dict(model, "train_stats", train_metrics)

            sil_stats = {
                "elite_samples": 0,
                "sil_updates": 0,
                "sil_loss_mean": 0.0,
            }
            if args.use_self_imitation:
                elite_buffer = collect_elite_repair_samples(
                    model,
                    train_graphs,
                    repair_cfg,
                    args,
                    seed + trained_steps + phase_idx * 1000,
                    max_episodes=int(args.sil_collect_episodes),
                )
                sil_stats = run_self_imitation_updates(
                    model,
                    elite_buffer,
                    args,
                    sil_rng,
                )

            sil_stats_history.append(sil_stats)
            _safe_record_dict(model, "self_imitation", sil_stats)

            standard_eval = eval_standard(
                model,
                eval_graphs,
                standard_cfg,
                seed + 5000 + trained_steps,
                args.standard_horizon,
            )
            last_standard_eval = standard_eval
            _safe_record_dict(model, "eval/standard", standard_eval)

            save_checkpoint(model, latest_path)
            _safe_record(model, "checkpoint/latest_saved", 1.0)
            _safe_record(model, "checkpoint/latest_step", trained_steps)

            metric = float(standard_eval[checkpoint_metric_name])
            _safe_record(model, "checkpoint/selection_metric", metric)
            _safe_record(model, "checkpoint/best_metric_so_far_before_update", best_metric)
            _safe_record(model, "checkpoint/is_best", 0.0)

            if metric < best_metric:
                best_metric = metric
                best_standard_eval = standard_eval
                best_step = trained_steps
                save_checkpoint(model, best_path)

                _safe_record(model, "checkpoint/is_best", 1.0)
                _safe_record(model, "checkpoint/best_step", best_step)
                _safe_record(model, "checkpoint/best_metric_so_far", best_metric)

            _safe_record(model, "train/trained_steps", trained_steps)
            _safe_record(model, "runtime/elapsed_sec", time.perf_counter() - runtime_start)
            _safe_record(model, "phase/index_current", phase_idx)
            _safe_dump(model, trained_steps)

        phase_start = phase_target

    runtime_sec = time.perf_counter() - runtime_start

    selected_model = model
    if best_path.exists():
        device_for_loading = str(repair_cfg.get("device", "auto"))
        selected_model = MaskablePPO.load(str(best_path), device=device_for_loading)

    if best_standard_eval is None:
        best_standard_eval = eval_standard(
            selected_model,
            eval_graphs,
            standard_cfg,
            seed + 5000,
            args.standard_horizon,
        )
        best_step = total_steps
        best_metric = float(best_standard_eval[checkpoint_metric_name])

    selected_repair_eval = eval_repair(
        selected_model,
        eval_graphs,
        repair_cfg,
        seed + 6000 + best_step,
        args.repair_perturb_steps,
        args.repair_perturb_attempts,
        args.repair_horizon,
    )

    if trained_steps == total_steps and last_standard_eval is not None:
        final_standard_eval = last_standard_eval
    else:
        final_standard_eval = eval_standard(
            model,
            eval_graphs,
            standard_cfg,
            seed + 7000,
            args.standard_horizon,
        )

    final_train_stats = train_stats(model)

    _safe_record_dict(model, "final/train", final_train_stats)
    _safe_record_summary_dict(model, "final/standard_eval_best", best_standard_eval)
    _safe_record_summary_dict(model, "final/repair_eval_best_checkpoint", selected_repair_eval)
    _safe_record_summary_dict(model, "final/standard_eval_last_model", final_standard_eval)

    _safe_record(model, "final/runtime_sec", runtime_sec)
    _safe_record(model, "final/best_checkpoint_step", best_step)
    _safe_record(model, "final/best_checkpoint_metric", best_metric)
    _safe_record(model, "final/normalize_reward", bool(args.normalize_reward))
    _safe_record(model, "final/reward_clip", float(args.reward_clip))

    _safe_record(model, "final/schedule_enabled", bool(args.use_lr_ent_schedule))
    _safe_record(model, "final/schedule_initial_learning_rate", init_lr)
    _safe_record(model, "final/schedule_final_learning_rate", final_lr)
    _safe_record(model, "final/schedule_initial_ent_coef", init_ent)
    _safe_record(model, "final/schedule_final_ent_coef", final_ent)

    _safe_record(model, "final/self_imitation_enabled", bool(args.use_self_imitation))
    _safe_record(
        model,
        "final/self_imitation_mean_elite_samples",
        float(np.mean([s["elite_samples"] for s in sil_stats_history])) if sil_stats_history else 0.0,
    )
    _safe_record(
        model,
        "final/self_imitation_mean_sil_updates",
        float(np.mean([s["sil_updates"] for s in sil_stats_history])) if sil_stats_history else 0.0,
    )
    _safe_record(
        model,
        "final/self_imitation_mean_sil_loss",
        float(np.mean([s["sil_loss_mean"] for s in sil_stats_history])) if sil_stats_history else 0.0,
    )

    _safe_dump(model, trained_steps)

    result = {
        "runtime_sec": runtime_sec,
        "train": final_train_stats,
        "standard_eval": best_standard_eval,
        "repair_eval": selected_repair_eval,
        "final_standard_eval": final_standard_eval,
        "best_checkpoint_step": int(best_step),
        "best_checkpoint_metric_name": checkpoint_metric_name,
        "best_checkpoint_metric": float(best_metric),
        "best_checkpoint_path": str(best_path),
        "latest_checkpoint_path": str(latest_path),
        "normalize_reward": bool(args.normalize_reward),
        "reward_clip": float(args.reward_clip),
        "schedule": {
            "enabled": bool(args.use_lr_ent_schedule),
            "initial_learning_rate": init_lr,
            "final_learning_rate": final_lr,
            "initial_ent_coef": init_ent,
            "final_ent_coef": final_ent,
        },
        "self_imitation": {
            "enabled": bool(args.use_self_imitation),
            "mean_elite_samples": float(np.mean([s["elite_samples"] for s in sil_stats_history])) if sil_stats_history else 0.0,
            "mean_sil_updates": float(np.mean([s["sil_updates"] for s in sil_stats_history])) if sil_stats_history else 0.0,
            "mean_sil_loss": float(np.mean([s["sil_loss_mean"] for s in sil_stats_history])) if sil_stats_history else 0.0,
        },
    }

    vec_env.close()
    return result


def summarize_repair_runs(seed_runs):
    rows = []
    for seed, payload in seed_runs:
        rows.append(
            {
                "seed": seed,
                "runtime_sec": payload["runtime_sec"],
                "train": payload["train"],
                "standard_eval": payload["standard_eval"],
                "repair_eval": payload["repair_eval"],
            }
        )
    return {
        "seeds": rows,
        "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in rows])),
        "mean_train": mean_dict([row["train"] for row in rows]),
        "mean_standard_eval": mean_dict([row["standard_eval"] for row in rows]),
        "mean_repair_eval": mean_dict([row["repair_eval"] for row in rows]),
    }


def dataset_graph_root_and_format(dataset_type):
    project_root = ROOT
    if dataset_type == "rome":
        return project_root / "graphs" / "rome_filtered" / "splits" / "data", "gml"
    if dataset_type == "extended_BA":
        return project_root / "graphs" / "extended_BA_filtered" / "data", "gml"
    if dataset_type == "contest":
        return project_root / "graphs" / "contest_filtered" / "data", "json"
    raise ValueError("Invalid dataset_type. Must be 'rome', 'extended_BA', or 'contest'.")


def read_graph_names_from_file(list_path):
    list_path = Path(list_path)
    suffix = list_path.suffix.lower()
    names = []
    if suffix == ".txt":
        with list_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    elif suffix == ".csv":
        with list_path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delimiter = ";" if sample.count(";") >= sample.count(",") else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames:
                fieldnames = [name.strip() for name in reader.fieldnames]
                key = None
                for candidate in ("instance", "graph", "file", "filename", "path"):
                    if candidate in fieldnames:
                        key = candidate
                        break
                if key is not None:
                    for row in reader:
                        value = row.get(key, "")
                        if value:
                            names.append(value.strip())
                else:
                    f.seek(0)
                    plain_reader = csv.reader(f, delimiter=delimiter)
                    next(plain_reader, None)
                    for row in plain_reader:
                        if row and row[0].strip():
                            names.append(row[0].strip())
            else:
                f.seek(0)
                plain_reader = csv.reader(f, delimiter=delimiter)
                for row in plain_reader:
                    if row and row[0].strip():
                        names.append(row[0].strip())
    else:
        raise ValueError(f"Unsupported list file format: {list_path}")
    return names


def load_graphs_from_list_file(list_path, dataset_type):
    graph_root, file_format = dataset_graph_root_and_format(dataset_type)
    names = read_graph_names_from_file(list_path)
    graphs = []
    missing = []
    for raw_name in names:
        graph_name = Path(raw_name).name
        graph_path = graph_root / graph_name
        if not graph_path.exists():
            missing.append(graph_name)
            continue
        if file_format == "gml":
            g_orig = nx.read_gml(graph_path)
            graph = nx.convert_node_labels_to_integers(g_orig, label_attribute="original_label")
        elif file_format == "gexf":
            g_orig = nx.read_gexf(graph_path)
            graph = nx.convert_node_labels_to_integers(g_orig, label_attribute="original_label")
        elif file_format == "json":
            from util.load_graph import load_graph as load_json_graph

            graph, _, _ = load_json_graph(graph_path)
        else:
            raise ValueError(f"Unsupported graph file format: {file_format}")
        graphs.append(graph)
    return names, graphs, missing


def load_graphs_from_split_count(split_type, dataset_type, count):
    if int(count) <= 0:
        return []
    dataset = load_split_dataset(split_type, dataset_type=dataset_type)
    max_count = min(int(count), len(dataset))
    return [dataset[i] for i in range(max_count)]


def load_mixed_graphs_from_counts(args):
    train_graphs = []
    eval_graphs = []
    mix_config = {
        "train_rome_count": int(args.train_rome_count),
        "train_ba_count": int(args.train_ba_count),
        "eval_rome_count": int(args.eval_rome_count),
        "eval_ba_count": int(args.eval_ba_count),
    }
    if mix_config["train_rome_count"] > 0:
        train_graphs.extend(load_graphs_from_split_count("train", "rome", mix_config["train_rome_count"]))
    if mix_config["train_ba_count"] > 0:
        train_graphs.extend(load_graphs_from_split_count("train", "extended_BA", mix_config["train_ba_count"]))
    if mix_config["eval_rome_count"] > 0:
        eval_graphs.extend(load_graphs_from_split_count("test", "rome", mix_config["eval_rome_count"]))
    if mix_config["eval_ba_count"] > 0:
        eval_graphs.extend(load_graphs_from_split_count("test", "extended_BA", mix_config["eval_ba_count"]))
    return train_graphs, eval_graphs, mix_config


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config_repair_ppo.json"))
    config_args, _ = config_parser.parse_known_args()
    config_defaults, repair_base_config = load_repair_config_bundle(config_args.config)

    parser = argparse.ArgumentParser(description="Repair PPO scratch experiment for move_distance.")
    parser.add_argument("--config", type=str, default=config_args.config, help="Path to a JSON config file")
    parser.add_argument("--dataset", type=str, default=config_defaults["dataset"], 
                        help="Dataset type (rome, extended_BA, or contest). Only needed for --train-list-file mode. "
                             "For mixed Rome+BA training, use --train-rome-count and --train-ba-count instead.")
    parser.add_argument("--train-count", type=int, default=config_defaults["train_count"])
    parser.add_argument("--eval-count", type=int, default=config_defaults["eval_count"])
    parser.add_argument("--train-rome-count", type=int, default=config_defaults["train_rome_count"])
    parser.add_argument("--train-ba-count", type=int, default=config_defaults["train_ba_count"])
    parser.add_argument("--eval-rome-count", type=int, default=config_defaults["eval_rome_count"])
    parser.add_argument("--eval-ba-count", type=int, default=config_defaults["eval_ba_count"])
    parser.add_argument("--train-list-file", type=str, default=config_defaults["train_list_file"])
    parser.add_argument("--eval-list-file", type=str, default=config_defaults["eval_list_file"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config_defaults["seeds"])
    parser.add_argument("--ppo-steps", type=int, default=config_defaults["ppo_steps"])
    parser.add_argument("--n-envs", type=int, default=config_defaults["n_envs"])
    parser.add_argument("--shortlist-size", type=int, default=config_defaults["shortlist_size"])
    parser.add_argument("--optimization-goal", choices=["local", "global"], default=config_defaults["optimization_goal"])
    parser.add_argument(
        "--node-selection-strategy",
        choices=["random", "heuristic", "heuristic_new", "heuristic_global"],
        default=config_defaults["node_selection_strategy"],
    )
    parser.add_argument("--best-local-bonus", type=float, default=config_defaults["best_local_bonus"])
    parser.add_argument("--local-weight", type=float, default=config_defaults["local_weight"])
    parser.add_argument("--sizemax-weight", type=float, default=config_defaults["sizemax_weight"])
    parser.add_argument("--global-weight", type=float, default=config_defaults["global_weight"])
    parser.add_argument("--standard-horizon", type=int, default=config_defaults["standard_horizon"])
    parser.add_argument("--repair-horizon", type=int, default=config_defaults["repair_horizon"])
    parser.add_argument("--repair-perturb-steps", type=int, default=config_defaults["repair_perturb_steps"])
    parser.add_argument("--repair-perturb-attempts", type=int, default=config_defaults["repair_perturb_attempts"])
    parser.add_argument("--plateau-patience", type=int, default=config_defaults["plateau_patience"])
    parser.add_argument("--plateau-bank-size", type=int, default=config_defaults["plateau_bank_size"])
    parser.add_argument("--broadened-repair", action="store_true", default=bool(config_defaults["broadened_repair"]))
    parser.add_argument("--plateau-only-replay", action="store_true", default=bool(config_defaults["plateau_only_replay"]))
    parser.add_argument("--eval-outer-restarts", type=int, default=config_defaults["eval_outer_restarts"])
    parser.add_argument("--eval-restart-perturb-steps", type=int, default=config_defaults["eval_restart_perturb_steps"])
    parser.add_argument("--eval-batch-size", type=int, default=config_defaults["eval_batch_size"])
    parser.add_argument("--eval-freq-steps", type=int, default=config_defaults["eval_freq_steps"])
    parser.add_argument("--checkpoint-root", type=str, default=config_defaults["checkpoint_root"])
    parser.add_argument("--device", type=str, default=config_defaults["device"], help="Training device: auto, cpu, or cuda")
    parser.add_argument("--normalize-reward", action="store_true", default=bool(config_defaults["normalize_reward"]))
    parser.add_argument("--no-normalize-reward", dest="normalize_reward", action="store_false")
    parser.add_argument("--reward-clip", type=float, default=config_defaults["reward_clip"])
    parser.add_argument("--use-lr-ent-schedule", action="store_true", default=bool(config_defaults["use_lr_ent_schedule"]))
    parser.add_argument("--final-learning-rate", type=float, default=config_defaults["final_learning_rate"])
    parser.add_argument("--final-ent-coef", type=float, default=config_defaults["final_ent_coef"])
    parser.add_argument("--use-self-imitation", action="store_true", default=bool(config_defaults["use_self_imitation"]))
    parser.add_argument("--sil-collect-episodes", type=int, default=config_defaults["sil_collect_episodes"])
    parser.add_argument("--sil-buffer-size", type=int, default=config_defaults["sil_buffer_size"])
    parser.add_argument("--sil-batch-size", type=int, default=config_defaults["sil_batch_size"])
    parser.add_argument("--sil-epochs", type=int, default=config_defaults["sil_epochs"])
    parser.add_argument("--sil-weight", type=float, default=config_defaults["sil_weight"])
    parser.add_argument("--sil-stochastic-collect", action="store_true", default=bool(config_defaults["sil_stochastic_collect"]))
    parser.add_argument("--output-json", type=str, default=config_defaults["output_json"])
    parser.add_argument("--output-jsonl", type=str, default=config_defaults["output_jsonl"])
    parser.add_argument("--exp-name", "--exp_name", type=str, default=config_defaults.get("exp_name"), help="Experiment name for TensorBoard logs")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path to continue training from")
    args = parser.parse_args()

    missing_graphs = {"train": [], "eval": []}
    use_mixed_counts = any(
        int(value) > 0
        for value in (
            args.train_rome_count,
            args.train_ba_count,
            args.eval_rome_count,
            args.eval_ba_count,
        )
    )
    if use_mixed_counts and (args.train_list_file or args.eval_list_file):
        raise ValueError("Mixed Rome/BA count flags cannot be combined with --train-list-file or --eval-list-file.")

    mix_config = None
    if use_mixed_counts:
        train_graphs, eval_graphs, mix_config = load_mixed_graphs_from_counts(args)
        if len(train_graphs) == 0 or len(eval_graphs) == 0:
            raise ValueError("Mixed Rome/BA count flags must load at least one training graph and one evaluation graph.")
        train_names = None
        eval_names = None
    elif args.train_list_file:
        if not args.dataset:
            raise ValueError("--dataset must be specified when using --train-list-file")
        train_names, train_graphs, missing_graphs["train"] = load_graphs_from_list_file(args.train_list_file, args.dataset)
        if args.eval_list_file:
            eval_names, eval_graphs, missing_graphs["eval"] = load_graphs_from_list_file(args.eval_list_file, args.dataset)
        else:
            eval_dataset = load_split_dataset("test", dataset_type=args.dataset)
            eval_graphs = [eval_dataset[i] for i in range(args.eval_count)]
            eval_names = None
    else:
        if not args.dataset:
            raise ValueError("--dataset must be specified when not using --train-rome-count/--train-ba-count (mixed counts)")
        dataset = load_split_dataset("train", dataset_type=args.dataset)
        actual_train_count = min(args.train_count, len(dataset))
        train_graphs = [dataset[i] for i in range(actual_train_count)]
        train_names = None
        if args.eval_list_file:
            eval_names, eval_graphs, missing_graphs["eval"] = load_graphs_from_list_file(args.eval_list_file, args.dataset)
        else:
            if args.train_list_file:
                eval_dataset = load_split_dataset("test", dataset_type=args.dataset)
                actual_eval_count = min(args.eval_count, len(eval_dataset))
                eval_graphs = [eval_dataset[i] for i in range(actual_eval_count)]
            else:
                eval_dataset = load_split_dataset("test", dataset_type=args.dataset)
                actual_eval_count = min(args.eval_count, len(eval_dataset))
                eval_graphs = [eval_dataset[i] for i in range(actual_eval_count)]
            eval_names = None

    config = {
        "dataset": args.dataset,
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "mixed_counts": mix_config,
        "train_list_file": args.train_list_file,
        "eval_list_file": args.eval_list_file,
        "train_graphs_loaded": len(train_graphs),
        "eval_graphs_loaded": len(eval_graphs),
        "missing_graphs": missing_graphs,
        "seeds": args.seeds,
        "ppo_steps": args.ppo_steps,
        "n_envs": args.n_envs,
        "shortlist_size": args.shortlist_size,
        "optimization_goal": args.optimization_goal,
        "node_selection_strategy": args.node_selection_strategy,
        "best_local_bonus": args.best_local_bonus,
        "local_weight": args.local_weight,
        "sizemax_weight": args.sizemax_weight,
        "global_weight": args.global_weight,
        "standard_horizon": args.standard_horizon,
        "repair_horizon": args.repair_horizon,
        "repair_perturb_steps": args.repair_perturb_steps,
        "repair_perturb_attempts": args.repair_perturb_attempts,
        "plateau_patience": args.plateau_patience,
        "plateau_bank_size": args.plateau_bank_size,
        "broadened_repair": args.broadened_repair,
        "plateau_only_replay": args.plateau_only_replay,
        "eval_outer_restarts": args.eval_outer_restarts,
        "eval_restart_perturb_steps": args.eval_restart_perturb_steps,
        "eval_batch_size": args.eval_batch_size,
        "eval_freq_steps": args.eval_freq_steps,
        "checkpoint_root": args.checkpoint_root,
        "normalize_reward": args.normalize_reward,
        "reward_clip": args.reward_clip,
        "use_lr_ent_schedule": args.use_lr_ent_schedule,
        "final_learning_rate": args.final_learning_rate,
        "final_ent_coef": args.final_ent_coef,
        "use_self_imitation": args.use_self_imitation,
        "sil_collect_episodes": args.sil_collect_episodes,
        "sil_buffer_size": args.sil_buffer_size,
        "sil_batch_size": args.sil_batch_size,
        "sil_epochs": args.sil_epochs,
        "sil_weight": args.sil_weight,
        "sil_stochastic_collect": args.sil_stochastic_collect,
        "resume_from": args.resume_from,
    }

    all_results = []
    start = time.perf_counter()
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        payload = run_repair_ppo(train_graphs, eval_graphs, seed, args, base_config=repair_base_config)
        all_results.append((seed, payload))
        event = {"variant": "repair_ppo_scratch", "seed": seed, "payload": payload}
        print(json.dumps(event), flush=True)
        append_jsonl(args.output_jsonl, {"event": "variant", **event})

    summary_config = {
        "dataset": {
            "name": repair_base_config["dataset"]["name"],
            "split": repair_base_config["dataset"]["split"],
        },
        "sampling": {
            "dataset": args.dataset,
            "train_count": args.train_count,
            "eval_count": args.eval_count,
            "train_rome_count": args.train_rome_count,
            "train_ba_count": args.train_ba_count,
            "eval_rome_count": args.eval_rome_count,
            "eval_ba_count": args.eval_ba_count,
            "train_list_file": args.train_list_file,
            "eval_list_file": args.eval_list_file,
            "train_graphs_loaded": len(train_graphs),
            "eval_graphs_loaded": len(eval_graphs),
            "missing_graphs": missing_graphs,
            "mixed_counts": mix_config,
        },
        "run": {
            "seeds": args.seeds,
            "ppo_steps": args.ppo_steps,
            "n_envs": args.n_envs,
            "device": args.device,
            "exp_name": getattr(args, "exp_name", None),
            "resume_from": args.resume_from,
        },
        "experiment": {
            "shortlist_size": args.shortlist_size,
            "optimization_goal": args.optimization_goal,
            "node_selection_strategy": args.node_selection_strategy,
            "best_local_bonus": args.best_local_bonus,
            "local_weight": args.local_weight,
            "sizemax_weight": args.sizemax_weight,
            "global_weight": args.global_weight,
            "standard_horizon": args.standard_horizon,
            "repair_horizon": args.repair_horizon,
            "repair_perturb_steps": args.repair_perturb_steps,
            "repair_perturb_attempts": args.repair_perturb_attempts,
            "plateau_patience": args.plateau_patience,
            "plateau_bank_size": args.plateau_bank_size,
            "broadened_repair": args.broadened_repair,
            "plateau_only_replay": args.plateau_only_replay,
            "eval_outer_restarts": args.eval_outer_restarts,
            "eval_restart_perturb_steps": args.eval_restart_perturb_steps,
            "eval_batch_size": args.eval_batch_size,
        },
        "training": {
            "eval_freq_steps": args.eval_freq_steps,
            "checkpoint_root": args.checkpoint_root,
            "normalize_reward": args.normalize_reward,
            "reward_clip": args.reward_clip,
            "use_lr_ent_schedule": args.use_lr_ent_schedule,
            "final_learning_rate": args.final_learning_rate,
            "final_ent_coef": args.final_ent_coef,
            "use_self_imitation": args.use_self_imitation,
            "sil_collect_episodes": args.sil_collect_episodes,
            "sil_buffer_size": args.sil_buffer_size,
            "sil_batch_size": args.sil_batch_size,
            "sil_epochs": args.sil_epochs,
            "sil_weight": args.sil_weight,
            "sil_stochastic_collect": args.sil_stochastic_collect,
        },
    }

    summary = {
        "config": repair_base_config,  # Use the authoritative loaded config
        "namespaces": summary_config,
        "total_runtime_sec": float(time.perf_counter() - start),
        "repair_ppo_scratch": summarize_repair_runs(all_results),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    append_jsonl(args.output_jsonl, {"event": "summary", "payload": summary})
    print("REPAIR_PPO_EXPERIMENT_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
