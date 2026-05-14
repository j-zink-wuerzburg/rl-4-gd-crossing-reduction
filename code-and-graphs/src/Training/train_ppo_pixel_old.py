import atexit
import os
import shutil
import sys
import json
import argparse
import threading
import traceback
import subprocess
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stable_baselines3.common.logger as sb3_logger

import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import collections
import random

try:
    from pyinstrument import Profiler
except ImportError:
    Profiler = None

# Ensure 'src' is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class NodeSelectionStatsCallback(BaseCallback):
    """
    Custom callback for logging node selection statistics to TensorBoard.
    Aggregates stats over each rollout step and logs the mean value.
    """
    def __init__(self, log_heatmap=False, heatmap_freq=10, log_video=False, video_freq=1000, verbose=0):
        super().__init__(verbose)
        self.stats = collections.defaultdict(list)
        self.last_heatmap_data = None
        self.last_video_data = None
        self.log_heatmap = log_heatmap
        self.heatmap_freq = heatmap_freq
        self.log_video = log_video
        self.video_freq = video_freq
        self.heatmap_episode_count = 0
        self.video_episode_count = 0
        
    def _on_step(self) -> bool:
        # Retrieve infos and dones
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False]*len(infos))
        
        for done, info in zip(dones, infos):
            if "node_degree" in info:
                self.stats["node_degree"].append(info["node_degree"])
            if "candidates_count" in info:
                self.stats["candidates_count"].append(info["candidates_count"])
            if "is_random_fallback" in info:
                self.stats["is_random_fallback"].append(1.0 if info["is_random_fallback"] else 0.0)
            if "coverage" in info:
                self.stats["coverage"].append(info["coverage"])
            if "edge_jumps" in info:
                self.stats["edge_jumps"].append(info["edge_jumps"])
            if "reward_is_sparse" in info:
                self.stats["reward_is_sparse"].append(info["reward_is_sparse"])
            if "step_displacement" in info:
                self.stats["step_displacement"].append(info["step_displacement"])
            if "idle_streak" in info:
                self.stats["idle_streak"].append(info["idle_streak"])
            if "max_visits" in info:
                self.stats["max_visits"].append(info["max_visits"])
            if "entropy" in info:
                self.stats["entropy"].append(info["entropy"])
            if "step_size" in info:
                self.stats["step_size"].append(info["step_size"])
            if "spiral_repairs_count" in info:
                self.stats["spiral_repairs_count"].append(info["spiral_repairs_count"])
            if "node_visit_counts_array" in info and self.log_heatmap:
                self.heatmap_episode_count += 1
                if self.heatmap_episode_count % self.heatmap_freq == 0:
                    self.last_heatmap_data = {
                        "graph_pos": info["graph_pos"],
                        "graph_edges": info["graph_edges"],
                        "node_visit_counts_array": info["node_visit_counts_array"]
                    }
            if "history" in info and "graph_edges" in info and self.log_video:
                self.video_episode_count += 1
                self.last_video_data = {
                    "history": info.get("history", []),
                    "graph_edges": info.get("graph_edges", [])
                }
                
            # Log best-improvement metrics ONLY when the episode ends
            if done:
                if "best_improvement_steps" in info:
                    self.stats["best_improvements_count"].append(len(info.get("best_improvement_steps", [])))
                if "first_best_improvement_step" in info:
                    v = info.get("first_best_improvement_step", None)
                    if v is not None:
                        self.stats["first_best_improvement_step"].append(float(v))
                if "last_best_improvement_step" in info:
                    v = info.get("last_best_improvement_step", None)
                    if v is not None:
                        self.stats["last_best_improvement_step"].append(float(v))
                
        return True
        
    def _on_rollout_end(self) -> None:
        """
        Log the calculated means at the end of each rollout to TensorBoard.
        """
        # Defines how each metric is aggregated. Default is ["mean"].
        agg_types = {
            "edge_jumps": ["sum"],
            "spiral_repairs_count": ["sum"],
            "max_visits": ["max"],
            "idle_streak": ["max"],
            "best_improvements_count": ["mean", "max"]
        }

        for key, values in self.stats.items():
            if len(values) == 0:
                continue
                
            aggs = agg_types.get(key, ["mean"])
            
            for agg in aggs:
                if agg == "sum":
                    val = float(np.sum(values))
                elif agg == "max":
                    val = float(np.max(values))
                else:
                    val = float(np.mean(values))
                    
                self.logger.record(f"env_stats/{key}_{agg}", val)

        self.stats.clear()
        
        if self.last_heatmap_data is not None:
            # Generate networkx plot
            fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
            G = nx.Graph()
            G.add_edges_from(self.last_heatmap_data["graph_edges"])
            
            # Reconstruct pos dict
            pos = self.last_heatmap_data["graph_pos"]
            node_colors = self.last_heatmap_data["node_visit_counts_array"]
            
            # Map node_colors list to dict if needed by networkx, but we can just use nodelist
            nodelist = list(pos.keys())
            colors = [node_colors[n] for n in nodelist]
            
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
            nodes = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist, node_color=colors, cmap=plt.cm.Reds, node_size=50)
            cbar = fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Visit Counts', rotation=270, labelpad=15)
            ax.set_title("Node Visit Heatmap")
            ax.axis('off')
            
            self.logger.record("node_visits/heatmap", sb3_logger.Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            self.last_heatmap_data = None

        if self.last_video_data is not None:
            print("[Video] Rendering episode video for TensorBoard...")
            history = self.last_video_data["history"]
            edges = self.last_video_data["graph_edges"]
            
            if len(history) > 200:
                indices = np.linspace(0, len(history) - 1, 200, dtype=int)
                history = [history[i] for i in indices]
                
            frames = []
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            G = nx.Graph()
            G.add_edges_from(edges)
            
            for step_data in history:
                ax.clear()
                current_pos = step_data["pos"]
                moved_node = step_data["node"]
                is_jump = step_data.get("is_jump", False)
                reward      = step_data.get("reward", 0.0)
                
                nodelist = list(current_pos.keys())
                colors = []
                for n in nodelist:
                    if n == moved_node:
                        colors.append('orange' if is_jump else 'red')
                    else:
                        colors.append('blue')
                sizes = [150 if n == moved_node else 20 for n in nodelist]

                nx.draw_networkx_edges(G, current_pos, ax=ax, alpha=0.3)
                nx.draw_networkx_nodes(G, current_pos, ax=ax, nodelist=nodelist, node_size=sizes, node_color=colors)
                title_str = f"Node {moved_node} | Reward: {reward:.4f}"
                if is_jump:
                    title_str += " (Jumping Edge!)"
                ax.set_title(title_str)
                ax.axis('off')
                fig.canvas.draw()
                
                buf_rgba = np.asarray(fig.canvas.buffer_rgba())
                buf_rgb = buf_rgba[..., :3].copy()
                frames.append(buf_rgb.transpose(2, 0, 1))
                
            plt.close(fig)
            
            if len(frames) > 0:
                video_array = np.stack(frames, axis=0)
                # Expand to (1, T, C, H, W) as sb3 logger Video expects this shape in newer versions
                video_tensor = torch.tensor(video_array).unsqueeze(0)
                self.logger.record("node_visits/episode_video", sb3_logger.Video(video_tensor, fps=10), exclude=("stdout", "log", "json", "csv"))
            self.last_video_data = None
        
    def log_before_after_comparison(
        self,
        initial_pos: dict,
        final_pos: dict,
        graph_edges: list,
        initial_crossings: int,
        final_crossings: int,
        tag: str = "layout/before_after",
    ):
        """
        Logs a side-by-side before/after graph layout comparison to TensorBoard.
        Left = initial layout, Right = final layout after agent episode.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

        G = nx.Graph()
        G.add_edges_from(graph_edges)

        for ax, pos, title, crossings in [
            (axes[0], initial_pos, "Before", initial_crossings),
            (axes[1], final_pos,   "After",  final_crossings),
        ]:
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color="gray")
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30,
                                node_color="steelblue")
            ax.set_title(f"{title}  |  crossings: {crossings}", fontsize=11)
            ax.axis("off")

        fig.tight_layout()
        self.logger.record(
            tag,
            sb3_logger.Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
            )

class PhaseManager(BaseCallback):
    """Adjusts difficulty (p_hard), Learning Rate, and Entropy over time."""
    def __init__(self, sampler, phase_steps, lr_schedule=None, ent_schedule=None, verbose=1):
        super().__init__(verbose)
        self.sampler = sampler
        self.phase_steps = phase_steps
        self.lr_schedule = lr_schedule or (lambda p: None)
        self.ent_schedule = ent_schedule or (lambda p: None)
        self.phase = "warmup"
        self.phase_start_steps = 0

    def set_lr(self, new_lr: float | None):
        if new_lr is None: return
        for g in self.model.policy.optimizer.param_groups:
            g["lr"] = float(new_lr)
        self.model.lr_schedule = lambda _: float(new_lr)

    def set_ent_coef(self, new_ent: float | None):
        if new_ent is None: return
        self.model.ent_coef = float(new_ent)
        if hasattr(self.model, "logger"):
            self.model.logger.record("train/ent_coef", float(new_ent))

    def _enter_phase(self, name):
        self.phase = name
        self.phase_start_steps = self.model.num_timesteps
        
        # Adjust difficulty based on phase
        if name == "warmup":
            self.sampler.set_p_hard(0.0)   # 100% Easy
        elif name == "mixed":
            self.sampler.set_p_hard(0.5)   # 50/50 Mix
        elif name == "hard":
            self.sampler.set_p_hard(1.0)   # 100% Hard

        # Apply schedules
        lr = self.lr_schedule(name)
        self.set_lr(lr)
        ent = self.ent_schedule(name)
        self.set_ent_coef(ent)
        
        if self.verbose: 
            print(f"[Phase] Enter '{name}' at {self.model.num_timesteps} steps; p_hard={self.sampler.p_hard:.2f}, LR={lr}, Ent={ent}")

    def _on_training_start(self) -> None:
        g = int(self.model.num_timesteps)
        w = int(self.phase_steps.get("warmup", 0))
        m = int(self.phase_steps.get("mixed", 0))
        h = int(self.phase_steps.get("hard", 0))
        
        if g < w:
            phase, done = "warmup", g
        elif g < w + m:
            phase, done = "mixed", g - w
        else:
            phase, done = "hard", g - w - m
            
        self._enter_phase(phase)
        self.phase_start_steps = self.model.num_timesteps - done

    def _on_step(self) -> bool:
        steps = self.model.num_timesteps - self.phase_start_steps

        if self.phase == "warmup" and steps >= self.phase_steps["warmup"]:
            self._enter_phase("mixed")
        elif self.phase == "mixed" and steps >= self.phase_steps["mixed"]:
            self._enter_phase("hard")
        elif self.phase == "hard" and steps >= self.phase_steps["hard"]:
            if self.verbose: print("[Phase] Final phase finished. Stopping.")
            return False

        return True
    
class StepTimingCallback(BaseCallback):
    def _on_training_start(self):
        import time
        self._t0 = time.perf_counter()
        self._rollout_start = time.perf_counter()

    def _on_rollout_start(self):
        import time
        self._rollout_start = time.perf_counter()

    def _on_rollout_end(self):
        import time
        elapsed = time.perf_counter() - self._rollout_start
        n_steps = self.model.n_steps
        n_envs = self.model.n_envs
        total_steps = n_steps * n_envs
        print(
            f"[Timing] Rollout: {elapsed:.2f}s | "
            f"{total_steps} steps | "
            f"{total_steps/elapsed:.0f} steps/s | "
            f"~{elapsed/n_steps*1000:.1f}ms per env-step",
            flush=True
        )

    def _on_step(self):
        return True
    
class CurriculumSampler:
    """
    Single-pool curriculum sampler.
    Difficulty is based on graph size = number of nodes.

    At curriculum_progress = 0.0, only the smallest graphs are sampled.
    At curriculum_progress = 1.0, the full dataset is available.
    """
    def __init__(self, graphs, shuffle_within_window=True):
        self.graphs = sorted(list(graphs), key=lambda g: g.number_of_nodes())
        if len(self.graphs) == 0:
            raise ValueError("CurriculumSampler received an empty graph list.")

        self.shuffle_within_window = shuffle_within_window
        self.curriculum_progress = 0.0
        self._idx = 0
        self._active_graphs = []
        self._refresh_active_graphs()

    def _refresh_active_graphs(self):
        n_total = len(self.graphs)
        n_active = max(1, int(np.ceil((0.05 + 0.95 * self.curriculum_progress) * n_total)))
        self._active_graphs = self.graphs[:n_active]

        if self.shuffle_within_window:
            random.shuffle(self._active_graphs)

        self._idx = 0

    def set_progress(self, p):
        p = float(np.clip(p, 0.0, 1.0))
        if p != self.curriculum_progress:
            self.curriculum_progress = p
            self._refresh_active_graphs()

    def sample(self):
        if len(self._active_graphs) == 0:
            raise RuntimeError("No active graphs available in CurriculumSampler.")

        g = self._active_graphs[self._idx % len(self._active_graphs)]
        self._idx += 1
        return g

    @property
    def max_allowed_nodes(self):
        return max(g.number_of_nodes() for g in self._active_graphs)

    @property
    def min_allowed_nodes(self):
        return min(g.number_of_nodes() for g in self._active_graphs)

    @property
    def mean_allowed_nodes(self):
        return float(np.mean([g.number_of_nodes() for g in self._active_graphs]))


class LinearCurriculumCallback(BaseCallback):
    def __init__(self, sampler, total_timesteps, config, verbose=1):
        super().__init__(verbose)
        self.sampler = sampler
        self.total_timesteps = total_timesteps

        self.lr_start = config["ppo"].get("learning_rate", 3e-4)
        self.lr_end = 1e-4
        self.ent_start = config["ppo"].get("ent_coef", 0.05)
        self.ent_end = 0.01

        self.curriculum_fraction = config["training"].get("curriculum_fraction", 0.7)

    def _on_step(self) -> bool:
        progress = np.clip(
            self.num_timesteps / max(1, self.total_timesteps * self.curriculum_fraction),
            0.0,
            1.0,
        )

        # 1) Update graph-size curriculum
        self.sampler.set_progress(progress)

        # 2) Update LR
        current_lr = self.lr_start + progress * (self.lr_end - self.lr_start)
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = current_lr

        # 3) Update entropy
        current_ent = self.ent_start + progress * (self.ent_end - self.ent_start)
        self.model.ent_coef = current_ent

        if self.n_calls % 1000 == 0:
            self.logger.record("curriculum/progress", progress)
            self.logger.record("curriculum/lr", current_lr)
            self.logger.record("curriculum/ent_coef", current_ent)
            self.logger.record("curriculum/min_allowed_nodes", self.sampler.min_allowed_nodes)
            self.logger.record("curriculum/max_allowed_nodes", self.sampler.max_allowed_nodes)
            self.logger.record("curriculum/mean_allowed_nodes", self.sampler.mean_allowed_nodes)

        return True
    
from env import create_graph_layout_env
from Training.dataloader import load_split_dataset
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x, inplace=True)

class CustomGraphLayoutExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the GraphLayoutEnvPixel.
    Processes the 'pixel_map' (2, 31, 31) through a custom 2D CNN.
    Concatenates the CNN output (128 dims) with the remaining 1D feature vectors (40+8+8+1+1).
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

        # ── defaults ────────────────────────────────────────────────────────
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        if cnn_res_blocks is None:
            cnn_res_blocks = [0, 1, 1]
        assert len(cnn_channels) == len(cnn_res_blocks), (
            "cnn_channels and cnn_res_blocks must have the same length"
        )

        # ── CNN branch ──────────────────────────────────────────────────────
        pixel_space = observation_space.spaces[pixel_map_key]
        in_ch = pixel_space.shape[0]  # channels-first, e.g. 3

        cnn_layers: list[nn.Module] = []
        for i, (out_ch, n_res) in enumerate(zip(cnn_channels, cnn_res_blocks)):
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            for _ in range(n_res):
                cnn_layers.append(ResidualBlock(out_ch))
            in_ch = out_ch

        cnn_layers += [
            nn.AdaptiveAvgPool2d((4, 4)),   # output always 4×4 regardless of patch_size
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, cnn_out_dim, bias=True),
            nn.ReLU(inplace=True),
        ]
        self.cnn = nn.Sequential(*cnn_layers)

        # ── Tabular MLP branch ──────────────────────────────────────────────
        flat_dim = sum(
            space.shape[0]
            for key, space in observation_space.spaces.items()
            if key != pixel_map_key
        )

        self.tab_mlp = nn.Sequential(
            nn.Linear(flat_dim, tab_hidden_dim, bias=True),
            nn.LayerNorm(tab_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(tab_hidden_dim, tab_out_dim, bias=True),
            nn.ReLU(inplace=True),
        )

        # ── Fusion MLP ──────────────────────────────────────────────────────
        fusion_in = cnn_out_dim + tab_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, features_dim, bias=True),
            nn.ReLU(inplace=True),
        )

        # ── Weight initialisation ───────────────────────────────────────────
        self._init_weights()

    # -----------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    def forward(self, observations: dict) -> torch.Tensor:
        # CNN branch
        cnn_out = self.cnn(observations[self.pixel_map_key])

        # Tabular branch – concatenate all non-image keys in stable dict order
        tab_tensors = [
            tensor
            for key, tensor in observations.items()
            if key != self.pixel_map_key
        ]
        tab_out = self.tab_mlp(torch.cat(tab_tensors, dim=1))

        # Fuse and return
        return self.fusion(torch.cat([cnn_out, tab_out], dim=1))

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResidualBlockV2(nn.Module):
    def __init__(self, channels: int, use_se: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.se(out)

        out = out + identity
        return self.act(out)


class DownsampleBlock(nn.Module):
    """
    Learned downsampling via stride-2 conv instead of MaxPool.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialCarefulCNN(nn.Module):
    """
    Variant B:
    - Keep high resolution longer
    - Residual processing before/around downsampling
    - Learned downsampling only
    - Final adaptive pooling for variable pixel-map sizes
    """
    def __init__(
        self,
        in_channels: int,
        channels=(32, 64, 128),
        blocks_per_stage=(2, 2, 2),
        pool_out=(4, 4),
        cnn_out_dim=256,
        use_se=True,
        dropout_p=0.0,
    ):
        super().__init__()
        assert len(channels) == len(blocks_per_stage)

        c1, c2, c3 = channels
        b1, b2, b3 = blocks_per_stage

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c1, kernel_size=3, stride=1),
            ConvNormAct(c1, c1, kernel_size=3, stride=1),
        )

        stage1 = [ResidualBlockV2(c1, use_se=use_se, dropout_p=dropout_p) for _ in range(b1)]
        self.stage1 = nn.Sequential(*stage1)

        self.down1 = DownsampleBlock(c1, c2)
        stage2 = [ResidualBlockV2(c2, use_se=use_se, dropout_p=dropout_p) for _ in range(b2)]
        self.stage2 = nn.Sequential(*stage2)

        self.down2 = DownsampleBlock(c2, c3)
        stage3 = [ResidualBlockV2(c3, use_se=use_se, dropout_p=dropout_p) for _ in range(b3)]
        self.stage3 = nn.Sequential(*stage3)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_out),
            nn.Flatten(),
            nn.Linear(c3 * pool_out[0] * pool_out[1], cnn_out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)     # keep full resolution
        x = self.stage1(x)   # still full resolution
        x = self.down1(x)    # H,W / 2
        x = self.stage2(x)
        x = self.down2(x)    # H,W / 4
        x = self.stage3(x)
        x = self.head(x)
        return x


class CustomGraphLayoutExtractorV2(BaseFeaturesExtractor):
    """
    Size-agnostic extractor for variable pixel_map sizes (e.g. 31, 63, ...).
    Replaces aggressive MaxPool with learned downsampling and late adaptive pooling.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        pixel_map_key: str = "pixel_map",
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        cnn_blocks_per_stage: tuple[int, int, int] = (2, 2, 2),
        cnn_pool_out: tuple[int, int] = (4, 4),
        cnn_out_dim: int = 256,
        tab_hidden_dim: int = 128,
        tab_out_dim: int = 128,
        fusion_hidden_dim: int = 512,
        use_se: bool = True,
        cnn_dropout_p: float = 0.0,
        tab_keys: list[str] | None = None,
    ):
        super().__init__(observation_space, features_dim)
        self.pixel_map_key = pixel_map_key

        pixel_space = observation_space.spaces[pixel_map_key]
        in_ch = pixel_space.shape[0]

        self.cnn = SpatialCarefulCNN(
            in_channels=in_ch,
            channels=cnn_channels,
            blocks_per_stage=cnn_blocks_per_stage,
            pool_out=cnn_pool_out,
            cnn_out_dim=cnn_out_dim,
            use_se=use_se,
            dropout_p=cnn_dropout_p,
        )

        if tab_keys is None:
            tab_keys = [k for k in observation_space.spaces.keys() if k != pixel_map_key]
        self.tab_keys = tab_keys

        flat_dim = 0
        for key in self.tab_keys:
            space = observation_space.spaces[key]
            flat_dim += int(torch.tensor(space.shape).prod().item())

        self.tab_mlp = nn.Sequential(
            nn.Linear(flat_dim, tab_hidden_dim, bias=True),
            nn.LayerNorm(tab_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(tab_hidden_dim, tab_out_dim, bias=True),
            nn.SiLU(inplace=True),
        )

        fusion_in = cnn_out_dim + tab_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim, bias=True),
            nn.LayerNorm(fusion_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(fusion_hidden_dim, features_dim, bias=True),
            nn.SiLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_out = self.cnn(observations[self.pixel_map_key])

        tab_tensors = []
        for key in self.tab_keys:
            x = observations[key]
            if x.ndim > 2:
                x = x.view(x.shape[0], -1)
            tab_tensors.append(x)

        if len(tab_tensors) > 0:
            tab_in = torch.cat(tab_tensors, dim=1)
            tab_out = self.tab_mlp(tab_in)
            fused = torch.cat([cnn_out, tab_out], dim=1)
        else:
            fused = cnn_out

        return self.fusion(fused)

def make_env(graph_idx, config, sampler, seed=0):
    def _init():
        # Get a starting graph for space initialization
        G_init = sampler.sample()

        # Create environment using factory function (configured via config["env"]["type"])
        env = create_graph_layout_env(G_init, config=config)

        # Wrapper to handle resampling every reset
        class CurriculumWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                return self.env.reset(Graph=sampler.sample(), **kwargs)

        return CurriculumWrapper(env)
    return _init

class TBSyncCallback(BaseCallback):
    """Syncs local TB dir to NFS in a background thread — main thread never blocks on NFS."""

    def __init__(self, local_dir: str, nfs_dir: str, every_n_rollouts: int = 5):
        super().__init__()
        self.local_dir = local_dir
        self.nfs_dir = nfs_dir
        self.every_n = every_n_rollouts
        self._n = 0
        self._sync_thread = None  # track last sync thread

    def _do_sync(self):
        try:
            shutil.copytree(self.local_dir, self.nfs_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"[TBSync] Non-fatal sync error: {e}", flush=True)

    def _on_rollout_end(self) -> None:
        self._n += 1
        if self._n % self.every_n != 0:
            return

        # Don't stack up syncs — skip if previous is still running
        if self._sync_thread is not None and self._sync_thread.is_alive():
            print("[TBSync] Previous sync still running, skipping this one.", flush=True)
            return

        self._sync_thread = threading.Thread(
            target=self._do_sync,
            name="TBSyncThread",
            daemon=True  # won't block process exit
        )
        self._sync_thread.start()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # Block here at the very end to ensure final sync completes
        print("[TBSync] Final sync to NFS...", flush=True)
        self._do_sync()  # intentional blocking call — training is done
        print("[TBSync] Done.", flush=True)


def run_post_training_evaluation(
    config: dict,
    final_model_path: str,
    eval_graph_folder: str,
    eval_list_file: str | None,
    eval_workers: int,
    eval_time_limit: int,
    eval_output: str | None,
) -> None:
    """Run runner.py once with only the freshly trained model."""
    this_file = Path(__file__).resolve()
    sng_root = this_file.parents[2]
    eval_root = sng_root.parent / "rlgd-evaluation"
    runner_path = eval_root / "runner.py"

    if not runner_path.exists():
        print(f"[PostEval] Skipping: runner not found at {runner_path}")
        return

    env_type = str(config.get("env", {}).get("type", "pixel")).lower()
    algo_name = "ppo_pixel_with_distance" if env_type in ("move_distance", "distance") else "ppo_pixel"

    cmd = [
        sys.executable,
        str(runner_path),
        str(eval_graph_folder),
        "-a",
        algo_name,
        "-w",
        str(max(1, int(eval_workers))),
        "-t",
        str(max(1, int(eval_time_limit))),
    ]

    if eval_list_file:
        cmd.extend(["-l", str(eval_list_file)])
    if eval_output:
        cmd.extend(["--output", str(eval_output)])

    eval_env = os.environ.copy()
    # runner.py resolves model path through these env vars; set both for compatibility.
    eval_env["PPO_PIXEL_MODEL_PATH"] = str(final_model_path)
    eval_env["PPO_PIXEL_WITH_DISTANCE_MODEL_PATH"] = str(final_model_path)

    print("[PostEval] Starting evaluation with newly trained model...")
    print(f"[PostEval] Algorithm: {algo_name}")
    print(f"[PostEval] Model: {final_model_path}")
    print(f"[PostEval] Command: {' '.join(cmd)}")

    subprocess.run(cmd, cwd=str(eval_root), env=eval_env, check=True)
    print("[PostEval] Evaluation finished successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--exp_name", type=str, default="ppo_pixel_exp", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profile", action="store_true", help="Enable Pyinstrument profiling")
    parser.add_argument("--run_eval", action="store_true", help="Run runner.py after training using the newly saved model")
    parser.add_argument("--eval_graph_folder", type=str, default=None, help="Graph folder passed to runner.py")
    parser.add_argument("--eval_list", type=str, default=None, help="Optional graph whitelist file for runner.py --list")
    parser.add_argument("--eval_workers", type=int, default=1, help="Workers passed to runner.py --workers")
    parser.add_argument("--eval_time_limit", type=int, default=600, help="Per-graph time limit passed to runner.py --time-limit")
    parser.add_argument("--eval_output", type=str, default=None, help="Optional CSV output path for runner.py --output")
    args = parser.parse_args()

    print("[Startup] Parsed CLI arguments", flush=True)
    print(f"[Startup] config={args.config}", flush=True)
    print(f"[Startup] exp_name={args.exp_name}", flush=True)
    print(f"[Startup] seed={args.seed}", flush=True)

    with open(args.config, 'r') as f:
        config = json.load(f)

    env_type = str(config.get("env", {}).get("type", "pixel"))
    n_envs = int(config.get("training", {}).get("n_envs", 1))
    print(f"[Startup] Loaded config. env.type={env_type}, training.n_envs={n_envs}", flush=True)

    log_dir = os.path.join("results", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Startup] Log dir: {log_dir}", flush=True)
    
    # Save the config to the log directory
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print("[Startup] Saved resolved config copy", flush=True)

    print("[Startup] Loading training dataset split...", flush=True)
    rome = load_split_dataset("train", dataset_type="rome") 
    print(f"[Startup] Dataset ready: {len(rome)} graphs", flush=True)
    
    sampler = CurriculumSampler(rome)
    print("[Startup] Curriculum sampler initialized", flush=True)

    n_envs = config["training"]["n_envs"]
    print(f"[Startup] Creating SubprocVecEnv with n_envs={n_envs}...", flush=True)
    vec_env = SubprocVecEnv([make_env(i, config, sampler, seed=args.seed) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    print("[Startup] VecEnv and VecMonitor ready", flush=True)


    extractor_cfg = config["policy"]["features_extractor"]

    # # Attach our custom CNN extractor to MultiInputPolicy
    # policy_kwargs = dict(
    # features_extractor_class=CustomGraphLayoutExtractor,
    #     features_extractor_kwargs=dict(
    #         features_dim        = extractor_cfg.get("features_dim",      512),
    #         cnn_channels        = extractor_cfg.get("cnn_channels",      [32, 64, 128]),
    #         cnn_res_blocks      = extractor_cfg.get("cnn_res_blocks",    [0, 1, 1]),
    #         cnn_out_dim         = extractor_cfg.get("cnn_out_dim",       256),
    #         tab_hidden_dim      = extractor_cfg.get("tab_hidden_dim",    128),
    #         tab_out_dim         = extractor_cfg.get("tab_out_dim",       128),
    #         fusion_hidden_dim   = extractor_cfg.get("fusion_hidden_dim", 512),
    #         pixel_map_key       = extractor_cfg.get("pixel_map_key",     "pixel_map"),
    #     ),
    # )
    
    # # Pass along standard actor/critic layers
    # raw_net_arch = config["ppo"].get("policy_kwargs", {}).get("net_arch")
    # if raw_net_arch is not None:
    #      policy_kwargs["net_arch"] = dict(pi=raw_net_arch, vf=raw_net_arch)

    # nfs_tb_dir = os.path.join(log_dir, "tb")
    # os.makedirs(nfs_tb_dir, exist_ok=True)
    # print(f"[Startup] TensorBoard dir: {nfs_tb_dir}", flush=True)

    extractor_cfg = config["policy"]["features_extractor"]

    tab_keys = extractor_cfg.get("tab_keys", [
        k for k in vec_env.observation_space.spaces.keys()
        if k != extractor_cfg.get("pixel_map_key", "pixel_map")
    ])

    policy_kwargs = dict(
        features_extractor_class=CustomGraphLayoutExtractorV2,
        features_extractor_kwargs=dict(
            features_dim=extractor_cfg.get("features_dim", 512),
            pixel_map_key=extractor_cfg.get("pixel_map_key", "pixel_map"),
            cnn_channels=tuple(extractor_cfg.get("cnn_channels", [32, 64, 128])),
            cnn_blocks_per_stage=tuple(extractor_cfg.get("cnn_blocks_per_stage", [2, 2, 2])),
            cnn_pool_out=tuple(extractor_cfg.get("cnn_pool_out", [4, 4])),
            cnn_out_dim=extractor_cfg.get("cnn_out_dim", 256),
            tab_hidden_dim=extractor_cfg.get("tab_hidden_dim", 128),
            tab_out_dim=extractor_cfg.get("tab_out_dim", 128),
            fusion_hidden_dim=extractor_cfg.get("fusion_hidden_dim", 512),
            use_se=extractor_cfg.get("use_se", True),
            cnn_dropout_p=extractor_cfg.get("cnn_dropout_p", 0.0),
            tab_keys=tab_keys,
        ),
    )

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy if config["ppo"]["policy"] == "MultiInputPolicy" else config["ppo"]["policy"],
        vec_env,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=nfs_tb_dir,
        verbose=1,
        seed=args.seed,
    )
    print(f"[Startup] Model initialized on device: {model.device}", flush=True)

    # def _monitor_tb_thread(model, interval=30):
    #     import time
    #     while True:
    #         time.sleep(interval)
    #         all_threads = {t.name for t in threading.enumerate()}
    #         print(f"[TB Monitor] Active threads: {all_threads}", flush=True)

    # monitor = threading.Thread(target=_monitor_tb_thread, args=(model,), daemon=True)
    # monitor.start()

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config["training"]["save_freq"] // n_envs),
        save_path=os.path.join(log_dir, "models"),
        name_prefix="rl_model"
    )

    # eval_cb = EvalCallback(
    #     eval_env,
    #     best_model_save_path=save_dir,
    #     log_path=save_dir,
    #     eval_freq=max(1, (50_000 // n_envs)),
    #     n_eval_episodes=8,
    #     deterministic=True,
    #     render=False,
    # )

    curriculum_callback = LinearCurriculumCallback(
        sampler=sampler, 
        total_timesteps=config["training"]["total_timesteps"], 
        config=config
    )
    
    log_heatmap = config["training"].get("log_heatmap", False)
    heatmap_freq = config["training"].get("heatmap_freq", 10)
    log_video = config["training"].get("log_video", False)
    video_freq = config["training"].get("video_freq", 1000)
    node_stats_callback = NodeSelectionStatsCallback(log_heatmap=log_heatmap, heatmap_freq=heatmap_freq, log_video=log_video, video_freq=video_freq)

    profiler = None
    if args.profile:
        if Profiler is None:
            print("Warning: --profile passed but pyinstrument is not installed. Skipping profiling.")
        else:
            profiler = Profiler()
            profiler.start()

    def _save_profile_and_exit(signum, frame):
        if profiler is not None:
            profiler.stop()
            profile_path = os.path.join(log_dir, "profile_report.html")
            with open(profile_path, "w") as f:
                f.write(profiler.output_html())
            print(f"\nProfiling complete (signal {signum}). Report saved to: {profile_path}")
        sys.exit(0)

    import signal as _signal

    def _dump_all_stacks(signum, frame):
        print("\n===== THREAD DUMP =====", flush=True)
        for thread in threading.enumerate():
            print(f"\n--- Thread: {thread.name} (id={thread.ident}) ---", flush=True)
            stack = sys._current_frames().get(thread.ident)
            if stack:
                traceback.print_stack(stack)
        print("===== END DUMP =====\n", flush=True)

    _signal.signal(_signal.SIGINT,  _save_profile_and_exit)  # Ctrl-C / scancel --signal=INT
    _signal.signal(_signal.SIGTERM, _save_profile_and_exit)  # scancel default signal
    _signal.signal(_signal.SIGINT, _dump_all_stacks)
    print(f"[Train] Starting training on device: {model.device}", flush=True)

    # tensorboard_sync_callback = TBSyncCallback(local_tb_dir, nfs_tb_dir, every_n_rollouts=5)

    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=CallbackList([checkpoint_callback, node_stats_callback, curriculum_callback]),
    )
    final_model_base = os.path.join(log_dir, "final_model")
    model.save(final_model_base)

    final_model_path = final_model_base + ".zip"
    if not os.path.exists(final_model_path):
        final_model_path = final_model_base

    auto_eval_cfg = bool(config.get("training", {}).get("auto_eval", False))
    should_run_eval = bool(args.run_eval or auto_eval_cfg)

    if should_run_eval:
        eval_graph_folder = (
            args.eval_graph_folder
            or config.get("training", {}).get("eval_graph_folder")
            or str(Path(__file__).resolve().parents[2] / "graphs" / "rome_filtered")
        )
        eval_list_file = args.eval_list or config.get("training", {}).get("eval_list")
        eval_workers = int(args.eval_workers or config.get("training", {}).get("eval_workers", 1))
        eval_time_limit = int(args.eval_time_limit or config.get("training", {}).get("eval_time_limit", 600))
        eval_output = args.eval_output or config.get("training", {}).get("eval_output")

        run_post_training_evaluation(
            config=config,
            final_model_path=final_model_path,
            eval_graph_folder=eval_graph_folder,
            eval_list_file=eval_list_file,
            eval_workers=eval_workers,
            eval_time_limit=eval_time_limit,
            eval_output=eval_output,
        )

    if profiler is not None:
        profiler.stop()
        profile_path = os.path.join(log_dir, "profile_report.html")
        with open(profile_path, "w") as f:
            f.write(profiler.output_html())
        print(f"Profiling complete. Report saved to: {profile_path}")

if __name__ == "__main__":
    main()