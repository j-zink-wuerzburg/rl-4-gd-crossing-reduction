"""
Graph Layout Environment Module

This module provides a factory pattern for creating graph layout environments with
unified configuration. All environments extend BaseGraphLayoutEnv and implement only
the action_masks() and step() methods specific to their action space.

Environment Types:
  - "single_step" (default): Discrete 8-direction single-pixel movement
  - "multi_scale": Discrete x MultiDiscrete for direction × distance pairs
  - "move_over_edge": Discrete 8-direction movement with automatic edge selection

Configuration:
  All environments are configured through a unified config dict with the following structure:
  {
      "env": {
          "type": "single_step" | "multi_scale" | "move_over_edge",
          "width": 1000,                                            # Canvas width
          "height": 1000,                                           # Canvas height
          "patch_size": 31,                                         # Observation patch size
          "pixel_decay_alpha": 0.5,                                 # Pixel decay rate
          "skip_edge_repeats": 1,                                   # Edge repetition skip
          "node_selection_strategy": "random" | "heuristic",        # Node selection
          "node_visit_repeat_count": 1,                              # Reuse the same node for N consecutive steps
          "step_limit": 2048,                                       # Episode length
          "n_distances": 5,                                         # (multi_scale only) Number of distance scales
          "reward": {                                               # Reward function weights
              "global_weight": 1.0,
              "local_weight": 10.0,
              "incident_weight": 2.0,
              "sizemax_weight": 0.1,
              "sparse_penalty": -0.001
          },
          "reward_scale": 1.0  # Global reward scaling factor
      }
  }

Example Usage:
  from env import create_graph_layout_env
  
  env = create_graph_layout_env(graph, config)
  obs, info = env.reset()
  action_mask = env.action_masks()
  obs, reward, done, truncated, info = env.step(action)
"""

import copy
import json
from functools import lru_cache
from pathlib import Path

from .BaseEnv import BaseGraphLayoutEnv
from .SingleStepEnv import SingleStepEnv
from .MultiScaleEnv import MultiScaleEnv
from .EdgeCrossingEnv import EdgeCrossingEnv


# Environment registry: maps type names to classes
ENV_REGISTRY = {
    "single_step": SingleStepEnv,
    "pixel": SingleStepEnv,
    "default": SingleStepEnv,
    "multi_scale": MultiScaleEnv,
    "move_distance": MultiScaleEnv,
    "distance": MultiScaleEnv,
    "move_over_edge": EdgeCrossingEnv,
    "edge_crossing": EdgeCrossingEnv,
    "along_edge": EdgeCrossingEnv,
}


ENV_DEFAULT_CONFIG_FILES = {
    "single_step": "config_ppo_pixel.json",
    "multi_scale": "config_ppo_pixel_move_distance.json",
    "move_over_edge": "config_ppo_pixel_move_over_edge.json",
}


@lru_cache(maxsize=None)
def _load_default_env_config_from_file(env_type):
    """Load default env settings from the canonical config JSON file."""
    filename = ENV_DEFAULT_CONFIG_FILES[env_type]
    config_path = Path(__file__).resolve().parents[2] / "configs" / filename

    with config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)

    env_config = config_data.get("env")
    if not isinstance(env_config, dict):
        raise ValueError(f"Missing or invalid 'env' section in {config_path}")

    return env_config


def _get_default_env_config(env_type):
    """Return a defensive copy so callers cannot mutate cached defaults."""
    return copy.deepcopy(_load_default_env_config_from_file(env_type))


def create_graph_layout_env(graph, config=None, env_type=None, **kwargs):
    """
    Factory function to create a graph layout environment.
    
    Args:
        graph (nx.Graph): NetworkX graph to layout
        config (dict, optional): Configuration dict containing "env" key with settings
        env_type (str, optional): Override environment type. If None, uses config["env"]["type"]
        **kwargs: Additional arguments passed to environment constructor
        
    Returns:
        GraphLayoutEnvBase: Instantiated environment
        
    Raises:
        ValueError: If env_type is unknown or config is malformed
        
    Example:
        >>> import networkx as nx
        >>> G = nx.erdos_renyi_graph(20, 0.3)
        >>> config = {"env": {"type": "pixel", "width": 1000, "height": 1000}}
        >>> env = create_graph_layout_env(G, config)
    """
    # Determine environment type
    if env_type is None:
        if config is None:
            env_type = "pixel"
        else:
            env_type = config.get("env", {}).get("type", "pixel")
    
    env_type = str(env_type).lower().strip()
    
    if env_type not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(
            f"Unknown environment type '{env_type}'. "
            f"Available types: {available}"
        )
    
    env_class = ENV_REGISTRY[env_type]
    return env_class(graph, config=config, **kwargs)


def get_available_env_types():
    """Return list of available environment types."""
    return sorted(list(set(ENV_REGISTRY.values().__class__.__name__ for _ in ENV_REGISTRY.values())))


def get_env_info(env_type=None):
    """
    Get information about an environment type.
    
    Args:
        env_type (str, optional): Environment type. If None, returns all.
        
    Returns:
        dict: Environment metadata and default config
    """
    env_info = {
        "single_step": {
            "name": "Single-Step Discrete Movement",
            "description": "8-directional discrete movement with 1 pixel per step",
            "action_space": "Discrete(8)",
            "requires": [],
            "default_config": _get_default_env_config("single_step"),
        },
        "multi_scale": {
            "name": "Multi-Scale Distance Movement",
            "description": "8 directions × N distance scales (power-of-2 multipliers)",
            "action_space": "MultiDiscrete([8, n_distances])",
            "requires": ["n_distances"],
            "default_config": _get_default_env_config("multi_scale"),
        },
        "move_over_edge": {
            "name": "Edge-Crossing Movement",
            "description": "Direction-based movement with automatic incident edge selection",
            "action_space": "Discrete(8)",
            "requires": [],
            "default_config": _get_default_env_config("move_over_edge"),
        }
    }
    
    if env_type is not None:
        env_type = str(env_type).lower()
        if env_type not in env_info:
            raise ValueError(f"Unknown environment type: {env_type}")
        return env_info[env_type]
    
    return env_info


__all__ = [
    "BaseGraphLayoutEnv",
    "SingleStepEnv",
    "MultiScaleEnv",
    "EdgeCrossingEnv",
    "create_graph_layout_env",
    "get_available_env_types",
    "get_env_info",
    "ENV_REGISTRY",
]
