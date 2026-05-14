import numpy as np
import networkx as nx
import scipy.stats
from gymnasium import spaces
from gymnasium.spaces import Box, Dict

from .BaseEnv import BaseGraphLayoutEnv, n_directions


class SingleStepEnv(BaseGraphLayoutEnv):
    """Single-pixel movement environment with Discrete(8) actions."""

    def __init__(
        self,
        graph: nx.Graph,
        config=None,
        width=1000,
        height=1000,
        patch_size=31,
        pixel_decay_alpha=0.5,
        skip_edge_repeats=1,
        node_selection_strategy="random",
        step_limit=2048,
        reward_weights=None,
    ):
        super().__init__()

        env_cfg = (config or {}).get("env", {})
        width = env_cfg.get("width", width)
        height = env_cfg.get("height", height)
        patch_size = env_cfg.get("patch_size", patch_size)
        pixel_decay_alpha = env_cfg.get("pixel_decay_alpha", pixel_decay_alpha)
        skip_edge_repeats = env_cfg.get("skip_edge_repeats", skip_edge_repeats)
        node_selection_strategy = env_cfg.get("node_selection_strategy", node_selection_strategy)
        node_visit_repeat_count = env_cfg.get("node_visit_repeat_count", 1)
        step_limit = env_cfg.get("step_limit", step_limit)
        reward_config = env_cfg.get("reward", None)

        self.reward_scale = float(env_cfg.get("reward_scale", 1.0))
        self.track_history = env_cfg.get("track_history", (config or {}).get("training", {}).get("log_video", False))
        self.optimization_goal = env_cfg.get("optimization_goal", "global")  # "global" or "local"
        self.reset_unsuccessful_moves_threshold = env_cfg.get("reset_unsuccessful_moves_threshold", None)
        self.use_int_grid = env_cfg.get("use_int_grid", True)  # Default to integer grid

        # Configure reward weights based on optimization goal using BaseEnv helper
        self.configure_reward_weights(reward_config)
        self.step_limit = step_limit
        self.patch_size = patch_size
        self.patch_half = patch_size // 2
        self.pixel_decay_alpha = pixel_decay_alpha
        self.skip_edge_repeats = skip_edge_repeats
        self.node_selection_strategy = node_selection_strategy
        self.node_visit_repeat_count = node_visit_repeat_count

        self.action_space = spaces.Discrete(n_directions)
        self.observation_space = Dict(
            {
                "pixel_map": Box(0.0, 1.0, shape=(3, self.patch_size, self.patch_size), dtype=np.float32),
                "cross_map": Box(0.0, 1.0, shape=(8,), dtype=np.float32),
                "cross_map_local": Box(0.0, 1.0, shape=(8,), dtype=np.float32),
                "local_view": Box(0.0, np.inf, shape=(40,), dtype=np.float32),
                "local_crossings": Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "global_crossings": Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            }
        )

        self._precompute_pixel_grids()
        self._init_for_graph(graph, width, height)

    def action_masks(self) -> np.ndarray:
        """Return valid movement directions for the current node."""
        mask = np.zeros(8, dtype=bool)
        node = self.current_node
        cur_pos = self.pos[node].astype(np.int64)

        if self.width is None or self.height is None:
            return np.ones(8, dtype=bool)

        occupied = {
            (int(self.pos[n][0]), int(self.pos[n][1]))
            for n in self.graph.nodes()
            if n != node
        }

        for action in range(8):
            action_world = self.obs_dir_to_world(action)
            delta = np.array(self.DIRS_INT[action_world], dtype=np.int64)
            new_pos = cur_pos + delta

            out_of_bounds = not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height)
            cell_occupied = (int(new_pos[0]), int(new_pos[1])) in occupied
            if out_of_bounds or cell_occupied:
                continue
            mask[action] = True

        if not mask.any():
            mask[0] = True
        return mask

    def step(self, action):
        node = self.current_node
        self.node_visit_counts[self.node_index[node]] += 1
        old_pos = self.pos[node].astype(np.float64).copy()

        before_global = self.global_crossings
        before_local = self.local_crossings
        before_sizemax = len(self.E_star)

        action_world = self.obs_dir_to_world(action)
        delta = np.array(self.DIRS_INT[action_world], dtype=np.int64)

        cur = self.pos[node].astype(np.int64)
        new_pos = cur + delta
        if self.width is not None and self.height is not None:
            new_pos[0] = int(np.clip(new_pos[0], 0, max(0, int(self.width) - 1)))
            new_pos[1] = int(np.clip(new_pos[1], 0, max(0, int(self.height) - 1)))

        self.pos[node] = np.asarray(new_pos, dtype=np.int32)
        self.pos_arr[self.node_index[node]] = self.pos[node]

        jumps_performed = 0
        for _ in range(getattr(self, "skip_edge_repeats", 1)):
            if self.is_node_position_legal(node) and not self._is_node_strictly_on_edge(node):
                break

            new_pos = new_pos + delta
            if self.width is not None and self.height is not None:
                new_pos_clamped = np.copy(new_pos)
                new_pos_clamped[0] = int(np.clip(new_pos[0], 0, max(0, int(self.width) - 1)))
                new_pos_clamped[1] = int(np.clip(new_pos[1], 0, max(0, int(self.height) - 1)))
                if np.array_equal(new_pos_clamped, self.pos[node]):
                    break
                new_pos = new_pos_clamped

            self.pos[node] = np.asarray(new_pos, dtype=np.int32)
            self.pos_arr[self.node_index[node]] = self.pos[node]
            jumps_performed += 1

            if self.track_history:
                self.history.append(
                    {
                        "pos": {self.node_index[k]: tuple(v) for k, v in self.pos.items()},
                        "node": self.node_index[node],
                        "is_jump": True,
                        "reward": 0.0,
                    }
                )

        if not self.is_node_position_legal(node):
            pref_dir_idx = self._dominant_dir_index_from_delta(delta)
            candidate = self._find_legal_position_spiral(node, start_pos=self.pos[node], start_dir_idx=pref_dir_idx)
            self.pos[node] = candidate.astype(np.int32)
            self.pos_arr[self.node_index[node]] = self.pos[node]

        if node in self.bboxes:
            self.rtree_index.delete(node, self.bboxes[node])
        new_bb = self._compute_bbox(node)
        self.rtree_index.insert(node, new_bb)
        self.bboxes[node] = new_bb
        self._update_edges_for_node(node)

        removed_pairs = set(self.remove_crossings_for_node(node))
        added_pairs = set(self.recompute_crossings_for_node(node))
        self._apply_crossing_deltas(list(added_pairs), list(removed_pairs))

        after_global = self.global_crossings
        after_local = self.local_crossings

        after_sizemax = len(self.E_star)
        global_delta = before_global - after_global
        local_delta = before_local - after_local
        sizemax_delta = (before_sizemax - after_sizemax) if local_delta == 0 else 0

        # Use only relevant weights depending on optimization goal (incident removed)
        if self.optimization_goal == "local":
            reward = (
                self.reward_weights.get("local_weight", 0.0) * local_delta
                + self.reward_weights.get("sizemax_weight", 0.0) * sizemax_delta
                + self.reward_weights.get("global_weight", 0.0) * global_delta
            )
        else:
            reward = self.reward_weights.get("global_weight", 0.0) * global_delta

        if reward == 0.0:
            reward = self.reward_weights.get("sparse_penalty", -0.01)
            self.idle_streak += 1
            reward_is_sparse = 1.0
        else:
            self.idle_streak = 0
            reward_is_sparse = 0.0

        reward *= self.reward_scale
        step_displacement = float(np.linalg.norm(self.pos[node].astype(np.float64) - old_pos))
        self.last_crossings = after_global

        if self.is_layout_better(after_global, after_local):
            self._handle_new_best(after_global, after_local)
        else:
            self.unsuccessful_moves += 1
        
        # Check if reset should be triggered (EdgeCrossingEnv only via threshold)
        self.check_and_handle_reset()

        if self.track_history:
            self.history.append(
                {
                    "pos": {self.node_index[k]: tuple(v) for k, v in self.pos.items()},
                    "node": self.node_index[node],
                    "reward": float(reward),
                }
            )

        self.select_next_node()
        self.step_count += 1

        obs = self.get_observation()
        done = self.global_crossings == 0
        truncated = self.step_count >= self.step_limit

        info = {
            "global_crossings": self.global_crossings,
            "local_crossings": self.local_crossings,
            "best_global_crossings": self.best_crossings,
            "best_local_crossings": self.best_local_crossings,
            "edge_jumps": jumps_performed,
            "reward_is_sparse": reward_is_sparse,
            "step_displacement": step_displacement,
            "idle_streak": float(self.idle_streak),
            "spiral_repairs_count": self.spiral_repairs_count,
        }

        if done or truncated:
            coverage = float(np.count_nonzero(self.node_visit_counts) / len(self.node_visit_counts))
            max_visits = float(np.max(self.node_visit_counts))
            entropy = float(scipy.stats.entropy(self.node_visit_counts)) if np.sum(self.node_visit_counts) > 0 else 0.0
            info["coverage"] = coverage
            info["max_visits"] = max_visits
            info["entropy"] = entropy
            info["graph_pos"] = {self.node_index[k]: tuple(v) for k, v in self.pos.items()}
            info["graph_edges"] = [(self.node_index[u], self.node_index[v]) for u, v in self.graph.edges()]
            info["node_visit_counts_array"] = list(self.node_visit_counts)
            if self.track_history:
                info["history"] = self.history

        # Best-improvement logging
        info["best_improvement_steps"] = list(self.best_improvement_steps)
        info["first_best_improvement_step"] = self.best_improvement_steps[0] if len(self.best_improvement_steps) > 0 else None
        info["last_best_improvement_step"] = self.last_improvement_step

        if hasattr(self, "last_selection_stats"):
            info.update(self.last_selection_stats)

        return obs, reward, done, truncated, info
