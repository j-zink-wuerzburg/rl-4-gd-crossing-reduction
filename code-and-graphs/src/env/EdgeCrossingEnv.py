import math
import numpy as np
import networkx as nx
from gymnasium import spaces
from gymnasium.spaces import Box, Dict

from .BaseEnv import BaseGraphLayoutEnv, n_directions


class EdgeCrossingEnv(BaseGraphLayoutEnv):
    """Direction-only environment that moves along the best matching incident edge."""

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
        self.optimization_goal = env_cfg.get("optimization_goal", "global") 
        self.reset_unsuccessful_moves_threshold = env_cfg.get("reset_unsuccessful_moves_threshold", None)
        self.use_int_grid = env_cfg.get("use_int_grid", True)

        # Configure reward weights via BaseEnv helper
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
        return np.ones(n_directions, dtype=bool)

    def _find_best_incident_edge(self, node, direction_idx):
        incident = self.incident_by_node.get(node, ())
        if not incident:
            return None

        dir_x, dir_y = self.DIRS_UNIT[direction_idx]
        target_angle = np.degrees(np.arctan2(dir_y, dir_x))

        best_edge = None
        best_angle_diff = float("inf")
        node_pos = self.pos[node].astype(float)

        for edge in incident:
            other = edge[1] if edge[0] == node else edge[0]
            edge_vec = self.pos[other].astype(float) - node_pos
            if np.allclose(edge_vec, 0):
                continue

            edge_angle = np.degrees(np.arctan2(edge_vec[1], edge_vec[0]))
            angle_diff = abs((edge_angle - target_angle + 180) % 360 - 180)
            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best_edge = edge

        return best_edge

    def step(self, action):
        node = self.current_node
        self.node_visit_counts[self.node_index[node]] += 1

        action = int(action)
        action_world = self.obs_dir_to_world(action)
        chosen_edge = self._find_best_incident_edge(node, action_world)
        incident = self.incident_by_node.get(node, ())

        before_global = self.global_crossings
        before_local = self.local_crossings
        before_sizemax = len(self.E_star)

        dx, dy = self.DIRS_UNIT[action_world]
        move_dir = np.array([dx, dy], dtype=float)

        if chosen_edge is None:
            new_pos = self.pos[node] + move_dir * self.move_step
        else:
            nodex, nodey = self.pos[node].astype(float)
            edge_other = chosen_edge[1] if chosen_edge[0] == node else chosen_edge[0]
            ex, ey = self.pos[edge_other].astype(float)
            edge_dx, edge_dy = ex - nodex, ey - nodey
            edge_len = np.hypot(edge_dx, edge_dy)

            if edge_len < 1e-6:
                new_pos = self.pos[node] + move_dir * self.move_step
            else:
                mx, my = edge_dx / edge_len, edge_dy / edge_len
                min_d = None
                for e in incident:
                    for f in self.crossings.get(e, ()):
                        pt = self.get_crossing_point(e[0], e[1], f[0], f[1])
                        if pt is None:
                            continue
                        px, py = pt
                        d_along = (px - nodex) * mx + (py - nodey) * my
                        if d_along > 1e-6 and (min_d is None or d_along < min_d):
                            min_d = d_along

                if self.use_int_grid:
                    new_pos = self._next_int_grid_position_just_past_crossing(node, edge_other, min_d, action_world)
                else:
                    if min_d is None:
                        new_pos = self.pos[node] + np.array([mx, my], dtype=float) * self.move_step
                    else:
                        epsilon = np.random.uniform(0.01, 0.10)
                        new_pos = self.pos[node] + np.array([mx, my], dtype=float) * min_d * (1 + epsilon)

        if self.use_int_grid:
            if getattr(self, "width", None) is not None and getattr(self, "height", None) is not None:
                new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
                new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
            self.pos[node] = np.asarray(new_pos, dtype=np.int32)
        else:
            if getattr(self, "width", None) is not None and getattr(self, "height", None) is not None:
                new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
                new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
            self.pos[node] = np.asarray(new_pos, dtype=np.float64)
        self.pos_arr[self.node_index[node]] = self.pos[node]

        if node in self.bboxes:
            try:
                self.rtree_index.delete(node, self.bboxes[node])
            except Exception:
                pass
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

        if self.is_layout_better(after_global, after_local):
            self._handle_new_best(after_global, after_local)
        else:
            self.unsuccessful_moves += 1

        self.check_and_handle_reset()
        self.last_crossings = after_global

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
            "reward_is_sparse": reward_is_sparse,
            "idle_streak": float(self.idle_streak),
            "spiral_repairs_count": getattr(self, "spiral_repairs_count", 0),
        }

        if done or truncated:
            coverage = float(np.count_nonzero(self.node_visit_counts) / len(self.node_visit_counts))
            max_visits = float(np.max(self.node_visit_counts))
            info["coverage"] = coverage
            info["max_visits"] = max_visits
            if self.track_history:
                info["history"] = self.history

        info["best_improvement_steps"] = list(self.best_improvement_steps)
        info["first_best_improvement_step"] = self.best_improvement_steps[0] if len(self.best_improvement_steps) > 0 else None
        info["last_best_improvement_step"] = self.last_improvement_step

        if hasattr(self, "last_selection_stats"):
            info.update(self.last_selection_stats)

        return obs, reward, done, truncated, info

    def _next_int_grid_position_just_past_crossing(self, node, edge_other, min_d, action_world):
        """Move just past the nearest crossing by rounding in the direction of the movement"""
        start = self.pos[node].astype(np.float64)
        end = self.pos[edge_other].astype(np.float64)
        delta = end - start
        edge_len = float(np.hypot(delta[0], delta[1]))

        if edge_len < 1e-6:
            return (start + np.array(self.DIRS_INT[action_world], dtype=np.float64)).astype(np.int32)

        unit = delta / edge_len

        if min_d is None:
            # No crossing found: step one unit along the edge
            target = start + unit * max(1.0, edge_len - 1.0)
        else:
            target = start + unit * min_d

        # round each axis in the direction of movement (ceil positive, floor negative)
        candidate = np.empty(2, dtype=np.int64)
        for i in range(2):
            if unit[i] > 1e-9:
                candidate[i] = math.ceil(target[i])
            elif unit[i] < -1e-9:
                candidate[i] = math.floor(target[i])
            else:
                candidate[i] = int(round(target[i]))

        # Ensure we moved at least one step from start
        start_int = start.astype(np.int64)
        if np.array_equal(candidate, start_int):
            candidate = start_int + np.array(self.DIRS_INT[action_world], dtype=np.int64)

        if self.width is not None and self.height is not None:
            candidate[0] = np.clip(candidate[0], 0, int(self.width) - 1)
            candidate[1] = np.clip(candidate[1], 0, int(self.height) - 1)

        return candidate.astype(np.int32)