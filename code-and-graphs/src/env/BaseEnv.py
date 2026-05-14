"""Comprehensive base class for graph layout environments.

Centralizes all shared logic:
- Graph initialization and state management
- Crossing computation and crossing maintenance
- Geometry and legality checks
- Observation computation
- Node selection strategies
- R-tree management for spatial indexing

Child classes only implement:
- action_space and observation_space definitions in __init__
- action_masks() method
- step() function
"""

import math
import copy
import heapq
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
from rtree import index
import scipy.stats

try:
    from pybindCode.graph_utils import (
        compute_crossings,
        find_crossings_for_edges,
        edge_ray_octant_distances as cpp_edge_ray_octant_distances,
        pixel_edge_min_distances as cpp_pixel_edge_min_distances,
        edge_pair_crossing_points as cpp_edge_pair_crossing_points,
        pixel_crossing_min_distances as cpp_pixel_crossing_min_distances,
        batch_octant_observations as cpp_batch_octant_observations,
        batch_pixel_maps as cpp_batch_pixel_maps,
    )
except Exception:
    from src.pybindCode.graph_utils import (
        compute_crossings,
        find_crossings_for_edges,
        edge_ray_octant_distances as cpp_edge_ray_octant_distances,
        pixel_edge_min_distances as cpp_pixel_edge_min_distances,
        edge_pair_crossing_points as cpp_edge_pair_crossing_points,
        pixel_crossing_min_distances as cpp_pixel_crossing_min_distances,
        batch_octant_observations as cpp_batch_octant_observations,
        batch_pixel_maps as cpp_batch_pixel_maps,
    )


def get_best_layout_by_crossings(G):
    """Find the layout with minimum crossings from available candidates."""
    cached = G.graph.get("_best_layout_by_crossings_cache")
    if cached is not None:
        return {n: cached[n].copy() for n in G.nodes()}

    node_order = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_order)}
    layouts = {
        "kamada_kawai": nx.kamada_kawai_layout(G),
    }
    has_json_pos = all(('x' in G.nodes[n] and 'y' in G.nodes[n]) for n in G.nodes()) and len(G.nodes()) > 0
    if has_json_pos:
        layouts["json"] = {n: np.array([float(G.nodes[n]['x']), float(G.nodes[n]['y'])], dtype=float) for n in G.nodes()}

    best_layout = None
    min_crossings = float('inf')

    for name, pos in layouts.items():
        pos_arr = np.array([pos[n] for n in node_order])
        edges_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        _, _, crossings = compute_crossings(pos_arr, edges_idx)
        if crossings < min_crossings:
            min_crossings = crossings
            best_layout = {n: pos[n] for n in G.nodes()}

    best_layout = {
        n: np.asarray(best_layout[n], dtype=float).copy()
        for n in G.nodes()
    }
    G.graph["_best_layout_by_crossings_cache"] = best_layout
    return {n: best_layout[n].copy() for n in G.nodes()}


def calculate_distance(point1, point2, epsilon=1e-6):
    """Calculate Euclidean distance with minimum epsilon."""
    return max(np.linalg.norm(point1 - point2), epsilon)


n_directions = 8


class BaseGraphLayoutEnv(gym.Env):
    """Comprehensive base class for graph layout environments.
    
    Centralizes all shared logic; child classes define only:
    - action_space and observation_space in __init__
    - action_masks() method
    - step() method
    """

    def __init__(self):
        """Initialize common instance variables and direction constants."""
        # Graph state
        self.graph = None
        self.n_nodes = None
        self.pos = None
        self.initial_pos = None
        self.best_pos = None
        self.pos_arr = None

        # Crossing tracking
        self.crossings = {}
        self.crossing_points = {}
        self.c_e = {}
        self.E_star = set()
        self.global_crossings = 0
        self.local_crossings = 0
        self.best_crossings = float('inf')
        self.best_local_crossings = float('inf')
        self.best_sizemax = float('inf')
        self.initial_crossings = None
        self.initial_local_crossings = None
        self.last_crossings = None

        # Node state
        self.current_node = None
        self.node_visit_counts = None
        self.idle_streak = 0
        self.unsuccessful_moves = 0
        self.spiral_repairs_count = 0

        # Canvas
        self.width = None
        self.height = None

        # Spatial indexing
        self.rtree_index = None
        self.bboxes = {}
        self.edge_rtree_index = None
        self.edge_bboxes = {}
        self.edge_id = {}
        self.id_edge = {}

        # Node/edge caching
        self.index_node = None
        self.node_index = None
        self.edges = None
        self.edge_idx_pair = None
        self.idxpair_to_edge = None
        self.incident_by_node = None

        # Training state
        self.step_count = 0
        self.track_history = False
        self.history = None
        self.last_selection_stats = {}
        self.node_visit_repeat_count = 1
        self._node_visit_repeat_remaining = 0

        # Configuration
        # Reward weights MUST be provided via config; keep empty until configured
        self.reward_weights = {}
        # Track when the episode found a new best layout (step indices, 1-based)
        self.best_improvement_steps = []
        self.last_improvement_step = None
        self.reward_scale = 1.0
        self.node_selection_strategy = "random"
        self.step_limit = 2048
        self.move_step = 0.1
        self.min_bbox_size = 1
        self.optimization_goal = "global"  # "global" or "local"
        self.reset_unsuccessful_moves_threshold = None  # None = disabled, int = threshold
        self.use_int_grid = True  # Enforce integer grid coordinates
        self.node_visit_repeat_count = 1
        self._node_visit_repeat_remaining = 0
        self.heuristic_new_visit_penalty_coef = 0.5
        self.defer_next_node_selection = False
        self.defer_step_observation = False

        # Direction constants
        self.ANGLES = [i * (360 / n_directions) for i in range(n_directions)]
        self.DIRS_UNIT = ((1, 0), (.5, .5), (0, 1), (-.5, .5), (-1, 0), (-.5, -.5), (0, -1), (.5, -.5))
        self.DIRS_INT = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
        self.DIRS_UNIT_ARR = np.asarray(self.DIRS_UNIT, dtype=np.float64)
        self.OCTANT_THETAS = np.arange(n_directions, dtype=np.float64) * (2.0 * np.pi / n_directions)
        self.OCTANT_COS = np.cos(self.OCTANT_THETAS)
        self.OCTANT_SIN = np.sin(self.OCTANT_THETAS)
        base_idx = np.arange(n_directions, dtype=np.int64)
        self.OCTANT_ROTATE_IDX = np.stack([np.roll(base_idx, -k) for k in range(n_directions)], axis=0)
        self._last_max_idx = 0

    def _rtree_props(self):
        """Create R-tree properties for spatial indexing."""
        p = index.Property()
        p.leaf_capacity = 64
        p.index_capacity = 64
        p.near_minimum_overlap_factor = 32
        p.ensure_tight_mbr = True
        p.storage = index.RT_Memory
        return p

    def _precompute_static(self):
        """Precompute static node and edge caches."""
        self.index_node = list(self.graph.nodes())
        self.node_index = {v: i for i, v in enumerate(self.index_node)}
        self.edges = [tuple(sorted(e)) for e in self.graph.edges()]
        self.edge_idx_pair = {e: (self.node_index[e[0]], self.node_index[e[1]]) for e in self.edges}
        self.idxpair_to_edge = {ij: e for e, ij in self.edge_idx_pair.items()}
        self.edge_idx_arr = np.asarray([self.edge_idx_pair[e] for e in self.edges], dtype=np.int32)
        self.incident_by_node = {
            v: [tuple(sorted((v, u))) for u in self.graph.neighbors(v)]
            for v in self.graph.nodes()
        }
        self.node_degree = {
            v: int(self.graph.degree(v))
            for v in self.graph.nodes()
        }
        self.max_node_degree = max(self.node_degree.values(), default=1)
        self.incident_set_by_node = {
            v: set(edges)
            for v, edges in self.incident_by_node.items()
        }
        self.incident_edge_other_by_node = {
            v: [(tuple(sorted((v, u))), u) for u in self.graph.neighbors(v)]
            for v in self.graph.nodes()
        }
        self.incident_edge_idx_by_node = {
            v: np.asarray([self.edge_idx_pair[e] for e in self.incident_by_node[v]], dtype=np.int32)
            if self.incident_by_node[v] else np.empty((0, 2), dtype=np.int32)
            for v in self.graph.nodes()
        }
        self.incident_other_idx_by_node = {
            v: np.asarray([self.node_index[u] for _, u in self.incident_edge_other_by_node[v]], dtype=np.int32)
            if self.incident_edge_other_by_node[v] else np.empty((0,), dtype=np.int32)
            for v in self.graph.nodes()
        }
        self.neighbor_indices_by_node = {
            v: np.fromiter((self.node_index[u] for u in self.graph.neighbors(v)), dtype=np.int64)
            for v in self.graph.nodes()
        }

    def _sync_pos_arrays(self):
        """Mirror pos dict to pos_arr numpy array."""
        N = len(self.index_node)
        self.pos_arr = np.zeros((N, 2), dtype=np.float64)
        for v, i in self.node_index.items():
            self.pos_arr[i] = self.pos[v]

    def _rebuild_spatial_indices(self):
        """Rebuild node/edge spatial caches after bulk position changes."""
        self._sync_pos_arrays()
        self._update_rtree()
        self._init_edge_index()

    def _compute_bbox(self, node):
        """Compute tight bbox for a single node."""
        x, y = self.pos[node]
        x, y = float(x), float(y)
        half = 0.5
        return (x - half, y - half, x + half, y + half)

    def _update_rtree(self):
        """Rebuild node R-tree."""
        self.rtree_index = index.Index()
        self.bboxes = {}
        for node in self.graph.nodes():
            bbox = self._compute_bbox(node)
            self.rtree_index.insert(node, bbox)
            self.bboxes[node] = bbox

    def _create_edge_rtree_entry(self, edge, position):
        """Create R-tree entry for an edge."""
        u, v = edge
        pu = position[u] if isinstance(position, dict) else position[self.node_index[u]]
        pv = position[v] if isinstance(position, dict) else position[self.node_index[v]]
        ax = float(pu[0])
        ay = float(pu[1])
        bx = float(pv[0])
        by = float(pv[1])
        xmin = ax if ax < bx else bx
        ymin = ay if ay < by else by
        xmax = bx if ax < bx else ax
        ymax = by if ay < by else ay
        return edge, (float(xmin), float(ymin), float(xmax), float(ymax))

    def _edge_entries_for_bulkload(self):
        """Generator for R-tree bulk load of edge entries."""
        for i, e in enumerate(self.edges):
            _, bbox = self._create_edge_rtree_entry(e, self.pos)
            yield i, bbox, e

    def _init_edge_index(self):
        """Initialize edge R-tree."""
        self.edge_id = {e: i for i, e in enumerate(self.edges)}
        self.id_edge = {i: e for e, i in self.edge_id.items()}
        self.incident_edge_id_by_node = {
            v: np.asarray([self.edge_id[e] for e in self.incident_by_node[v]], dtype=np.int32)
            if self.incident_by_node[v] else np.empty((0,), dtype=np.int32)
            for v in self.graph.nodes()
        }
        self.edge_bboxes = {}
        for e in self.edges:
            _, bb = self._create_edge_rtree_entry(e, self.pos)
            self.edge_bboxes[e] = bb
        props = self._rtree_props()
        self.edge_rtree_index = index.Index(self._edge_entries_for_bulkload(), properties=props)

    def _update_edges_for_node(self, node):
        """Update edge R-tree for edges incident to node."""
        for e in self.incident_by_node.get(node, ()):
            eid = self.edge_id[e]
            old_bb = self.edge_bboxes[e]
            self.edge_rtree_index.delete(eid, old_bb)
            u_idx, v_idx = self.edge_idx_pair[e]
            ax, ay = self.pos_arr[u_idx]
            bx, by = self.pos_arr[v_idx]
            xmin = float(ax if ax < bx else bx)
            ymin = float(ay if ay < by else by)
            xmax = float(bx if ax < bx else ax)
            ymax = float(by if ay < by else ay)
            new_bb = (xmin, ymin, xmax, ymax)
            self.edge_bboxes[e] = new_bb
            self.edge_rtree_index.insert(eid, new_bb)

    def _union_bbox(self, edges):
        """Compute union bbox for a set of edges."""
        xmin = ymin = 1e100
        xmax = ymax = -1e100
        for e in edges:
            bx, by, Bx, By = self.edge_bboxes[e]
            if bx < xmin: xmin = bx
            if by < ymin: ymin = by
            if Bx > xmax: xmax = Bx
            if By > ymax: ymax = By
        return (xmin, ymin, xmax, ymax)

    def query_edges_overlaps_any_ids(self, edges):
        """Query R-tree for edges overlapping given edge set."""
        ub = self._union_bbox(edges)
        return list(self.edge_rtree_index.intersection(ub))

    def compute_crossings(self, edges=None):
        """Compute crossings for relevant edge set using C++ kernel."""
        if edges is None:
            relevant_edges = list(self.edges)
        else:
            to_update = [tuple(sorted(e)) for e in edges]
            if not to_update:
                return {}, None, 0
            cand_ids = self.query_edges_overlaps_any_ids(to_update)
            overlap_edges = [self.id_edge[eid] for eid in cand_ids]
            relevant_edges = list({*to_update, *overlap_edges})

        positions = self.pos_arr
        rel_edges_idx = [self.edge_idx_pair[e] for e in relevant_edges]

        crossings_map_idx, max_edge_idx, global_crossings = compute_crossings(positions, rel_edges_idx)

        idx_to_edge = {self.edge_idx_pair[e]: e for e in relevant_edges}

        crossings = {}
        for (i, j), crossers in crossings_map_idx.items():
            e = idx_to_edge[(i, j)]
            S = crossings.setdefault(e, set())
            for (u, v) in crossers:
                f = idx_to_edge[(u, v)]
                S.add(f)

        if edges is None:
            self.crossings = crossings
            self.crossing_points = {}
            unique_pairs = set()
            for e, S in self.crossings.items():
                for f in S:
                    unique_pairs.add(self._crossing_pair_key(e, f))
            self._refresh_crossing_points(unique_pairs)
        else:
            rel_set = set(relevant_edges)
            for S in self.crossings.values():
                S.difference_update(rel_set)
            for e in relevant_edges:
                self.crossings[e] = set()
            for e, S in crossings.items():
                for f in S:
                    self.crossings.setdefault(e, set()).add(f)
                    self.crossings.setdefault(f, set()).add(e)

            updated_pairs = set()
            for e, S in crossings.items():
                for f in S:
                    updated_pairs.add(self._crossing_pair_key(e, f))
            self._refresh_crossing_points(updated_pairs)

        if max_edge_idx is not None and max_edge_idx != (-1, -1) and max_edge_idx != [-1, -1] and tuple(max_edge_idx) in idx_to_edge:
            max_crossing_edge = idx_to_edge[tuple(max_edge_idx)]
        else:
            max_crossing_edge = None

        return self.crossings, max_crossing_edge, global_crossings

    def _rebuild_counts_from_crossings(self):
        """Rebuild c_e, global_crossings, local_crossings from self.crossings."""
        self.c_e = {e: len(self.crossings.get(e, ())) for e in self.edges}
        self.global_crossings = sum(self.c_e.values()) // 2
        self._count_freq = {}
        for c in self.c_e.values():
            self._count_freq[c] = self._count_freq.get(c, 0) + 1
        if self._count_freq:
            m = max(self._count_freq.keys())
            self.local_crossings = m
            self.E_star = {e for e, c in self.c_e.items() if c == m}
        else:
            self.local_crossings = 0
            self.E_star = set()

    @staticmethod
    def _crossing_pair_key(e, f):
        return (e, f) if e < f else (f, e)

    def _refresh_crossing_points(self, pairs):
        if not pairs:
            return

        pair_list = list(pairs)
        edge_pair_arr = np.asarray(
            [(e[0], e[1], f[0], f[1]) for e, f in pair_list],
            dtype=np.int32,
        )
        point_arr = cpp_edge_pair_crossing_points(self.pos_arr, edge_pair_arr)
        for idx, (e, f) in enumerate(pair_list):
            px = float(point_arr[idx, 0])
            py = float(point_arr[idx, 1])
            if not (math.isfinite(px) and math.isfinite(py)):
                continue
            pair = self._crossing_pair_key(e, f)
            self.crossing_points[pair] = (px, py)

    def _remove_crossing_points(self, pairs):
        for e, f in pairs:
            self.crossing_points.pop(self._crossing_pair_key(e, f), None)

    def _apply_crossing_deltas(self, added_pairs, removed_pairs):
        """Apply crossing deltas and recompute counts."""
        if not added_pairs and not removed_pairs:
            return

        if not hasattr(self, "_count_freq") or not self._count_freq:
            self._rebuild_counts_from_crossings()

        touched = set()
        for e, f in removed_pairs:
            touched.add(e)
            touched.add(f)
        for e, f in added_pairs:
            touched.add(e)
            touched.add(f)
        old_counts = {e: int(self.c_e.get(e, 0)) for e in touched}
        old_local = int(self.local_crossings)

        for e, f in removed_pairs:
            if e in self.crossings:
                self.crossings[e].discard(f)
            if f in self.crossings:
                self.crossings[f].discard(e)
        for e, f in added_pairs:
            self.crossings.setdefault(e, set()).add(f)
            self.crossings.setdefault(f, set()).add(e)

        if removed_pairs:
            self._remove_crossing_points(removed_pairs)
        if added_pairs:
            self._refresh_crossing_points(added_pairs)

        for e in touched:
            if not self.crossings.get(e):
                self.crossings.pop(e, None)

        self.global_crossings += int(len(added_pairs) - len(removed_pairs))
        if self.global_crossings <= 0:
            self.global_crossings = 0
            for e in touched:
                old = old_counts[e]
                new = 0
                self._count_freq[old] = self._count_freq.get(old, 0) - 1
                if self._count_freq[old] <= 0:
                    self._count_freq.pop(old, None)
                self._count_freq[new] = self._count_freq.get(new, 0) + 1
                self.c_e[e] = new
            self.local_crossings = 0
            self.E_star = set()
            return

        max_touched_count = 0
        for e in touched:
            old = old_counts[e]
            new = len(self.crossings.get(e, ()))
            if old != new:
                self._count_freq[old] = self._count_freq.get(old, 0) - 1
                if self._count_freq[old] <= 0:
                    self._count_freq.pop(old, None)
                self._count_freq[new] = self._count_freq.get(new, 0) + 1
                self.c_e[e] = new
            if new > max_touched_count:
                max_touched_count = new

        if self._count_freq.get(old_local, 0) <= 0:
            positive_counts = [count for count, freq in self._count_freq.items() if count > 0 and freq > 0]
            if positive_counts:
                self.local_crossings = max(positive_counts)
                self.E_star = {e for e, count in self.c_e.items() if count == self.local_crossings}
            else:
                self.local_crossings = 0
                self.E_star = set()
            return

        if max_touched_count > old_local:
            self.local_crossings = max_touched_count
            self.E_star = {e for e in touched if self.c_e[e] == self.local_crossings}
            return

        self.local_crossings = old_local
        for e in touched:
            if old_counts[e] == old_local and self.c_e[e] != old_local:
                self.E_star.discard(e)
        for e in touched:
            if self.c_e[e] == old_local:
                self.E_star.add(e)

    def remove_crossings_for_node(self, node):
        """Return crossing pairs for edges incident to node WITHOUT modifying self.crossings."""
        E_incident = {tuple(sorted(e)) for e in self.graph.edges(node)}
        removed = set()

        for e in E_incident:
            S = self.crossings.get(e)
            if not S:
                continue
            for f in S:
                a, b = (e, f) if e < f else (f, e)
                removed.add((a, b))

        return list(removed)

    @staticmethod
    def _ranges_overlap(a1, a2, b1, b2):
        """Check if two 1D ranges overlap."""
        if a1 > a2: a1, a2 = a2, a1
        if b1 > b2: b1, b2 = b2, b1
        return not (a2 < b1 or b2 < a1)

    def recompute_crossings_for_node(self, node):
        """Recompute crossings for edges incident to node."""
        inc = self.incident_by_node[node]
        if not inc:
            return []

        ub = self._union_bbox(inc)
        cand_ids = set(self.edge_rtree_index.intersection(ub))

        cand_edges = []
        for eid in cand_ids:
            f = self.id_edge[eid]
            if f in inc:
                continue
            cand_edges.append(f)

        incident_idx = [self.edge_idx_pair[e] for e in inc]
        candidate_idx = [self.edge_idx_pair[e] for e in cand_edges]
        if not incident_idx or not candidate_idx:
            return []

        crossings_map = find_crossings_for_edges(self.pos_arr, incident_idx, candidate_idx)

        added = set()
        idx_to_edge = {}
        for e in inc:
            idx_to_edge[self.edge_idx_pair[e]] = e
        for f in cand_edges:
            idx_to_edge[self.edge_idx_pair[f]] = f

        for e_idx, crossed_set in crossings_map.items():
            e = idx_to_edge[tuple(e_idx)]
            eu, ev = e
            for f_idx in crossed_set:
                f = idx_to_edge[tuple(f_idx)]
                if eu in f or ev in f:
                    continue
                a, b = (e, f) if e < f else (f, e)
                added.add((a, b))

        return list(added)

    @staticmethod
    def _line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
        """Compute line intersection point."""
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        x3 = float(x3)
        y3 = float(y3)
        x4 = float(x4)
        y4 = float(y4)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return None
        a = (x1 * y2 - y1 * x2)
        b = (x3 * y4 - y3 * x4)
        px = (a * (x3 - x4) - (x1 - x2) * b) / denom
        py = (a * (y3 - y4) - (y1 - y2) * b) / denom
        return px, py

    def get_crossing_point(self, u1, v1, u2, v2):
        """Get crossing point between two edges."""
        i1 = self.node_index[u1]
        i2 = self.node_index[v1]
        i3 = self.node_index[u2]
        i4 = self.node_index[v2]
        x1, y1 = self.pos_arr[i1]
        x2, y2 = self.pos_arr[i2]
        x3, y3 = self.pos_arr[i3]
        x4, y4 = self.pos_arr[i4]
        return self._line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4)

    def _as_int_point(self, p):
        """Convert point to integer grid."""
        return np.asarray(np.rint(np.asarray(p, dtype=np.float64)), dtype=np.int64)

    def _grid_xy(self, node):
        """Return integer grid coordinates for a node without allocations when possible."""
        pos = self.pos[node]
        if self.use_int_grid:
            return int(pos[0]), int(pos[1])
        rounded = self._as_int_point(pos)
        return int(rounded[0]), int(rounded[1])

    @staticmethod
    def _point_on_segment_grid_xy(ax, ay, bx, by, px, py):
        """Check if integer point lies on integer segment using scalar arithmetic."""
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross != 0:
            return False
        return (min(ax, bx) <= px <= max(ax, bx)) and (min(ay, by) <= py <= max(ay, by))

    def _point_on_segment_grid(self, a, b, p):
        """Check if integer point is on integer segment."""
        ax, ay = int(a[0]), int(a[1])
        bx, by = int(b[0]), int(b[1])
        px, py = int(p[0]), int(p[1])
        return self._point_on_segment_grid_xy(ax, ay, bx, by, px, py)

    def find_nodes_on_incident_edges(self, node):
        """Find nodes lying on incident edges of given node."""
        offenders = []
        inc = self.incident_by_node.get(node, [])
        if not inc:
            return offenders

        seen = set()

        if hasattr(self, "rtree_index") and self.rtree_index is not None:
            edge_data = []
            xmin = ymin = 10**18
            xmax = ymax = -(10**18)
            for e in inc:
                u, v = e
                ax, ay = self._grid_xy(u)
                bx, by = self._grid_xy(v)
                exmin = ax if ax < bx else bx
                exmax = bx if ax < bx else ax
                eymin = ay if ay < by else by
                eymax = by if ay < by else ay
                if exmin < xmin:
                    xmin = exmin
                if exmax > xmax:
                    xmax = exmax
                if eymin < ymin:
                    ymin = eymin
                if eymax > ymax:
                    ymax = eymax
                edge_data.append((e, u, v, ax, ay, bx, by))

            for w in self.rtree_index.intersection((xmin, ymin, xmax, ymax)):
                px, py = self._grid_xy(w)
                for e, u, v, ax, ay, bx, by in edge_data:
                    if w in (u, v):
                        continue
                    key = (w, e)
                    if key in seen:
                        continue
                    if self._point_on_segment_grid_xy(ax, ay, bx, by, px, py):
                        offenders.append((w, e))
                        seen.add(key)
            return offenders

        all_nodes = list(self.graph.nodes())
        for e in inc:
            u, v = e
            ax, ay = self._grid_xy(u)
            bx, by = self._grid_xy(v)
            for w in all_nodes:
                if w in (u, v):
                    continue
                px, py = self._grid_xy(w)
                if self._point_on_segment_grid_xy(ax, ay, bx, by, px, py):
                    key = (w, e)
                    if key not in seen:
                        offenders.append((w, e))
                        seen.add(key)
        return offenders

    def _is_node_strictly_on_edge(self, node):
        """Check if node is strictly placed on an edge."""
        x, y = self._grid_xy(node)

        if hasattr(self, "edge_rtree_index") and self.edge_rtree_index is not None and hasattr(self, "id_edge"):
            cand_ids = list(self.edge_rtree_index.intersection((x, y, x, y)))
            candidate_edges = (self.id_edge[eid] for eid in cand_ids)
        else:
            candidate_edges = self.edges

        for (u, v) in candidate_edges:
            if node in (u, v):
                continue
            ax, ay = self._grid_xy(u)
            bx, by = self._grid_xy(v)
            if self._point_on_segment_grid_xy(ax, ay, bx, by, x, y):
                return True

        if self.find_nodes_on_incident_edges(node):
            return True

        return False

    def is_node_position_legal(self, node, check_incident_edges=True):
        """Check if node position is legal (no overlaps or edge violations)."""
        x, y = self._grid_xy(node)

        if hasattr(self, "rtree_index") and self.rtree_index is not None:
            for other in self.rtree_index.intersection((x, y, x, y)):
                if other == node:
                    continue
                ox, oy = self._grid_xy(other)
                if ox == x and oy == y:
                    return False
        else:
            for other in self.graph.nodes():
                if other == node:
                    continue
                ox, oy = self._grid_xy(other)
                if ox == x and oy == y:
                    return False

        if hasattr(self, "edge_rtree_index") and self.edge_rtree_index is not None and hasattr(self, "id_edge"):
            cand_ids = list(self.edge_rtree_index.intersection((x, y, x, y)))
            candidate_edges = (self.id_edge[eid] for eid in cand_ids)
        else:
            candidate_edges = self.edges

        for (u, v) in candidate_edges:
            if node in (u, v):
                continue
            ax, ay = self._grid_xy(u)
            bx, by = self._grid_xy(v)
            if self._point_on_segment_grid_xy(ax, ay, bx, by, x, y):
                return False

        if check_incident_edges:
            if self.find_nodes_on_incident_edges(node):
                return False

        return True

    def query_neighbors_within_radius(self, node):
        """Query neighbors within max incident edge distance."""
        center = self.pos[node]
        neigh = list(self.graph.neighbors(node))
        longest = max(calculate_distance(center, self.pos[u]) for u in neigh) if neigh else self.min_bbox_size
        radius = max(longest, self.min_bbox_size)
        bbox = (center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius)
        candidates = self.rtree_index.intersection(bbox)
        return [n for n in candidates if n != node and calculate_distance(center, self.pos[n]) <= radius]

    def _dominant_dir_index_from_delta(self, delta: np.ndarray) -> int:
        """Map delta vector to dominant direction index."""
        dx, dy = float(delta[0]), float(delta[1])
        if dx == 0.0 and dy == 0.0:
            return 0
        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        return int(round(angle / 45.0)) % 8

    def _find_legal_position_spiral(self, node, start_pos: np.ndarray, start_dir_idx: int, max_radius: int = None) -> np.ndarray:
        """Find nearest legal position via spiral search."""
        self.spiral_repairs_count += 1
        if max_radius is None:
            cw = int(self.width) if self.width is not None else 1000
            ch = int(self.height) if self.height is not None else 1000
            max_radius = max(cw, ch)

        def in_bounds(p):
            if self.width is None or self.height is None: return True
            return 0 <= p[0] <= int(self.width) - 1 and 0 <= p[1] <= int(self.height) - 1

        original = self._as_int_point(start_pos)
        old_pos = self.pos[node].copy()
        
        self.pos[node] = original.astype(np.int32)
        self.pos_arr[self.node_index[node]] = self.pos[node]
        if in_bounds(original) and self.is_node_position_legal(node):
            return original

        cardinal_steps = [np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0])]

        for r in range(1, int(max_radius) + 1):
            p = original + np.array([r, -r], dtype=np.int64)
            
            for step_vector in cardinal_steps:
                for _ in range(2 * r):
                    if in_bounds(p):
                        self.pos[node] = p.astype(np.int32)
                        self.pos_arr[self.node_index[node]] = self.pos[node]
                        if self.is_node_position_legal(node):
                            return p.astype(np.int64)
                    p = p + step_vector

        # Fallback: restore original
        self.pos[node] = old_pos
        self.pos_arr[self.node_index[node]] = self.pos[node]
        return original

    @staticmethod
    def convert_to_integer_grid(pos, canvas_width=None, canvas_height=None):
        """Convert floating point positions to integer grid."""
        if not pos:
            return {}

        min_x = min(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        shifted_pos = {node: np.array([p[0] - min_x, p[1] - min_y], dtype=float)
                       for node, p in pos.items()}

        if canvas_width is not None and canvas_height is not None:
            usable_w = 0.8 * float(canvas_width)
            usable_h = 0.8 * float(canvas_height)
            margin_x = 0.1 * float(canvas_width)
            margin_y = 0.1 * float(canvas_height)

            max_x = max(p[0] for p in shifted_pos.values())
            max_y = max(p[1] for p in shifted_pos.values())

            if max_x == 0 and max_y == 0:
                center = np.array([margin_x + usable_w / 2.0,
                                   margin_y + usable_h / 2.0])
                rounded_pos = {node: np.round(center).astype(np.int32) for node in pos}
                return rounded_pos

            scale_x = (usable_w / max_x) if max_x > 0 else float('inf')
            scale_y = (usable_h / max_y) if max_y > 0 else float('inf')
            scaling_factor = min(scale_x, scale_y)

            final_pos = {}
            for node, p in shifted_pos.items():
                scaled = p * scaling_factor
                translated = scaled + np.array([margin_x, margin_y])
                final_pos[node] = translated

            rounded_pos = {node: np.round(coords).astype(np.int32) for node, coords in final_pos.items()}
            max_x_int = max(0, int(np.floor(canvas_width)) - 1)
            max_y_int = max(0, int(np.floor(canvas_height)) - 1)
            for node, coords in rounded_pos.items():
                coords[0] = np.clip(coords[0], 0, max_x_int)
                coords[1] = np.clip(coords[1], 0, max_y_int)

            return rounded_pos

        scaling_factor = 1.0
        max_iterations = 1000
        iteration = 0

        while iteration < max_iterations:
            scaled_pos = {node: p * scaling_factor for node, p in shifted_pos.items()}
            rounded_pos = {node: np.round(p).astype(np.int32) for node, p in scaled_pos.items()}

            positions = {}
            has_overlap = False
            for node, coords in rounded_pos.items():
                pos_tuple = (int(coords[0]), int(coords[1]))
                if pos_tuple in positions:
                    has_overlap = True
                    break
                positions[pos_tuple] = node

            if not has_overlap:
                return rounded_pos

            scaling_factor *= 1.5
            iteration += 1

    def get_observation(self):
        """
        Returns the observation for the current node, including local view,
        pixel map, crossing information, and local/global statistics.
        
        Returns:
            Dict with keys:
                - "pixel_map": (3, patch_size, patch_size) pixel representation
                - "cross_map": (8,) octant-based crossing heatmap
                - "cross_map_local": (8,) local crossing heatmap
                - "local_view": (40,) local observation features
                - "local_crossings": (1,) number of crossings in local view
                - "global_crossings": (1,) total graph crossings
        """
        pixel_map, cross_oct, cross_oct_local, base_obs = self.get_observation_components()

        local_crossings_obs = np.array([float(self.local_crossings)], dtype=np.float32)
        global_crossings_obs = np.array([float(self.global_crossings)], dtype=np.float32)

        return {
            "pixel_map": pixel_map,
            "cross_map": cross_oct,
            "cross_map_local": cross_oct_local,
            "local_view": base_obs,
            "local_crossings": local_crossings_obs,
            "global_crossings": global_crossings_obs,
        }

    def get_observation_components(self):
        """Return the current-node observation components without packaging them into a dict."""
        base_obs, cross_oct, cross_oct_local = self._get_observation_Octant()
        pixel_map = self._get_pixel_map()
        return pixel_map, cross_oct, cross_oct_local, base_obs

    def _get_pixel_map(self):
        """Compute pixel map representation with decay-based channels."""
        idx = self.obs_dir_to_world(0)
        batch = self._get_pixel_map_batch([self.current_node], [idx])
        if batch.shape[0] == 0:
            return np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        return batch[0]

    def _collect_crossing_points_for_node(self, node, bbox, edge_ids, incident_ids):
        if edge_ids.size == 0 or incident_ids.size == 0:
            return []

        nearby_mask = np.zeros(len(self.edges), dtype=np.bool_)
        nearby_mask[edge_ids] = True
        processed_crossings = set()
        crossing_points = []
        pair_crossing_points = self.crossing_points

        for edge, eid in zip(self.incident_by_node.get(node, ()), incident_ids.tolist()):
            if not nearby_mask[eid]:
                continue
            for other in self.crossings.get(edge, ()):
                other_id = self.edge_id[other]
                if not nearby_mask[other_id]:
                    continue
                pair = self._crossing_pair_key(edge, other)
                if pair in processed_crossings:
                    continue
                processed_crossings.add(pair)

                cp = pair_crossing_points.get(pair)
                if cp is None:
                    cp = self.get_crossing_point(edge[0], edge[1], other[0], other[1])
                if cp is None:
                    continue

                cpx, cpy = cp
                if bbox[0] - 2 <= cpx <= bbox[2] + 2 and bbox[1] - 2 <= cpy <= bbox[3] + 2:
                    crossing_points.append((cpx, cpy))
        return crossing_points

    def _get_pixel_map_batch(self, nodes, rotations):
        if not nodes:
            return np.empty((0, 3, self.patch_size, self.patch_size), dtype=np.float32)

        node_indices = np.asarray([self.node_index[node] for node in nodes], dtype=np.int32)
        rotation_indices = np.asarray([int(rot) % 8 for rot in rotations], dtype=np.int32)

        edge_offsets = [0]
        edge_ids_flat = []
        incident_offsets = [0]
        incident_ids_flat = []
        crossing_offsets = [0]
        crossing_points_flat = []

        radius = self.pixel_query_radius
        for node in nodes:
            node_idx = self.node_index[node]
            nx, ny = self.pos_arr[node_idx]
            bbox = (nx - radius, ny - radius, nx + radius, ny + radius)
            edge_ids = np.fromiter(self.edge_rtree_index.intersection(bbox), dtype=np.int32)
            edge_ids_flat.extend(edge_ids.tolist())
            edge_offsets.append(len(edge_ids_flat))

            incident_ids = self.incident_edge_id_by_node[node]
            incident_ids_flat.extend(incident_ids.tolist())
            incident_offsets.append(len(incident_ids_flat))

            crossing_points = self._collect_crossing_points_for_node(node, bbox, edge_ids, incident_ids)
            crossing_points_flat.extend(crossing_points)
            crossing_offsets.append(len(crossing_points_flat))

        edge_ids_arr = np.asarray(edge_ids_flat, dtype=np.int32) if edge_ids_flat else np.empty((0,), dtype=np.int32)
        incident_ids_arr = np.asarray(incident_ids_flat, dtype=np.int32) if incident_ids_flat else np.empty((0,), dtype=np.int32)
        crossing_points_arr = (
            np.asarray(crossing_points_flat, dtype=np.float64)
            if crossing_points_flat
            else np.empty((0, 2), dtype=np.float64)
        )

        return cpp_batch_pixel_maps(
            self.pos_arr,
            self.edge_idx_arr,
            node_indices,
            rotation_indices,
            np.asarray(edge_offsets, dtype=np.int32),
            edge_ids_arr,
            np.asarray(incident_offsets, dtype=np.int32),
            incident_ids_arr,
            np.asarray(crossing_offsets, dtype=np.int32),
            crossing_points_arr,
            self.pixel_dx_flat_arr,
            self.pixel_dy_flat_arr,
            self.patch_size,
            float(self.pixel_decay_alpha),
        )

    def _get_observation_Octant(self):
        """Compute octant-based observation features."""
        node = self.current_node
        i = self.node_index[node]
        node_pos = self.pos_arr[i]  # (2,)
        P = self.pos_arr  # (N,2)
        V = P - node_pos  # vectors to all
        # Build neighbor mask over all nodes first
        neigh = np.zeros(len(self.index_node), dtype=bool)
        neigh_idx = self.neighbor_indices_by_node.get(node)
        if neigh_idx is not None and neigh_idx.size > 0:
            neigh[neigh_idx] = True

        # Exclude self from subsequent computations (angles/bins/distances/counts)
        mask_others = np.ones(len(P), dtype=bool)
        mask_others[i] = False
        V_others = V[mask_others]
        ang = (np.degrees(np.arctan2(V_others[:, 1], V_others[:, 0])) + 360.0) % 360.0
        bins = (ang // 45).astype(np.int64) % 8
        d = np.linalg.norm(V_others, axis=1)
        neigh_others = neigh[mask_others]

        counts = np.bincount(bins, minlength=8).astype(np.float32)
        neigh_d = np.full(8, np.inf, dtype=np.float32)
        nonneigh_d = np.full(8, np.inf, dtype=np.float32)

        if d.size > 0:
            if np.any(neigh_others):
                np.minimum.at(neigh_d, bins[neigh_others], d[neigh_others].astype(np.float32, copy=False))
            if np.any(~neigh_others):
                np.minimum.at(nonneigh_d, bins[~neigh_others], d[~neigh_others].astype(np.float32, copy=False))

        abs_counts = counts.copy()
        neigh_d[~np.isfinite(neigh_d)] = 0.0
        nonneigh_d[~np.isfinite(nonneigh_d)] = 0.0
        rel_counts = counts / counts.sum() if counts.sum() > 0 else counts
        rel_abs_counts = abs_counts / abs_counts.max() if abs_counts.max() > 0 else abs_counts

        #computed octant data as before
        #now compute octant crossing map
        cross_oct = np.zeros(8, dtype=np.float32)
        cross_oct_local = np.zeros(8, dtype=np.float32)
        nx_, ny_ = node_pos
        for edge, other in self.incident_edge_other_by_node[node]:

            # Octant from node -> neighbor direction
            dx_e = self.pos[other][0] - nx_
            dy_e = self.pos[other][1] - ny_
            if dx_e == 0.0 and dy_e == 0.0:
                continue
            angle = (math.degrees(math.atan2(dy_e, dx_e)) + 360.0) % 360.0
            bin_idx = int(angle // 45.0) % 8

            # Crossing count for this incident edge
            crossed_set = self.crossings.get(edge, ())
            c = float(len(crossed_set))

            # Aggregate: sum for the octant, and track the max per octant
            cross_oct[bin_idx] += c
            if c > cross_oct_local[bin_idx]:
                cross_oct_local[bin_idx] = c

        # Normalize each to [0,1] (independently) for scale stability before rotation
        if cross_oct.max() > 0:
            cross_oct /= cross_oct.max()
        if cross_oct_local.max() > 0:
            cross_oct_local /= cross_oct_local.max()

        # compute absolute distance to first edge hit for each octant ray
        edge_ray = self._edge_ray_octant_distances(node)

        #shift and rotate everything according to first data, i.e., crossing octant?
        cross_oct, cross_oct_local, rel_counts, neigh_d, nonneigh_d, rel_abs_counts, edge_ray = \
            self.shift_and_rotate_octants(cross_oct, cross_oct_local, rel_counts, neigh_d, nonneigh_d, rel_abs_counts, edge_ray)

        obs = np.round(np.concatenate((rel_counts, neigh_d, nonneigh_d, rel_abs_counts, edge_ray)), 3).astype(np.float32)

        return obs, cross_oct, cross_oct_local

    def _get_observation_Octant_batch(self, nodes):
        """Compute octant-based observation features for multiple nodes in one C++ call."""
        if not nodes:
            empty_rot = np.empty((0,), dtype=np.int32)
            empty_vec = np.empty((0, 40), dtype=np.float32)
            empty_cross = np.empty((0, 8), dtype=np.float32)
            return empty_vec, empty_cross, empty_cross.copy(), empty_rot

        node_indices = np.asarray([self.node_index[node] for node in nodes], dtype=np.int32)

        neighbor_offsets = [0]
        neighbor_flat = []
        incident_offsets = [0]
        incident_edges = []
        incident_other = []
        incident_cross = []
        for node in nodes:
            neigh_idx = self.neighbor_indices_by_node.get(node)
            if neigh_idx is not None and neigh_idx.size > 0:
                neighbor_flat.extend(neigh_idx.tolist())
            neighbor_offsets.append(len(neighbor_flat))

            inc_edges_arr = self.incident_edge_idx_by_node[node]
            inc_other_arr = self.incident_other_idx_by_node[node]
            if inc_edges_arr.size > 0:
                incident_edges.extend(inc_edges_arr.tolist())
                incident_other.extend(inc_other_arr.tolist())
                for edge in self.incident_by_node[node]:
                    incident_cross.append(float(self.c_e.get(edge, len(self.crossings.get(edge, ())))))
            incident_offsets.append(len(incident_other))

        if neighbor_flat:
            neighbor_arr = np.asarray(neighbor_flat, dtype=np.int32)
        else:
            neighbor_arr = np.empty((0,), dtype=np.int32)
        if incident_edges:
            incident_edges_arr = np.asarray(incident_edges, dtype=np.int32)
            incident_other_arr = np.asarray(incident_other, dtype=np.int32)
            incident_cross_arr = np.asarray(incident_cross, dtype=np.float32)
        else:
            incident_edges_arr = np.empty((0, 2), dtype=np.int32)
            incident_other_arr = np.empty((0,), dtype=np.int32)
            incident_cross_arr = np.empty((0,), dtype=np.float32)

        mins = self.pos_arr.min(axis=0)
        maxs = self.pos_arr.max(axis=0)
        diag = float(np.hypot(maxs[0] - mins[0], maxs[1] - mins[1]))
        ray_length = max(diag, 1.0) + 1.0

        base_obs, cross_oct, cross_oct_local, rotations = cpp_batch_octant_observations(
            self.pos_arr,
            self.edge_idx_arr,
            node_indices,
            np.asarray(neighbor_offsets, dtype=np.int32),
            neighbor_arr,
            np.asarray(incident_offsets, dtype=np.int32),
            incident_edges_arr,
            incident_other_arr,
            incident_cross_arr,
            self.DIRS_UNIT_ARR,
            self.OCTANT_COS,
            self.OCTANT_SIN,
            ray_length,
        )
        return (
            np.round(np.asarray(base_obs, dtype=np.float32), 3),
            np.asarray(cross_oct, dtype=np.float32),
            np.asarray(cross_oct_local, dtype=np.float32),
            np.asarray(rotations, dtype=np.int32),
        )

    def _edge_ray_octant_distances(self, node, eps=1e-9):
        """
        For each of the 8 octants, shoot a ray from `node` along the octant center
        direction and return the absolute distance to the first non-incident edge.
        If no edge is hit, returns a large value equal to the query ray length (diag).
        """
        i = self.node_index[node]
        mins = self.pos_arr.min(axis=0)
        maxs = self.pos_arr.max(axis=0)
        diag = float(np.hypot(maxs[0] - mins[0], maxs[1] - mins[1]))
        R = max(diag, 1.0) + 1.0

        return cpp_edge_ray_octant_distances(
            self.pos_arr,
            self.edge_idx_arr,
            self.incident_edge_idx_by_node[node],
            i,
            self.DIRS_UNIT_ARR,
            R,
            eps,
        )

    @staticmethod
    def _ray_segment_intersection_param(o, d, a, b, eps=1e-9):
        """
        Ray/segment intersection parameter along the ray.
        Returns t >= 0 if ray o + t d intersects segment [a,b], else None.
        d must be unit length. Excludes hits with t < eps. Handles collinearity.
        """
        r = d
        s = b - a

        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]

        denom = cross(r, s)
        ao = a - o

        if abs(denom) < eps:
            # Parallel; check collinearity
            if abs(cross(ao, r)) >= eps:
                return None
            # Collinear: project endpoints onto ray
            t0 = float(np.dot(a - o, r))
            t1 = float(np.dot(b - o, r))
            lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
            if hi < eps:
                return None  # segment entirely behind
            if lo >= eps:
                return lo
            # overlaps across origin; nearest forward point
            return hi if hi >= eps else None

        t = cross(ao, s) / denom  # along ray
        u = cross(ao, r) / denom  # along segment
        if t >= eps and 0.0 <= u <= 1.0:
            return float(t)
        return None

    def _legalize_initial_layout(self, max_passes: int = 3):
        """Ensure no nodes are illegal by spiraling them to nearest legal positions.
        Updates node and edge R-trees as nodes move. Runs a few passes to resolve cascades.
        """
        # Legalization is an integer-grid repair pass; skip entirely for float layouts.
        if not self.use_int_grid:
            return

        for _ in range(int(max_passes)):
            any_fix = False
            for node in self.graph.nodes():
                if not self.is_node_position_legal(node):
                    cand = self._find_legal_position_spiral(node, start_pos=self.pos[node], start_dir_idx=0)
                    if (int(cand[0]) != int(self.pos[node][0])) or (int(cand[1]) != int(self.pos[node][1])):
                        any_fix = True
                    self.pos[node] = np.asarray(cand, dtype=np.int32)
                    self.pos_arr[self.node_index[node]] = self.pos[node]
                    # update node rtree and edges
                    if node in self.bboxes:
                        self.rtree_index.delete(node, self.bboxes[node])
                    new_bb = self._compute_bbox(node)
                    self.rtree_index.insert(node, new_bb)
                    self.bboxes[node] = new_bb
                    self._update_edges_for_node(node)

                    # Repair offenders on incident edges
                    offenders = self.find_nodes_on_incident_edges(node)
                    seen = set()
                    for w, _ in offenders:
                        if w in seen:
                            continue
                        seen.add(w)
                        cand_w = self._find_legal_position_spiral(w, start_pos=self.pos[w], start_dir_idx=0)
                        if (int(cand_w[0]) != int(self.pos[w][0])) or (int(cand_w[1]) != int(self.pos[w][1])):
                            any_fix = True
                        self.pos[w] = np.asarray(cand_w, dtype=np.int32)
                        self.pos_arr[self.node_index[w]] = self.pos[w]
                        if w in self.bboxes:
                            self.rtree_index.delete(w, self.bboxes[w])
                        newbbw = self._compute_bbox(w)
                        self.rtree_index.insert(w, newbbw)
                        self.bboxes[w] = newbbw
                        self._update_edges_for_node(w)
            if not any_fix:
                break

    def _validate_positions_on_grid(self):
        """Check if all node positions are on integer grid. Returns True if valid."""
        if not self.use_int_grid:
            return True
        
        for node in self.graph.nodes():
            pos = self.pos[node]
            # Check if position is integer-valued
            if not (np.isclose(pos[0], np.round(pos[0])) and np.isclose(pos[1], np.round(pos[1]))):
                return False
        return True

    def _init_for_graph(self, G, width, height, seed=None):
        """Initialize environment state for a new graph."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.graph = G
        self.n_nodes = G.number_of_nodes()

        self.height = height if height is not None else 100
        self.width = width if width is not None else 100

        cache_key = (int(self.width), int(self.height), self.use_int_grid)
        graph_cache = self.graph.graph.setdefault("_layout_env_initial_pos_cache", {})
        cached_initial_pos = graph_cache.get(cache_key)
        if cached_initial_pos is None:
            float_pos = get_best_layout_by_crossings(self.graph)
            if self.use_int_grid:
                cached_initial_pos = {
                    n: np.array(coords, dtype=np.int32)
                    for n, coords in self.convert_to_integer_grid(float_pos, self.width, self.height).items()
                }
            else:
                cached_initial_pos = {
                    n: np.array(coords, dtype=np.float64)
                    for n, coords in float_pos.items()
                }
            graph_cache[cache_key] = cached_initial_pos

        dtype = np.int32 if self.use_int_grid else np.float64
        self.initial_pos = {n: np.array(cached_initial_pos[n], dtype=dtype) for n in self.graph.nodes()}
        self.pos = {n: np.array(self.initial_pos[n], dtype=dtype) for n in self.graph.nodes()}

        self._precompute_static()
        self._sync_pos_arrays()

        self._update_rtree()
        self._init_edge_index()

        if self.use_int_grid and hasattr(self, "_legalize_initial_layout"):
            self._legalize_initial_layout()

        dtype = np.int32 if self.use_int_grid else np.float64
        self.initial_pos = {n: np.array(self.pos[n], dtype=dtype) for n in self.graph.nodes()}
        self._sync_pos_arrays()

        self.crossings.clear()
        self.compute_crossings()
        self._rebuild_counts_from_crossings()

        self.current_node = self._choose_next_node()
        self._reset_node_visit_repeat_state()
        self.step_count = 0
        self.idle_streak = 0
        self.unsuccessful_moves = 0
        self.best_improvement_steps = []
        self.last_improvement_step = None
        self.initial_crossings = self.global_crossings
        self.initial_local_crossings = self.local_crossings
        self.last_crossings = self.global_crossings
        self.best_crossings = self.global_crossings
        self.best_local_crossings = self.local_crossings
        self.best_sizemax = len(self.E_star)
        self.best_pos = copy.deepcopy(self.pos)
        self.spiral_repairs_count = 0

    def select_next_node(self):
        """Select a fresh node or keep the current node for repeated visits."""
        repeat_count = max(1, int(getattr(self, "node_visit_repeat_count", 1)))
        self.node_visit_repeat_count = repeat_count

        if self.current_node is not None and self._node_visit_repeat_remaining > 0:
            self._node_visit_repeat_remaining -= 1
            self.last_selection_stats = {
                "is_random_fallback": False,
                "node_degree": self.graph.degree(self.current_node),
                "candidates_count": self.last_selection_stats.get("candidates_count", 0),
                "repeat_node_visit": True,
                "node_visit_repeat_remaining": self._node_visit_repeat_remaining,
            }
            return self.current_node

        self.current_node = self._choose_next_node()
        self._reset_node_visit_repeat_state()
        self.last_selection_stats["node_degree"] = self.graph.degree(self.current_node)
        return self.current_node

    def translate_back_action(self, action):
        """Normalize action to direction index."""
        if isinstance(action, (list, tuple, np.ndarray)):
            arr = np.asarray(action)
            if arr.size == 0:
                return action
            if hasattr(self, "translate_back_action_vec"):
                try:
                    return int(self.translate_back_action_vec(arr))
                except Exception:
                    pass
            return int(arr.flat[0])

        try:
            return int(action)
        except Exception:
            return action

    def shift_and_rotate_octants(self, *arrays):
        """Rotate octant arrays to align first array's mode to bin 0."""
        if len(arrays) == 0:
            return tuple()

        neigh_counts = arrays[0]
        vx = float(np.dot(neigh_counts, self.OCTANT_COS))
        vy = float(np.dot(neigh_counts, self.OCTANT_SIN))
        angle = math.atan2(vy, vx)
        k = int(round(angle / (2 * np.pi / 8))) % 8

        rotate_idx = self.OCTANT_ROTATE_IDX[k]
        rotated = tuple(arr[rotate_idx] for arr in arrays)
        self._last_max_idx = k
        return rotated

    def obs_dir_to_world(self, dir_idx: int) -> int:
        """Map an observation-frame direction index to world-frame direction index."""
        return (int(dir_idx) + int(self._last_max_idx)) % n_directions

    def translate_back_action_vec(self, action_vec: np.ndarray) -> np.ndarray:
        """Undo observation rotation for vector actions."""
        offset = self._last_max_idx % n_directions
        return np.roll(action_vec, offset)

    def is_layout_better(self, after_global, after_local):
        """
        Check if current layout is better than best layout based on optimization goal.
        
        Returns: bool indicating if layout should be saved as new best
        """
        if self.optimization_goal == "local":
            # Optimize local first, then sizemax, then global
            if after_local < self.best_local_crossings:
                return True
            if after_local == self.best_local_crossings:
                # Check sizemax (length of E_star)
                if len(self.E_star) < self.best_sizemax:
                    return True
                if len(self.E_star) == self.best_sizemax and after_global < self.best_crossings:
                     return True
            return False
        else:  # "global" is default
            # Optimize global first, then local
            if after_global < self.best_crossings:
                return True
            if after_global == self.best_crossings and after_local < self.best_local_crossings:
                return True
            return False

    def reset_to_best_position(self):
        """Reset node positions to the best saved layout."""
        if self.best_pos is None:
            return
        
        self.pos = copy.deepcopy(self.best_pos)
        self._rebuild_spatial_indices()
        
        # Recompute crossings after reset
        self.crossings.clear()
        self.compute_crossings()
        self._rebuild_counts_from_crossings()
        
        # Reset unsuccessful moves counter
        self.unsuccessful_moves = 0

        # Optionally perform a short force-directed relaxation initialized
        # at the best layout to avoid overly clamped nodes. Only accept the
        # relaxed layout if it does not make the layout worse. Controlled by config.
        if getattr(self, 'relax_after_reset', False):
            try:
                self._relax_layout_if_beneficial(n_iter=getattr(self, 'relax_iters', 10))
            except Exception:
                # If relaxation fails for any reason, keep the original best layout
                pass

    def _relax_layout_if_beneficial(self, n_iter: int = 10):
        """Run a short force-directed relaxation (spring layout) starting from
        the current positions and accept it only if it does not worsen the
        crossings counts.

        The method temporarily replaces positions, recomputes crossings, and
        restores the original state if the relaxed layout is worse.
        """
        if self.graph is None or len(self.graph) == 0:
            return

        # Capture current state
        old_pos = {n: self.pos[n].copy() for n in self.graph.nodes()}
        old_pos_arr = self.pos_arr.copy() if self.pos_arr is not None else None
        old_crossings = copy.deepcopy(self.crossings)
        old_crossing_points = copy.deepcopy(getattr(self, 'crossing_points', {}))
        old_c_e = copy.deepcopy(getattr(self, 'c_e', {}))
        old_count_freq = copy.deepcopy(getattr(self, '_count_freq', {}))
        old_E_star = set(getattr(self, 'E_star', set()))
        old_global = int(self.global_crossings)
        old_local = int(self.local_crossings)

        # Run networkx spring_layout starting from current positions
        try:
            init_pos = {n: np.array(self.pos[n], dtype=float) for n in self.graph.nodes()}
            relaxed_pos = nx.spring_layout(self.graph, pos=init_pos, iterations=int(n_iter), seed=42)
        except Exception:
            return

        # Apply relaxed positions and recompute crossings
        try:
            if self.use_int_grid:
                for n, p in relaxed_pos.items():
                    self.pos[n] = np.round(p).astype(np.int32)
            else:
                for n, p in relaxed_pos.items():
                    self.pos[n] = np.asarray(p, dtype=np.float64)
            self._rebuild_spatial_indices()
            # Recompute crossings and counts
            self.crossings.clear()
            self.compute_crossings()
            self._rebuild_counts_from_crossings()
        except Exception:
            # Restore on failure
            self.pos = old_pos
            self._rebuild_spatial_indices()
            if old_pos_arr is not None:
                self.pos_arr = old_pos_arr
            self.crossings = old_crossings
            self.crossing_points = old_crossing_points
            self.c_e = old_c_e
            self._count_freq = old_count_freq
            self.E_star = old_E_star
            self.global_crossings = old_global
            self.local_crossings = old_local
            return

        # Accept relaxed layout only if not worse
        if self.global_crossings <= old_global and self.local_crossings <= old_local:
            # If it's strictly better according to optimization goal, record it
            if self.is_layout_better(self.global_crossings, self.local_crossings):
                self._handle_new_best(self.global_crossings, self.local_crossings)
            # else keep current as-is (relaxed layout accepted)
            return
        else:
            # Revert to old layout
            self.pos = old_pos
            self._rebuild_spatial_indices()
            if old_pos_arr is not None:
                self.pos_arr = old_pos_arr
            self.crossings = old_crossings
            self.crossing_points = old_crossing_points
            self.c_e = old_c_e
            self._count_freq = old_count_freq
            self.E_star = old_E_star
            self.global_crossings = old_global
            self.local_crossings = old_local
            return

    def check_and_handle_reset(self):
        """
        Check if reset should be triggered (only if threshold is set).
        Returns: bool indicating if reset occurred
        """
        if self.reset_unsuccessful_moves_threshold is None:
            return False
        
        if self.unsuccessful_moves >= self.reset_unsuccessful_moves_threshold:
            self.reset_to_best_position()
            return True
        
        return False

    def _handle_new_best(self, after_global: int, after_local: int):
        """Update stored best layout and record the (1-based) step index when it occurred."""
        self.best_crossings = after_global
        self.best_local_crossings = after_local
        self.best_sizemax = len(self.E_star)
        self.best_pos = {n: self.pos[n].copy() for n in self.graph.nodes()}
        self.unsuccessful_moves = 0
        step_num = getattr(self, "step_count", 0) + 1
        self.best_improvement_steps.append(step_num)
        self.last_improvement_step = step_num

    def configure_reward_weights(self, config_reward_dict):
        """
        Configure reward weights based on optimization_goal.

        Accepts either the new two-key format:
        {
            "global_optimization": { ... },
            "local_optimization": { ... }
        }
        or a legacy flat dict of weights. The selected sub-dict is assigned to
        `self.reward_weights` depending on `self.optimization_goal`.
        """
        if not config_reward_dict:
            # Keep existing defaults in self.reward_weights
            return

        if "global_optimization" in config_reward_dict and "local_optimization" in config_reward_dict:
            if self.optimization_goal == "local":
                self.reward_weights = config_reward_dict.get("local_optimization", {})
            else:
                self.reward_weights = config_reward_dict.get("global_optimization", {})
        else:
            # Legacy single dict: use as-is
            self.reward_weights = config_reward_dict

    def _precompute_pixel_grids(self):
        """Precompute rotated patch grids used by pixel-map observations."""
        self.pixel_dx = {}
        self.pixel_dy = {}
        self.pixel_dx_flat = {}
        self.pixel_dy_flat = {}
        self.pixel_query_radius = int(np.ceil(self.patch_half * np.sqrt(2))) + 1

        c_vals = np.arange(-self.patch_half, self.patch_half + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(c_vals, c_vals)
        for i in range(8):
            theta = i * (np.pi / 4.0)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            self.pixel_dx[i] = grid_x * sin_t + grid_y * cos_t
            self.pixel_dy[i] = -grid_x * cos_t + grid_y * sin_t
            self.pixel_dx_flat[i] = self.pixel_dx[i].reshape(-1)
            self.pixel_dy_flat[i] = self.pixel_dy[i].reshape(-1)

        self.pixel_dx_flat_arr = np.stack([self.pixel_dx_flat[i].astype(np.float64) for i in range(8)], axis=0)
        self.pixel_dy_flat_arr = np.stack([self.pixel_dy_flat[i].astype(np.float64) for i in range(8)], axis=0)

    def _reset_node_visit_repeat_state(self):
        """Initialize the repeat counter for the current node selection."""
        repeat_count = max(1, int(getattr(self, "node_visit_repeat_count", 1)))
        self.node_visit_repeat_count = repeat_count
        self._node_visit_repeat_remaining = repeat_count - 1

    def _critical_candidate_nodes(self):
        """Candidate pool centered on current worst crossing edges."""
        candidate_nodes = set()
        for edge in self.E_star:
            candidate_nodes.update(edge)
            for crossed_edge in self.crossings.get(edge, set()):
                candidate_nodes.update(crossed_edge)
        return candidate_nodes

    def _global_candidate_nodes(self):
        """Candidate pool for global minimization: nodes on any currently crossing edge."""
        candidate_nodes = set()
        for edge, count in self.c_e.items():
            if count <= 0:
                continue
            candidate_nodes.update(edge)
        if candidate_nodes:
            return candidate_nodes

        for edge, crossed in self.crossings.items():
            if crossed:
                candidate_nodes.update(edge)
        return candidate_nodes

    def _heuristic_candidate_scores(self, candidate_nodes, strategy: str):
        """Return a node->score mapping for heuristic node selection strategies."""
        scores = {}

        if strategy == "heuristic":
            for node in candidate_nodes:
                deg = self.graph.degree(node)
                if deg > 0:
                    scores[node] = 1.0 / float(deg)
            return scores

        if strategy == "heuristic_global":
            c_e = self.c_e
            incident_by_node = self.incident_by_node
            node_degree = self.node_degree
            node_index = self.node_index
            node_visit_counts = self.node_visit_counts
            visit_penalty_coef = float(getattr(self, "heuristic_new_visit_penalty_coef", 0.5))

            for node in candidate_nodes:
                if node_degree.get(node, self.graph.degree(node)) <= 0:
                    continue

                incident_edges = incident_by_node.get(node, ())
                if not incident_edges:
                    continue

                total_cross = 0.0
                for edge in incident_edges:
                    edge_cross = float(c_e.get(edge, 0))
                    if edge_cross <= 0.0:
                        continue
                    total_cross += edge_cross

                if total_cross <= 0.0:
                    continue

                visit_penalty = 1.0
                if (
                    node_visit_counts is not None
                    and len(node_visit_counts) == len(self.index_node)
                ):
                    visit_penalty = 1.0 + visit_penalty_coef * float(node_visit_counts[node_index[node]])

                score = total_cross / max(1.0, visit_penalty)

                if score > 0.0:
                    scores[node] = score

            return scores

        crossing_point_cache = self.crossing_points
        crossings = self.crossings
        c_e = self.c_e
        e_star = self.E_star
        incident_by_node = self.incident_by_node
        node_degree = self.node_degree
        node_index = self.node_index
        pos_arr = self.pos_arr
        node_visit_counts = self.node_visit_counts
        visit_penalty_coef = float(getattr(self, "heuristic_new_visit_penalty_coef", 0.5))

        for node in candidate_nodes:
            deg = node_degree.get(node, self.graph.degree(node))
            if deg <= 0:
                continue

            incident_edges = incident_by_node.get(node, ())
            if not incident_edges:
                continue

            total_cross = 0.0
            max_cross = 0.0
            critical_hits = 0.0
            critical_cross_hits = 0.0
            nearest_cross_dist_sq = float("inf")
            node_pos = pos_arr[node_index[node]]
            node_x = float(node_pos[0])
            node_y = float(node_pos[1])

            for edge in incident_edges:
                crossed = crossings.get(edge, ())
                edge_cross = float(c_e.get(edge, len(crossed)))
                total_cross += edge_cross
                if edge_cross > max_cross:
                    max_cross = edge_cross
                edge_is_critical = edge in e_star
                if edge_is_critical:
                    critical_hits += 1.0
                saw_critical_cross = False
                if edge_is_critical or crossed:
                    for other in crossed:
                        other_is_critical = other in e_star
                        if not edge_is_critical and not other_is_critical:
                            continue
                        saw_critical_cross = True
                        pair = (edge, other) if edge <= other else (other, edge)
                        cp = crossing_point_cache.get(pair)
                        if cp is None:
                            cp = self.get_crossing_point(edge[0], edge[1], other[0], other[1])
                            if cp is None:
                                continue
                            crossing_point_cache[pair] = cp
                        dx = float(cp[0]) - node_x
                        dy = float(cp[1]) - node_y
                        d_sq = dx * dx + dy * dy
                        if d_sq < nearest_cross_dist_sq:
                            nearest_cross_dist_sq = d_sq
                if saw_critical_cross:
                    critical_cross_hits += 1.0

            visit_penalty = 1.0
            if (
                node_visit_counts is not None
                and len(node_visit_counts) == len(self.index_node)
            ):
                visit_penalty = 1.0 + visit_penalty_coef * float(node_visit_counts[node_index[node]])

            degree_penalty = math.sqrt(float(deg))
            distance_bonus = 1.0
            if math.isfinite(nearest_cross_dist_sq):
                distance_bonus = 1.0 / (1.0 + math.sqrt(nearest_cross_dist_sq))
            score = (
                total_cross
                + 0.5 * max_cross
                + 5.0 * critical_hits
                + 2.0 * critical_cross_hits
            ) * distance_bonus / max(1.0, degree_penalty * visit_penalty)

            if score > 0.0:
                scores[node] = score

        return scores

    def get_node_shortlist(self, k: int, strategy: str | None = None):
        """Return up to k candidate nodes ranked by the configured selection strategy."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return []

        strategy = strategy or self.node_selection_strategy
        k = max(1, int(k))

        if strategy in {"heuristic", "heuristic_new", "heuristic_global"}:
            if strategy == "heuristic_global":
                candidate_nodes = self._global_candidate_nodes()
            else:
                candidate_nodes = self._critical_candidate_nodes()
            scores = self._heuristic_candidate_scores(candidate_nodes, strategy)
            if scores:
                ranked = heapq.nlargest(
                    k,
                    scores.items(),
                    key=lambda kv: (kv[1], -self.node_index[kv[0]]),
                )
                ranked.sort(key=lambda kv: (-kv[1], self.node_index[kv[0]]))
                return [(node, float(score)) for node, score in ranked[:k]]
            if candidate_nodes:
                ranked_nodes = sorted(candidate_nodes, key=lambda node: self.node_index[node])
                return [(node, 0.0) for node in ranked_nodes[:k]]

        all_nodes = sorted(self.graph.nodes(), key=lambda node: self.node_index[node])
        if strategy == "random":
            random.shuffle(all_nodes)
        return [(node, 0.0) for node in all_nodes[:k]]

    def _choose_next_node(self):
        """Select a fresh node according to the configured strategy."""
        self.last_selection_stats = {
            "is_random_fallback": False,
            "node_degree": 0,
            "repeat_node_visit": False,
            "node_visit_repeat_remaining": 0,
        }

        if self.node_selection_strategy in {"heuristic", "heuristic_new", "heuristic_global"}:
            if self.node_selection_strategy == "heuristic_global":
                candidate_nodes = self._global_candidate_nodes()
            else:
                candidate_nodes = self._critical_candidate_nodes()

            self.last_selection_stats["candidates_count"] = len(candidate_nodes)
            probabilities = self._heuristic_candidate_scores(candidate_nodes, self.node_selection_strategy)

            if probabilities:
                return random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]

            self.last_selection_stats["is_random_fallback"] = True
            return random.choice(list(self.graph.nodes()))

        candidate_nodes = set()
        for edge_tuple, crossed_set in self.crossings.items():
            if len(crossed_set) > 0:
                candidate_nodes.update(edge_tuple)

        self.last_selection_stats["candidates_count"] = len(candidate_nodes)
        if candidate_nodes:
            return random.choice(list(candidate_nodes))

        self.last_selection_stats["is_random_fallback"] = True
        return random.choice(list(self.graph.nodes()))

    def reset(self, seed=None, Graph=None, Width=None, Height=None, **kwargs):
        """Shared reset logic for all graph-layout environments."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if Graph is not None:
            w = self.width if Width is None else Width
            h = self.height if Height is None else Height
            self._init_for_graph(Graph, w, h, seed=seed)
        else:
            dtype = np.int32 if self.use_int_grid else np.float64
            self.pos = {n: np.array(self.initial_pos[n], dtype=dtype) for n in self.graph.nodes()}
            self._sync_pos_arrays()
            self._update_rtree()
            self._init_edge_index()
            self.crossings.clear()
            self.compute_crossings()
            self._rebuild_counts_from_crossings()
            self.current_node = self._choose_next_node()
            self._reset_node_visit_repeat_state()
            self.step_count = 0
            self.idle_streak = 0
            self.unsuccessful_moves = 0
            self.best_improvement_steps = []
            self.last_improvement_step = None
            self.last_crossings = self.global_crossings
            self.best_crossings = self.global_crossings
            self.best_local_crossings = self.local_crossings
            self.best_sizemax = len(self.E_star)
            self.best_pos = copy.deepcopy(self.pos)
            self.spiral_repairs_count = 0

        self.node_visit_counts = np.zeros(self.graph.number_of_nodes())

        if self.track_history:
            self.history = [{"pos": {self.node_index[k]: tuple(v) for k, v in self.pos.items()}, "node": None}]
        else:
            self.history = None

        return self.get_observation(), {}
