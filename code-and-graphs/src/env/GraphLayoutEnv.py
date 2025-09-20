import sys
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import numpy as np
import networkx as nx
from rtree import index
import math

from rewards.fr import compute_node_forces, compute_layout_energy
from util.plot_graph import plot_graph_reward
import torch
from torch_geometric.nn import DeepGraphInfomax
from util.gat_prototype import build_data_from_graph, GATEncoder
import random
from util.layout_similarity import layout_similarity_distance_correlation
from gnn.gat_prototype import create_gat_embedding
from sklearn.decomposition import PCA
from pybindCode.graph_utils import compute_crossings, count_crossings_involving_node
from pybindCode.graph_utils import find_crossings_for_edges
import copy
from util.plot_graph import plot_graph
from PIL import Image, ImageDraw

def get_best_layout_by_crossings(G):
    # always use the same sorted order
    node_order = sorted(G.nodes())
    layouts = {
        #"spring": nx.spring_layout(G, seed=42),
        "kamada_kawai": nx.kamada_kawai_layout(G),
        #"spectral": nx.spectral_layout(G),
        #"random": nx.random_layout(G, seed=42),
        #"circular": nx.circular_layout(G),
        #"shell": nx.shell_layout(G),
    }
    best_layout = None
    min_crossings = float('inf')

    for name, pos in layouts.items():
        pos_arr = np.array([pos[n] for n in node_order])
        edges_idx = [(node_order.index(u), node_order.index(v)) for u, v in G.edges()]
        _, _, crossings = compute_crossings(pos_arr, edges_idx)
        # print(f"{name} layout crossings: {crossings}")
        if crossings < min_crossings:
            min_crossings = crossings
            best_layout = {n: pos[n] for n in G.nodes()}  # still return dict
    return best_layout






# method that, given node pos, checks no nodes have the same coordinates
def check_no_overlap(graph, pos):
    """
    Check if any two nodes have the same position and return overlapping nodes.
    """
    seen = {}
    overlapping_nodes = []
    for node, coords in pos.items():
        coords_tuple = tuple(np.round(coords, 6))  # Round to avoid floating point issues
        if coords_tuple in seen:
            overlapping_nodes.append((seen[coords_tuple], node))
        else:
            seen[coords_tuple] = node
    return len(overlapping_nodes) == 0, overlapping_nodes


def initialize_positions_from_embeddings(graph, embeddings):
    """
    Initialize node positions deterministically based on GAT embeddings.

    Parameters:
        graph: networkx.Graph
        embeddings: np.ndarray - Node embeddings (shape: [n_nodes, embedding_dim])

    Returns:
        dict: A dictionary mapping node IDs to 2D positions.
    """
    # Reduce embeddings to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Normalize positions to [0, 1]
    min_vals = reduced_embeddings.min(axis=0)
    max_vals = reduced_embeddings.max(axis=0)
    normalized_positions = (reduced_embeddings - min_vals) / (max_vals - min_vals)

    # Map positions to nodes
    positions = {node: normalized_positions[i] for i, node in enumerate(graph.nodes())}
    return positions



def calculate_angle(origin, point):
    delta = point - origin
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return (angle + 360) % 360


def calculate_distance(point1, point2, epsilon=1e-6):
    return max(np.linalg.norm(point1 - point2), epsilon)


class GraphLayoutEnv(gym.Env):
    def __init__(self, graph: nx.Graph, opt_type):
        super().__init__()
        self.opt_type = opt_type

        self.action_space = spaces.Discrete(16)
        self.observation_space = Dict({
            "cross_map": Box(0.0, 1.0, shape=(8,), dtype=np.float32),
            "cross_map_local": Box(0.0, 1.0, shape=(8,), dtype=np.float32),
            "local_view": Box(0.0, 1.0, shape=(32,), dtype=np.float32),
            "gat_embedding": Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "local_crossings": Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "global_crossings": Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            # "max_crossing_edge_direction": Box(0.0, 8.0, shape=(1,), dtype=np.int32),
            #"force": Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            #"local_patch": Box(0.0, 1.0, shape=(32, 32, 3), dtype=np.float32),
            #'current_pos': Box(-1, 1, shape=(2,), dtype=np.float32),
        })

        # static constants / misc state that do NOT depend on the specific graph
        self.current_node = None
        self.crossings = {}
        self.initial_crossings = None
        self.initial_local_crossings = None
        self.last_crossings = None
        self.best_crossings = float('inf')
        self.pos = None
        self.best_pos = None
        self.initial_pos = None
        self.unsuccessful_moves = 0
        self.local_crossings = None
        self.best_local_crossings = float('inf')
        self.move_step = 0.1
        self.min_bbox_size = 1
        self._last_max_idx = 0
        self.ANGLES16 = [i * 22.5 for i in range(16)]
        self.DIRS_UNIT = tuple((math.cos(math.radians(a)), math.sin(math.radians(a))) for a in self.ANGLES16)

        self._init_for_graph(graph)


    def _init_for_graph(self, G, seed=None):
        """Fully (re)initialize the environment state for a new graph G."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.graph = G
        self.n_nodes = G.number_of_nodes()

        self.gat_embeddings = self._compress_gat_embeddings(create_gat_embedding(self.graph))
        self.initial_pos = get_best_layout_by_crossings(self.graph)
        self.pos = {n: np.array(self.initial_pos[n], dtype=np.float64) for n in self.graph.nodes()}

        self._precompute_static()  # index_node, node_index, edges, edge_idx_pair, incident_by_node
        self._sync_pos_arrays()  # builds self.pos_arr from self.pos

        self._update_rtree()  # node rtree
        self._init_edge_index()  # edge rtree

        self.crossings.clear()
        self.compute_crossings()  # fills self.crossings globally for current graph
        self._rebuild_counts_from_crossings()

        self.current_node = 0
        self.step_count = 0
        self.unsuccessful_moves = 0
        self.initial_crossings = self.global_crossings
        self.initial_local_crossings = self.local_crossings
        self.last_crossings = self.global_crossings
        self.best_crossings = self.global_crossings
        self.best_local_crossings = self.local_crossings
        self.best_pos = copy.deepcopy(self.pos)



    def _rtree_props(self):
        # set rtree properties
        p = index.Property()
        # Tunables – good defaults for 2D straight-line segments
        p.leaf_capacity = 64
        p.index_capacity = 64
        p.near_minimum_overlap_factor = 32
        p.ensure_tight_mbr = True
        # Keep in-memory index (no disk files)
        p.storage = index.RT_Memory
        return p

    def _precompute_static(self):
        """Static, once-per-graph caches."""
        # nodes
        self.index_node = list(self.graph.nodes())
        self.node_index = {v: i for i, v in enumerate(self.index_node)}
        # edges (sorted label form) and their index-pairs (i,j)
        self.edges = [tuple(sorted(e)) for e in self.graph.edges()]
        self.edge_idx_pair = {e: (self.node_index[e[0]], self.node_index[e[1]]) for e in self.edges}
        self.idxpair_to_edge = {ij: e for e, ij in self.edge_idx_pair.items()}
        # incident edges per node (as sorted tuples)
        self.incident_by_node = {
            v: [tuple(sorted((v, u))) for u in self.graph.neighbors(v)]
            for v in self.graph.nodes()
        }

    def _sync_pos_arrays(self):
        """Mirror dict -> numpy (N,2) once, then mutate rows in-place on moves."""
        N = len(self.index_node)
        self.pos_arr = np.zeros((N, 2), dtype=np.float64)
        for v, i in self.node_index.items():
            self.pos_arr[i] = self.pos[v]


    def reset(self, seed=None, Graph=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        if Graph is not None:
            self._init_for_graph(Graph, seed=seed)
        else:
            self.pos = {n: np.array(self.initial_pos[n], dtype=np.float64) for n in self.graph.nodes()}
            self._sync_pos_arrays()
            self._update_rtree()
            self._init_edge_index()
            self.crossings.clear()
            self.compute_crossings()
            self._rebuild_counts_from_crossings()
            self.current_node = 0
            self.step_count = 0
            self.unsuccessful_moves = 0
            self.last_crossings = self.global_crossings
            self.best_crossings = self.global_crossings
            self.best_local_crossings = self.local_crossings
            self.best_pos = copy.deepcopy(self.pos)

        return self.get_observation(), {}

    def step(self, action):
        node = self.current_node

        # rotate back to original orientation, reverse of observation vector rotation
        action = self.translate_back_action(action)

        dx, dy = self.DIRS_UNIT[int(action)]
        move_dir = np.array((dx, dy))

        # incident = [tuple(sorted(e)) for e in self.graph.edges(node)]
        incident = self.incident_by_node[node]

        before_global = self.global_crossings
        before_local = self.local_crossings
        before_sizemax = len(self.E_star)

        min_d = None
        nodex, nodey = self.pos[node]
        mx, my = float(dx), float(dy)
        for e in incident:
            for f in self.crossings.get(e, ()):
                u1, v1 = e
                u2, v2 = f
                pt = self.get_crossing_point(u1, v1, u2, v2)
                if pt is None:
                    continue
                px, py = pt
                d_along = (px - nodex) * mx + (py - nodey) * my
                if d_along > 1e-6 and (min_d is None or d_along < min_d):
                    min_d = d_along

        if min_d is None:
            self.pos[node] += move_dir * self.move_step

        else:
            epsilon = np.random.uniform(0.01, 0.10)
            self.pos[node] += move_dir * min_d * (1 + epsilon)

        self.pos_arr[self.node_index[node]] = self.pos[node]

        old = self.bboxes[node]
        self.rtree_index.delete(node, old)
        new_bb = self._compute_bbox(node)
        self.rtree_index.insert(node, new_bb)
        self.bboxes[node] = new_bb
        self._update_edges_for_node(node)

        # === incremental crossing maintenance ===
        removed_pairs = self.remove_crossings_for_node(node)
        added_pairs = self.recompute_crossings_for_node(node)
        self._apply_crossing_deltas(added_pairs, removed_pairs)
        # assert self._check_invariants()

        # if (self.step_count % 200) == 0 or True:
        #     old_g, old_l = self.global_crossings, self.local_crossings
        #     self._sanity_full_recompute()
        #     if (self.global_crossings != old_g) or (self.local_crossings != old_l):
        #         print(f"[sanity] drift detected: glob {old_g}->{self.global_crossings} "
        #               f"loc {old_l}->{self.local_crossings}")
        #         assert False

        after_global = self.global_crossings
        after_local = self.local_crossings
        after_sizemax = len(self.E_star)

        global_delta = before_global - after_global  # whole-graph change (exact)
        local_delta = before_local - after_local
        sizemax_delta = after_sizemax - before_sizemax if local_delta == 0 else 0

        # Reward weights
        if self.opt_type == "Local":
            local_weight = 10.0
            sizemax_weight = 0.1
            global_weight = 1.0/(len(self.edges))
            reward = local_weight * local_delta + sizemax_weight * sizemax_delta + global_weight * global_delta
            if reward == 0.0:
                reward = -0.001 #account for small global crossing weight
        else:
            assert self.opt_type == "Global"
            reward = global_delta
            if reward == 0.0:
                reward = -0.001

        # if reward > 0:
        #     print("Reward:", reward, "from", local_delta, sizemax_delta, global_delta)

        self.last_crossings = after_global

        # Update best_pos and best_local_crossings based on local crossings
        if (self.opt_type == "Local" and (after_local < self.best_local_crossings or (after_local == self.best_local_crossings and after_global < self.best_crossings)))\
                or (self.opt_type == "Global" and after_global < self.best_crossings):
            self.best_crossings = after_global
            self.best_local_crossings = after_local
            self.best_pos = {n: self.pos[n].copy() for n in self.graph.nodes()}

            self.unsuccessful_moves = 0  # Reset counter on success
        else:
            self.unsuccessful_moves += 1

        if self.unsuccessful_moves >= 400:
            self.unsuccessful_moves = 0

            self.pos = copy.deepcopy(self.best_pos)
            self._sync_pos_arrays()
            # recompute local and global crossings
            self._update_rtree()
            self._init_edge_index()
            self.crossings.clear()
            _, _, _ = self.compute_crossings()
            self._rebuild_counts_from_crossings()

            self.last_crossings = self.global_crossings


        if self.opt_type == "Local":
            self.select_next_node()
        else:
            assert self.opt_type == "Global"
            self.select_next_node_global_crossings_number()
        self.step_count += 1

        obs = self.get_observation()
        done = (self.best_local_crossings == 1)
        truncated = (self.step_count >= 2000)

        info = {
            "global_crossings": self.global_crossings,
            "local_crossings": self.local_crossings,
            "best_global_crossings": self.best_crossings,
            "best_local_crossings": self.best_local_crossings,
        }
        return obs, reward, done, truncated, info

    def select_next_node(self):
        """
        Select the next node based on the edges with the highest crossing count.
        Candidate nodes are those connected to the edges with the highest crossings,
        as well as nodes connected to any edge that crosses these edges.
        Nodes with lower degrees have a higher probability of being selected.
        """

        # Collect all candidate nodes connected to these edges with max local crossing
        candidate_nodes = set()
        for edge in self.E_star:
            candidate_nodes.update(edge)
            # Also include nodes from edges that cross this edge
            for crossed_edge in self.crossings.get(edge, set()):
                candidate_nodes.update(crossed_edge)

        # Assign probabilities inversely proportional to node degree
        probabilities = {node: (1.0 / self.graph.degree(node)) for node in candidate_nodes if self.graph.degree(node) > 0}

        # Select a node based on the computed probabilities
        if probabilities:
            self.current_node = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
        else:
            # Fallback to a random node if no candidates exist
            self.current_node = random.choice(list(self.graph.nodes()))

        return self.current_node

    # Method added by JZ
    def select_next_node_global_crossings_number(self):
        nodes = list(self.graph.nodes())
        ease_scores = []

        for v in nodes:
            deg = self.graph.degree(v)
            if deg == 0:
                ease_scores.append(0)
            else:
                inc = [tuple(sorted(e)) for e in self.graph.edges(v)]
                crossings = sum(len(self.crossings.get(e, ())) for e in inc)
                ease_scores.append(crossings / deg)

        if sum(ease_scores) == 0:
            self.current_node = random.choice(nodes)
        else:
            self.current_node = random.choices(nodes, weights=ease_scores, k=1)[0]

        return self.current_node

    @staticmethod
    def _line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
        # Returns (px, py) or None if parallel/collinear (within tol)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return None
        a = (x1 * y2 - y1 * x2)
        b = (x3 * y4 - y3 * x4)
        px = (a * (x3 - x4) - (x1 - x2) * b) / denom
        py = (a * (y3 - y4) - (y1 - y2) * b) / denom
        return px, py

    def get_crossing_point(self, u1, v1, u2, v2):
        x1, y1 = self.pos[u1]
        x2, y2 = self.pos[v1]
        x3, y3 = self.pos[u2]
        x4, y4 = self.pos[v2]
        return self._line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4)

    def _compute_gat_embeddings(self):
        data = build_data_from_graph(self.graph)
        encoder = GATEncoder(in_feats=8, hidden_feats=16, out_feats=32)
        model = DeepGraphInfomax(
            hidden_channels=32,
            encoder=encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)
        )
        model.eval()
        with torch.no_grad():
            embeddings = model.encoder(data.x, data.edge_index)
        return embeddings

    def get_observation(self):
        """
        Returns the observation for the current node, including local view,
        GAT embedding, force, and the rendered local patch.
        Now also includes the direction of the local crossing max edge.
        """
        node = self.current_node
        base_obs, cross_oct, cross_oct_local = self._get_observation_Octant()
        gat_feat = self.gat_embeddings[node].astype(np.float32)

        # Add local and global crossing numbers to observation
        local_crossings_obs = np.array([float(self.local_crossings)], dtype=np.float32)
        global_crossings_obs = np.array([float(self.global_crossings)], dtype=np.float32)

        #todo: normalize everything
        return {
            "cross_map": cross_oct,
            "cross_map_local": cross_oct_local,
            "local_view": base_obs,
            "gat_embedding": gat_feat,
            "local_crossings": local_crossings_obs,
            "global_crossings": global_crossings_obs,
        }

    def _get_observation_Octant(self):
        node = self.current_node
        i = self.node_index[node]
        node_pos = self.pos_arr[i]  # (2,)
        P = self.pos_arr  # (N,2)
        V = P - node_pos  # vectors to all
        V[i, :] = 0.0  # ignore self
        ang = (np.degrees(np.arctan2(V[:, 1], V[:, 0])) + 360.0) % 360.0
        bins = (ang // 45).astype(np.int64) % 8
        d = np.linalg.norm(V, axis=1)
        d[i] = np.inf

        # neighbors mask
        neigh = np.zeros(len(self.index_node), dtype=bool)
        for u in self.graph.neighbors(node):
            neigh[self.node_index[u]] = True

        counts = np.bincount(bins, minlength=8).astype(np.float32)
        neigh_d = np.full(8, np.inf, dtype=np.float32)
        nonneigh_d = np.full(8, np.inf, dtype=np.float32)

        for b in range(8):
            mask = (bins == b)
            if not mask.any(): continue
            db = d[mask]
            nb = neigh[mask]
            if nb.any():     neigh_d[b] = db[nb].min()
            if (~nb).any():  nonneigh_d[b] = db[~nb].min()

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
        for (u1, v1) in self.incident_by_node[node]:
            # Determine the neighbor endpoint on this incident edge
            # (incident_by_node stores sorted tuples; one of them must be 'node')
            other = v1 if u1 == node else (u1 if v1 == node else None)
            if other is None:
                continue  # safety, should not happen

            # Octant from node -> neighbor direction
            dx_e = self.pos[other][0] - nx_
            dy_e = self.pos[other][1] - ny_
            if dx_e == 0.0 and dy_e == 0.0:
                continue
            angle = (math.degrees(math.atan2(dy_e, dx_e)) + 360.0) % 360.0
            bin_idx = int(angle // 45.0) % 8

            # Crossing count for this incident edge
            crossed_set = self.crossings.get((u1, v1), ())
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
        #shift and rotate everything according to first data, i.e., crossing octant?
        cross_oct, cross_oct_local, rel_counts, neigh_d, nonneigh_d, rel_abs_counts = \
            self.shift_and_rotate_octants(cross_oct, cross_oct_local, rel_counts, neigh_d, nonneigh_d, rel_abs_counts)

        obs = np.round(np.concatenate((rel_counts, neigh_d, nonneigh_d, rel_abs_counts)), 3).astype(np.float32)
        cross_oct = cross_oct.astype(np.float32)
        cross_oct_local = cross_oct_local.astype(np.float32)
        return obs, cross_oct, cross_oct_local


    def _compute_bbox(self, node):
        x, y = self.pos[node]
        neigh = list(self.graph.neighbors(node))
        longest = max(calculate_distance(self.pos[node], self.pos[u]) for u in neigh) if neigh else self.min_bbox_size
        half = max(longest, self.min_bbox_size)
        return (x-half, y-half, x+half, y+half)

    def _update_rtree(self):
        self.rtree_index = index.Index()
        self.bboxes = {}
        for node in self.graph.nodes():
            bbox = self._compute_bbox(node)
            self.rtree_index.insert(node, bbox)
            self.bboxes[node] = bbox


    def render(self, reward, mode="human"):
        plot_graph_reward(self.graph, self.pos, reward)

    def query_neighbors_within_radius(self, node):
        center = self.pos[node]
        neigh = list(self.graph.neighbors(node))
        longest = max(calculate_distance(center, self.pos[u]) for u in neigh) if neigh else self.min_bbox_size
        radius = max(longest, self.min_bbox_size)
        bbox = (center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius)
        candidates = self.rtree_index.intersection(bbox)
        return [n for n in candidates if n != node and calculate_distance(center, self.pos[n]) <= radius]

    def shift_and_rotate_octants(self, *arrays):
        """
        Rotate ALL passed 8-bin octant arrays so that the weighted direction of the
        FIRST array (assumed to be a 'counts' distribution) lands at index 0.

        Returns the rotated arrays in the same order.
        """
        if len(arrays) == 0:
            return tuple()

        neigh_counts = arrays[0]
        # Compute bin center angles for 8 octants
        thetas = np.arange(8) * (2 * np.pi / 8)
        # Weighted direction from the first array
        vx = float(np.sum(neigh_counts * np.cos(thetas)))
        vy = float(np.sum(neigh_counts * np.sin(thetas)))
        angle = math.atan2(vy, vx)  # [-pi, pi]
        k = int(round(angle / (2 * np.pi / 8))) % 8  # nearest octant

        rotated = tuple(np.roll(arr, -k) for arr in arrays)
        self._last_max_idx = k  # keep for action translation if you use it
        return rotated

    def translate_back_action(self, action):
        """
        Map the action index in the rotated frame back to the original orientation.
        Observation vector is rotated in steps of 45 degrees, directions are steps of 22.5
        """
        return (action + 2 * self._last_max_idx) % 16

    def check_no_overlap(self):
        """
        Check if any two nodes have the same position and return overlapping nodes.
        """
        overlapping_nodes = []
        for node in self.graph.nodes():
            nearby_nodes = self.query_neighbors_within_radius(node)
            for candidate in nearby_nodes:
                if np.allclose(self.pos[node], self.pos[candidate], atol=1e-6):
                    overlapping_nodes.append((node, candidate))
        return len(overlapping_nodes) == 0, overlapping_nodes

    def compute_crossings(self, edges=None):
        """
        Compute crossings using a consistent node order:
        - positions is self.pos_arr in self.index_node order
        - edges are passed as index pairs into positions
        Uses a single R-tree query (IDs only) to expand by bbox overlaps.
        """
        # Normalize and validate target edges (label/tuple form)
        if edges is None:
            # Full compute: just use all edges you have
            relevant_edges = list(self.edges)
        else:
            to_update = [tuple(sorted(e)) for e in edges]
            if not to_update:
                return {}, None, 0

            # Single query over the union bbox of the to_update set (IDs only)
            cand_ids = self.query_edges_overlaps_any_ids(to_update)
            overlap_edges = [self.id_edge[eid] for eid in cand_ids]

            # Final relevant set: targets + overlaps (dedup)
            relevant_edges = list({*to_update, *overlap_edges})

        # Build positions once (already mirrored)
        positions = self.pos_arr  # shape (N,2), synced in _sync_pos_arrays

        # Convert relevant label-edges -> index pairs via cached map
        # (edge_idx_pair was precomputed as { (u,v_sorted): (i,j) })
        rel_edges_idx = [self.edge_idx_pair[e] for e in relevant_edges]

        # Call the C++ routine; it returns index-based dict + max-edge index + global total
        crossings_map_idx, max_edge_idx, global_crossings = compute_crossings(positions, rel_edges_idx)

        # Fast map back from index pairs -> label-edges for just the relevant set
        idx_to_edge = {self.edge_idx_pair[e]: e for e in relevant_edges}

        # Rebuild the crossings dict (only for relevant edges)
        crossings = {}
        for (i, j), crossers in crossings_map_idx.items():
            e = idx_to_edge[(i, j)]
            S = crossings.setdefault(e, set())
            for (u, v) in crossers:
                f = idx_to_edge[(u, v)]
                S.add(f)

        # Replace the stored crossing map (keep only what we recomputed)
        # If you want a global map, merge instead of overwrite:
        self.crossings = crossings

        # Map back the max-crossing edge (may be None)
        max_crossing_edge = idx_to_edge[max_edge_idx] if max_edge_idx is not None else None

        return self.crossings, max_crossing_edge, global_crossings

    def render_local_patch(self, u, patch_size=32, world_fov=0.75):
        """
        Returns a (patch_size × patch_size × 3) uint8 array centered on node u.
        - world_fov: total width/height in world‐units spanned by the patch.
        """
        cx, cy = self.pos[u]  # node center in world coords

        # prepare a blank RGB image
        img = Image.new("RGB", (patch_size, patch_size), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # world‐to‐pixel scaling
        scale = patch_size / world_fov  # px per world‐unit

        def wp2p(wx, wy):
            # translate so (cx,cy) maps to (patch_size/2, patch_size/2)
            px = (wx - (cx - world_fov / 2)) * scale
            py = (wy - (cy - world_fov / 2)) * scale
            return px, patch_size - py  # flip y if needed

        # draw edges
        for (u1, u2) in self.graph.edges():
            x1, y1 = self.pos[u1]
            x2, y2 = self.pos[u2]
            p1 = wp2p(x1, y1)
            p2 = wp2p(x2, y2)
            draw.line([p1, p2], fill=(200, 200, 200), width=1)

        # draw all other nodes as blue circles (except u)
        for node, pos in self.pos.items():
            if node == u:
                continue
            px, py = wp2p(*pos)
            draw.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill=(0, 0, 255))

        # only consider edges incident to node u
        inc_edges = [tuple(sorted(e)) for e in self.graph.edges(u)]

        # draw crossings *only* between u’s edges and the edges they cross
        for e in inc_edges:
            crossed = self.crossings.get(e, set())
            for f in crossed:
                # compute intersection point of e and f
                cp = self.get_crossing_point(e[0], e[1], f[0], f[1])
                if cp is not None:
                    px, py = wp2p(*cp)
                    # small red dot at that location
                    draw.ellipse([(px - 1, py - 1), (px + 1, py + 1)], fill=(255, 0, 0))

        # draw the node itself in green
        px, py = wp2p(cx, cy)
        draw.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill=(0, 255, 0))

        return np.array(img, dtype=np.uint8)


    def _compress_gat_embeddings(self, embeddings):
        from sklearn.decomposition import PCA
        """
        Compresses the GAT embeddings from 32D to 4D using PCA.

        Parameters:
            embeddings (torch.Tensor): The GAT embeddings to compress.

        Returns:
            np.ndarray: Compressed embeddings with shape (n_nodes, 4).
        """
        pca = PCA(n_components=4)
        # Ensure tensor is detached, moved to CPU, and converted to NumPy
        if embeddings.is_cuda:
            embeddings = embeddings.detach().cpu()
        embeddings_np = embeddings.numpy()

        # Debugging: Check shape and type
        assert len(embeddings_np.shape) == 2, f"Expected 2D array, got {embeddings_np.shape}"
        assert embeddings_np.dtype == np.float32 or embeddings_np.dtype == np.float64, \
            f"Expected float32 or float64, got {embeddings_np.dtype}"

        compressed_embeddings = pca.fit_transform(embeddings_np)
        return compressed_embeddings

    def create_edge_rtree_entry(self, edge, position):
        """
        Create an R-tree entry for a given edge.

        Parameters:
            edge (tuple): A tuple (node1, node2) representing the edge.
            position (dict): A dictionary mapping node IDs to their (x, y) positions.

        Returns:
            tuple: (edge_id, bbox), where bbox is (min_x, min_y, max_x, max_y).
        """
        u, v = edge
        # pos can be dict or array; support both
        pu = position[u] if isinstance(position, dict) else position[self.node_index[u]]
        pv = position[v] if isinstance(position, dict) else position[self.node_index[v]]
        coords = np.vstack((pu, pv))
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        return edge, (float(xmin), float(ymin), float(xmax), float(ymax))

    def _edge_entries_for_bulkload(self):
        for i, e in enumerate(self.edges):
            _, bbox = self.create_edge_rtree_entry(e, self.pos)
            yield i, bbox, e

    def _init_edge_index(self):
        # Stable ids for edges
        self.edge_id = {e: i for i, e in enumerate(self.edges)}
        self.id_edge = {i: e for e, i in self.edge_id.items()}

        # Build bboxes cache
        self.edge_bboxes = {}
        for e in self.edges:
            _, bb = self.create_edge_rtree_entry(e, self.pos)
            self.edge_bboxes[e] = bb

        # Bulk-load R-tree in memory
        props = self._rtree_props()
        self.edge_rtree_index = index.Index(self._edge_entries_for_bulkload(), properties=props)

    def _update_edges_for_node(self, node):
        """Update only edges incident to `node` in the R-tree."""
        # Iterate incident edges as *sorted* tuples to match your canonical edge keys
        for e in (tuple(sorted(x)) for x in self.graph.edges(node)):
            eid = self.edge_id[e]
            old_bb = self.edge_bboxes[e]
            # Delete old bbox
            self.edge_rtree_index.delete(eid, old_bb)
            # Compute and cache new bbox
            _, new_bb = self.create_edge_rtree_entry(e, self.pos)
            self.edge_bboxes[e] = new_bb
            # Reinsert with same id
            self.edge_rtree_index.insert(eid, new_bb, obj=e)

    def _union_bbox(self, edges):
        """Tight bbox covering all given edges (list/iter of sorted tuples)."""
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
        """
        One rtree query for the union bbox; returns edge IDs (ints).
        Much cheaper than objects=True and avoids per-edge queries.
        """
        ub = self._union_bbox(edges)
        # objects=False returns ids directly, avoids _get_bounds/_get_data per hit
        return list(self.edge_rtree_index.intersection(ub))

    def _rebuild_counts_from_crossings(self):
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

    def _apply_crossing_deltas(self, added_pairs, removed_pairs):
        # Symmetric update
        for e, f in removed_pairs:
            if e in self.crossings: self.crossings[e].discard(f)
            if f in self.crossings: self.crossings[f].discard(e)
        for e, f in added_pairs:
            self.crossings.setdefault(e, set()).add(f)
            self.crossings.setdefault(f, set()).add(e)

        # Recompute per-edge counts ONLY for touched edges
        touched = {edge for pair in (removed_pairs + added_pairs) for edge in pair}
        for e in touched:
            self.c_e[e] = len(self.crossings.get(e, ()))

        # Global crossings: exact (no +/- deltas)
        self.global_crossings = sum(len(S) for S in self.crossings.values()) // 2

        # Local max and E*
        if self.crossings:
            m = max((len(S) for S in self.crossings.values()), default=0)
            self.local_crossings = m
            self.E_star = {e for e, S in self.crossings.items() if len(S) == m}
        else:
            self.local_crossings = 0
            self.E_star = set()

    def remove_crossings_for_node(self, node):
        """
        Remove all crossing entries involving any edge incident to `node`.
        Returns: list of removed (e, f) pairs (unique, e<f).
        """
        E_incident = {tuple(sorted(e)) for e in self.graph.edges(node)}
        removed = set()  # ensure uniqueness

        # For each incident edge e, remove e from every f that currently crosses it,
        # then drop e from self.crossings.
        for e in E_incident:
            S = self.crossings.get(e)
            if not S:
                self.crossings.pop(e, None)
                continue
            for f in list(S):
                # remove symmetric link
                Fs = self.crossings.get(f)
                if Fs is not None:
                    Fs.discard(e)
                # store the pair once, ordered
                a, b = (e, f) if e < f else (f, e)
                removed.add((a, b))
            # finally delete e
            self.crossings.pop(e, None)

        return list(removed)

    @staticmethod
    def _ranges_overlap(a1, a2, b1, b2):
        if a1 > a2: a1, a2 = a2, a1
        if b1 > b2: b1, b2 = b2, b1
        return not (a2 < b1 or b2 < a1)

    # def recompute_crossings_for_node(self, node):
    #     incident_edges = self.incident_by_node[node]
    #     if not incident_edges:
    #         return []
    #
    #     # One R-tree query for union bbox of incident edges
    #     cand_ids = self.query_edges_overlaps_any_ids(incident_edges)
    #
    #     # Map IDs back to edges, skip incident & any edge sharing an endpoint with **any** incident edge
    #     incident_endpoints = set().union(*incident_edges)  # {node, all neighbors}
    #     cand_edges = []
    #     for eid in cand_ids:
    #         f = self.id_edge[eid]
    #         if f in incident_edges:
    #             continue
    #         if not set(f).isdisjoint(incident_endpoints):  # shares any endpoint -> cannot cross
    #             continue
    #         cand_edges.append(f)
    #     if not cand_edges:
    #         return []
    #
    #     # Optional cheap AABB overlap prefilter per-axis (keeps correctness)
    #     filtered = []
    #     for f in cand_edges:
    #         fu, fv = f
    #         fx1, fy1 = self.pos[fu]
    #         fx2, fy2 = self.pos[fv]
    #         keep = False
    #         for e in incident_edges:
    #             eu, ev = e
    #             ex1, ey1 = self.pos[eu]
    #             ex2, ey2 = self.pos[ev]
    #             if (min(ex1, ex2) <= max(fx1, fx2) and min(fx1, fx2) <= max(ex1, ex2) and
    #                     min(ey1, ey2) <= max(fy1, fy2) and min(fy1, fy2) <= max(ey1, ey2)):
    #                 keep = True
    #                 break
    #         if keep:
    #             filtered.append(f)
    #
    #     incident_idx = [self.edge_idx_pair[e] for e in incident_edges]
    #     candidate_idx = [self.edge_idx_pair[e] for e in filtered]
    #
    #     crossings_map = find_crossings_for_edges(self.pos_arr, incident_idx, candidate_idx)
    #
    #     # Map back and form unique pairs
    #     idx_to_edge = {self.edge_idx_pair[e]: e for e in incident_edges}
    #     idx_to_edge.update({self.edge_idx_pair[e]: e for e in filtered})  # use filtered
    #     added = set()
    #     for e_idx, crossed_set in crossings_map.items():
    #         e = idx_to_edge[tuple(e_idx)]
    #         for f_idx in crossed_set:
    #             f = idx_to_edge[tuple(f_idx)]
    #             a, b = (e, f) if e < f else (f, e)
    #             added.add((a, b))
    #     return list(added)

    def recompute_crossings_for_node(self, node):
        inc = self.incident_by_node[node]
        if not inc:
            return []

        # 1) R-tree candidates: union of per-edge bbox queries
        cand_ids = set()
        for e in inc:
            bb = self.edge_bboxes[e]  # bbox updated in _update_edges_for_node(node)
            cand_ids.update(self.edge_rtree_index.intersection(bb))

        # 2) Map to edges, keep everything except the incident edges themselves
        cand_edges = []
        for eid in cand_ids:
            f = self.id_edge[eid]
            if f in inc:
                continue
            cand_edges.append(f)

        # 3) Build index lists (no global “shares endpoint with ANY incident edge” filter!)
        incident_idx = [self.edge_idx_pair[e] for e in inc]
        candidate_idx = [self.edge_idx_pair[e] for e in cand_edges]

        # 4) Batch predicate in C++, then filter adjacency PER (e,f) pair
        crossings_map = find_crossings_for_edges(self.pos_arr, incident_idx, candidate_idx)

        added = set()
        # Fast reverse map
        idx_to_edge = {}
        for e in inc:         idx_to_edge[self.edge_idx_pair[e]] = e
        for f in cand_edges:  idx_to_edge[self.edge_idx_pair[f]] = f

        for e_idx, crossed_set in crossings_map.items():
            e = idx_to_edge[tuple(e_idx)]
            eu, ev = e
            for f_idx in crossed_set:
                f = idx_to_edge[tuple(f_idx)]
                fu, fv = f
                # Skip adjacency **per pair** (cannot be a proper crossing)
                if eu in f or ev in f:
                    continue
                a, b = (e, f) if e < f else (f, e)
                added.add((a, b))

        return list(added)

    def _repair_and_recount(self):
        # enforce symmetry and no self-loops
        for e, S in list(self.crossings.items()):
            if e in S: S.discard(e)
            for f in list(S):
                self.crossings.setdefault(f, set()).add(e)

        self.c_e = {e: len(S) for e, S in self.crossings.items()}
        self.global_crossings = sum(self.c_e.values()) // 2
        self.local_crossings = max(self.c_e.values(), default=0)
        self.E_star = {e for e, c in self.c_e.items() if c == self.local_crossings}

    def _check_invariants(self):
        C = sum(len(S) for S in self.crossings.values()) // 2
        L = max((len(S) for S in self.crossings.values()), default=0)
        if L > C or C != self.global_crossings or L != self.local_crossings:
            print(f"[repair] L={L} C={C} (stored glob={self.global_crossings}) -> repairing")
            self._repair_and_recount()
            return False
        return True

    def _sanity_full_recompute(self):
        # full recompute using C++ kernel
        self.compute_crossings(edges=None)
        self._rebuild_counts_from_crossings()