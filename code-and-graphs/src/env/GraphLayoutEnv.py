import sys
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import numpy as np
import networkx as nx
from rtree import index

from util.plot_graph import plot_graph_reward
import torch
from torch_geometric.nn import DeepGraphInfomax
from util.gat_prototype import build_data_from_graph, GATEncoder
import random
from gnn.gat_prototype import create_gat_embedding
from sklearn.decomposition import PCA
from pybindCode.graph_utils import compute_crossings
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
        print(f"{name} layout crossings: {crossings}")
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
    def __init__(self, graph: nx.Graph, move_step=0.1, threshold_margin=0.02, energy_margin=0.02, eps_proc=0.1, gat_epochs=500, gat_lr=1e-3):
        super().__init__()
        self.step_count = 0
        self.graph = graph
        self.gat_embeddings = self._compress_gat_embeddings(create_gat_embedding(self.graph))
        self.n_nodes = graph.number_of_nodes()
        self.initial_pos = initialize_positions_from_embeddings(self.graph, self.gat_embeddings) #get_best_layout_by_crossings(self.graph)
        self.pos = copy.deepcopy(self.initial_pos)
        self.ideal_pos = None #get_best_layout_by_crossings(self.graph)
        self.move_step = move_step
        self.current_node = 0
        self.min_bbox_size = 1


        # Edge crossing reference
        self.edges = [tuple(sorted(e)) for e in graph.edges()]

        # Observation dim: original 56 + 2 local net_force dims
        self.action_space = spaces.Discrete(8)
        self.observation_space = Dict({
            "cross_map": Box(0.0, 1.0, shape=(3, 3, 1), dtype=np.float32),#TODO change here as well
            "local_view": Box(0.0, 1.0, shape=(32,), dtype=np.float32),
            "gat_embedding": Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "max_crossing_edge_direction": Box(0.0, 8.0, shape=(1,), dtype=np.int32),
            #"force": Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            #"local_patch": Box(0.0, 1.0, shape=(32, 32, 3), dtype=np.float32),
            #'current_pos': Box(-1, 1, shape=(2,), dtype=np.float32),
        })

        self._update_rtree()
        self._init_edge_index()
#        self._update_edge_rtree()


        # For symmetry normalization
        self._last_max_idx = 0
        self._last_next_idx = 0

        # in GraphLayoutEnv.__init__:
        self.crossings = {}  # maps edge → set of edges it crosses
        self.affected_edges = set(self.graph.edges())
        self.cached_intersections = {}
        self.last_crossings = None
        self.best_crossings = float('inf')
        self.best_pos = copy.deepcopy(self.initial_pos)

        self.unsuccessful_moves = 0

        # Track local crossings (max crossings on any edge)
        self.local_crossings = None
        self.best_local_crossings = float('inf')

    def reset(self, seed=None, Graph=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        if Graph is not None:
            self.graph = Graph   # TODO: not implemented yet
        self.pos = copy.deepcopy(self.initial_pos)
        self.step_count = 0
        self._update_rtree()
        self._init_edge_index()
#        self._update_edge_rtree()
        self.current_node = 0
        self.crossings.clear()
        self.affected_edges = set(self.graph.edges())
        self.cached_intersections.clear()

        # ** FULL initial crossing computation **
        # this populates self.crossings for every edge pair in the graph
        _, max_crossing_edge, global_crossings = self.compute_crossings()

        # Track local crossings (max crossings on any edge)
        local_crossings = 0
        if max_crossing_edge is not None:
            local_crossings = len(self.crossings.get(max_crossing_edge, []))
        self.local_crossings = local_crossings
        self.best_local_crossings = local_crossings
        self.last_crossings = global_crossings
        self.best_crossings = global_crossings
        self.best_pos = copy.deepcopy(self.initial_pos)
        self.unsuccessful_moves = 0

        return self.get_observation(), {}

    def step(self, action):
        node = self.current_node
        #TODO: account for rotation of observation!
        dirs = [
            np.array((1, 0)), np.array((1, 1)), np.array((0, 1)), np.array((-1, 1)),
            np.array((-1, 0)), np.array((-1, -1)), np.array((0, -1)), np.array((1, -1)),
        ]
        move_dir = dirs[int(action)] / np.linalg.norm(dirs[int(action)])

        incident = [tuple(sorted(e)) for e in self.graph.edges(node)]
        before_node_crossings = sum(len(self.crossings.get(e, [])) for e in incident)
        before_global = sum(len(v) for v in self.crossings.values()) // 2

        # Track local crossings before move (max crossings on any edge)
        before_local = 0
        if self.crossings:
            before_local = max(len(v) for v in self.crossings.values())

        min_d = None
        for e in incident:
            for f in self.crossings.get(e, []):
                u1, v1 = e
                u2, v2 = f
                pt = self.get_crossing_point(u1, v1, u2, v2)
                if pt is None:
                    continue
                vec = pt - self.pos[node]
                d_along = np.dot(vec, move_dir)
                if d_along > 1e-6:
                    if min_d is None or d_along < min_d:
                        min_d = d_along


        after_global = before_global  # Default value in case no move is made
        after_local = before_local    # Default value in case no move is made
        reward = 0.0
        if min_d is None:
            self.pos[node] += move_dir * self.move_step

        else:
            epsilon = np.random.uniform(0.01, 0.10)
            self.pos[node] += move_dir * min_d * (1 + epsilon)

        old = self.bboxes[node]
        self.rtree_index.delete(node, old)
        new_bb = self._compute_bbox(node)
        self.rtree_index.insert(node, new_bb)
        self.bboxes[node] = new_bb

        self._update_edges_for_node(node)
#        self._update_edge_rtree()
        self.remove_crossings_for_node(node)
        self.recompute_crossings_for_node(node)

        after_node_crossings = sum(len(self.crossings.get(e, [])) for e in incident)
        after_global = sum(len(v) for v in self.crossings.values()) // 2
        after_local = 0
        if self.crossings:
            after_local = max(len(v) for v in self.crossings.values())

        # Reward weights
        local_weight = 1.0
        # global_weight = 0.2
        global_weight = 1/(len(self.edges))
        local_delta = before_node_crossings - after_node_crossings
        global_delta = before_global - after_global
        if global_delta != 0 or local_delta != 0:
            reward = local_weight * local_delta + global_weight * global_delta
        else:
            reward = -0.001 #account for small global crossing weight


        self.last_crossings = after_global

        # if reward > 0:
        #     self.unsuccessful_moves = 0  # Reset counter on success
        # else:
        #     self.unsuccessful_moves += 1

        # Track local crossings (max crossings on any edge)
        self.local_crossings = after_local

        # Update best_pos and best_local_crossings based on local crossings
        if after_local < self.best_local_crossings:
            self.best_local_crossings = after_local
            self.best_pos = {n: self.pos[n].copy() for n in self.graph.nodes()}

            self.unsuccessful_moves = 0  # Reset counter on success
        else:
            self.unsuccessful_moves += 1

        # (Keeping global crossings logic for reference)
        # if after_global < self.best_crossings:
        #     self.best_crossings = after_global
        #     self.best_pos = {n: self.pos[n].copy() for n in self.graph.nodes()}

        if self.unsuccessful_moves >= 400:
            self.pos = copy.deepcopy(self.best_pos)
            self.unsuccessful_moves = 0


        # if self.step_count % 5 == 0:
        self.select_next_node() #todo: think about this
        self.step_count += 1

        obs = self.get_observation()
        done = (self.best_local_crossings == 1)
        truncated = (self.step_count >= 2000)

        return obs, reward, done, truncated, {}

    def select_next_node(self):
        """
        Select the next node based on the edges with the highest crossing count.
        Candidate nodes are those connected to the edges with the highest crossings,
        as well as nodes connected to any edge that crosses these edges.
        Nodes with lower degrees have a higher probability of being selected.
        """
        # Find the edges with the highest crossing count
        # max_crossings = max((len(self.crossings.get(e, ())) for e in self.edges), default=0) #TODO: make sure local_crossings has what we want here
        highest_crossing_edges = [e for e in self.edges if len(self.crossings.get(e, ())) == self.local_crossings]

        # Collect all candidate nodes connected to these edges
        candidate_nodes = set()
        for edge in highest_crossing_edges:
            candidate_nodes.update(edge)
            # Also include nodes from edges that cross this edge
            for crossed_edge in self.crossings.get(edge, set()):
                candidate_nodes.update(crossed_edge)

        # Assign probabilities inversely proportional to node degree
        # degrees = {node: self.graph.degree(node) for node in candidate_nodes}
        # total_weight = sum(1.0 / degrees[node] for node in candidate_nodes if degrees[node] > 0)
        # probabilities = {node: (1.0 / degrees[node]) / total_weight for node in candidate_nodes if degrees[node] > 0}
        probabilities = {node: (1.0 / self.graph.degree(node)) for node in candidate_nodes if self.graph.degree(node) > 0}

        # Select a node based on the computed probabilities
        if probabilities:
            self.current_node = random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]
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

    def get_crossing_point(self, u1, v1, u2, v2):
        x1, y1 = self.pos[u1]  # Use self.pos instead of self.positions
        x2, y2 = self.pos[v1]
        x3, y3 = self.pos[u2]
        x4, y4 = self.pos[v2]

        # Denominator for the intersection formulas
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom < 1e-12:
            # Lines are parallel or collinear—no unique crossing
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return np.array([px, py])

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
        base_obs = self._get_observation_Octant()
        gat_feat = self.gat_embeddings[node].astype(np.float32)
        #_, net_force = compute_node_forces(self.graph, self.pos, node)

        # --- New: Compute direction of the local crossing max edge ---
        # Find the edge incident to the current node with the most crossings
        incident = [tuple(sorted(e)) for e in self.graph.edges(node)]
        max_crossings = -1
        max_edge = None
        for e in incident:
            num_cross = len(self.crossings.get(e, []))
            if num_cross > max_crossings:
                max_crossings = num_cross
                max_edge = e

        # Default direction is 0 if no crossings
        max_crossing_edge_direction = 0.0
        if max_edge is not None and max_crossings > 0:
            # Find the closest crossing point for this edge
            min_dist = float('inf')
            closest_angle = 0.0
            for f in self.crossings.get(max_edge, []):
                u1, v1 = max_edge
                u2, v2 = f
                pt = self.get_crossing_point(u1, v1, u2, v2)
                if pt is not None:
                    dist = np.linalg.norm(pt - self.pos[node])
                    if dist < min_dist:
                        min_dist = dist
                        vec = pt - self.pos[node]
                        closest_angle = int(((np.degrees(np.arctan2(vec[1], vec[0])) + 360) % 360) // 45) % 8
            max_crossing_edge_direction = closest_angle

        return {
            "cross_map": self._get_cross_map(), #TODO: also rotate cross_map or move it into base_obs
            "local_view": base_obs,
            "gat_embedding": gat_feat,
            "max_crossing_edge_direction": np.array([max_crossing_edge_direction], dtype=np.int32), #todo replace this with below
            #todo: add to observation vector the local crossing number of incident edges in octant

            #todo: add also local crossing number (and global crossing number)
            #todo: use same observation vector for global and local

            #"force": net_force,
            #"local_patch": patch_normalized,
            #"current_pos": self.pos[node],
        }

    def _get_observation_Octant(self):
        node = self.current_node
        node_pos = self.pos[node]
        neighbors = set(self.graph.neighbors(node))
        nearby = self.graph.nodes #self.query_neighbors_within_radius(node)

        # Compute raw counts/distances per octant
        counts = np.zeros(8, dtype=np.float32)
        neigh_d    = np.full(8, np.inf, dtype=np.float32)
        nonneigh_d = np.full(8, np.inf, dtype=np.float32)
        # first_d removed, replaced by abs_counts
        abs_counts = np.zeros(8, dtype=np.float32)
        dists = {u: calculate_distance(node_pos, self.pos[u]) for u in nearby}

        for u, dist in dists.items():
            idx = int(calculate_angle(node_pos, self.pos[u]) // 45) % 8
            if u in neighbors:
                counts[idx] += 1
                neigh_d[idx] = min(neigh_d[idx], dist)
            else:
                nonneigh_d[idx] = min(nonneigh_d[idx], dist)
            abs_counts[idx] += 1  # Count all nodes (neighbors and non-neighbors)
        # zero‐fill empties
        neigh_d[neigh_d == np.inf]    = 0
        nonneigh_d[nonneigh_d == np.inf] = 0
        # first_d removed
        # normalize
        # radius = max(max(dists.values()) if dists else self.min_bbox_size, self.min_bbox_size)
        rel_counts = counts / counts.sum() if counts.sum() > 0 else counts
        # rel_neigh_d = neigh_d / radius
        # rel_nonneigh_d = nonneigh_d / nonneigh_d.max() if nonneigh_d.max() > 0 else nonneigh_d
        # rel_first replaced by normalized abs_counts
        if abs_counts.max() > 0:
            rel_abs_counts = abs_counts / abs_counts.max()
        else:
            rel_abs_counts = abs_counts
        # apply symmetry shift/rotation
        rel_counts, rel_neigh_d, rel_nonneigh, rel_first = \
           self.shift_and_rotate_octants(rel_counts, neigh_d, nonneigh_d, rel_abs_counts)
        # build obs
        obs = np.round(np.concatenate((rel_counts,
                                       neigh_d,
                                       nonneigh_d,
                                       rel_abs_counts)), 3).astype(np.float32)
        return obs


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

    def _get_direction_vectors(self):
        return {0:(1,-1),1:(1,0),2:(0,1),3:(-1,0),4:(0,-1),5:(1,1),6:(-1,1),7:(-1,-1)}

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

    def shift_and_rotate_octants(self, neigh_counts, neigh_dist, nonneigh_counts, first_dist):
        """
        Align the histogram features by rotating all octant-based arrays so that
        the net-force (centroid) direction is at index 0. This yields a canonical
        orientation based on the weighted mean direction of neighbors.
        """
        # Compute bin center angles for 8 octants
        thetas = np.arange(8) * (2 * np.pi / 8)
        # Use neighbor counts as weights for the directional vector
        vx = np.sum(neigh_counts * np.cos(thetas))
        vy = np.sum(neigh_counts * np.sin(thetas))
        # Compute overall angle of the weighted vector
        angle = np.arctan2(vy, vx)  # range [-pi, pi]
        # Determine the nearest octant index
        k = int(np.round(angle / (2 * np.pi / 8))) % 8
        # Rotate all feature arrays by -k to bring that direction to index 0
        for arr in (neigh_counts, neigh_dist, nonneigh_counts, first_dist):
            arr[:] = np.roll(arr, -k)
        # Store for action translation
        self._last_max_idx = k
        return neigh_counts, neigh_dist, nonneigh_counts, first_dist

    def translate_back_action(self, action, max_idx):
        """
        Map the action index in the rotated frame back to the original orientation.
        We ignore next_idx under the new scheme.
        """
        return (action + max_idx) % 8

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
        Compute crossings for the graph using the imported compute_crossings function.

        Parameters:
            edges (list): A list of edges to update. If None, recompute for all edges.

        Returns:
            tuple: (crossings_map, max_crossing_edge, global_crossings)
        """
        # Normalize the list of edges to update
        if edges is None:
            to_update = self.edges
        else:
            to_update = [tuple(sorted(e)) for e in edges]

        # Step 1: Query the R-tree for overlapping edges
        relevant_edges = set(to_update)
        for e in to_update:
            overlapping_edges = self.query_edge_bbox_overlaps(e)
            relevant_edges.update(overlapping_edges)

        # Step 2: Use the imported compute_crossings function
        positions = np.array([self.pos[n] for n in self.graph.nodes()])
        relevant_edges = list(relevant_edges)  # Convert to list for compatibility
        crossings_map, max_crossing_edge, global_crossings = compute_crossings(positions, relevant_edges)

        # Step 3: Update the internal crossings tracker
        self.crossings = {tuple(sorted(k)): set(map(tuple, v)) for k, v in crossings_map.items()}

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

    def _get_cross_map(self):
        grid = np.zeros((3, 3), dtype=np.float32)

        # 1) Extract crossings for the current node
        incident = [tuple(sorted(e)) for e in self.graph.edges(self.current_node)]
        cmap = {e: self.crossings.get(e, set()) for e in incident}

        # 2) Process crossings to populate the grid
        for e, crossed_edges in cmap.items():
            for f in crossed_edges:
                u1, v1 = e
                u2, v2 = f
                # Compute the actual crossing point
                pt = self.get_crossing_point(u1, v1, u2, v2)
                if pt is None:
                    continue

                vec = pt - self.pos[self.current_node]
                angle = (np.degrees(np.arctan2(vec[1], vec[0])) + 360) % 360
                idx = int(((angle + 22.5) % 360) // 45) #TODO make this fit into octant based view

                compass_to_grid = {
                    0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 0),
                    4: (1, 0), 5: (2, 0), 6: (2, 1), 7: (2, 2)
                }
                gi, gj = compass_to_grid[idx]
                grid[gi, gj] += 1

        # 3) Normalize so brightest = 1.0
        if grid.max() > 0:
            grid /= grid.max()

        return grid[..., None]

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

    def create_edge_rtree_entry(self, edge, positions):
        """
        Create an R-tree entry for a given edge.

        Parameters:
            edge (tuple): A tuple (node1, node2) representing the edge.
            positions (dict): A dictionary mapping node IDs to their (x, y) positions.

        Returns:
            tuple: (edge_id, bbox), where bbox is (min_x, min_y, max_x, max_y).
        """
        node1, node2 = edge
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]

        # Compute the bounding box
        bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        # Use the edge itself as the unique ID
        return edge, bbox

    def _init_edge_index(self):
        self.edge_rtree_index = index.Index()
        self.edge_bboxes = {}
        self.edge_id = {}
        for i, e in enumerate(self.edges):
            self.edge_id[e] = i
            _, bb = self.create_edge_rtree_entry(e, self.pos)
            self.edge_bboxes[e] = bb
            self.edge_rtree_index.insert(i, bb, obj=e)

    def _update_edges_for_node(self, node):
        for e in (tuple(sorted(x)) for x in self.graph.edges(node)):
            eid = self.edge_id[e]
            self.edge_rtree_index.delete(eid, self.edge_bboxes[e])
            _, bb = self.create_edge_rtree_entry(e, self.pos)
            self.edge_bboxes[e] = bb
            self.edge_rtree_index.insert(eid, bb, obj=e)

    def _update_edge_rtree(self):
        """
        Update the R-tree for edges by inserting their bounding boxes.
        """
        self.edge_rtree_index = index.Index()
        self.edge_bboxes = {}
        for edge in self.edges:
            edge_id, bbox = self.create_edge_rtree_entry(edge, self.pos)
            self.edge_rtree_index.insert(id(edge_id), bbox, obj=edge_id)
            self.edge_bboxes[edge_id] = bbox

    def query_edge_bbox_overlaps(self, edge):
        """
        Query the R-tree for edges whose bounding boxes overlap with the given edge's bounding box.

        Parameters:
            edge (tuple): The edge to query (node1, node2).

        Returns:
            list: A list of edges that overlap with the given edge's bounding box.
        """
        # Get the bounding box of the given edge
        bbox = self.edge_bboxes[edge]

        # Query the R-tree for overlapping edges
        overlapping_edges = [
            obj.object for obj in self.edge_rtree_index.intersection(bbox, objects=True)
            if obj.object != edge  # Exclude the edge itself
        ]

        return overlapping_edges

    def remove_crossings_for_node(self, node):
        """
        Efficiently remove all crossing‐entries for any edge incident to `node`.

        Algorithm:
          1. Build E_incident = { sorted(edge) for every edge touching `node` }.
          2. Iterate once over all keys in self.crossings:
               • If key ∈ E_incident: skip it for now.
               • Otherwise: do `self.crossings[key] -= E_incident`.
          3. Finally delete every key in E_incident from self.crossings if it exists.

        After this call, no entry involving node’s edges will remain in self.crossings.
        """
        # 1) Collect all incident edges (sorted) into a set
        E_incident = {tuple(sorted(e)) for e in self.graph.edges(node)}

        # 2) Single pass over all keys to remove references to any incident edge
        for other_edge in list(self.crossings.keys()):
            if other_edge in E_incident:
                # we’ll delete it in step 3
                continue
            # remove any incident edge from other_edge’s crossing‐set
            self.crossings[other_edge].difference_update(E_incident)

        # 3) Delete the crossing‐entries for the incident edges themselves
        for e in E_incident:
            self.crossings.pop(e, None)

    def recompute_crossings_for_node(self, node):
        """
        Recompute (and insert) any new crossings between edges incident to `node`
        and all other edges (except incident edges). Any newly found
        crossing pair (e, f) is inserted into self.crossings[e] and self.crossings[f].

        Uses a C++ function for the heavy crossing computation.
        """
        # 1) Gather all edges incident to `node`, always stored as sorted tuples.
        incident_edges = [tuple(sorted(e)) for e in self.graph.edges(node)]

        # 2) Gather all candidate edges whose bbox overlaps any incident edge
        candidate_edges = set()
        for e in incident_edges:
            overlaps = self.query_edge_bbox_overlaps(e)
            candidate_edges.update(tuple(sorted(f)) for f in overlaps)
        # Remove incident edges from candidates (no self-crossing)
        candidate_edges.difference_update(incident_edges)
        candidate_edges = list(candidate_edges)

        # 3) Call the C++ function to find crossings
        node_list = list(self.graph.nodes())
        node_idx = {n: i for i, n in enumerate(node_list)}
        def edge_to_idx(e): return (node_idx[e[0]], node_idx[e[1]])
        positions = np.array([self.pos[n] for n in node_list])
        incident_edges_idx = [edge_to_idx(e) for e in incident_edges]
        candidate_edges_idx = [edge_to_idx(e) for e in candidate_edges]

        crossings_map = find_crossings_for_edges(
            positions, incident_edges_idx, candidate_edges_idx
        )

        idx_to_edge = {edge_to_idx(e): e for e in incident_edges + candidate_edges}
        for e_idx, crossed_set in crossings_map.items():
            e = idx_to_edge[tuple(e_idx)]
            self.crossings.setdefault(e, set())
            for f_idx in crossed_set:
                f = idx_to_edge[tuple(f_idx)]
                self.crossings.setdefault(f, set())
                self.crossings[e].add(f)
                self.crossings[f].add(e)
