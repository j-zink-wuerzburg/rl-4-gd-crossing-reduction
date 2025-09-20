"""
crossings.py

Provides functions to compute global crossing counts and a local, continuous
crossing‐reduction reward suitable for RL environments.
"""
import numpy as np
from shapely.geometry import LineString
from rtree import index


def count_global_crossings(G, pos):
    """
    Count total edge‐edge crossings in a straight‐line drawing.

    Args:
        G   : networkx.Graph
        pos : dict mapping node -> (x,y)

    Returns:
        int: number of intersecting edge pairs
    """
    edges = list(G.edges())
    lines = [LineString([tuple(pos[u]), tuple(pos[v])]) for u, v in edges]
    n_cross = 0
    m = len(lines)
    for i in range(m):
        Li = lines[i]
        for j in range(i+1, m):
            if Li.crosses(lines[j]):
                n_cross += 1
    return n_cross


def compute_crossing_reward(G, old_pos, new_pos, node,
                            alpha=1.0, beta=1.0,
                            clip_range=(-1.0, 1.0)):
    """
    Compute a local, dense reward for reducing edge crossings by moving `node`.
    Only incident edges are tested against non‐incident edges whose bounding boxes overlap.

    Reward is composed of:
      - continuous distance change: +alpha * (d_new - d_old)
      - discrete bonus/penalty: +beta for resolving a crossing, -beta for creating one

    Args:
        G        : networkx.Graph
        old_pos  : dict node->(x,y) before move
        new_pos  : dict node->(x,y) after move
        node     : the moved node
        alpha    : weight for continuous distance reward
        beta     : weight for discrete crossing bonus
        clip_range: (min,max) clip on final reward

    Returns:
        float: normalized reward ∈ clip_range
    """
    # Prepare edges and R-tree of old positions
    edges = list(G.edges())
    idx = index.Index()
    for eid, (u, v) in enumerate(edges):
        line = LineString([tuple(old_pos[u]), tuple(old_pos[v])])
        idx.insert(eid, line.bounds)

    # Identify edges incident on moved node
    incident = [eid for eid, (u, v) in enumerate(edges) if u == node or v == node]
    total_reward = 0.0
    count = 0

    for eid in incident:
        u, v = edges[eid]
        # Old and new segment
        L_old = LineString([tuple(old_pos[u]), tuple(old_pos[v])])
        L_new = LineString([tuple(new_pos[u]), tuple(new_pos[v])])
        # Query only overlapping edges
        for other in idx.intersection(L_old.bounds):
            if other == eid:
                continue
            u2, v2 = edges[other]
            # skip edges sharing the moved node
            if u2 == node or v2 == node:
                continue
            # Other segment old/new
            M_old = LineString([tuple(old_pos[u2]), tuple(old_pos[v2])])
            M_new = LineString([tuple(new_pos[u2]), tuple(new_pos[v2])])

            # distances
            d_old = L_old.distance(M_old)
            d_new = L_new.distance(M_new)
            # crossings
            cross_old = L_old.crosses(M_old)
            cross_new = L_new.crosses(M_new)

            # accumulate
            total_reward += alpha * (d_new - d_old)
            if cross_old and not cross_new:
                total_reward += beta
            if not cross_old and cross_new:
                total_reward -= beta
            count += 1

    # normalize and clip
    if count > 0:
        total_reward = total_reward / count
    low, high = clip_range
    return float(np.clip(total_reward, low, high))


# Example usage
if __name__ == '__main__':
    import networkx as nx
    G = nx.barabasi_albert_graph(20, 1)
    pos = nx.spring_layout(G)
    # artificially perturb one node
    old = pos.copy()
    new = pos.copy()
    new[0] = new[0] + np.array([0.1, -0.05])

    print("Global crossings:", count_global_crossings(G, pos))
    print("Crossing reward:", compute_crossing_reward(G, old, new, node=0))
