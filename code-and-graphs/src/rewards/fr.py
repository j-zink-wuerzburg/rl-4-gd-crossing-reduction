import numpy as np

def fr_force_vector(G, positions, node, k):
    """
    Compute the Fruchterman-Reingold net force vector for a single node.

    Parameters:
        G         : networkx.Graph
        positions : dict {node: np.array([x,y])}
        node      : node to compute force for
        k         : optimal distance constant

    Returns:
        np.ndarray: 2D force vector for this node
    """
    pos_v = np.array(positions[node], dtype=float)
    disp = np.zeros(2, dtype=float)

    # Repulsive forces: between this node and all others
    for u in G.nodes():
        if u == node:
            continue
        delta = pos_v - positions[u]
        dist = np.linalg.norm(delta) + 1e-6
        # repulsive: k^2 / dist
        rep = (k ** 2) / dist
        disp += (delta / dist) * rep

    # Attractive forces: along edges
    for u in G.neighbors(node):
        delta = pos_v - positions[u]
        dist = np.linalg.norm(delta) + 1e-6
        # attractive: dist^2 / k
        attr = (dist ** 2) / k
        disp -= (delta / dist) * attr

    return disp


import math
import itertools
import numpy as np


def compute_layout_energy(graph, pos, k_s=1.0, k_r=1.0, eps=1e-9):
    """
    Compute the Fruchterman–Reingold layout energy:
      E = ½ k_s ∑_{(i,j)∈E} ||x_i - x_j||^2
          - k_r ∑_{i<j} ln(||x_i - x_j|| + eps)

    Args:
        graph: networkx.Graph
        pos:   dict mapping node -> np.array([x,y])
        k_s:   spring constant (attractive)
        k_r:   repulsion constant
        eps:   small term to avoid log(0)

    Returns:
        float total energy
    """
    E_attr = 0.0
    # Attractive term over edges
    for u, v in graph.edges():
        duv = pos[u] - pos[v]
        d2 = np.dot(duv, duv)
        E_attr += 0.5 * k_s * d2

    E_rep = 0.0
    # Repulsive term over all unordered node pairs
    nodes = list(graph.nodes())
    for i, j in itertools.combinations(nodes, 2):
        d = np.linalg.norm(pos[i] - pos[j]) + eps
        E_rep -= k_r * math.log(d)

    return E_attr + E_rep


def compute_node_forces(graph, positions, node, k=None):
    """
    Computes the Fruchterman-Reingold scalar force and net force vector for a single node,
    with defensive checks against infinities/NaNs.

    Parameters:
        graph: networkx.Graph
        positions: dict
        node: node
        k: float

    Returns:
        scalar_force: float
        net_force: np.ndarray (shape (2,))
    """
    if k is None:
        k = 1 / np.sqrt(graph.number_of_nodes())


    pos_node = np.array(positions[node], dtype=float)
    scalar_force = 0.0
    net_force = np.zeros(2, dtype=float)

    # Attractive forces (edges)
    for neighbor in graph.neighbors(node):
        pos_nb = np.array(positions[neighbor], dtype=float)
        dist = np.linalg.norm(pos_node - pos_nb) + 1e-6
        force_magnitude = (dist ** 2) / k
        direction = (pos_nb - pos_node) / dist
        # guard infinities/NaNs
        if np.isfinite(force_magnitude) and np.all(np.isfinite(direction)):
            net_force += force_magnitude * direction
            scalar_force += force_magnitude

    # Repulsive forces (all other nodes)
    for other in graph.nodes:
        if other == node:
            continue
        pos_oth = np.array(positions[other], dtype=float)
        dist = np.linalg.norm(pos_node - pos_oth) + 1e-6
        force_magnitude = - (k ** 2) / dist
        direction = (pos_node - pos_oth) / dist
        if np.isfinite(force_magnitude) and np.all(np.isfinite(direction)):
            net_force += force_magnitude * direction
            scalar_force += force_magnitude

    return scalar_force, net_force
