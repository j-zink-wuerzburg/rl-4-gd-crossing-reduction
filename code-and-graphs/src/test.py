import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



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


def test_fr_movement():
    # Create graph
    G = nx.barabasi_albert_graph(10, 1, seed=42)
    n = G.number_of_nodes()
    area = 100.0

    # FR constants
    k = np.sqrt(area / n)
    initial_temp = np.sqrt(area)
    cooling = 0.95
    iterations = 100

    # Initialize random positions
    side = np.sqrt(area)
    positions = {v: np.random.rand(2) * side for v in G.nodes()}

    # Plot initial layout
    plt.figure(figsize=(6, 6))
    nx.draw(G, positions, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Initial Layout")
    plt.show()

    # Iteratively update positions
    t = initial_temp
    for i in range(iterations):
        new_pos = {}
        for v in G.nodes():
            disp = fr_force_vector(G, positions, v, k)
            norm = np.linalg.norm(disp)
            if norm > 0:
                # clamp displacement by temperature
                disp = disp / norm * min(norm, t)
            new_pos[v] = positions[v] + disp
        positions = new_pos
        t *= cooling

    # Plot final layout
    plt.figure(figsize=(6, 6))
    nx.draw(G, positions, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title("Fruchterman-Reingold Layout")
    plt.show()

if __name__ == '__main__':
    test_fr_movement()
