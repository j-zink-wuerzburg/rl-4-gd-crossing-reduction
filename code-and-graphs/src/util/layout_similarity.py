import numpy as np
from scipy.linalg import orthogonal_procrustes
import networkx as nx
from util.plot_graph import plot_graph

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist

def layout_similarity_distance_correlation(posA, posB):
    """
    Returns a % similarity based on the Pearson corr. of all pairwise distances.

    After Procrustes-aligning A→B, it builds two
    (n choose 2)-length vectors
    The similarity is then computed as the
    Pearson correlation between these two vectors.

    """
    nodes = list(posA.keys())
    A = np.array([posA[n] for n in nodes])
    B = np.array([posB[n] for n in nodes])

    # Center and normalize A
    A -= A.mean(axis=0)
    norm_A = np.linalg.norm(A)
    if norm_A > 0:
        A /= norm_A

    # Center and normalize B
    B -= B.mean(axis=0)
    norm_B = np.linalg.norm(B)
    if norm_B > 0:
        B /= norm_B

    # Procrustes alignment
    R, _ = orthogonal_procrustes(A, B)
    A_rot = A.dot(R)

    # Vectorize pairwise distances
    dA = pdist(A_rot)
    dB = pdist(B)

    # Correlation → percent
    corr = np.corrcoef(dA, dB)[0, 1]
    return max(0, corr) * 100


# Code to test layout_similarity_procrustes function
if __name__ == "__main__":
    # Create two graphs with the same nodes but different layouts
    G = nx.Graph()


    G = nx.barabasi_albert_graph(10, 1, seed=42)

    posA = nx.spring_layout(G, seed=42)
    #posB = nx.spring_layout(G1, seed=43)  # Different seed for different layout
    posB = nx.spectral_layout(G)
    #posB = nx.random_layout(G1, seed=43)



    # Plot the graphs
    plot_graph(G, posA, title="Graph A")
    plot_graph(G, posB, title="Graph B")

    similarity = layout_similarity_distance_correlation(posA, posB)
    print(f"Layout similarity: {similarity:.2f}%")
