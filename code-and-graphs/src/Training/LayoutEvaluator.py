import networkx as nx
import pandas as pd
import numpy as np
import gdMetriX as metrics


def mean_edge_length(G, pos):
    edge_lengths = [
        np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        for u, v in G.edges()
    ]
    return np.mean(edge_lengths) if edge_lengths else 0.0

class LayoutEvaluatorCore:
    """
    Core evaluation of 2D graph layouts with a concise set of normalized metrics.
    Metrics included:
      - crossing_score           : 1 - (#crossings / max_crossings)
      - crossing_angle_score     : crossing angular resolution / 90
      - angular_resolution_score : angular resolution / 180
      - edge_length_uniformity   : 1 / (1 + std_dev(edge_lengths))
      - node_separation_score    : min node distance / avg edge length
      - aspect_ratio_score       : 1 - |1 - aspect_ratio|
      - compactness_score        : 1 - (tight_area / bounding_box_area)
      - stress_score             : 1 - (stress / sum(d_uv^2))
    """
    def __init__(self, G: nx.Graph, pos: dict):
        self.G = G
        self.pos = pos

    from pybindCode import graph_utils

    def evaluate(self) -> pd.DataFrame:
        scores = {}
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()

        # Convert graph edges and positions to the required format
        edges = list(self.G.edges())
        positions = np.array([self.pos[node] for node in self.G.nodes()])

        # Compute crossings using graph_utils
        crossings, max_crossing_edge, global_crossings = self.graph_utils.compute_crossings(positions, edges)

        # Absolute global crossings
        scores['global_crossings'] = global_crossings

        # k-planarity number (max crossings for a single edge)
        k_planarity = len(crossings[max_crossing_edge]) if max_crossing_edge in crossings else 0
        scores['k_planarity_number'] = k_planarity




        # Stress (distance preservation)
        dists = dict(nx.all_pairs_shortest_path_length(self.G))
        stress = 0.0
        denom = 0.0
        for u in self.G.nodes():
            for v, d_uv in dists[u].items():
                if u >= v:
                    continue
                p_u = np.array(self.pos[u])
                p_v = np.array(self.pos[v])
                euc = np.linalg.norm(p_u - p_v)
                stress += (euc - d_uv) ** 2
                denom += d_uv ** 2
        sts = 1.0 - (stress / denom) if denom > 0 else 1.0
        scores['stress_score'] = float(np.clip(sts, 0, 1))

        # Assemble DataFrame
        df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
        return df

if __name__ == '__main__':
    import networkx as nx
    from util.plot_graph import plot_graph

    G = nx.barabasi_albert_graph(100, 1)
    pos_good = nx.spring_layout(G)
    pos_rand = nx.random_layout(G)

    evalr = LayoutEvaluatorCore(G, pos_good)
    print("Spring layout:\n", evalr.evaluate())

    evalr2 = LayoutEvaluatorCore(G, pos_rand)
    print("Random layout:\n", evalr2.evaluate())

    plot_graph(G, pos_good)
    plot_graph(G, pos_rand)
