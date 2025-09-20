import networkx as nx
import igraph as ig
import leidenalg
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class ClusterNode:
    def __init__(self, cluster_id, children, level):
        self.id = cluster_id
        self.level = level
        self.children = children  # List of node IDs or other ClusterNodes
        self.color = cluster_id
        self.x = None
        self.y = None

    def compute_position(self, node_positions): # Dummy for now
        """
        Computes average position from contained children.
        """
        if not node_positions:
            return

        positions = []

        for child in self.children:
            if isinstance(child, ClusterNode):
                child.compute_position(node_positions)
                if child.x is not None and child.y is not None:
                    positions.append((child.x, child.y))
            else:
                # Base node (level 0)
                if child in node_positions:
                    positions.append(node_positions[child])

        if positions:
            pos_arr = np.array(positions)
            self.x, self.y = np.mean(pos_arr[:, 0]), np.mean(pos_arr[:, 1])


class GraphLevel:
    def __init__(self, level_id, graph, cluster_map, parent_level=None):
        self.level_id = level_id
        self.graph = graph
        self.cluster_map = cluster_map  # node → cluster_id
        self.clusters = defaultdict(list)
        for node, cluster_id in cluster_map.items():
            self.clusters[cluster_id].append(node)
        self.cluster_nodes = {}  # cluster_id → ClusterNode
        self.parent_level = parent_level

        # Placeholder for node positions (optional)
        self.node_positions = {}

    def build_cluster_nodes(self):
        for cluster_id, members in self.clusters.items():
            children = []
            for node in members:
                if isinstance(node, ClusterNode):
                    children.append(node)
                else:
                    children.append(node)
            self.cluster_nodes[cluster_id] = ClusterNode(cluster_id, children, self.level_id)

        # Compute cluster positions
        for cluster in self.cluster_nodes.values():
            cluster.compute_position(self.node_positions)

    def get_super_graph(self):
        """
        Returns a new NetworkX graph of supernodes (clusters).
        """
        super_graph = nx.Graph()
        for cluster_id in self.clusters:
            super_graph.add_node(cluster_id, color=cluster_id)

        for u, v in self.graph.edges():
            cu = self.cluster_map[u]
            cv = self.cluster_map[v]
            if cu != cv:
                if super_graph.has_edge(cu, cv):
                    super_graph[cu][cv]['weight'] += 1
                else:
                    super_graph.add_edge(cu, cv, weight=1)

        return super_graph


def apply_leiden_clustering(graph):
    igraph_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(igraph_graph, leidenalg.ModularityVertexPartition)
    cluster_map = {node: cluster for cluster, nodes in enumerate(partition) for node in nodes}
    return cluster_map, len(partition)


def build_hierarchy(initial_graph, max_levels=5):
    levels = []
    current_graph = initial_graph
    current_positions = nx.spring_layout(current_graph, seed=42)

    for level_id in range(max_levels):
        cluster_map, num_clusters = apply_leiden_clustering(current_graph)
        graph_level = GraphLevel(level_id, current_graph, cluster_map)
        graph_level.node_positions = current_positions
        graph_level.build_cluster_nodes()

        levels.append(graph_level)

        # Stop if no more coarsening possible
        if num_clusters <= 2:
            break

        # Build next coarsened graph
        super_graph = graph_level.get_super_graph()
        current_graph = super_graph
        current_positions = nx.spring_layout(super_graph, seed=42)

    return levels


def test():
    G = nx.barabasi_albert_graph(200, 1)
    hierarchy = build_hierarchy(G, max_levels=5)

    for level in hierarchy:
        print(f"Level {level.level_id}:")
        print(f" - Nodes: {len(level.graph.nodes())}")
        print(f" - Clusters: {len(level.clusters)}")

        # visualize
        pos = nx.spring_layout(level.graph, seed=42)
        colors = [level.cluster_map[n] for n in level.graph.nodes()]
        plt.figure(figsize=(6, 6))
        nx.draw(level.graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.tab20)
        plt.title(f"Graph Level {level.level_id}")
        plt.show()


if __name__ == "__main__":
    test()
