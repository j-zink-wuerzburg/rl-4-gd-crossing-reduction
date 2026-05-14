import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
import json
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx


def load_graph(graph_path):
    with open(graph_path, 'r') as f:
        data = json.load(f)

    G = nx.Graph()
    # Nodes may be under 'nodes' or 'points'
    nodes_key = 'nodes' if 'nodes' in data else ('points' if 'points' in data else None)
    if nodes_key is None:
        raise ValueError("JSON does not contain 'nodes' or 'points' array")

    # Accumulate min/max for optional width/height derivation
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for node in data[nodes_key]:
        nid = node.get('id')
        if nid is None:
            raise ValueError("Node missing 'id'")
        x = node.get('x')
        y = node.get('y')
        if x is None or y is None:
            # If missing, default to 0
            x = 0
            y = 0
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        G.add_node(nid, x=x, y=y)

    # Edges
    edges = data.get('edges', [])
    for edge in edges:
        u = edge.get('source')
        v = edge.get('target')
        if u is None or v is None:
            continue
        if u == v:
            # Skip self-loops for planarity purposes
            continue
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # Width/Height optional; derive if missing
    width = data.get('width')
    height = data.get('height')
    if width is None or height is None:
        # Derive canvas from node bbox with small margins; ensure positive ints
        if min_x == float('inf') or min_y == float('inf'):
            width = width or 1000
            height = height or 1000
        else:
            # +1 to ensure max coordinate fits; add 1% margin (at least 10)
            span_x = max(1, int(max_x - min_x + 1))
            span_y = max(1, int(max_y - min_y + 1))
            margin_x = max(10, int(0.01 * span_x))
            margin_y = max(10, int(0.01 * span_y))
            width = span_x + 2 * margin_x
            height = span_y + 2 * margin_y

    return G, width, height

def preprocess_graph(G):
    pyg_data = from_networkx(G)
    G_processed = to_networkx(pyg_data)
    # Create a new undirected graph to store the preprocessed graph
    preprocessed_graph = nx.Graph()

    # Add nodes to the preprocessed graph
    preprocessed_graph.add_nodes_from(G_processed.nodes(data=True))

    # Add edges to the preprocessed graph, ensuring no duplicates
    for u, v in G_processed.edges():
        if not preprocessed_graph.has_edge(u, v):
            preprocessed_graph.add_edge(u, v)

    preprocessed_graph = round_graph_layout(preprocessed_graph)
    return preprocessed_graph


def round_graph_layout(G):
    pos = nx.get_node_attributes(G, 'pos')
    rounded_pos = {node: (round(coord[0]), round(coord[1])) for node, coord in pos.items()}
    nx.set_node_attributes(G, rounded_pos, 'pos')
    return G
