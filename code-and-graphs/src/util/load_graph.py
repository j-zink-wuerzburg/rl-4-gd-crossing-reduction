import json
import networkx as nx




def load_graph(graph_path):
    with open(graph_path, 'r') as f:
        data = json.load(f)

    G = nx.Graph()

    for node in data['nodes']:
        G.add_node(node['id'], x=node['x'], y=node['y'])

    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])

    return G
