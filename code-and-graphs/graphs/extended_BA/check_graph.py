import os
import networkx as nx

# Change work directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Directory containing the .gml files
directory = 'data'

def find_highest_edge_count(directory):
    highest_edge_count = 0
    graph_with_highest_edges = None

    for filename in os.listdir(directory):
        if filename.endswith('.gml'):
            filepath = os.path.join(directory, filename)
            try:
                # Load the graph
                graph = nx.read_gml(filepath)
                edge_count = graph.number_of_edges()

                # Check if this graph has the highest edge count
                if edge_count > highest_edge_count:
                    highest_edge_count = edge_count
                    graph_with_highest_edges = filename
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return graph_with_highest_edges, highest_edge_count

def check_graph_connectivity(directory):
    disconnected_graphs = []
    for filename in os.listdir(directory):
        if filename.endswith('.gml'):
            filepath = os.path.join(directory, filename)
            try:
                graph = nx.read_gml(filepath)
                if not nx.is_connected(graph):
                    disconnected_graphs.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return disconnected_graphs

if __name__ == "__main__":

    disconnected = check_graph_connectivity(directory)
    if disconnected:
        print("Disconnected graphs found:")
        for g in disconnected:
            print(f" - {g}")
    else:
        print("All graphs are connected.")
