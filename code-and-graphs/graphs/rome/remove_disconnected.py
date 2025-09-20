import os
import networkx as nx

def remove_disconnected(graphs_dir='./splits/data', trash_dir=None):
    """
    Remove all disconnected .gexf graphs in the specified directory.
    Args:
        graphs_dir (str): Directory containing .gexf files. Defaults to './splits/data'.
        trash_dir (str, optional): Directory to move disconnected graphs into instead of deleting.
    """
    # Ensure target directory exists
    if not os.path.isdir(graphs_dir):
        raise NotADirectoryError(f"Graphs directory not found: {graphs_dir}")
    # Ensure trash_dir exists if provided
    if trash_dir:
        os.makedirs(trash_dir, exist_ok=True)

    for fname in os.listdir(graphs_dir):
        if not fname.endswith('.gexf'):
            continue
        fpath = os.path.join(graphs_dir, fname)
        try:
            G = nx.read_gexf(fpath)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
        # Check connectivity (treat as undirected)
        if not nx.is_connected(G.to_undirected()):
            print(f"Removing disconnected graph: {fname}")
            if trash_dir:
                os.rename(fpath, os.path.join(trash_dir, fname))
            else:
                os.remove(fpath)

if __name__ == '__main__':
    # Default to './splits/data'
    print("Removing disconnected graphs from './splits/data'")
    remove_disconnected()
    print('Done.')