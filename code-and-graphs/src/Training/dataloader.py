import os
from torch.utils.data import Dataset
import networkx as nx
import re

from util.plot_graph import plot_graph

class GraphPathDataset(Dataset):
    """
    Dataset yielding NetworkX graphs from a split list, handling both absolute and relative paths.
    Relabels nodes to integers to ensure numeric node IDs for spatial indexing.
    """
    def __init__(self, split_list_path, root_dir=None, file_format='gexf', sort_by_n=False):
        """
        Args:
            split_list_path (str): Path to a .txt file listing one graph file per line.
            root_dir (str, optional): Base directory to prepend to each listed relative path.
            file_format (str): Format of the graph files ('gexf' or 'gml').
            sort_by_n (bool): Whether to sort the graph paths by the 'n' value in the filename.
        """
        with open(split_list_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        # Only sort if requested (for extended_BA)
        if sort_by_n:
            def extract_n(filename):
                match = re.search(r'_n(\d+)_', filename)
                return int(match.group(1)) if match else float('inf')
            paths.sort(key=extract_n)
        self.graph_paths = paths
        self.root = root_dir or ''
        self.file_format = file_format

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        # Read the raw path and normalize separators for current OS
        rel_path = self.graph_paths[idx].replace('\\', os.sep).replace('/', os.sep)
        # Extract only filename to avoid duplicated directories
        filename = os.path.basename(rel_path)
        # Construct absolute path under root_dir
        abs_path = os.path.normpath(os.path.join(self.root, filename))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Graph file not found: {abs_path}")
        # Read the graph based on the file format
        if self.file_format == 'gexf':
            G_orig = nx.read_gexf(abs_path)
        elif self.file_format == 'gml':
            G_orig = nx.read_gml(abs_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        # Relabel nodes to integers for R-tree compatibility
        G = nx.convert_node_labels_to_integers(G_orig, label_attribute="original_label")
        return G

    def get_abs_paths(self):
        """Return absolute file paths the same way __getitem__ resolves them."""
        abs_paths = []
        for rel in self.graph_paths:
            rel_norm = rel.replace('\\', os.sep).replace('/', os.sep)
            fname = os.path.basename(rel_norm)
            abs_path = os.path.normpath(os.path.join(self.root, fname))
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"Graph file not found: {abs_path}")
            abs_paths.append(abs_path)
        return abs_paths


def load_split_dataset(split_type, dataset_type='rome'):
    """
    Loads a dataset object for the specified split and dataset type.
    """
    # Resolve project root based on this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    if dataset_type == 'rome':
        splits_dir = os.path.join(project_root, 'graphs', 'rome_filtered', 'splits')
        data_dir = os.path.join(splits_dir, 'data')
        file_format = 'gml'
        sort_by_n = False
    elif dataset_type == 'extended_BA':
        splits_dir = os.path.join(project_root, 'graphs', 'extended_BA_filtered')
        data_dir = os.path.join(splits_dir, 'data')  # Corrected path
        file_format = 'gml'
        sort_by_n = True
    else:
        raise ValueError("Invalid dataset_type. Must be 'rome' or 'extended_BA'.")

    split_file = os.path.join(splits_dir, f"{split_type}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    return GraphPathDataset(split_file, root_dir=data_dir, file_format=file_format, sort_by_n=sort_by_n)

# Example usage:
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import re
    # change working dir to ../
    print("Current working directory:", os.getcwd())
    # set work dir ../
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    # Instantiate dataset for the 'train' split from 'rome'
    rome_dataset = load_split_dataset('train', dataset_type='rome')
    rome_loader = DataLoader(rome_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Instantiate dataset for the 'train' split from 'extended_BA'
    extended_ba_dataset = load_split_dataset('train', dataset_type='extended_BA')
    extended_ba_loader = DataLoader(extended_ba_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Print the first 10 filenames and their n values to verify sorting for extended_BA
    print("First 10 extended_Ba filenames and their n values (should be sorted by n):")
    for path in extended_ba_dataset.graph_paths[:10]:
        match = re.search(r'_n(\d+)_', path)
        n_val = int(match.group(1)) if match else None
        print(f"{path} (n={n_val})")

    # Print the first 5 rome filenames to verify loading is unchanged
    print("First 5 rome filenames (should be in original order):")
    for path in rome_dataset.graph_paths[:5]:
        print(path)

    # Load a sample graph and visualize
    G = extended_ba_dataset[2]
    from env.GraphLayoutEnv import GraphLayoutEnv
    env = GraphLayoutEnv(graph=G)
    env.reset(Graph=G)
    spring_pos = nx.spring_layout(G)
    plot_graph(G, spring_pos)
    plot_graph(G, env.pos)
    env = GraphLayoutEnv(graph=rome_dataset[0])
    plot_graph(rome_dataset[0], env.pos)
