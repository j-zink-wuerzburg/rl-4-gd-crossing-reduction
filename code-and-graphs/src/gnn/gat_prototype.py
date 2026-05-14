import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GATv2Conv, DeepGraphInfomax, global_mean_pool
from torch_geometric.utils import from_networkx

# Add the project root directory to the Python path
def add_project_root_to_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.append(project_root)

def get_project_root():
    """Returns the absolute path to the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', '..'))


def build_data_from_graph(G, noise_scale=0.01):
    """
    Convert a NetworkX graph to a PyG Data object with handcrafted node features.
    Introduces deterministic per-node noise based on node id to ensure cross-graph consistency.
    """
    # Safe eigenvector centrality: fallback to power-method or zeros on failure
    try:
        centrality = nx.eigenvector_centrality_numpy(G)
    except Exception:
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        except Exception:
            centrality = {n: 0.0 for n in G.nodes()}
    # Other metrics (generally safe on disconnected)
    try:
        betweenness = nx.betweenness_centrality(G)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}
    try:
        clustering_coeff = nx.clustering(G)
    except Exception:
        clustering_coeff = {n: 0.0 for n in G.nodes()}
    # Eccentricity for disconnected graphs: compute per-component
    try:
        if nx.is_connected(G) or G.number_of_nodes() == 0:
            eccentricity = nx.eccentricity(G)
        else:
            eccentricity = {}
            for comp in nx.connected_components(G):
                sub = G.subgraph(comp)
                ecc_sub = nx.eccentricity(sub)
                eccentricity.update(ecc_sub)
    except Exception:
        eccentricity = {n: 0.0 for n in G.nodes()}
    try:
        harmonic_centrality = nx.harmonic_centrality(G)
    except Exception:
        harmonic_centrality = {n: 0.0 for n in G.nodes()}
    try:
        core_number = nx.core_number(G)
    except Exception:
        core_number = {n: 0 for n in G.nodes()}

    features = []
    for node in G.nodes():
        # Deterministic noise based on node label/id
        seed = (hash(str(node)) & 0xFFFFFFFF)
        rng = np.random.RandomState(seed)
        noise = float(rng.randn() * noise_scale)

        feat = [
            G.degree[node],
            noise,
            centrality.get(node, 0.0),
            clustering_coeff.get(node, 0.0),
            betweenness.get(node, 0.0),
            eccentricity.get(node, 0.0),
            harmonic_centrality.get(node, 0.0),
            core_number.get(node, 0),
        ]
        features.append(feat)

    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    return data


class GATEncoder(nn.Module):
    """
    Multi-layer GATv2 with LayerNorm, dropout, and residual connections.
    """
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4, dropout=0.6, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.ModuleDict({
            'conv': GATv2Conv(in_feats, hidden_feats, heads=heads, dropout=dropout),
            'norm': nn.LayerNorm(hidden_feats * heads)
        }))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.ModuleDict({
                'conv': GATv2Conv(hidden_feats * heads, hidden_feats, heads=heads, dropout=dropout),
                'norm': nn.LayerNorm(hidden_feats * heads)
            }))
        # Output layer
        self.layers.append(nn.ModuleDict({
            'conv': GATv2Conv(hidden_feats * heads, out_feats, heads=1, concat=False, dropout=dropout),
            'norm': nn.LayerNorm(out_feats)
        }))

        self.dropout = dropout
        self.res_projs = nn.ModuleList([
            nn.Linear(in_feats if i == 0 else hidden_feats * heads, hidden_feats * heads)
            for i in range(num_layers - 1)
        ])

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, layer in enumerate(self.layers):
            conv = layer['conv']
            norm = layer['norm']
            h_new = conv(h, edge_index, edge_weight) if edge_weight is not None else conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if i < len(self.res_projs):
                res = self.res_projs[i](h)
                h = h_new + res
            else:
                h = h_new
        return h


class DGIModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4, dropout=0.6, noise_std=0.01):
        super().__init__()
        self.encoder = GATEncoder(in_feats, hidden_feats, out_feats, heads, dropout)
        self.noise_std = noise_std
        self.dgi = DeepGraphInfomax(
            hidden_channels=out_feats,
            encoder=self.encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(
                global_mean_pool(
                    z,
                    torch.zeros(z.size(0), dtype=torch.long, device=z.device)
                )
            ),
            corruption=lambda x, edge_index, edge_weight=None: (
                x[torch.randperm(x.size(0))],
                edge_index,
                edge_weight
            )
        )

    def forward(self, data):
        return self.dgi(data.x, data.edge_index, getattr(data, 'edge_attr', None))

    def embed(self, data, apply_noise=False):      #   <----- With or without noise???
        self.eval()
        with torch.no_grad():
            z = self.encoder(data.x, data.edge_index, getattr(data, 'edge_attr', None))
            if apply_noise:
                noise = torch.randn_like(z) * self.noise_std
                z = z + noise
            return z


def train_dgi_with_gat(G_list, epochs=100, lr=1e-3, batch_size=8, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datas = [build_data_from_graph(G).to(device) for G in G_list]
    loader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    model = DGIModel(in_feats=8, hidden_feats=16, out_feats=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 1

    # Optionally resume
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    model.train()
    print("Starting training...")
    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0
        for batch in loader:
            for data in batch:
                optimizer.zero_grad()
                pos_z, neg_z, summary = model(data)
                loss = model.dgi.loss(pos_z, neg_z, summary)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch:4d} | Avg Loss: {total_loss / len(G_list):.4f}")

    print("Training complete.")
    return model, optimizer, epoch


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Checkpoint saved (epoch {epoch}) to {path}")


def create_gat_embedding(graph, model_path='src/gnn/models/dgi_gat_romeLongCrossGen.pt', noise_std=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Always resolve model_path from project root
    project_root = get_project_root()
    # Ensure model_path is always relative to project root
    abs_model_path = os.path.join(project_root, model_path) if not os.path.isabs(model_path) else model_path
    try:
        ckpt = torch.load(abs_model_path, map_location=device)
        model = DGIModel(in_feats=8, hidden_feats=16, out_feats=32, noise_std=noise_std).to(device)
        model.load_state_dict(ckpt['state_dict'])
        data = build_data_from_graph(graph).to(device)
        return model.embed(data)
    except Exception:
        # Fallback: return dummy zero embeddings with expected latent dim (32)
        n = graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0
        return torch.zeros((n, 32), dtype=torch.float32, device=device)


def test_gat_embedding(graphs, model_path='src/gnn/models/dgi_gat_romeLongCrossGen.pt', noise_std=0.01):
    """
    For each graph, plot the original graph and its node embeddings (PCA-reduced) side by side.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Always resolve model_path from project root
    project_root = get_project_root()
    abs_model_path = os.path.join(project_root, model_path) if not os.path.isabs(model_path) else model_path
    ckpt = torch.load(abs_model_path, map_location=device)
    model = DGIModel(in_feats=8, hidden_feats=16, out_feats=32, noise_std=noise_std).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    for idx, graph in enumerate(graphs):
        data = build_data_from_graph(graph).to(device)
        embeddings = model.embed(data).cpu().numpy()
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500)
        plt.title(f"Original Graph {idx+1}")

        plt.subplot(1, 2, 2)
        plt.scatter(reduced[:, 0], reduced[:, 1], c='lightcoral', edgecolors='k', s=100)
        for i, (x, y) in enumerate(reduced):
            plt.text(x, y, str(i), ha='center', va='center', fontsize=9)
        plt.title("GAT + DGI Embeddings (2D)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.show()


def train_model():
    add_project_root_to_path()
    from Training.dataloader import load_split_dataset

    # load graphs
    ds = load_split_dataset('train')
    graphs = [ds[i] for i in range(len(ds))]

    # train/resume
    project_root = get_project_root()
    checkpoint_path = os.path.join(project_root, 'src/gnn/models/dgi_gat_romeLongCrossGen.pt')
    model, optimizer, last_epoch = train_dgi_with_gat(
        graphs,
        epochs=200,
        lr=1e-3,
        batch_size=8,
        resume_path=checkpoint_path
    )

    # save final checkpoint
    save_checkpoint(model, optimizer, last_epoch, checkpoint_path)


if __name__ == '__main__':

    G1 = nx.barabasi_albert_graph(10, 1, seed=42)
    G2 = nx.barabasi_albert_graph(10, 1, seed=42)

    Gs = [G1, G2]

    # Always resolve model path from project root
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'src/gnn/models/dgi_gat_romeLongCrossGen.pt')
    test_gat_embedding(Gs, model_path, noise_std=0.01)
