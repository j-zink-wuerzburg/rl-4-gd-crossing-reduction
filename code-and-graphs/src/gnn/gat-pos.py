import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, from_networkx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from util.gat_prototype import build_data_from_graph

"""
Based on https://arxiv.org/abs/2105.04037
"""

class PositionalEmbeddingModel(nn.Module):
    """
    Simple MLP to refine raw positional lookup embeddings into final p_v.
    """
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(p0))
        p = self.fc2(h)  # [N, emb_dim]
        return p


def unsupervised_skipgram_loss(pos: torch.Tensor,
                                edge_index: torch.Tensor,
                                num_nodes: int,
                                num_neg: int = 5) -> torch.Tensor:
    """
    Skip-gram style loss over edges; for each edge (i,j) maximize log sigmoid(p_i^T p_j)
    and for num_neg negative samples minimize log sigmoid(-p_i^T p_neg).
    """
    row, col = edge_index
    device = pos.device
    pos_norm = pos / (pos.norm(dim=1, keepdim=True) + 1e-8)
    loss = 0.0
    sampler = torch.distributions.Categorical(torch.ones(num_nodes, device=device))
    for i, j in zip(row.tolist(), col.tolist()):
        score_pos = torch.sigmoid((pos_norm[i] * pos_norm[j]).sum())
        neg_idx = sampler.sample((num_neg,))
        neg_pos = pos_norm[neg_idx]
        score_neg = torch.sigmoid(- (pos_norm[i].unsqueeze(0) * neg_pos).sum(dim=1)).mean()
        loss += -torch.log(score_pos + 1e-15) - torch.log(score_neg + 1e-15)
    return loss / row.size(0)

class GATPOSConv(MessagePassing):
    """
    Graph Attention layer that incorporates learned positional embeddings into attention.
    Implements eqs. (2) & (3) from GAT-POS: attention coefficients over [W x + U p].
    """
    def __init__(self,
                 in_channels: int,
                 pos_channels: int,
                 out_channels: int,
                 heads: int = 4,
                 dropout: float = 0.6,
                 concat: bool = True):
        # node_dim=0 makes propagate index over the 1st axis (nodes)
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout

        self.lin_x = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_p = nn.Linear(pos_channels, heads * out_channels, bias=False)
        self.att   = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_x.weight)
        nn.init.xavier_uniform_(self.lin_p.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self,
                x: torch.Tensor,
                p: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        # add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        N = x.size(0)
        H, C = self.heads, self.out_channels
        # project content and position
        x_proj = self.lin_x(x).view(N, H, C)  # [N, H, C]
        p_proj = self.lin_p(p).view(N, H, C)  # [N, H, C]
        # combine streams
        x_pos = x_proj + p_proj
        # message passing
        out = self.propagate(edge_index,
                             x=x_proj,
                             x_pos=x_pos)
        # aggregate
        if self.concat:
            return out.view(N, H * C)
        else:
            return out.mean(dim=1)

    def message(self,
                x_j: torch.Tensor,
                x_pos_i: torch.Tensor,
                x_pos_j: torch.Tensor,
                index: torch.Tensor,
                ptr: torch.Tensor,
                size_i: int) -> torch.Tensor:
        alpha = torch.cat([x_pos_i, x_pos_j], dim=-1)  # [E, H, 2C]
        alpha = (alpha * self.att).sum(dim=-1)           # [E, H]
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

class GATPOSEncoder(nn.Module):
    """
    Two-stream GAT-POS encoder: content from x, structural from learned lookup p0.
    """
    def __init__(self,
                 num_nodes: int,
                 in_feats:  int,
                 p0_dim:    int,
                 pos_hidden:int,
                 pos_dim:   int,
                 gat_hidden:int,
                 gat_out:   int,
                 heads:     int = 4,
                 dropout:   float = 0.6):
        super().__init__()
        # positional lookup & MLP
        self.p_lookup   = nn.Embedding(num_nodes, p0_dim)
        self.pos_mlp    = PositionalEmbeddingModel(p0_dim, pos_hidden, pos_dim)
        # GAT-POS layers
        self.conv1 = GATPOSConv(in_feats, pos_dim, gat_hidden, heads, dropout, concat=True)
        self.conv2 = GATPOSConv(gat_hidden * heads, pos_dim, gat_out, 1, dropout, concat=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # 1) generate positional embedding p
        N = x.size(0)
        node_ids = torch.arange(N, device=x.device)
        p0 = self.p_lookup(node_ids)          # [N, p0_dim]
        p  = self.pos_mlp(p0)                 # [N, pos_dim]

        # 2) GAT-POS content stream
        h = F.dropout(x,    p=self.dropout, training=self.training)
        h = self.conv1(h, p, edge_index)
        h = F.elu(h)
        h = F.dropout(h,    p=self.dropout, training=self.training)
        h = self.conv2(h, p, edge_index)
        return h, p

# Example & test
def train_and_visualize():
    # Build sample graph
    G = nx.barabasi_albert_graph(20, 1, seed=42)
    G = nx.convert_node_labels_to_integers(G)
    data = from_networkx(G)

    # Extract features from the Data object returned by build_data_from_graph
    features = build_data_from_graph(G)

    # If features is a Data object, extract its 'x' attribute
    if isinstance(features, Data):
        features = features.x

    # Convert features to a tensor
    data.x = features.clone().detach().float()  # Avoid the UserWarning
    data.y = torch.randint(0, 2, (data.num_nodes,), dtype=torch.long)  # Random binary labels

    # Instantiate model
    model = GATPOSEncoder(
        num_nodes=data.num_nodes,
        in_feats=data.x.size(1),  # Match the number of features in data.x
        p0_dim=32,
        pos_hidden=32,
        pos_dim=8,
        gat_hidden=8,
        gat_out=16,
        heads=4,
        dropout=0.6
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    lambda_pos = 0.1  # Weight for positional loss

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        h, p = model(data.x, data.edge_index)
        L_task = criterion(h, data.y)  # Supervised loss
        L_pos = unsupervised_skipgram_loss(p, data.edge_index, data.num_nodes)
        loss = L_task + lambda_pos * L_pos
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, L_task: {L_task.item():.4f}, L_pos: {L_pos.item():.4f}")

    # Visualize learned embeddings and the graph
    model.eval()
    with torch.no_grad():
        h, p = model(data.x, data.edge_index)

    # PCA for content embeddings h
    pca_h = PCA(n_components=2)
    reduced_h = pca_h.fit_transform(h.detach().numpy())

    # PCA for positional embeddings p
    pca_p = PCA(n_components=2)
    reduced_p = pca_p.fit_transform(p.detach().numpy())

    # Graph layout
    pos = nx.spring_layout(G)

    # Plot
    plt.figure(figsize=(18, 6))

    # Plot the graph
    plt.subplot(1, 3, 1)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Graph Structure")

    # Plot content embeddings
    plt.subplot(1, 3, 2)
    plt.scatter(reduced_h[:, 0], reduced_h[:, 1], c='lightblue', edgecolors='k', s=100)
    for i, (x, y) in enumerate(reduced_h):
        plt.text(x, y, str(i), ha='center', va='center', fontsize=9)
    plt.title("Content Embeddings (h)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Plot positional embeddings
    plt.subplot(1, 3, 3)
    plt.scatter(reduced_p[:, 0], reduced_p[:, 1], c='lightcoral', edgecolors='k', s=100)
    for i, (x, y) in enumerate(reduced_p):
        plt.text(x, y, str(i), ha='center', va='center', fontsize=9)
    plt.title("Positional Embeddings (p)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
