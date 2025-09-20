import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from gat_prototype import build_data_from_graph, train_dgi_with_gat, DGIModel

# build two identical graphs (same structure & labels)
G1 = nx.barabasi_albert_graph(20, 2, seed=42)
G2 = nx.barabasi_albert_graph(20, 2, seed=42)

# train for just a couple of epochs to get a quick demo model
model, _ = train_dgi_with_gat([G1, G2], epochs=2, lr=1e-3, batch_size=2)

# embed without noise: should be nearly identical for each node pair
data1 = build_data_from_graph(G1)
data2 = build_data_from_graph(G2)
z1_clean = model.embed(data1, apply_noise=False)
z2_clean = model.embed(data2, apply_noise=False)

# embed with desired noise_std (default 0.01): should jitter a bit
z1_noisy = model.embed(data1, apply_noise=True)
z2_noisy = model.embed(data2, apply_noise=True)

# compute average cosine similarity across corresponding nodes
sim_clean = np.diag(cosine_similarity(z1_clean.cpu(), z2_clean.cpu())).mean()
sim_noisy = np.diag(cosine_similarity(z1_noisy.cpu(), z2_noisy.cpu())).mean()

print(f"Avg cosine (clean embeddings): {sim_clean:.4f}")
print(f"Avg cosine (with noise):      {sim_noisy:.4f}")
