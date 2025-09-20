import os
import copy
import numpy as np
import pandas as pd
import networkx as nx
from stable_baselines3 import PPO

from Training.dataloader import load_split_dataset
from env.GraphLayoutEnv import GraphLayoutEnv
from runners.AgentRunner import AgentRunner
from pybindCode.graph_utils import compute_crossings

# classical layout functions
LAYOUT_FUNCS = {
    "spring": nx.spring_layout,
    "random": nx.random_layout,
    "circular": nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "spectral": nx.spectral_layout,
    "shell": nx.shell_layout,
}


RL_MODEL_PATH = "src/models/ppo_it"
rl_model = PPO.load(RL_MODEL_PATH)

def benchmark_layouts(split_type="mini", output_csv="results/layout_benchmark.csv"):
    from pybindCode.graph_utils import compute_crossings

    dataset = load_split_dataset(split_type)
    n_graphs = len(dataset)
    records = []


    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for idx in range(n_graphs):
        G = dataset[idx]          # explicit __getitem__
        graph_id = f"{split_type}_{idx}"
        n, m = G.number_of_nodes(), G.number_of_edges()
        print(f"[{idx+1}/{n_graphs}] {graph_id}  n={n}  m={m}")

        # build a fixed node→index mapping
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # 1) find best classical layout
        best_init_name = None
        best_init_pos = None
        best_init_cross = float("inf")

        for name, func in LAYOUT_FUNCS.items():
            if name == "ppo_agent":
                continue
            pos_dict = func(G)
            node_order = sorted(G.nodes())
            pos_arr = np.array([pos_dict[n] for n in node_order])
            edges_idx = [(node_order.index(u), node_order.index(v)) for u, v in G.edges()]
            _, _, cross = compute_crossings(pos_arr, edges_idx)

            if cross < best_init_cross:
                best_init_cross = cross
                best_init_name = name
                best_init_pos = {n: pos_dict[n].copy() for n in nodes}

        print(f"  → best init: {best_init_name} with {best_init_cross} crossings")

        env = GraphLayoutEnv(G)
        obs, _ = env.reset()
        initial_crossings = env.best_crossings
        print(f"initial_crossings = {initial_crossings}")

        runner = AgentRunner(rl_model, env)
        runner.run(render=False)
        final_best = env.best_crossings
        print(f"  → after RL, best_crossings = {final_best}")

        record = {
            "graph_id": graph_id,
            "initial_algorithm": best_init_name,
            "initial_crossings": initial_crossings,
            "final_best_crossings": final_best,
            "improvement": initial_crossings - final_best,
            "num_nodes": n,
            "num_edges": m,
        }
        records.append(record)

        # Write the current record to the CSV file
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"→ wrote {output_csv}")


if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, "src")
    os.chdir(src_dir)
    benchmark_layouts(
        split_type="test",
        output_csv="results/layout_benchmark.csv"
    )
