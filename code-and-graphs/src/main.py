# sb3 or Ray RLlib
import random
import time
import matplotlib
render = False
#matplotlib.use('TkAgg'); render = True
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO as Agent
from env.GraphLayoutEnv import GraphLayoutEnv
from runners.AgentRunner import AgentRunner
import networkx as nx
from util.plot_graph import plot_graph
from util.gat_prototype import GATEncoder, train_dgi_with_gat
from Training.dataloader import load_split_dataset
import cProfile
import pstats
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from Training.dataloader import load_split_dataset
from util.export_graph import export_graph_to_json  # added for planar export
from util.load_graph import load_graph



def eval_one_graph(model_path, graph_path, max_steps=2000, deterministic=True, plot=True, opt_type="Local"):
    # Load graph (supports .gml/.gexf and contest .json)
    width = height = None
    if graph_path.endswith(".gml"):
        G = nx.read_gml(graph_path)
    elif graph_path.endswith(".gexf"):
        G = nx.read_gexf(graph_path)
    elif graph_path.endswith(".json"):
        try:
            from util.load_graph import load_graph as _load_json_graph
        except Exception:
            from src.util.load_graph import load_graph as _load_json_graph
        G, width, height = _load_json_graph(graph_path)
    else:
        raise ValueError(f"Unsupported graph format: {graph_path}")
    G = nx.convert_node_labels_to_integers(G, label_attribute="original_label")
    # G = nx.erdos_renyi_graph(20, 0.2)

    # Early planar fast-path: export planar layout snapped to integer grid
    try:
        is_planar, _ = nx.check_planarity(G)
    except Exception:
        is_planar = False
    if is_planar:
        float_pos = nx.planar_layout(G)
        int_pos = GraphLayoutEnv.convert_to_integer_grid(float_pos, width, height)
        # Export only when we know canvas size
        if width is not None and height is not None:
            base = os.path.splitext(os.path.basename(graph_path))[0]
            out_name = f"{base}_exported.json"
            export_graph_to_json(G, int_pos, width, height, out_name, export_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exports')))
            print(f"Planar graph detected. Exported planar layout to exports/{out_name}.")
        else:
            print("Planar graph detected. Returned planar integer-grid layout (no export: missing width/height).")
        if plot:
            plot_graph(G, int_pos, title="NetworkX Planar Layout (Integer Grid)")
        return G, int_pos

    # Env
    if width is not None and height is not None:
        env = GraphLayoutEnv(G, width=width, height=height, opt_type=opt_type)
    else:
        env = GraphLayoutEnv(G, opt_type=opt_type)
    obs, _ = env.reset()
    initial_crossings = env.best_crossings
    initial_local_crossings = env.best_local_crossings
    print("Initial global crossings:", initial_crossings)
    print("Initial local crossings :", initial_local_crossings)

    # Load model
    model = Agent.load(model_path)

    best_local = env.best_local_crossings
    best_global = env.best_crossings
    best_pos = env.best_pos

    n_dirs = len(env.DIRS_INT)

    def to_env_action(a):
        # Convert model output to the env's continuous action vector
        if isinstance(a, (int, np.integer)):
            vec = np.zeros(n_dirs, dtype=np.float32)
            vec[int(a) % n_dirs] = 1.0
            return vec
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size == 1:
            vec = np.zeros(n_dirs, dtype=np.float32)
            vec[int(a.item()) % n_dirs] = 1.0
            return vec
        if a.size != n_dirs:
            raise ValueError(f"Unexpected action shape {a.shape}; expected ({n_dirs},) or scalar index.")
        return a

    for t in range(max_steps):
        action, _state = model.predict(obs, deterministic=deterministic)
        env_action = to_env_action(action)
        obs, reward, done, truncated, info = env.step(env_action)

        # detect improvements
        if (opt_type == "Local" and (env.local_crossings < best_local or (env.local_crossings == best_local and env.global_crossings < best_global))) \
                or (opt_type == "Global" and env.global_crossings < best_global):
            best_local = env.local_crossings
            best_global = env.global_crossings
            best_pos = env.best_pos
            print(f"[t={t}] improved -> local {best_local}, global {best_global}, reward {reward:.3f}")

        if done or truncated:
            break

    print("Initial global crossings:", initial_crossings)
    print("Initial local crossings :", initial_local_crossings)
    print("Final global crossings:", env.global_crossings)
    print("Final local crossings :", env.local_crossings)
    print("Best global crossings:", env.best_crossings)
    print("Best local crossings :", env.best_local_crossings)

    if plot:
        plot_graph(G, env.initial_pos, title="Initial layout")
        plot_graph(G, best_pos, title="Best layout")
        plot_graph(G, env.pos, title="Final layout")

    return G, env.best_pos


def test_convert_to_integer_grid():
    import networkx as nx
    import matplotlib.pyplot as plt
    from env.GraphLayoutEnv import GraphLayoutEnv

    # Generate a random graph
    G = nx.barabasi_albert_graph(random.uniform(10, 30), 1)
    # Get float positions
    float_pos = nx.kamada_kawai_layout(G)
    print("Float positions:")
    for n, p in float_pos.items():
        print(f"{n}: {p}")

    # Convert to integer grid
    int_pos = GraphLayoutEnv.convert_to_integer_grid(float_pos, 25, 25)
    print("\nInteger grid positions:")
    for n, p in int_pos.items():
        print(f"{n}: {p}")

    # Plot both layouts
    plt.figure(figsize=(10, 5))
    # Float layout
    plt.subplot(1, 2, 1)
    nx.draw(G, float_pos, with_labels=True, node_color='skyblue')
    plt.title('Float (Kamada-Kawai)')
    ax1 = plt.gca()
    xs = [p[0] for p in float_pos.values()]
    ys = [p[1] for p in float_pos.values()]
    x_min, x_max = int(np.floor(min(xs))) - 1, int(np.ceil(max(xs))) + 2
    y_min, y_max = int(np.floor(min(ys))) - 1, int(np.ceil(max(ys))) + 2
    for x in range(x_min, x_max):
        ax1.axvline(x, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
    for y in range(y_min, y_max):
        ax1.axhline(y, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    # Integer grid layout
    plt.subplot(1, 2, 2)
    nx.draw(G, int_pos, with_labels=True, node_color='orange')
    plt.title('Integer Grid')
    ax2 = plt.gca()
    xs = [p[0] for p in int_pos.values()]
    ys = [p[1] for p in int_pos.values()]
    x_min, x_max = int(np.floor(min(xs))) - 1, int(np.ceil(max(xs))) + 2
    y_min, y_max = int(np.floor(min(ys))) - 1, int(np.ceil(max(ys))) + 2
    for x in range(x_min, x_max):
        ax2.axvline(x, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
    for y in range(y_min, y_max):
        ax2.axhline(y, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    plt.show()



def test_node_position_legal():
    """
    Test the is_node_position_legal method in GraphLayoutEnv.
    Forces crossings at env init using a disjoint K3,3 subgraph to avoid the KeyError,
    but keeps the original square-based legality checks intact.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from env.GraphLayoutEnv import GraphLayoutEnv
    import numpy as np

    # Base square on nodes 0..3
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    # Disjoint K3,3 on nodes 10..15 to guarantee crossings at env init
    left = [10, 11, 12]
    right = [13, 14, 15]
    G.add_nodes_from(left + right)
    for u in left:
        for v in right:
            G.add_edge(u, v)

    G.add_edge(0, 10)  # Connect the two components

    # Initialize the environment (compute_crossings runs here; K3,3 forces crossings)
    env = GraphLayoutEnv(G, width=10, height=10, opt_type="Local")


    # Setup plot
    plt.figure(figsize=(12, 10))

    # Use an isolated test node id
    test_node = 99
    G.add_node(test_node)

    # Test 1: Legal position (center, away from square edges and nodes)
    env.pos[test_node] = np.array([0.5, 0.5], dtype=np.float64)
    is_legal = env.is_node_position_legal(test_node)
    plt.subplot(2, 2, 1)
    nx.draw(G, pos=env.pos, with_labels=True,
            node_color=['blue' if n != test_node else ('green' if is_legal else 'red') for n in G.nodes()])
    plt.title(f"Test 1: Position at center (0.5, 0.5)\nLegal: {is_legal}")

    # Test 2: Illegal position (on another node)
    env.pos[test_node] = env.pos[3]  # Same as node 3
    is_legal = env.is_node_position_legal(test_node)
    plt.subplot(2, 2, 2)
    nx.draw(G, pos=env.pos, with_labels=True,
            node_color=['blue' if n != test_node else ('green' if is_legal else 'red') for n in G.nodes()])
    plt.title(f"Test 2: Position on node 3 (0, 0)\nLegal: {is_legal}")

    # Test 3: Illegal position (on an edge)
    env.pos[test_node] = np.array([env.pos[10][0] + 1, env.pos[10][1] + 1], dtype=np.int32)  # same height as 15, x = 15.x - 1
    env._sync_pos_arrays()
    env._update_rtree()
    is_legal = env.is_node_position_legal(10)
    plt.subplot(2, 2, 3)
    nx.draw(G, pos=env.pos, with_labels=True,
            node_color=['blue' if n != test_node else ('green' if is_legal else 'red') for n in G.nodes()])
    plt.title(f"Test 3: Position on edge (0.5, 0)\nLegal: {is_legal}")

    # Test 4: Illegal position (near an edge, within tolerance)
    env.pos[test_node] = np.array([env.pos[15][0] - 1, env.pos[10][1] - 1], dtype=np.float32)
    is_legal = env.is_node_position_legal(test_node)
    plt.subplot(2, 2, 4)
    nx.draw(G, pos=env.pos, with_labels=True,
            node_color=['blue' if n != test_node else ('green' if is_legal else 'red') for n in G.nodes()])
    plt.title(f"Test 4: Position near edge (0.5, 0.001)\nLegal: {is_legal}")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Launch the UI instead of running tests
    from ui.app import main as ui_main

    # opt_type = "Local"
    # eval_one_graph(f"Training/runs/ppo_test_cluster_{opt_type}/int_grid_model.zip",
    #                # "../graphs/rome_filtered/splits/data/grafo3451.43.gml",
    #                "../graphs/GD-Test/test-6.json",
    #                # "../graphs/extended_BA_filtered/data/ba_03_n94_m3.gml",
    #                opt_type=opt_type,
    #                max_steps=2000,
    #                )

    ui_main()





