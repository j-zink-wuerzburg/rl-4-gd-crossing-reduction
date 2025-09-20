
# sb3 or Ray RLlib
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



def eval_one_graph(model_path, graph_path, max_steps=2000, deterministic=True, plot=True, opt_type="Local"):
    # Load graph
    if graph_path.endswith(".gml"):
        G = nx.read_gml(graph_path)
    else:
        G = nx.read_gexf(graph_path)
    G = nx.convert_node_labels_to_integers(G, label_attribute="original_label")

    # Env
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

    for t in range(max_steps):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(int(action))

        # detect improvements
        if (opt_type == "Local" and (env.local_crossings < best_local or env.local_crossings == best_local and env.global_crossings < best_local)) \
                or (opt_type == "Global" and env.global_crossings < best_global):
            best_local = env.local_crossings
            best_global = env.global_crossings
            best_pos = env.best_pos
            print(f"[t={t}] improved -> local {best_local}, global {best_global}, reward {reward:.3f}")

        if done:# or truncated:
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


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # main()

    opt_type = "Local"
    # opt_type = "Global"
    eval_one_graph(f"Training/runs/ppo_test_cluster_{opt_type}/final_model.zip",
                   "../graphs/rome_filtered/splits/data/grafo3451.43.gml",
                   # "../graphs/extended_BA_filtered/data/ba_03_n94_m3.gml",
                   opt_type=opt_type,
                   max_steps=20000,
                   )

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)



