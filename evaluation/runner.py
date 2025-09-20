#!/usr/bin/env python3
"""
Usage
------
python layout_benchmark.py graphs/         # default algo = fr
python layout_benchmark.py graphs/ -a fr   # Fruchterman-Reingold
python layout_benchmark.py graphs/ -a kk   # Kamada-Kawai
python layout_benchmark.py graphs/ -a smartgd -w 4   # your GNN
"""

from __future__ import annotations

import concurrent.futures
import sys, time, random, shutil, json, os, signal, math, ast, resource
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Callable, Tuple


import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import gdMetriX

from gml2json import gml_to_json, gexf_to_json
from metrics import Metrics


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Helpers shared by every algorithm
# ──────────────────────────────────────────────────────────────────────────────
def count_crossings(G: nx.Graph, pos: Dict[int, Tuple[float, float]]) -> Tuple[int, Dict[frozenset, int]]:
    edge_counts = {frozenset(e): 0 for e in G.edges()}
    lines = {frozenset(e): LineString([pos[e[0]], pos[e[1]]]) for e in G.edges()}
    total = 0
    for e1, e2 in combinations(G.edges(), 2):
        if set(e1) & set(e2):               # share a vertex → skip
            continue
        k1, k2 = frozenset(e1), frozenset(e2)
        if lines[k1].crosses(lines[k2]):
            total += 1
            edge_counts[k1] += 1
            edge_counts[k2] += 1
    return total, edge_counts

def local_crossing_number(edge_counts: Dict[frozenset, int]) -> int:
    return max(edge_counts.values(), default=0)

def _ccw(A, B, C):
    """Return True iff the turn A→B→C is counter-clockwise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(p1, p2, p3, p4):
    """Proper intersection (excludes touching endpoints)."""
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)
            and _ccw(p1, p2, p3) != _ccw(p1, p2, p4))
def edge_crossings(G, pos):
    edges = list(G.edges())
    per_edge = {frozenset(e): 0 for e in edges}
    total = 0

    for (u1, v1), (u2, v2) in combinations(edges, 2):
        # skip adjacent edges (they share a node)
        if {u1, v1} & {u2, v2}:
            continue
        p1, p2 = pos[u1], pos[v1]
        p3, p4 = pos[u2], pos[v2]
        if segments_intersect(p1, p2, p3, p4):
            total += 1
            per_edge[frozenset((u1, v1))] += 1
            per_edge[frozenset((u2, v2))] += 1
    return total, per_edge


def get_crossing_function(G, pos, name=""):
    # total, edge_counts = count_crossings(G, pos)
    # total2, edge_counts2 = edge_crossings(G, pos)
    #crosses = gdMetriX.get_crossings(G, pos=pos)
    crosses = gdMetriX.get_crossings_quadratic(G, pos=pos)
    total3 = len(crosses)
    # lcn = local_crossing_number(edge_counts)
    # lcn2 = local_crossing_number(edge_counts2)
    per_e = {frozenset(e): 0 for e in G.edges()}
    for c in crosses:
        cstr = str(c).split("edges: ")[-1][:-1]
        edges = ast.literal_eval(cstr)
        for (u1, v1) in edges:
            per_e[frozenset((u1, v1))] += 1
    lcn3 = max(per_e.values(), default=0)
    # print(name, total, total2, total3, lcn, lcn2, lcn3)
    print(name, total3, lcn3)
    return  total3, lcn3


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Layout algorithm wrappers
#     Each returns (pos_dict, elapsed_seconds)
# ──────────────────────────────────────────────────────────────────────────────
def algo_fr(G: nx.Graph, **_):
    """NetworkX Fruchterman-Reingold"""
    t0 = time.perf_counter()
    pos = nx.fruchterman_reingold_layout(G)
    return pos, time.perf_counter() - t0

def algo_kk(G: nx.Graph, **_):
    """NetworkX Kamada-Kawai"""
    t0 = time.perf_counter()
    pos = nx.kamada_kawai_layout(G)
    return pos, time.perf_counter() - t0

# --- SmartGD GNN --------------------------------------------------------------
def algo_smartgd(G: nx.Graph, device="cpu", **_):
    """Your trained SmartGD model."""
    import torch
    from torch_geometric.data import Batch
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from smartgd.data import GraphDrawingData
    from smartgd.transformations import Compose, Center, NormalizeRotation, RescaleByStress
    from smartgd.model import Generator
    from smartgd.metrics import Crossings

    # static objects only once (module-level cache)
    if not hasattr(algo_smartgd, "_model"):
        algo_smartgd._model = Generator(
            params=Generator.Params(
                num_blocks=11,
                block_depth=3,
                block_width=8,
                block_output_dim=8,
                edge_net_depth=2,
                edge_net_width=16,
                edge_attr_dim=2,
                node_attr_dim=2,
            )
        ).to(device)
        algo_smartgd._model.load_state_dict(
            torch.load("../generator_xing_only.pt", map_location=device)
        )
        algo_smartgd._model.eval()
        algo_smartgd._canon = Compose(Center(), NormalizeRotation(), RescaleByStress())


    pos = {}
    elapsed = 0
    for comp in nx.connected_components(G):
        if len(comp) <= 1:
            n = next(iter(comp))
            pos[n] = np.zeros(2)
            continue
        compG = G.subgraph(comp).copy()
        gdd = GraphDrawingData(compG)
        gdd = GraphDrawingData.pre_transform(gdd)
        gdd = GraphDrawingData.dynamic_transform(gdd)
        gdd = GraphDrawingData.static_transform(gdd)

        batch = Batch.from_data_list([gdd]).to(device)
        edge_attr = torch.cat([batch.apsp_attr[:, None],
                               1 / batch.apsp_attr[:, None].square()], dim=-1)

        t0 = time.perf_counter()
        with torch.no_grad():
            # init_pos = batch.pos
            init_layout = nx.kamada_kawai_layout(G)
            node_order = list(compG.nodes())
            init_np = np.asarray([init_layout[n] for n in node_order], dtype=np.float32)
            init_pos = torch.as_tensor(init_np, device=device)

            init = algo_smartgd._canon(init_pos, batch.apsp_attr,
                                       batch.perm_index, batch.batch)
            pred = algo_smartgd._model(init_pos=init,
                                       edge_index=batch.perm_index,
                                       edge_attr=edge_attr,
                                       batch_index=batch.batch)
            final = algo_smartgd._canon(pred, batch.apsp_attr,
                                        batch.perm_index, batch.batch)

        elapsed += time.perf_counter() - t0
        for i, node in enumerate(compG.nodes()):
            pos[node] = final[i].cpu().numpy()
    return pos, elapsed


# ─── PPO / RL layout ────────────────────────────────────────────────────────
def algo_rlgc(G: nx.Graph, **_):
    """
    Pick the best classical layout as initial position, then improve it with
    the pretrained PPO agent.  Returns (pos_dict, elapsed_seconds).
    """
    import numpy as np
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../sng_LCN/src"))
    from pybindCode.graph_utils import compute_crossings
    from env.GraphLayoutEnv import GraphLayoutEnv
    from runners.AgentRunner import AgentRunner
    from stable_baselines3 import PPO

    RL_MODEL_PATH = "../../sng_LCN/src/Training/runs/ppo_test_cluster_Global/final_model.zip"

    # one-time global load of the RL model
    if not hasattr(algo_rlgc, "_model"):
        algo_rlgc._model = PPO.load(RL_MODEL_PATH, device="cpu")

    # ---------- 2) run the RL agent ---------------------------------------
    t0 = time.perf_counter()
    env = GraphLayoutEnv(G, opt_type="Global")
    obs, _ = env.reset()

    print("Initial global crossings:", env.best_crossings)
    print("Initial local crossings :", env.best_local_crossings)

    best_local = env.best_local_crossings
    best_global = env.best_crossings
    best_pos = env.best_pos

    max_steps = 2000
    for t in range(max_steps):
        action, _ = algo_rlgc._model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))

        if env.global_crossings < best_global:
            best_local = env.local_crossings
            best_global = env.global_crossings
            best_pos = env.pos

        if done:  # or truncated:
            break

    print("Best global crossings:", env.best_crossings)
    print("Best local crossings :", env.best_local_crossings)
    # return positions in the original node-label space
    final_pos_dict = best_pos if best_pos is not None else env.best_pos
    elapsed = time.perf_counter() - t0
    return final_pos_dict, elapsed


# ─── PPO / RL layout ────────────────────────────────────────────────────────
def algo_rllc(G: nx.Graph, **_):
    ### Same as algo_ppo but laods a different model for local crossing optimization!
    import numpy as np
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../sng_LCN/src"))
    from pybindCode.graph_utils import compute_crossings
    from env.GraphLayoutEnv import GraphLayoutEnv
    from runners.AgentRunner import AgentRunner
    from stable_baselines3 import PPO

    RL_MODEL_PATH = "../../sng_LCN/src/Training/runs/ppo_test_cluster_Local/final_model.zip"

    # one-time global load of the RL model
    if not hasattr(algo_rllc, "_model"):
        algo_rllc._model = PPO.load(RL_MODEL_PATH, device="cpu")

    # ---------- 2) run the RL agent ---------------------------------------
    t0 = time.perf_counter()
    env = GraphLayoutEnv(G, opt_type="Local")
    obs, _ = env.reset()

    print("Initial global crossings:", env.best_crossings)
    print("Initial local crossings :", env.best_local_crossings)

    best_local = env.best_local_crossings
    best_global = env.best_crossings
    best_pos = env.best_pos

    max_steps=2000
    for t in range(max_steps):
        action, _ = algo_rllc._model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))

        if env.local_crossings < best_local or (env.local_crossings == best_local and env.global_crossings < best_global):
            best_local = env.local_crossings
            best_global = env.global_crossings
            best_pos = env.pos

        if done:#  or truncated:
            break

    print("Best global crossings:", env.best_crossings)
    print("Best local crossings :", env.best_local_crossings)
    # return positions in the original node-label space
    final_pos_dict = best_pos if best_pos is not None else env.best_pos
    elapsed = time.perf_counter() - t0
    return final_pos_dict, elapsed


# --- external “edge_insertion” binary ----------------------------------------
def algo_edgeins(G: nx.Graph, *, graph_path: Path, time_limit: int = 600,
                 tmp_dir="edgeins_tmp", **_) -> Tuple[dict, float]:
    """
    Run the compiled C++ program `edge_insertion` on the *original* .gml file,
    read the produced *_out.gml*, and return the node positions.

    Parameters
    ----------
    graph_path : pathlib.Path      path to the input .gml file
    time_limit : int               seconds before SIGTERM, 5 s grace
    tmp_dir    : str | Path        where to write the *_out.gml*
    """
    import subprocess, time

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    if graph_path.suffix == ".gexf":
        tmp_gml = tmp_dir / f"{graph_path.stem}_gexf.gml"
        pos = nx.kamada_kawai_layout(G)
        for node, (x, y) in pos.items():
            G.nodes[node]["x"] = float(x)
            G.nodes[node]["y"] = float(y)
        nx.write_gml(G, tmp_gml)
        graph_path = tmp_gml

    out_gml = tmp_dir / f"{graph_path.stem}_out.gml"


    cmd = (
        "{ " + f"timeout -s SIGTERM -k 5s {time_limit} "
        f"./edge_insertion --in {graph_path} --gml {out_gml}" + " ;}"
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        lines = proc.stdout.splitlines()
        print(lines[-1])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running edge_insertion: {e}")
    elapsed = time.perf_counter() - t0



    assert os.path.exists(out_gml)
    G_out = nx.read_gml(out_gml).to_undirected()
    G_out = nx.convert_node_labels_to_integers(G_out)
    # pos = {n: (0.0, 0.0) for n in G_out.nodes()}
    pos = {}
    for n, attr in G_out.nodes(data=True):
        if "x" in attr and "y" in attr:
            x, y = attr["x"], attr["y"]
        else:
            gfx = attr.get("graphics", {})
            x, y = gfx.get("x", 0.0), gfx.get("y", 0.0)
        pos[n] = np.asarray((x, y))

    # stress = None
    # if nx.is_connected(G_out):
    #     try:
    #         apsp = dict(nx.all_pairs_shortest_path_length(G_out))
    #         stress = round(Metrics(G_out, pos, apsp).compute_stress_kruskal(), 4)
    #     except Exception as e:
    #         print(f"[!] {out_gml}: stress error → {e}")
    #
    # print(elapsed, end=" ")
    # gcn, lcn = get_crossing_function(G_out, pos, out_gml.name)
    # pos = {"GCN": gcn, "LCN": lcn, "stress" : stress}

    return pos, elapsed

# --- external “vertex_movement” binary ----------------------------------------
def algo_vermove(G: nx.Graph, *, graph_path: Path, time_limit: int = 600,
                 tmp_dir="vermove_tmp", **_) -> Tuple[dict, float]:
    """
    Run the compiled C++ program `vertex_movement` on the *original* .gml file,
    read the produced *_out.gml*, and return the node positions.

    Parameters
    ----------
    graph_path : pathlib.Path      path to the input .gml file
    time_limit : int               seconds before SIGTERM, 5 s grace
    tmp_dir    : str | Path        where to write the *_out.gml*
    """
    import subprocess, time

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    if graph_path.suffix == ".gexf":
        tmp_gml = tmp_dir / f"{graph_path.stem}_gexf.gml"
        pos = nx.kamada_kawai_layout(G)
        for node, (x, y) in pos.items():
            G.nodes[node]["x"] = float(x)
            G.nodes[node]["y"] = float(y)
        nx.write_gml(G, tmp_gml)
        graph_path = tmp_gml

    out_gml = tmp_dir / f"{graph_path.stem}_out.gml"


    cmd = (
        "{ " + f"timeout -s SIGTERM -k 5s {time_limit} "
        f"./vertex_movement --in {graph_path} --gml {out_gml}" + " ;}"
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        lines = proc.stdout.splitlines()
        print(lines[-1])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running vertex_movement: {e}")
    elapsed = time.perf_counter() - t0



    assert os.path.exists(out_gml)
    G_out = nx.read_gml(out_gml).to_undirected()
    G_out = nx.convert_node_labels_to_integers(G_out)
    # pos = {n: (0.0, 0.0) for n in G_out.nodes()}
    pos = {}
    for n, attr in G_out.nodes(data=True):
        if "x" in attr and "y" in attr:
            x, y = attr["x"], attr["y"]
        else:
            gfx = attr.get("graphics", {})
            x, y = gfx.get("x", 0.0), gfx.get("y", 0.0)
        pos[n] = np.asarray((x, y))

    # stress = None
    # if nx.is_connected(G_out):
    #     try:
    #         apsp = dict(nx.all_pairs_shortest_path_length(G_out))
    #         stress = round(Metrics(G_out, pos, apsp).compute_stress_kruskal(), 4)
    #     except Exception as e:
    #         print(f"[!] {out_gml}: stress error → {e}")
    #
    # print(elapsed, end=" ")
    # gcn, lcn = get_crossing_function(G_out, pos, out_gml.name)
    # pos = {"GCN": gcn, "LCN": lcn, "stress" : stress}

    return pos, elapsed


# --- external “edge_insertion” binary ----------------------------------------
def algo_upwards(G: nx.Graph, *, graph_path: Path, time_limit: int = 600,
                 tmp_dir="upwards_tmp", **_) -> Tuple[dict, float]:
    """
    Run the compiled C++ program `edge_insertion` on the *original* .gml file,
    read the produced *_out.gml*, and return the node positions.

    Parameters
    ----------
    graph_path : pathlib.Path      path to the input .gml file
    time_limit : int               seconds before SIGTERM, 5 s grace
    tmp_dir    : str | Path        where to write the *_out.gml*
    """
    import subprocess, time

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    out_gml = tmp_dir / f"{graph_path.stem}_out.json"

    # algorithm needs json input
    if graph_path.suffix != ".json":
        tmp_gml = tmp_dir / f"{graph_path.stem}_converted.json"
        if graph_path.suffix == ".gml":
            data = gml_to_json(graph_path)
        else:
            assert graph_path.suffix == ".gexf"
            data = gexf_to_json(graph_path)
        with open(tmp_gml, "w") as fp:
            json.dump(data, fp, indent=4)
        graph_path = tmp_gml

    cmd = (
        "{ " + f"timeout -s SIGTERM -k 5s {time_limit} "
               f"./upward_drawings_c -f {graph_path} -a random 10 1 -o {out_gml}"+ " ;}"
    )

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running edge_insertion: {e}")
    elapsed = time.perf_counter() - t0

    assert os.path.exists(out_gml)
    with open(out_gml) as fp:  # or json.loads(string)
        data = json.load(fp)
    G_out = nx.Graph()

    # 1) add nodes (with x,y attributes)
    for n in data["nodes"]:
        G_out.add_node(n["id"], x=n["x"], y=n["y"])

    # 2) add edges
    for e in data["edges"]:
        G_out.add_edge(e["source"], e["target"])

    G_out.to_undirected()
    G_out = nx.convert_node_labels_to_integers(G_out)
    pos = {n: (d["x"], d["y"]) for n, d in G_out.nodes(data=True)}

    return pos, elapsed


def algo_sgd2(G: nx.Graph, *,
        graph_path: Path | None = None,  # not used, but keeps the signature
        time_limit: int = 600,
        device: str = "cpu",
        **_,
    ):
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../SGD2"))
    import utils.weight_schedule as ws
    from gd2 import GD2
    import torch
    # --- 1.  set up a single-criterion schedule (crossings only) ----------
    max_iter = int(1e4)

    # stress + crossing from paper
    criteria_weights = dict(
        stress=ws.SmoothSteps([0, max_iter * 0.25, max_iter * 0.50],
                              [1.0, 1.0, 0.0]),
        crossings=ws.SmoothSteps([max_iter * 0.25, max_iter * 0.50, max_iter],
                                 [0.0, 0.2, 0.2])
    )
    # pure crossings
    # criteria_weights = dict(
    #     crossings = ws.SmoothSteps(
    #         [0,      max_iter*0.3, max_iter],     # iteration break-points
    #         [1.0,    1.0,         0.5],          # weights (here simply 1→0.5)
    #     )
    # )
    criteria = list(criteria_weights.keys())

    sample_sizes = dict(
        stress = 32,
        crossings = 128         # values of paper
    )

    # --- 2.  run GD-2 -------------------------------------------------------
    gd = GD2(G)

    init_dict = nx.kamada_kawai_layout(G)
    # turn that dict into a tensor in GD2’s internal order
    init_arr = np.asarray([init_dict[v] for v in list(G.nodes)], dtype=np.float32)
    init_tensor = torch.tensor(init_arr, device=gd.device, requires_grad=True)
    gd.pos = init_tensor


    t0 = time.perf_counter()
    gd.optimize(
        criteria_weights = criteria_weights,
        sample_sizes     = sample_sizes,
        evaluate         = criteria,          # only that metric
        max_iter         = max_iter,
        time_limit       = time_limit,             # seconds
        evaluate_interval= max_iter,              # no intermediate logging
        vis_interval     = -1,                    # disable plotting
        clear_output     = False,
        grad_clamp       = 20,
        optimizer_kwargs = dict(mode="SGD", lr=2),
        criteria_kwargs  = {},                    # none needed for crossings
    )
    elapsed = time.perf_counter() - t0

    # --- 3.  convert tensor -> dict ----------------------------------------
    pos_tensor = gd.pos.detach().cpu().numpy()
    pos_dict   = { n : pos_tensor[gd.k2i[n]].copy()
                   for n in gd.G.nodes() }

    return pos_dict, elapsed


def algo_oldrl(G: nx.Graph, *,
        graph_path: Path | None = None,  # not used, but keeps the signature
        time_limit: int = 600,
        device: str = "cpu",
        **_,):

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../practical/Python"))
    from main import compute_rl_layout

    # from utils.GraphVisualizer import GraphVisualizer
    # from utils.export_graph import export_graph_to_json
    # from utils.load_graph import load_graph, preprocess_graph
    # from utils.plot_graph import plot_graph
    # from utils.planar_graph import process_embedding
    # from env.graph_env import GraphEnv

    # one-time global load of the RL model
    if not hasattr(algo_oldrl, "_model"):
        from stable_baselines3 import PPO as RLagent
        RL_MODEL_PATH = "../../practical/Python/models/challenge_pp_model4.zip"
        algo_oldrl._model = RLagent.load(RL_MODEL_PATH, custom_objects={"clip_range": 0.2, "lr_schedule": 0.0003}, device=device)

    t0 = time.perf_counter()
    width = G.number_of_nodes() ** 2 # maybe choose this differently?
    height = width
    pos = compute_rl_layout(G, width=width, height=height, model_rl=algo_oldrl._model)
    elapsed = time.perf_counter() - t0

    return pos, elapsed
# --------------------------------------------------------------------------
ALGORITHMS: Dict[str, Callable] = {
    "fr"      : algo_fr,
    "kk"      : algo_kk,
    "smartgd" : algo_smartgd,
    "rlgc"    : algo_rlgc,
    "rllc"    : algo_rllc,
    "edgeins" : algo_edgeins,
    "vermove" : algo_vermove,
    "upwards" : algo_upwards,
    "sgd2"    : algo_sgd2,
    "oldrl"   : algo_oldrl,
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Per-graph worker
# ──────────────────────────────────────────────────────────────────────────────
def default_row(graph_name: str, time_limit: int) -> dict:
    return dict(instance = graph_name, solved = 0, time = time_limit, GCN = None, LCN = None, stress = None)

def evaluate_graph(gml_file: Path, layout_fun: Callable, time_limit: int, **layout_kw) -> Dict:
    name = gml_file.name
    print(f"[+] {name}")

    if not layout_fun in [algo_edgeins, algo_upwards]:
        used = resource.getrusage(resource.RUSAGE_SELF)
        cpu_used = used.ru_utime + used.ru_stime
        new_soft = math.ceil(cpu_used + time_limit + 10)
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (new_soft, hard))

    def _cpu_time_exceeded(signum, frame):
        print("Signal received")
        raise TimeoutError(f"RLIMIT_CPU exceeded ({time_limit}s)")
    signal.signal(signal.SIGXCPU, _cpu_time_exceeded)

    if gml_file.name.endswith(".gml"):
        G = nx.read_gml(gml_file).to_undirected()
    else:
        G = nx.read_gexf(gml_file).to_undirected()
    G = nx.convert_node_labels_to_integers(G)
    result = default_row(name, time_limit)
#    if not nx.is_connected(G):
#        print("Warning: Graph is not connected")
#        return result

    layout_kw |= dict(graph_path=gml_file)
    try:
        pos, elapsed = layout_fun(G, time_limit=time_limit, **layout_kw)
        result.update(solved=1, time=round(elapsed, 4))
    except Exception as e:
        print(f"[!] {name}: layout error → {e}")
        return result


    # don't timeout crossing calcualation
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (soft + 600, hard))


    if "GCN" in pos and "LCN" in pos: #sepcial cases for edge_ins, get GCN/LCN/STRESS in the function itself
        gcn = pos["GCN"]
        lcn = pos["LCN"]
    else:
        print(elapsed, end=" ")
        gcn, lcn = get_crossing_function(G, pos, gml_file.name)

    result.update(GCN=gcn, LCN=lcn)


    if nx.is_connected(G) and not ("GCN" in pos and "LCN" in pos):
        try:
            apsp = dict(nx.all_pairs_shortest_path_length(G))
            result["stress"] = round(Metrics(G, pos, apsp).compute_stress_kruskal(), 4)
        except Exception as e:
            print(f"[!] {name}: stress error → {e}")

    print(f"[-] {name}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse, os
    parser = argparse.ArgumentParser(description="Benchmark layout algorithms.")
    parser.add_argument("folder", type=Path, help="Folder with *.gml graphs")
    parser.add_argument("-a", "--algo", choices=ALGORITHMS.keys(), default="fr",
                        help="Which layout algorithm to benchmark")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(),
                        help="Parallel threads")
    parser.add_argument("-t", "--time-limit", type=int, default=600,
                        help="Per-graph wall-clock limit (seconds, unused for now)")
    parser.add_argument("-r", "--repeat", type=int, default=1,
                        help="Repeat the whole set N times")
    parser.add_argument("-l", "--list", type=Path,
                        help="Text file with graph **filenames** (one per line) "
                             "to include; if omitted every graph is used")
    parser.add_argument("--split", type=str,
                        help = "Run only a slice k/n of the job list, "
                             "e.g. 2/5 runs the second fifth.")
    args = parser.parse_args()

    # restrict to graphs from directory that appear in list
    whitelist: set[str] | None = None
    if args.list:
        if not args.list.is_file():
            sys.exit(f"List file {args.list} does not exist")
        whitelist = {line.strip().split("\\")[-1] for line in args.list.open()
                     if line.strip() and not line.startswith("#")}
        print(f"Using whitelist with {len(whitelist)} entries")

    graphs = sorted(list(args.folder.glob("*.gml")) + list(args.folder.glob("*.gexf")))
    if whitelist is not None:
        graphs = [p for p in graphs if p.name in whitelist]
    if not graphs:
        sys.exit("No .gml/.gexf files found")
    print(f"{len(graphs)} graph(s) queued for algorithm '{args.algo}'")

    job_args = graphs * args.repeat
    # job_args = job_args[:10]
    # job_args = [job_args[1]]
    # ---------------------------------------------------------------
    # optional k/n split
    # ---------------------------------------------------------------

    if args.split:
        try:
            k_str, n_str = args.split.split("/")
            k, n = int(k_str), int(n_str)
            if not (1 <= k <= n):
                raise ValueError
        except Exception:
            sys.exit("‣ --split must be of the form k/n with 1 ≤ k ≤ n")

        chunk_size = math.ceil(len(job_args) / n)
        start = (k - 1) * chunk_size
        end = min(len(job_args), k * chunk_size)
        job_args = job_args[start:end]
        print(f"Running chunk {k}/{n}: {len(job_args)} jobs {start}:{end})")

    print(len(job_args), "jobs")

    out_dir = Path("results") / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_part{k}of{n}" if args.split else ""
    if args.folder.name != "data":
        out_csv = out_dir / f"{args.folder.name}{suffix}.csv"
    else:
        if any("filtered" in part for part in args.folder.parts):
            suffix = "_filtered"+suffix

        if any("extended_BA" in part for part in args.folder.parts):
            out_csv = out_dir / f"extended_BA{suffix}.csv"
        elif any("rome" in part for part in args.folder.parts):
            out_csv = out_dir / f"rome{suffix}.csv"

    layout = ALGORITHMS[args.algo]


    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_graph, g, layout, args.time_limit): g for g in job_args}
        for fut, g in futures.items():
            try:
                results.append(fut.result(timeout=args.time_limit+15))
            except concurrent.futures.TimeoutError:
                fut.cancel()  # terminates the child process
                print(f"[!] {g.name} hard-killed after {args.time_limit}s")
                results.append(default_row(g.name, args.time_limit))

            except Exception as e:
                print(f"[!] {g.name} crashed: {e}")
                results.append(default_row(g.name, args.time_limit))

    df = pd.DataFrame(results).set_index("instance")
    df.to_csv(out_csv, sep=";", encoding="utf-8")
    print("Saved", out_csv)

if __name__ == "__main__":
    main()

# python ./runner.py ../../sng/graphs/rome/splits/data/ -a smartgd  -w 1 -r 1 -t 60 -l ../../sng/graphs/rome/splits/test.txt

## sbatch -w gpu-intel-pvc --job-name=run-graphs  --time=24:00:00 --output=runner.%j.out --error=runner.%j.out --wrap="python ./runner.py ../../sng/graphs/rome/splits/data/ -a smartgd  -w 1 -r 1 -t 900 -l ../../sng/graphs/rome/splits/test.txt  --split 1/4"
## sbatch -w gpu-intel-pvc --job-name=run-graphs  --time=24:00:00 --output=runner.%j.out --error=runner.%j.out --wrap="python ./runner.py ../../sng/graphs/extended_BA/data/ -a smartgd  -w 1 -r 1 -t 900 -l ../../sng/graphs/rome/extended_BA/test.txt  --split 1/4"
## sbatch -w gpu-intel-pvc --job-name=run-graphs  --time=24:00:00 --output=runner.%j.out --error=runner.%j.out --wrap="python ./runner.py ../../crossing_min/graphs/benchmark_small/ -a smartgd  -w 1 -r 1 -t 900"

## for index in {1..50}; do sbatch -w gpu-intel-pvc --job-name=run-graphs  --time=24:00:00 --output=runner.%j.out --error=runner.%j.out --wrap="python ./runner.py ../../sng/graphs/extended_BA/data/ -a ppo  -w 1 -r 1 -t 900 -l ../../sng/graphs/extended_BA/test.txt  --split ${index}/50"; done
