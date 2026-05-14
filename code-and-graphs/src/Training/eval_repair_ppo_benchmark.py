import argparse
import csv
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from Training.train_repair_ppo import ShortlistNodeMoveWrapper, load_base_cfg
from env import create_graph_layout_env


def _load_checkpoint_config(model_path):
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    candidates = []
    if path.is_file():
        candidates.append(path.parent / "config.json")
    else:
        candidates.append(path / "config.json")
        candidates.append(path / "seed_123" / "config.json")
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def read_graph_rows(csv_path, limit=None):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for idx, row in enumerate(reader):
            if limit is not None and idx >= int(limit):
                break
            rows.append(row)
    return rows


def read_graph_names(list_path, limit=None):
    names = [line.strip() for line in Path(list_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return names[:limit] if limit is not None else names


def load_rome_graph(name):
    path = ROOT / "graphs" / "rome_filtered" / "splits" / "data" / Path(name).name
    if not path.exists():
        return None
    g_orig = nx.read_gml(path)
    return nx.convert_node_labels_to_integers(g_orig, label_attribute="original_label")


def _stack_obs_batch(obs_list):
    keys = obs_list[0].keys()
    return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in keys}


def _run_rollout_batch(model, wrappers, obs_list, horizon, deterministic):
    done = np.zeros(len(wrappers), dtype=bool)
    truncated = np.zeros(len(wrappers), dtype=bool)
    steps = np.zeros(len(wrappers), dtype=np.int32)
    while True:
        active = [idx for idx in range(len(wrappers)) if steps[idx] < horizon and not (done[idx] or truncated[idx])]
        if not active:
            break
        batch_obs = _stack_obs_batch([obs_list[idx] for idx in active])
        batch_masks = np.stack([wrappers[idx].action_masks() for idx in active], axis=0)
        actions, _ = model.predict(batch_obs, action_masks=batch_masks, deterministic=deterministic)
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        for local_idx, wrapper_idx in enumerate(active):
            obs, _, done_flag, truncated_flag, _ = wrappers[wrapper_idx].step(actions[local_idx])
            obs_list[wrapper_idx] = obs
            done[wrapper_idx] = bool(done_flag)
            truncated[wrapper_idx] = bool(truncated_flag)
            steps[wrapper_idx] += 1
    return obs_list


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for repair-PPO models.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmark-csv", type=str, default="rome_filtered_edge_local.csv")
    parser.add_argument("--graph-list-file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compare-column", type=str, default="LCN")
    parser.add_argument("--metric", choices=["local", "global"], default=None)
    parser.add_argument("--standard-horizon", type=int, default=256)
    parser.add_argument("--restarts", type=int, default=7)
    parser.add_argument("--restart-jitter-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--best-local-bonus", type=float, default=5.0)
    parser.add_argument("--local-weight", type=float, default=10.0)
    parser.add_argument("--sizemax-weight", type=float, default=0.1)
    parser.add_argument("--global-weight", type=float, default=0.05)
    parser.add_argument("--optimization-goal", choices=["local", "global"], default=None)
    parser.add_argument(
        "--node-selection-strategy",
        choices=["random", "heuristic", "heuristic_new", "heuristic_global"],
        default=None,
    )
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    if args.graph_list_file:
        graph_names = read_graph_names(args.graph_list_file, limit=args.limit)
        csv_rows = None
    else:
        csv_rows = read_graph_rows(args.benchmark_csv, limit=args.limit)
        graph_names = [row["instance"] for row in csv_rows]

    compare_by_name = {}
    if csv_rows is not None:
        compare_by_name = {row["instance"]: row for row in csv_rows}
    else:
        all_rows = read_graph_rows(args.benchmark_csv)
        compare_by_name = {row["instance"]: row for row in all_rows}

    checkpoint_cfg = _load_checkpoint_config(args.model_path)
    optimization_goal = args.optimization_goal
    if optimization_goal is None and checkpoint_cfg is not None:
        optimization_goal = checkpoint_cfg.get("env", {}).get(
            "optimization_goal",
            checkpoint_cfg.get("experiment", {}).get("optimization_goal"),
        )
    if optimization_goal is None:
        optimization_goal = "global" if str(args.compare_column).upper() == "GCN" else "local"

    node_selection_strategy = args.node_selection_strategy
    if node_selection_strategy is None and checkpoint_cfg is not None:
        node_selection_strategy = checkpoint_cfg.get("env", {}).get("node_selection_strategy")

    cfg = load_base_cfg(
        step_limit=args.standard_horizon,
        reset_threshold=64,
        local_weight=args.local_weight,
        sizemax_weight=args.sizemax_weight,
        global_weight=args.global_weight,
        optimization_goal=optimization_goal,
        node_selection_strategy=node_selection_strategy,
        base_config=checkpoint_cfg,
    )
    cfg["experiment"] = {
        "shortlist_size": args.shortlist_size,
        "best_local_bonus": args.best_local_bonus,
        "optimization_goal": optimization_goal,
        "eval_outer_restarts": args.restarts,
        "eval_restart_perturb_steps": args.restart_jitter_steps,
    }
    metric = args.metric or ("global" if str(args.compare_column).upper() == "GCN" else "local")

    model = MaskablePPO.load(str(ROOT / args.model_path if not Path(args.model_path).is_absolute() else args.model_path), device="cpu")
    was_training = bool(getattr(model.policy, "training", False))
    model.policy.set_training_mode(False)

    rows = []
    missing = []
    start = time.perf_counter()
    try:
        out_csv_path = Path(args.output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "graph",
                "compare_value",
                "model_local_initial",
                "model_local",
                "model_global_best",
                "model_gap_to_compare",
                "model_better",
                "model_equal",
                "model_worse",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for start_idx in range(0, len(graph_names), int(args.batch_size)):
                batch_names = graph_names[start_idx:start_idx + int(args.batch_size)]
                wrappers = []
                obs_list = []
                batch_meta = []
                for offset, name in enumerate(batch_names):
                    graph = load_rome_graph(name)
                    if graph is None:
                        missing.append(name)
                        continue
                    compare_value = float(compare_by_name[name][args.compare_column])
                    env_obj = create_graph_layout_env(graph, config=cfg)
                    wrapper = ShortlistNodeMoveWrapper(
                        env_obj,
                        shortlist_size=cfg["experiment"]["shortlist_size"],
                        best_local_bonus=cfg["experiment"]["best_local_bonus"],
                        optimization_goal=cfg["experiment"]["optimization_goal"],
                        perturb_steps=0,
                        attempts=0,
                        seed=args.seed_base + start_idx + offset,
                    )
                    obs, _ = wrapper.reset(seed=args.seed_base + start_idx + offset)
                    wrappers.append(wrapper)
                    obs_list.append(obs)
                    batch_meta.append(
                        {
                            "graph": name,
                            "compare_value": compare_value,
                            "initial_local": int(wrapper.base.initial_local_crossings),
                            "best_global": int(wrapper.base.best_crossings),
                            "best_local": int(wrapper.base.best_local_crossings),
                        }
                    )

                for restart_idx in range(max(1, int(args.restarts))):
                    if wrappers:
                        obs_list = _run_rollout_batch(
                            model,
                            wrappers,
                            obs_list,
                            int(args.standard_horizon),
                            bool(args.deterministic),
                        )
                    for meta_idx, wrapper in enumerate(wrappers):
                        batch_meta[meta_idx]["best_global"] = min(batch_meta[meta_idx]["best_global"], int(wrapper.base.best_crossings))
                        batch_meta[meta_idx]["best_local"] = min(batch_meta[meta_idx]["best_local"], int(wrapper.base.best_local_crossings))
                    if restart_idx + 1 < int(args.restarts):
                        for meta_idx, wrapper in enumerate(wrappers):
                            if batch_meta[meta_idx]["best_global"] > 0:
                                wrapper.prepare_outer_restart(jitter_steps=int(args.restart_jitter_steps))
                                obs_list[meta_idx] = wrapper._stack_observation()

                for meta_idx, wrapper in enumerate(wrappers):
                    rec = {
                        "graph": batch_meta[meta_idx]["graph"],
                        "compare_value": batch_meta[meta_idx]["compare_value"],
                        "model_local_initial": float(batch_meta[meta_idx]["initial_local"]),
                        "model_local": float(batch_meta[meta_idx]["best_local"]),
                        "model_global_best": float(batch_meta[meta_idx]["best_global"]),
                        "model_gap_to_compare": float(
                            (batch_meta[meta_idx]["best_global"] if metric == "global" else batch_meta[meta_idx]["best_local"])
                            - batch_meta[meta_idx]["compare_value"]
                        ),
                        "model_better": bool(
                            (batch_meta[meta_idx]["best_global"] if metric == "global" else batch_meta[meta_idx]["best_local"])
                            < batch_meta[meta_idx]["compare_value"]
                        ),
                        "model_equal": bool(
                            (batch_meta[meta_idx]["best_global"] if metric == "global" else batch_meta[meta_idx]["best_local"])
                            == batch_meta[meta_idx]["compare_value"]
                        ),
                        "model_worse": bool(
                            (batch_meta[meta_idx]["best_global"] if metric == "global" else batch_meta[meta_idx]["best_local"])
                            > batch_meta[meta_idx]["compare_value"]
                        ),
                    }
                    writer.writerow(rec)
                    rows.append(rec)
                    wrapper.close()
                f.flush()
    finally:
        model.policy.set_training_mode(was_training)

    compare_vals = [r["compare_value"] for r in rows]
    model_vals = [r["model_global_best"] if metric == "global" else r["model_local"] for r in rows]
    summary = {
        "graphs_requested": len(graph_names),
        "graphs_evaluated": len(rows),
        "graphs_missing": len(missing),
        "missing_graphs": missing,
        "mean_model_selected": float(np.mean(model_vals)) if model_vals else None,
        "mean_model_local": float(np.mean([r["model_local"] for r in rows])) if rows else None,
        "mean_model_global": float(np.mean([r["model_global_best"] for r in rows])) if rows else None,
        "mean_compare_value": float(np.mean(compare_vals)) if compare_vals else None,
        "median_model_local": float(np.median([r["model_local"] for r in rows])) if rows else None,
        "median_model_global": float(np.median([r["model_global_best"] for r in rows])) if rows else None,
        "median_compare_value": float(np.median(compare_vals)) if compare_vals else None,
        "better_count": int(sum(r["model_better"] for r in rows)),
        "equal_count": int(sum(r["model_equal"] for r in rows)),
        "worse_count": int(sum(r["model_worse"] for r in rows)),
        "mean_gap_model_minus_compare": float(np.mean([r["model_gap_to_compare"] for r in rows])) if rows else None,
        "model_optimal1_count": int(sum(r["model_local"] == 1 for r in rows)),
        "runtime_sec": float(time.perf_counter() - start),
        "config": {
            "model_path": args.model_path,
            "benchmark_csv": args.benchmark_csv,
            "graph_list_file": args.graph_list_file,
            "limit": args.limit,
            "compare_column": args.compare_column,
            "metric": metric,
            "optimization_goal": optimization_goal,
            "node_selection_strategy": cfg["env"].get("node_selection_strategy"),
            "standard_horizon": args.standard_horizon,
            "restarts": args.restarts,
            "restart_jitter_steps": args.restart_jitter_steps,
            "batch_size": args.batch_size,
            "deterministic": bool(args.deterministic),
        },
    }
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
