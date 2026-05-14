import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from Training.train_repair_ppo import ShortlistNodeMoveWrapper, load_base_cfg
from env import create_graph_layout_env
from util.load_graph import load_graph as load_json_graph


def choose_eval_params(num_nodes, num_edges, default_horizon, default_restarts, default_jitter, adaptive):
    if not adaptive:
        return {
            "horizon": int(default_horizon),
            "restarts": int(default_restarts),
            "jitter": int(default_jitter),
        }

    if num_nodes <= 250 and num_edges <= 1000:
        return {"horizon": 256, "restarts": 7, "jitter": 4}
    if num_nodes <= 600 and num_edges <= 3500:
        return {"horizon": 192, "restarts": 5, "jitter": 4}
    if num_nodes <= 3000 and num_edges <= 8000:
        return {"horizon": 128, "restarts": 3, "jitter": 4}
    return {"horizon": 96, "restarts": 2, "jitter": 2}


def append_jsonl(path_str, payload):
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def evaluate_graph(model, graph_path, shortlist_size, best_local_bonus, local_weight, sizemax_weight, global_weight,
                   default_horizon, default_restarts, default_jitter, adaptive, seed):
    graph_path = Path(graph_path)
    graph, _, _ = load_json_graph(str(graph_path.resolve()))
    params = choose_eval_params(
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        default_horizon=default_horizon,
        default_restarts=default_restarts,
        default_jitter=default_jitter,
        adaptive=adaptive,
    )

    cfg = load_base_cfg(
        step_limit=params["horizon"],
        reset_threshold=64,
        local_weight=local_weight,
        sizemax_weight=sizemax_weight,
        global_weight=global_weight,
    )
    cfg["experiment"] = {
        "shortlist_size": int(shortlist_size),
        "best_local_bonus": float(best_local_bonus),
        "eval_outer_restarts": int(params["restarts"]),
        "eval_restart_perturb_steps": int(params["jitter"]),
    }

    t0 = time.perf_counter()
    env_obj = create_graph_layout_env(graph, config=cfg)
    wrapper = ShortlistNodeMoveWrapper(
        env_obj,
        shortlist_size=int(shortlist_size),
        best_local_bonus=float(best_local_bonus),
        perturb_steps=0,
        attempts=0,
        seed=int(seed),
    )
    obs, _ = wrapper.reset(seed=int(seed))
    initial_local = int(wrapper.base.initial_local_crossings)
    initial_global = int(wrapper.base.initial_crossings)

    try:
        for restart_idx in range(int(params["restarts"])):
            for _ in range(int(params["horizon"])):
                action_mask = wrapper.action_masks()
                action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                obs, _, done, truncated, _ = wrapper.step(action)
                if done or truncated:
                    break
            if restart_idx + 1 < int(params["restarts"]) and int(wrapper.base.best_crossings) > 0:
                wrapper.prepare_outer_restart(jitter_steps=int(params["jitter"]))
                obs = wrapper._stack_observation()

        best_local = int(wrapper.base.best_local_crossings)
        best_global = int(wrapper.base.best_crossings)
        runtime_sec = float(time.perf_counter() - t0)
    finally:
        wrapper.close()

    local_improvement = int(initial_local - best_local)
    global_improvement = int(initial_global - best_global)
    local_improvement_ratio = None
    if initial_local > 0:
        local_improvement_ratio = float(local_improvement / initial_local)
    global_improvement_ratio = None
    if initial_global > 0:
        global_improvement_ratio = float(global_improvement / initial_global)

    return {
        "graph": graph_path.name,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "horizon": int(params["horizon"]),
        "restarts": int(params["restarts"]),
        "jitter": int(params["jitter"]),
        "initial_local": initial_local,
        "best_local": best_local,
        "local_improvement": local_improvement,
        "local_improvement_ratio": local_improvement_ratio,
        "initial_global": initial_global,
        "best_global": best_global,
        "global_improvement": global_improvement,
        "global_improvement_ratio": global_improvement_ratio,
        "runtime_sec": runtime_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate repair-PPO on graph files one at a time.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--graph-dir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*.json")
    parser.add_argument("--include-list-file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--best-local-bonus", type=float, default=5.0)
    parser.add_argument("--local-weight", type=float, default=10.0)
    parser.add_argument("--sizemax-weight", type=float, default=0.1)
    parser.add_argument("--global-weight", type=float, default=0.05)
    parser.add_argument("--standard-horizon", type=int, default=256)
    parser.add_argument("--restarts", type=int, default=7)
    parser.add_argument("--restart-jitter-steps", type=int, default=4)
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default=None)
    args = parser.parse_args()

    model_path = ROOT / args.model_path if not Path(args.model_path).is_absolute() else Path(args.model_path)
    graph_dir = ROOT / args.graph_dir if not Path(args.graph_dir).is_absolute() else Path(args.graph_dir)
    output_csv_path = ROOT / args.output_csv if not Path(args.output_csv).is_absolute() else Path(args.output_csv)
    output_json_path = ROOT / args.output_json if not Path(args.output_json).is_absolute() else Path(args.output_json)
    output_jsonl_path = None
    if args.output_jsonl:
        output_jsonl_path = ROOT / args.output_jsonl if not Path(args.output_jsonl).is_absolute() else Path(args.output_jsonl)
    include_names = None
    if args.include_list_file:
        list_path = ROOT / args.include_list_file if not Path(args.include_list_file).is_absolute() else Path(args.include_list_file)
        include_names = {
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    graph_paths = sorted(graph_dir.glob(args.glob))
    if include_names is not None:
        graph_paths = [path for path in graph_paths if path.name in include_names]
    if args.limit is not None:
        graph_paths = graph_paths[: int(args.limit)]

    model = MaskablePPO.load(str(model_path), device="cpu")
    was_training = bool(getattr(model.policy, "training", False))
    model.policy.set_training_mode(False)

    results = []
    failures = []
    start = time.perf_counter()
    out_csv_path = output_csv_path
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "graph",
            "nodes",
            "edges",
            "horizon",
            "restarts",
            "jitter",
            "initial_local",
            "best_local",
            "local_improvement",
            "local_improvement_ratio",
            "initial_global",
            "best_global",
            "global_improvement",
            "global_improvement_ratio",
            "runtime_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for idx, graph_path in enumerate(graph_paths):
                event = {"event": "start_graph", "graph": graph_path.name, "index": idx}
                print(json.dumps(event), flush=True)
                append_jsonl(str(output_jsonl_path) if output_jsonl_path else None, event)
                try:
                    result = evaluate_graph(
                        model=model,
                        graph_path=graph_path,
                        shortlist_size=args.shortlist_size,
                        best_local_bonus=args.best_local_bonus,
                        local_weight=args.local_weight,
                        sizemax_weight=args.sizemax_weight,
                        global_weight=args.global_weight,
                        default_horizon=args.standard_horizon,
                        default_restarts=args.restarts,
                        default_jitter=args.restart_jitter_steps,
                        adaptive=bool(args.adaptive),
                        seed=args.seed_base + idx,
                    )
                    writer.writerow(result)
                    f.flush()
                    results.append(result)
                    graph_event = {"event": "graph_result", "payload": result}
                    print(json.dumps(graph_event), flush=True)
                    append_jsonl(str(output_jsonl_path) if output_jsonl_path else None, graph_event)
                except Exception as exc:
                    failure = {
                        "graph": graph_path.name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    fail_event = {"event": "graph_failure", "payload": failure}
                    print(json.dumps(fail_event), flush=True)
                    append_jsonl(str(output_jsonl_path) if output_jsonl_path else None, fail_event)
        finally:
            model.policy.set_training_mode(was_training)

    summary = {
        "graphs_requested": len(graph_paths),
        "graphs_completed": len(results),
        "graphs_failed": len(failures),
        "failures": failures,
        "mean_best_local": float(np.mean([row["best_local"] for row in results])) if results else None,
        "mean_initial_local": float(np.mean([row["initial_local"] for row in results])) if results else None,
        "mean_local_improvement": float(np.mean([row["local_improvement"] for row in results])) if results else None,
        "mean_best_global": float(np.mean([row["best_global"] for row in results])) if results else None,
        "mean_initial_global": float(np.mean([row["initial_global"] for row in results])) if results else None,
        "mean_global_improvement": float(np.mean([row["global_improvement"] for row in results])) if results else None,
        "total_runtime_sec": float(time.perf_counter() - start),
        "config": {
            "model_path": str(model_path),
            "graph_dir": str(graph_dir),
            "glob": args.glob,
            "adaptive": bool(args.adaptive),
            "standard_horizon": int(args.standard_horizon),
            "restarts": int(args.restarts),
            "restart_jitter_steps": int(args.restart_jitter_steps),
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_jsonl(str(output_jsonl_path) if output_jsonl_path else None, {"event": "summary", "payload": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
