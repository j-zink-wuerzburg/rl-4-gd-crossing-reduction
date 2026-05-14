import os

# Cap native math threads per worker (override via env if needed)
NUM_THREADS = int(os.environ.get("SNG_WORKER_NUM_THREADS", "1"))
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    # don't overwrite if user provided a smaller value externally
    if var not in os.environ:
        os.environ[var] = str(NUM_THREADS)
# Reduce OpenMP verbosity/affinity noise
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_SETTINGS", "0")

import numpy as np  # noqa: E402
from stable_baselines3 import PPO as Agent  # noqa: E402
import networkx as nx  # noqa: E402

# Support running from repo root or from within src
try:  # noqa: E402
    from env.GraphLayoutEnv import GraphLayoutEnv  # type: ignore
    from util.load_graph import load_graph  # type: ignore
    from util.export_graph import export_graph_to_json  # type: ignore
except Exception:  # noqa: E402
    from src.env.GraphLayoutEnv import GraphLayoutEnv  # type: ignore
    from src.util.load_graph import load_graph  # type: ignore
    from src.util.export_graph import export_graph_to_json  # type: ignore


def _to_env_action(a, n_dirs: int):
    """Map model output to environment action vector (one-hot over n_dirs)."""
    if isinstance(a, (int, np.integer)):
        vec = np.zeros(n_dirs, dtype=np.float32)
        vec[int(a) % n_dirs] = 1.0
        return vec
    arr = np.asarray(a, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        vec = np.zeros(n_dirs, dtype=np.float32)
        vec[int(arr.item()) % n_dirs] = 1.0
        return vec
    if arr.size != n_dirs:
        raise ValueError(f"Unexpected action shape {arr.shape}; expected ({n_dirs},) or scalar index.")
    return arr


def run_graph(file_path: str,
              export_dir: str,
              model_path: str,
              opt_type: str,
              max_steps: int,
              out_queue):
    """
    Worker entry: process a single graph file and report progress via out_queue.
    Sends tuples (kind, file_path, payload) where kind in {"progress", "finished", "finished_planar", "error"}.
    """
    try:
        # Torch thread control must happen in-process
        try:
            import torch
            # Keep per-worker CPU usage bounded to avoid oversubscription across processes
            torch.set_num_threads(max(1, NUM_THREADS))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        # Load graph
        G, width, height = load_graph(file_path)

        # Fast path: if graph is planar, export NetworkX's planar layout snapped to integer grid
        try:
            is_planar, _ = nx.check_planarity(G)
        except Exception:
            is_planar = False
        if is_planar:
            try:
                float_pos = nx.planar_layout(G)
                int_pos = GraphLayoutEnv.convert_to_integer_grid(float_pos, width, height)
                base = os.path.splitext(os.path.basename(file_path))[0]
                out_name = f"{base}_exported.json"
                export_graph_to_json(G, int_pos, width, height, out_name, export_dir=export_dir)
                out_path = os.path.join(export_dir, out_name)
                try:
                    out_queue.put(("finished_planar", file_path, out_path))
                except Exception:
                    pass
                return
            except Exception:
                # If anything about planar export fails, fall back to RL path
                pass

        # Create env and load model lazily in worker
        env = GraphLayoutEnv(G, width=width, height=height, opt_type=opt_type)
        model = Agent.load(model_path)

        # Derive action size from env
        try:
            n_dirs = len(env.DIRS_INT)
        except Exception:
            n_dirs = 8  # fallback

        obs, _ = env.reset()
        steps = 0
        done = False
        truncated = False

        while not done and not truncated and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            env_action = _to_env_action(action, n_dirs)
            obs, _reward, done, truncated, _info = env.step(env_action)
            steps += 1
            if (steps % 10) == 0:
                try:
                    out_queue.put(("progress", file_path, steps))
                except Exception:
                    pass

        base = os.path.splitext(os.path.basename(file_path))[0]
        out_name = f"{base}_exported.json"
        export_graph_to_json(G, env.best_pos, width, height, out_name, export_dir=export_dir)
        out_path = os.path.join(export_dir, out_name)
        try:
            out_queue.put(("finished", file_path, out_path))
        except Exception:
            pass
    except Exception as e:
        try:
            out_queue.put(("error", file_path, str(e)))
        except Exception:
            pass
