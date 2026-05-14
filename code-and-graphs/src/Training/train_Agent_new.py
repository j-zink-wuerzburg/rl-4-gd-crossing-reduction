import os, argparse, random, numpy as np, torch, sys
from locale import normalize

from stable_baselines3 import PPO as Agent
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import set_random_seed


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# os.chdir(parent_dir)
sys.path.append(parent_dir)  # Ensure the parent directory is in the Python path

from env.GraphLayoutEnv import GraphLayoutEnv
from Training.dataloader import load_split_dataset
import networkx as nx
import gymnasium as gym


def _read_graph(path: str):
    """Return a dict with graph and optional width/height inferred from file extension."""
    if path.endswith(".gml"):
        G = nx.read_gml(path)
        return {"graph": nx.convert_node_labels_to_integers(G, label_attribute="original_label"), "width": None, "height": None}
    elif path.endswith(".gexf"):
        G = nx.read_gexf(path)
        return {"graph": nx.convert_node_labels_to_integers(G, label_attribute="original_label"), "width": None, "height": None}
    elif path.endswith(".json"):
        try:
            from util.load_graph import load_graph as _load_json_graph
        except Exception:
            from src.util.load_graph import load_graph as _load_json_graph
        G, width, height = _load_json_graph(path)
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_label")
        return {"graph": G, "width": width, "height": height}
    else:
        raise ValueError(f"Unsupported graph format: {path}")


def mutate_graph(G: nx.Graph, edge_swap_frac: float = 0.05, max_frac: float = 0.2) -> nx.Graph:
    """
    Degree-preserving structural mutation using double_edge_swap.
    - edge_swap_frac: fraction of edges to rewire per mutation (clipped by max_frac)
    - max_frac: upper bound on fraction of edges to attempt swapping
    Safe for small graphs; returns a copied graph if mutation not applicable.
    """
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)
    H = G.copy()
    m = H.number_of_edges()
    # If too few edges, return unmodified
    if m < 4:
        return H
    frac = min(max(edge_swap_frac, 0.0), max_frac)
    nswap = max(1, int(round(m * frac)))
    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=nswap * 10)
    except Exception:
        # On failure, just return the original copy
        return H
    return H


class Sampler:
    def __init__(self, easy_graphs, hard_graphs, p_hard=0.1):
        self.easy = list(easy_graphs)
        self.hard = list(hard_graphs)
        self.p_hard = p_hard
        self.i_e = 0
        self.i_h = 0
        random.shuffle(self.easy)
        random.shuffle(self.hard)

    def set_p_hard(self, p):
        self.p_hard = float(np.clip(p, 0.0, 1.0))

    def sample(self):
        assert self.easy and self.hard
        is_hard = False
        if random.random() < self.p_hard:
            p = self.hard[self.i_h % len(self.hard)]
            self.i_h += 1
            is_hard = True
        else:
            p = self.easy[self.i_e % len(self.easy)]
            self.i_e += 1
        info = _read_graph(p)
        if is_hard:
            # Mutate only the hard graphs to increase variety
            info["graph"] = mutate_graph(info["graph"])  # width/height unchanged
        return info

def make_env(rank, seed, sampler, max_steps, opt_type):
    def _thunk():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        if rank == 0:
            print(f"[Main PID] {os.getpid()} creating workers…", flush=True)
        print(f"[Worker {rank}] PID={os.getpid()}", flush=True)
        info = sampler.sample()
        G, W, H = info["graph"], info.get("width"), info.get("height")
        if W is not None and H is not None:
            env = GraphLayoutEnv(graph=G, width=W, height=H, opt_type=opt_type)
        else:
            env = GraphLayoutEnv(graph=G, opt_type=opt_type)

        # rotate a new graph on every reset
        orig_reset = env.reset
        def reset_with_graph(**kwargs):
            info2 = sampler.sample()
            G2, W2, H2 = info2["graph"], info2.get("width"), info2.get("height")
            return orig_reset(Graph=G2, Width=W2, Height=H2, **kwargs)
        env.reset = reset_with_graph

        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env.reset(seed=seed + rank)
        return env
    return _thunk

def make_eval_env(seed, graphs, opt_type):
    fixed = graphs[:min(8, len(graphs))] or graphs
    def _thunk():
        idx = {"i": 0}
        class EvalEnv(GraphLayoutEnv):
            def reset(self, **kwargs):
                p = fixed[idx["i"] % len(fixed)]
                idx["i"] += 1
                info = _read_graph(p)
                return super().reset(Graph=info["graph"], Width=info.get("width"), Height=info.get("height"), **kwargs)
        first = _read_graph(fixed[0])
        if first.get("width") is not None and first.get("height") is not None:
            env = EvalEnv(first["graph"], width=first.get("width"), height=first.get("height"), opt_type=opt_type)
        else:
            env = EvalEnv(first["graph"], opt_type=opt_type)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk

def main():
    print("Start training")

    # options
    easy_split = "train"
    hard_split = "train"
    easy_type = "rome"
    hard_type = "contest"
    opt_type = "Local"
    # opt_type = "Global"
    # n_envs = 4
    n_envs = 16
    # total_steps = 50_000
    total_steps = 30_000_000
    max_ep_steps = 2_000
    phase_steps={"warmup": 15_000_000, "mixed": 10_000_000, "hard": 5_000_000}
    seed = 12345
    p_hard = 0.1
    logdir = "runs"
    run_name = f"ppo_test_cluster_{opt_type}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resume_training = True

    # Load datasets (with caching)
    easy_ds = load_split_dataset(easy_split, dataset_type=easy_type)
    hard_ds = load_split_dataset(hard_split, dataset_type=hard_type)
    # easy_graphs = [easy_ds[i] for i in range(len(easy_ds))]
    # hard_graphs = [hard_ds[i] for i in range(len(hard_ds))]

    easy_paths = easy_ds.get_abs_paths()
    hard_paths = hard_ds.get_abs_paths()

    print("Train on", len(easy_paths), "easy graphs and", len(hard_paths), "hard graphs")

    sampler = Sampler(easy_paths, hard_paths, p_hard=p_hard)

    os.environ.setdefault("OMP_NUM_THREADS", "4")  # learner can use 2-4 threads
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    torch.set_num_threads(4)
    set_random_seed(seed)

    env_fns = [make_env(i, seed, sampler, max_ep_steps, opt_type) for i in range(n_envs)]
    vec_env  = SubprocVecEnv(env_fns)
    vec_env  = VecMonitor(vec_env)

    n_steps = 256
    rollout = n_envs * n_steps
    for bs in (4096, 2048, 1024, 512, 256, 128, 64):
        if rollout % bs == 0:
            batch_size = bs
            break
    else:
        batch_size = rollout
    print(f"[PPO] n_envs={n_envs}, n_steps={n_steps}, rollout={rollout}, batch_size={batch_size}")

    eval_env = DummyVecEnv([make_eval_env(seed + 123, hard_paths, opt_type=opt_type)])
    eval_env = VecMonitor(eval_env)


    save_dir = os.path.join(logdir, run_name)
    # Also look under src/runs if top-level runs is empty for this run
    if not os.path.isdir(save_dir):
        alt_save_dir = os.path.join(parent_dir, 'runs', run_name)
        if os.path.isdir(alt_save_dir):
            print(f"[Resume] Using existing run dir: {alt_save_dir}")
            save_dir = alt_save_dir
    os.makedirs(save_dir, exist_ok=True)
    MODEL_PATH = os.path.join(save_dir, "int_grid_model.zip")

    import re
    def _latest_ckpt_and_steps(d):
        try:
            files = [f for f in os.listdir(d) if f.endswith('.zip')]
        except Exception:
            return None, None
        # Prefer ckpt_*_steps.zip with max step count
        ckpts = []
        for f in files:
            m = re.match(r"ckpt_(\d+)_steps\.zip$", f)
            if m:
                ckpts.append((int(m.group(1)), os.path.join(d, f)))
        if ckpts:
            step, path = max(ckpts, key=lambda t: t[0])
            return path, step
        # Else best_model.zip
        if 'best_model.zip' in files:
            return os.path.join(d, 'best_model.zip'), None
        return None, None

    total_steps = 30_000_000
    resume_path = None
    resume_at = None

    if os.path.exists(MODEL_PATH):
        resume_path = MODEL_PATH
    else:
        resume_path, parsed_steps = _latest_ckpt_and_steps(save_dir)
        if parsed_steps is not None:
            resume_at = int(parsed_steps)

    if resume_path and resume_training:
        print(f"[Resume] Loading {resume_path}")
        model = Agent.load(resume_path, env=vec_env, device=device)
        # If resume_at not parsed, use loaded model's counter; else enforce parsed
        if resume_at is None:
            resume_at = int(getattr(model, 'num_timesteps', 0))
        model.num_timesteps = int(resume_at)
        reset_steps = False
        total_steps = max(0, 30_000_000 - int(resume_at))
        print(f"[Resume] Starting from {model.num_timesteps} steps; remaining={total_steps}")
    else:
        policy_kwargs = dict(net_arch={"pi": [256, 256], "vf": [256, 256]}, normalize_images=False)  # simple and stable
        model = Agent(
            "MultiInputPolicy",
            vec_env,
            device=device,
            learning_rate=3e-4,
            n_steps=n_steps,             # per env => 256 * n_envs per update
            batch_size=batch_size,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.02,
            vf_coef=0.5,
            clip_range=0.2,
            tensorboard_log=os.path.join(logdir, run_name),
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        reset_steps = True


    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, (200_000 // n_envs)),  # every ~200k env steps
        save_path=save_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Seed baseline best so we don't overwrite best_model.zip on resume unless better
    from stable_baselines3.common.evaluation import evaluate_policy
    baseline_best = None
    best_file = os.path.join(save_dir, "best_model.zip")
    if os.path.exists(best_file):
        try:
            tmp_model = Agent.load(best_file, env=eval_env, device=device)
            mean_reward, _ = evaluate_policy(tmp_model, eval_env, n_eval_episodes=8, deterministic=True)
            baseline_best = float(mean_reward)
            print(f"[Eval] Baseline best from existing best_model.zip: {baseline_best:.4f}")
            del tmp_model
        except Exception as e:
            print(f"[Eval] Could not evaluate existing best model: {e}")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=max(1, (50_000 // n_envs)),
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    if baseline_best is not None:
        eval_cb.best_mean_reward = baseline_best

    class LCRLoggerCB(BaseCallback):
        def __init__(self, log_every=5000, success_threshold=1, verbose=0):
            super().__init__(verbose)
            self.log_every = int(log_every)
            self.success_threshold = success_threshold
            self._last = 0
            self._buf_lcr = []
            self._buf_gcr = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                lcr = info.get("best_local_crossings", None)
                if lcr is not None:
                    self._buf_lcr.append(lcr)
                gcr = info.get("best_global_crossings", None)
                if gcr is not None:
                    self._buf_gcr.append(gcr)

            n = self.model.num_timesteps
            if n - self._last >= self.log_every and self._buf_lcr:
                arr = np.array(self._buf_lcr, dtype=float)
                mean_lcr = float(arr.mean())
                med_lcr = float(np.median(arr))
                succ = float((arr <= self.success_threshold).mean())
                self.model.logger.record("lcr/mean", mean_lcr)
                self.model.logger.record("lcr/median", med_lcr)
                self.model.logger.record("lcr/success_rate", succ)
                self.model.logger.dump(n)
                self._buf_lcr.clear()

                arr = np.array(self._buf_gcr, dtype=float)
                mean_gcr = float(arr.mean())
                med_gcr = float(np.median(arr))
                succ = float((arr <= self.success_threshold).mean())
                self.model.logger.record("gcr/mean", mean_gcr)
                self.model.logger.record("gcr/median", med_gcr)
                self.model.logger.record("gcr/success_rate", succ)
                self.model.logger.dump(n)
                self._buf_gcr.clear()

                self._last = n
            return True

    class PhaseManager(BaseCallback):
        def __init__(self, sampler, phase_steps, lr_schedule=None, ent_schedule=None, verbose=1):
            """
            phase_steps: dict like {"warmup": 3_000_000, "mixed": 10_000_000, "hard": 3_000_000}
            """
            super().__init__(verbose)
            self.sampler = sampler
            self.phase_steps = phase_steps
            self.lr_schedule = lr_schedule or (lambda p: None)
            self.ent_schedule = ent_schedule or (lambda p: None)
            self.phase = "warmup"
            self.phase_start_steps = 0

        def set_lr(self, new_lr: float | None):
            """Set optimizer LR and sync SB3 lr_schedule if provided."""
            if new_lr is None:
                return
            for g in model.policy.optimizer.param_groups:
                g["lr"] = float(new_lr)
            self.model.lr_schedule = lambda _: float(new_lr)

        def set_ent_coef(self, new_ent: float | None):
            """Set entropy coefficient if provided."""
            if new_ent is None:
                return
            self.model.ent_coef = float(new_ent)
            if hasattr(model, "logger"):
                model.logger.record("train/ent_coef", float(new_ent))

        def _enter_phase(self, name):
            self.phase = name
            self.phase_start_steps = self.model.num_timesteps
            if name == "warmup":
                self.sampler.set_p_hard(0.5)
            elif name == "mixed":
                self.sampler.set_p_hard(0.5)
            elif name == "hard":
                self.sampler.set_p_hard(0.5)
            # apply schedules (optional)
            lr = self.lr_schedule(name)
            self.set_lr(lr)
            if self.verbose: print(f"[Phase] LR -> {lr}")
            ent = self.ent_schedule(name)
            self.set_ent_coef(ent)
            if self.verbose: print(f"[Phase] ent_coef -> {ent}")
            if self.verbose: print(
                f"[Phase] Enter '{name}' at {self.model.num_timesteps} steps; p_hard={self.sampler.p_hard:.2f}")

        def _on_training_start(self) -> None:
            # Resume correct phase and progress based on global step (TEMP one-off resume patch)
            g = int(self.model.num_timesteps)
            w = int(self.phase_steps.get("warmup", 0))
            m = int(self.phase_steps.get("mixed", 0))
            h = int(self.phase_steps.get("hard", 0))
            if g < w:
                phase, done = "warmup", g
            elif g < w + m:
                phase, done = "mixed", g - w
            elif g < w + m + h:
                phase, done = "hard", g - w - m
            else:
                phase, done = "hard", h
            self._enter_phase(phase)
            # Make phase counter reflect progress already done
            self.phase_start_steps = self.model.num_timesteps - done
            if self.verbose:
                print(f"[Phase] Resume: global={g}, phase='{phase}', done_in_phase={done}, start_at={self.phase_start_steps}")

        def _on_step(self) -> bool:
            steps = self.model.num_timesteps - self.phase_start_steps

            # early exits & transitions
            if self.phase == "warmup":
                if steps >= self.phase_steps["warmup"]:
                    self._enter_phase("mixed")

            elif self.phase == "mixed":
                if steps >= self.phase_steps["mixed"]:
                    self._enter_phase("hard")

            elif self.phase == "hard":
                if steps >= self.phase_steps["hard"]:
                    if self.verbose: print("[Phase] Final phases finished reached. Stopping.")
                    return False

            return True

    phase_cb = PhaseManager(
        sampler,
        phase_steps=phase_steps,
        lr_schedule=lambda p: {"warmup": 3e-4, "mixed": 2e-4, "hard": 1e-4}.get(p, None),
        ent_schedule=lambda p: {"warmup": 0.03, "mixed": 0.02, "hard": 0.01}.get(p, None),
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb, phase_cb, LCRLoggerCB(log_every=10_000)])

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True, reset_num_timesteps=reset_steps)

    model.save(MODEL_PATH)
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
