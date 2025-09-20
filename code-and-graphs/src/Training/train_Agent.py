import os
import torch
import sys
import gc
# Set the working directory to one level above 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(parent_dir)
sys.path.append(parent_dir)  # Ensure the parent directory is in the Python path

from torch.utils.data import DataLoader
import gymnasium as gym
from stable_baselines3 import PPO as Agent
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from Training.dataloader import load_split_dataset
from env.GraphLayoutEnv import GraphLayoutEnv

def make_env(graph):
    """
    Gym environment factory for a single graph.
    """
    def _init():
        env = GraphLayoutEnv(graph)
        return env
    return _init


def train(train_split='mini', timesteps_per_graph=10000, n_envs=8,
          model_path='models/ppo_graph_mini.pt', batch_size=100,
          epochs=1, resume=False, dataset_type='rome', start_batch=0):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Using device: {device}")

    # Load the dataset of graphs
    ds = load_split_dataset(train_split, dataset_type=dataset_type)
    graphs = [ds[i] for i in range(len(ds))]
    num_graphs = len(graphs)
    num_batches = (num_graphs + batch_size - 1) // batch_size

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_idx in range(start_batch, num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, num_graphs)
            batch_graphs = graphs[start:end]
            print(f"Processing batch {batch_idx + 1}/{num_batches}, graphs {start} to {end - 1}")

            for chunk_start in range(0, len(batch_graphs), n_envs):
                chunk = batch_graphs[chunk_start:chunk_start + n_envs]
                k = len(chunk)

                if 'vec_env' in locals():
                    del vec_env
                    gc.collect()
                    torch.cuda.empty_cache()


                env_fns = [make_env(g) for g in chunk]
                vec_env = SubprocVecEnv(env_fns)
                vec_env = VecMonitor(vec_env)

                if model is None:
                    if resume: # and os.path.exists(model_path + ".zip"):
                        model = Agent.load(model_path, env=vec_env, device=device)
                        print("Resuming training from checkpoint...")
                    else:
                        model = Agent(
                            'MultiInputPolicy',
                            vec_env,
                            verbose=1,
                            batch_size=64,
                            n_epochs=10,
                            learning_rate=3e-3,
                            device=device,
                            policy_kwargs=policy_kwargs
                        )
                else:
                    model.set_env(vec_env)

                try:
                    model.learn(total_timesteps=int(timesteps_per_graph * k))
                except RuntimeError as e:
                    if 'CUDA error: out of memory' in str(e):
                        print("CUDA out of memory. Freeing memory...")
                        del vec_env
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)
                print(f"Model saved to {model_path} for batch {batch_idx + 1}, chunk {chunk_start // n_envs + 1}")

if __name__ == '__main__':
    train(
        train_split='train',
        timesteps_per_graph=10000,
        n_envs=10,
        model_path='models/ppo_local_test.pt',
        batch_size=100,
        epochs=1,
        resume=True,
        dataset_type='extended_BA',  # Change to 'rome' for the Rome dataset
        start_batch=6
    )
