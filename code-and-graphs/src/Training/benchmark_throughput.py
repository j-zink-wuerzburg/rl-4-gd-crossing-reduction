import os
import sys
import time
import json
import argparse
import pandas as pd
from itertools import product

import stable_baselines3.common.logger as sb3_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

# Ensure 'src' is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Training.train_ppo_pixel import make_env, CustomGraphLayoutExtractor
from Training.dataloader import load_split_dataset

def run_benchmark(n_envs, batch_size, n_steps, config, dataset, total_timesteps=20000):
    print(f"--- Benchmarking: n_envs={n_envs}, batch_size={batch_size}, n_steps={n_steps} ---")
    
    # Temporarily suppress SB3 output so we can see the results clearly
    sb3_logger.configure(os.path.join("results", "benchmark_logs"), ["log"])
    
    config["training"]["n_envs"] = n_envs
    config["ppo"]["batch_size"] = batch_size
    config["ppo"]["n_steps"] = n_steps
    
    try:
        vec_env = SubprocVecEnv([make_env(i, config, dataset, seed=42) for i in range(n_envs)])
        vec_env = VecMonitor(vec_env)
        
        policy_kwargs = dict(
            features_extractor_class=CustomGraphLayoutExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        
        raw_net_arch = config["ppo"].get("policy_kwargs", {}).get("net_arch")
        if raw_net_arch is not None:
             policy_kwargs["net_arch"] = dict(pi=raw_net_arch, vf=raw_net_arch)

        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy if config["ppo"]["policy"] == "MultiInputPolicy" else config["ppo"]["policy"],
            vec_env,
            learning_rate=config["ppo"]["learning_rate"],
            n_steps=config["ppo"]["n_steps"],
            batch_size=config["ppo"]["batch_size"],
            n_epochs=config["ppo"]["n_epochs"],
            gamma=config["ppo"]["gamma"],
            gae_lambda=config["ppo"]["gae_lambda"],
            clip_range=config["ppo"]["clip_range"],
            ent_coef=config["ppo"]["ent_coef"],
            vf_coef=config["ppo"]["vf_coef"],
            max_grad_norm=config["ppo"]["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=0, # Turn off verbose
            seed=42,
        )
        
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps)
        end_time = time.time()
        
        vec_env.close()
        
        elapsed = end_time - start_time
        fps = total_timesteps / elapsed
        print(f"Result: {fps:.2f} FPS (Elapsed: {elapsed:.2f}s for {total_timesteps} steps)\n")
        return fps
        
    except Exception as e:
        print(f"Error during benchmarking: {e}\n")
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ppo_pixel.json", help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset_name = config.get("dataset", {}).get("name", "rome")
    dataset_split = config.get("dataset", {}).get("split", "train")
    dataset = load_split_dataset(dataset_split, dataset_type=dataset_name)

    # Disable callbacks/video/heatboards that cause I/O drops during benchmarks
    config["training"]["log_heatmap"] = False
    config["training"]["log_video"] = False
    
    # Parameter grid to test
    # Moving up from the previous best (n_envs=8, batch_size=2048)
    n_envs_list = [8, 16, 32, 64]
    batch_size_list = [2048, 4096, 8192, 16384]
    # We lock n_steps here, but adjust it if buffer size becomes a memory issue
    n_steps_list = [2048] 
    
    results = []
    
    # Warmup run to initialize CUDA context / dataset in RAM
    print("Performing CUDA initialization warmup...")
    run_benchmark(4, 256, 128, config, dataset, total_timesteps=1000)

    for n_envs, batch_size, n_steps in product(n_envs_list, batch_size_list, n_steps_list):
        # The rollout buffer size (n_envs * n_steps) must be perfectly divisible by batch_size!
        if (n_envs * n_steps) % batch_size != 0:
            print(f"Skipping n_envs={n_envs}, batch_size={batch_size}: Buffer size ({n_envs * n_steps}) not divisible by batch_size.")
            continue
            
        fps = run_benchmark(n_envs, batch_size, n_steps, config, dataset, total_timesteps=40000)
        results.append({
            "n_envs": n_envs,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "fps": fps
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by="fps", ascending=False)
    
    print("\n================== BENCHMARK RESULTS ==================")
    print(df.to_string(index=False))
    
    df.to_csv("results/benchmark_fps_results.csv", index=False)
    print("\nResults saved to results/benchmark_fps_results.csv")

if __name__ == "__main__":
    main()
