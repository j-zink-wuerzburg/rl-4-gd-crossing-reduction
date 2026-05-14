import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
import json
import argparse
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(parent_dir)
sys.path.append(parent_dir)

from stable_baselines3 import PPO
from src.env.GraphLayoutEnvPixel import GraphLayoutEnvPixel
from src.Training.dataloader import load_split_dataset
from src.Training.train_ppo_pixel import CustomGraphLayoutExtractor  # We must import to allow SB3 to deserialize it properly

def run_agent_on_graph(model, env, max_steps=2048):
    obs, info = env.reset(seed=42)
    initial_crossings = env.global_crossings
    
    steps = 0
    start_time = time.time()
    
    while steps < max_steps:
        # PPO predict
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        if done or truncated:
            break
            
    runtime = time.time() - start_time
    final_crossings = env.best_crossings
    improvement = initial_crossings - final_crossings
    
    return {
        'initial_crossings': initial_crossings,
        'final_crossings': final_crossings,
        'improvement': improvement,
        'steps_taken': steps,
        'solved': (final_crossings == 0),
        'runtime': runtime
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO Pixel agent on Rome dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model .zip file")
    parser.add_argument("--config", type=str, required=True, help="Path to config used for training")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on (e.g., test, train, test1000)")
    parser.add_argument("--max_graphs", type=int, default=10, help="Max graphs to evaluate")
    parser.add_argument("--csv_out", type=str, default="results/eval_ppo_pixel.csv", help="Where to save the results")
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    width = config["env"].get("width", 1000)
    height = config["env"].get("height", 1000)
        
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path)
    
    dataset_name = config.get("dataset", {}).get("name", "rome")
    print(f"Loading dataset: {dataset_name}, split: {args.split}")
    ds = load_split_dataset(args.split, dataset_type=dataset_name)
    
    n_graphs = min(args.max_graphs, len(ds))
    
    results = []
    
    print(f"Evaluating model on {n_graphs} graphs...")
    for i in tqdm(range(n_graphs)):
        G = ds[i]
        
        # Instantiate environment with config
        env = GraphLayoutEnvPixel(G, config=config)
        
        # Run agent
        res = run_agent_on_graph(model, env, max_steps=2048)
        
        graph_name = os.path.basename(ds.graph_paths[i])
        res["graph_name"] = graph_name
        
        results.append(res)
        
        del env
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.set_index("graph_name", inplace=True)
    
    print("\nEvaluation Summary:")
    print(f"Average Initial Crossings: {df['initial_crossings'].mean():.2f}")
    print(f"Average Final Crossings: {df['final_crossings'].mean():.2f}")
    print(f"Average Improvement: {df['improvement'].mean():.2f}")
    print(f"Average Solved (%): {df['solved'].mean() * 100:.2f}%")
    print(f"Average Steps: {df['steps_taken'].mean():.2f}")
    
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    df.to_csv(args.csv_out)
    print(f"\nSaved detailed results to {args.csv_out}")

if __name__ == "__main__":
    main()
