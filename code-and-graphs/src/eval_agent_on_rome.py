import os
import pandas as pd
from tqdm import tqdm
import torch
from stable_baselines3 import PPO as Agent
from sb3_contrib import RecurrentPPO as RecurrentAgent
from src.env.GraphLayoutEnv import GraphLayoutEnv
from src.Training.dataloader import load_split_dataset
import sys
import networkx as nx
import argparse
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(parent_dir)
sys.path.append(parent_dir)

# --------- CONFIGURATION ---------
# Model configurations for easy comparison
MODEL_CONFIGS = {
    "ppo_gnn_LSTM": {
        "model_path": "src/models/ppo_gnn_LSTM.pt",
        "gnn_path": "src/gnn/models/ppo_gnn_gnn_LSTM.pt",
        "agent_type": "recurrent",  # or "standard"
        "description": "PPO with GNN and LSTM"
    },
    "ppo_gnn_100": {
        "model_path": "src/models/ppo_gnn_100.pt",
        "gnn_path": "src/gnn/models/ppo_gnn_gnn.pt",
        "agent_type": "standard",
        "description": "PPO with GNN (100 nodes)"
    },
    "ppo_gnn": {
        "model_path": "src/models/ppo_gnn.pt",
        "gnn_path": "src/gnn/models/ppo_gnn_gnn.pt",
        "agent_type": "standard",
        "description": "Standard PPO with GNN"
    },
    "ppo_local": {
        "model_path": "src/models/ppo_local.pt",
        "gnn_path": None,  # No GNN for local model
        "agent_type": "standard",
        "description": "Standard PPO with local features only"
    },
}

# Default configuration - easily changeable
DEFAULT_MODEL = "ppo_local"
DEFAULT_SPLIT = "test1000"  # Changed from 'train' to 'test1000'
DEFAULT_CSV_PATH = "rome_ppo_local_noKK.csv"  # Updated CSV name
DEFAULT_MAX_GRAPHS = 1000  # Increased for test1000
# -----------------------------------

# Number of times to run the agent per graph
N_RUNS_PER_GRAPH = 3

def load_agent(model_path, gnn_path, env, agent_type="standard"):
    """Load agent with appropriate type (standard PPO or recurrent PPO)"""
    print(f"Loading {agent_type} agent from {model_path}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        if agent_type == "recurrent":
            model = RecurrentAgent.load(model_path, env=env)
        else:
            model = Agent.load(model_path, env=env)

        # Load GNN weights if available
        if gnn_path and os.path.exists(gnn_path):
            print(f"Loading GNN weights from {gnn_path}")
            gnn_state = torch.load(gnn_path, map_location="cpu")
            model.policy.features_extractor.gnn.load_state_dict(gnn_state)
        else:
            print(f"GNN weights not found at {gnn_path}, using model's built-in weights")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def run_agent_on_graph(model, env, max_steps=1000):
    """Run agent on a single graph and return best crossing count achieved"""
    import time
    obs, _ = env.reset()
    best_local = env.best_local_crossings
    initial_crossings = best_local
    initial_local_crossings = best_local  # Save initial local crossings
    steps = 0
    start_time = time.time()
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if env.best_local_crossings < best_local:
            best_local = env.best_local_crossings
        if terminated or truncated:
            break
        steps += 1
    runtime = time.time() - start_time
    improvement = initial_crossings - best_local
    return {
        'best_crossings': best_local,
        'initial_crossings': initial_crossings,
        'initial_local_crossings': initial_local_crossings,
        'improvement': improvement,
        'steps_taken': steps,
        'solved': terminated,
        'runtime': runtime
    }

def evaluate_model(model_name, config, split='test1000', max_graphs=None, csv_path=None):
    """Evaluate a single model on the specified dataset split"""

    # Load dataset
    print(f"Loading dataset split: {split}")
    ds = load_split_dataset(split, dataset_type='rome')
    n_graphs = min(max_graphs or len(ds), len(ds))

    # Prepare CSV
    csv_file = csv_path or DEFAULT_CSV_PATH
    # Check if file exists and is non-empty before reading
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        df = pd.read_csv(csv_file, index_col=0)
    else:
        df = pd.DataFrame()

    # Use graph filenames as index
    graph_names = [os.path.basename(ds.graph_paths[i]) for i in range(n_graphs)]
    if df.empty:
        df = pd.DataFrame(index=graph_names)
    else:
        # Ensure all graph_names are present
        for name in graph_names:
            if name not in df.index:
                df.loc[name] = None

    # Evaluate agent
    print(f"Evaluating model '{model_name}' ({config['description']}) on {n_graphs} rome_filtered graphs...")
    results = []
    detailed_results = []

    start_time = time.time()

    for i in tqdm(range(n_graphs), desc=f"Evaluating {model_name}"):
        best_result = None
        best_improvement = float('-inf')
        for run_idx in range(N_RUNS_PER_GRAPH):
            try:
                G = ds[i]
                env = GraphLayoutEnv(G)

                if model_name == "ppo_gnn_LSTM":
                    # Set appropriate environment settings
                    env.update_training_phase(0)  # Use full rewards for evaluation
                    env.use_lstm = (config['agent_type'] == 'recurrent')

                model = load_agent(config['model_path'], config['gnn_path'], env, config['agent_type'])
                result = run_agent_on_graph(model, env)
                if result['improvement'] > best_improvement:
                    best_improvement = result['improvement']
                    best_result = result

                # Clean up to prevent memory issues
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing graph {i} ({graph_names[i]}) run {run_idx+1}: {e}")
                continue
        if best_result is not None:
            results.append(best_result['best_crossings'])
            detailed_results.append(best_result)
        else:
            results.append(None)
            detailed_results.append({'best_crossings': None, 'error': 'All runs failed'})

    # Update DataFrame with results
    # Ensure the DataFrame has the correct index and columns
    if model_name not in df.columns:
        df[model_name] = None
    for idx, name in enumerate(graph_names):
        df.loc[name, model_name] = results[idx]

    # Save detailed results to separate columns in the desired order
    detail_columns = [
        f"{model_name}_initial_local_crossings",
        f"{model_name}_improvement",
        f"{model_name}",
        f"{model_name}_solved",
        f"{model_name}_runtime",
        f"{model_name}_steps"
    ]
    for col in detail_columns:
        if col not in df.columns:
            df[col] = None
    for j, detail in enumerate(detailed_results):
        df.loc[graph_names[j], f"{model_name}_initial_local_crossings"] = detail.get('initial_local_crossings', 0) if detail.get('best_crossings') is not None else None
        df.loc[graph_names[j], f"{model_name}_improvement"] = detail.get('improvement', 0) if detail.get('best_crossings') is not None else None
        df.loc[graph_names[j], f"{model_name}"] = detail.get('best_crossings', 0) if detail.get('best_crossings') is not None else None
        df.loc[graph_names[j], f"{model_name}_solved"] = detail.get('solved', False) if detail.get('best_crossings') is not None else None
        df.loc[graph_names[j], f"{model_name}_runtime"] = round(detail.get('runtime', 0.0), 2) if detail.get('best_crossings') is not None else None
        df.loc[graph_names[j], f"{model_name}_steps"] = detail.get('steps_taken', 0) if detail.get('best_crossings') is not None else None
    # Reorder columns for ppo_local if present
    if model_name == "ppo_local":
        ordered_cols = [
            f"{model_name}_initial_local_crossings",
            f"{model_name}_improvement",
            f"{model_name}",
            f"{model_name}_solved",
            f"{model_name}_runtime",
            f"{model_name}_steps"
        ]
        # Keep other columns after
        other_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + other_cols]
    # Save results
    df.to_csv(csv_file)

    # Print summary statistics
    valid_results = [r for r in results if r is not None]
    if valid_results:
        elapsed_time = time.time() - start_time
        print(f"\n=== Results Summary for {model_name} ===")
        print(f"Graphs evaluated: {len(valid_results)}/{n_graphs}")
        print(f"Average crossings: {sum(valid_results)/len(valid_results):.2f}")
        print(f"Best result: {min(valid_results)}")
        print(f"Worst result: {max(valid_results)}")
        print(f"Graphs solved (0 crossings): {sum(1 for r in detailed_results if r.get('solved', False))}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Results written to {csv_file}")

    return df

def compare_models(model_names, split='test1000', max_graphs=None, csv_path=None):
    """Compare multiple models on the same dataset"""
    csv_file = csv_path or DEFAULT_CSV_PATH

    for model_name in model_names:
        if model_name not in MODEL_CONFIGS:
            print(f"Warning: Model '{model_name}' not found in configurations. Skipping.")
            continue

        config = MODEL_CONFIGS[model_name]
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}: {config['description']}")
        print(f"{'='*60}")

        evaluate_model(model_name, config, split, max_graphs, csv_file)

    # Print comparison summary
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        for model_name in model_names:
            if model_name in df.columns:
                valid_results = df[model_name].dropna()
                if len(valid_results) > 0:
                    avg_crossings = valid_results.mean()
                    solved_count = sum(1 for _, row in df.iterrows()
                                     if f"{model_name}_solved" in df.columns and row[f"{model_name}_solved"])
                    print(f"{model_name:20} | Avg: {avg_crossings:6.2f} | Solved: {solved_count:3d}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agents on Rome graphs')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to evaluate')
    parser.add_argument('--models', nargs='+',
                       help='Multiple models to compare')
    parser.add_argument('--split', type=str, default=DEFAULT_SPLIT,
                       choices=['train', 'test', 'test1000'],
                       help='Dataset split to use')
    parser.add_argument('--max-graphs', type=int, default=DEFAULT_MAX_GRAPHS,
                       help='Maximum number of graphs to evaluate')
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH,
                       help='Output CSV file path')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model configurations')

    args = parser.parse_args()

    if args.list_models:
        print("Available model configurations:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name:15} - {config['description']}")
        return

    if args.models:
        # Compare multiple models
        compare_models(args.models, args.split, args.max_graphs, args.csv_path)
    else:
        # Evaluate single model
        if args.model not in MODEL_CONFIGS:
            print(f"Error: Model '{args.model}' not found in configurations.")
            print("Use --list-models to see available options.")
            return

        config = MODEL_CONFIGS[args.model]
        evaluate_model(args.model, config, args.split, args.max_graphs, args.csv_path)

if __name__ == "__main__":
    main()
