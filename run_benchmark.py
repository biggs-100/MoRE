import argparse
import torch
import numpy as np
from benchmark.stream import TaskStream
from benchmark.features import FeatureExtractor
from benchmark.baselines import MLPBaseline, EWCBaseline
from benchmark.more_wrapper import MoREBenchmarkWrapper
from benchmark.runner import run_experiment
import json

def main():
    parser = argparse.ArgumentParser(description='MoRE-3 Catastrophic Forgetting Benchmark')
    parser.add_argument('--mode', type=str, default='permuted_mnist', choices=['synthetic', 'permuted_mnist', 'split_mnist', 'split_cifar'])
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--model', type=str, default='more', choices=['mlp', 'ewc', 'er', 'more'])
    parser.add_argument('--theta', type=float, default=0.3)
    parser.add_argument('--n_experts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Setup Stream and Features
    print(f"Initializing {args.mode} stream with {args.n_tasks} tasks ({args.samples} samples/task)...")
    stream = TaskStream(n_tasks=args.n_tasks, mode=args.mode, max_samples_per_task=args.samples)
    feature_extractor = FeatureExtractor()
    
    # 2. Setup Model (detect dim from first batch)
    temp_X, _, _ = next(iter(stream))
    temp_features = feature_extractor.extract(temp_X)
    input_dim = temp_features.shape[1]
    
    n_classes = 100 # Max for CIFAR100
    if args.mode in ['permuted_mnist', 'split_mnist']: n_classes = 10
    if args.mode == 'synthetic': n_classes = 3
    
    if args.model == 'mlp':
        model = MLPBaseline(input_dim, n_classes)
    elif args.model == 'ewc':
        model = EWCBaseline(input_dim, n_classes, lambda_ewc=5000.0)
    elif args.model == 'er':
        from benchmark.baselines import ERBaseline
        model = ERBaseline(input_dim, n_classes, mem_size=200)
    else:
        model = MoREBenchmarkWrapper(d_input=input_dim, n_classes=n_classes, n_experts=args.n_experts, theta=args.theta)
        
    # 3. Run Experiment
    # Re-iterating stream since we consumed one batch
    stream = TaskStream(n_tasks=args.n_tasks, mode=args.mode, max_samples_per_task=args.samples)
    engine = run_experiment(model, stream, feature_extractor, args.n_tasks)
    
    # 4. Results
    acc = engine.calculate_acc()
    bwt = engine.calculate_bwt()
    
    print("\n" + "="*30)
    print(f"RESULTS FOR {args.model.upper()} ON {args.mode.upper()}")
    print(f"Average Accuracy (ACC): {acc:.4f}")
    print(f"Backward Transfer (BWT): {bwt:.4f}")
    print("="*30)
    
    # Save results
    res = {
        "model": args.model,
        "mode": args.mode,
        "theta": args.theta if args.model == 'more' else None,
        "acc": acc,
        "bwt": bwt,
        "matrix": engine.get_matrix()
    }
    suffix = f"_theta{args.theta}" if args.model == 'more' else ""
    filename = f"results_{args.model}_{args.mode}{suffix}.json"
    with open(filename, "w") as f:
        json.dump(res, f, indent=4)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
