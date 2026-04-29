"""
MoRE-3 Master Reproduction Script
Protocol for Absolute Truth: Statistical Validation & Multi-Seed Verification.
"""

import os
import json
import torch
import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Import benchmark components
from benchmark.stream import TaskStream
from benchmark.features import FeatureExtractor
from benchmark.baselines import MLPBaseline, EWCBaseline
from benchmark.more_wrapper import MoREBenchmarkWrapper
from benchmark.runner import run_experiment

console = Console()

def run_multi_seed_experiment(model_type, mode, n_tasks=5, samples=1000, n_seeds=3, theta=0.3):
    acc_list = []
    bwt_list = []
    
    with Progress(SpinnerColumn(), TextColumn(f"[bold cyan]{model_type.upper()} on {mode}..."), BarColumn(), transient=True) as progress:
        task = progress.add_task("Running seeds...", total=n_seeds)
        
        for seed in range(n_seeds):
            torch.manual_seed(seed + 100)
            np.random.seed(seed + 100)
            
            stream = TaskStream(n_tasks=n_tasks, mode=mode, max_samples_per_task=samples)
            feature_extractor = FeatureExtractor()
            
            # Setup Model
            temp_X, _, _ = next(iter(stream))
            temp_features = feature_extractor.extract(temp_X)
            input_dim = temp_features.shape[1]
            n_classes = 10 if mode in ['split_mnist', 'permuted_mnist'] else 3
            
            if model_type == 'mlp':
                model = MLPBaseline(input_dim, n_classes)
            elif model_type == 'ewc':
                model = EWCBaseline(input_dim, n_classes, lambda_ewc=5000.0)
            else:
                model = MoREBenchmarkWrapper(d_input=input_dim, n_classes=n_classes, n_experts=3, theta=theta)
            
            # Re-run stream
            stream = TaskStream(n_tasks=n_tasks, mode=mode, max_samples_per_task=samples)
            engine = run_experiment(model, stream, feature_extractor, n_tasks)
            
            acc_list.append(engine.calculate_acc())
            bwt_list.append(engine.calculate_bwt())
            progress.update(task, advance=1)
            
    return {
        'acc_mean': np.mean(acc_list),
        'acc_std': np.std(acc_list),
        'bwt_mean': np.mean(bwt_list),
        'bwt_std': np.std(bwt_list)
    }

def reproduce_all():
    console.print(Panel.fit("[bold rainbow]MoRE-3: MASTER REPRODUCTION SUITE[/bold rainbow]\n[italic]Statistical Validation Protocol[/italic]", border_style="bold magenta"))
    
    final_report = {}
    
    # --- 1. Split MNIST Benchmark ---
    console.print("\n[bold green]EXERCISE 1: Split MNIST Stability (N=3 Seeds)[/bold green]")
    models = ['more', 'mlp', 'ewc']
    mnist_results = {}
    
    for m in models:
        res = run_multi_seed_experiment(m, 'split_mnist', n_tasks=5, samples=500, n_seeds=3, theta=0.3)
        mnist_results[m] = res
        console.log(f"Model: [bold cyan]{m:4}[/bold cyan] | ACC: {res['acc_mean']:.4f} ± {res['acc_std']:.4f} | BWT: {res['bwt_mean']:.4f} ± {res['bwt_std']:.4f}")
    
    final_report['split_mnist'] = mnist_results

    # --- 2. FAISS Scaling Benchmark ---
    console.print("\n[bold yellow]EXERCISE 2: FAISS Scaling Audit[/bold yellow]")
    # We run a subset of faiss_benchmark logic
    import faiss
    from rperceptron import RPerceptron
    d, M = 384, 10000
    # Use the new signature with n_classes
    model = RPerceptron(d, M, n_classes=10, use_faiss=True, faiss_threshold=1)
    x = torch.randn(1, d)
    
    # Dense
    model.use_faiss = False
    start = time.perf_counter()
    for _ in range(100): model.forward(x)
    t_dense = (time.perf_counter() - start) * 10 # average in ms
    
    # FAISS
    model.use_faiss = True
    model._rebuild_index()
    start = time.perf_counter()
    for _ in range(100): model.forward(x)
    t_faiss = (time.perf_counter() - start) * 10
    
    scaling_res = {
        't_dense': t_dense,
        't_faiss': t_faiss,
        'speedup': t_dense / t_faiss
    }
    console.log(f"FAISS Speedup (M=10k): [bold green]{scaling_res['speedup']:.2f}x[/bold green]")
    final_report['scaling'] = scaling_res

    # --- FINAL REPORT GENERATION ---
    with open("reproduction_report.json", "w") as f:
        json.dump(final_report, f, indent=4)
    
    # Generate LaTeX Table Snippet
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Statistical Validation on Split MNIST (3-seed average)}
\begin{tabular}{lcc}
\hline
Model & ACC ($\uparrow$) & BWT ($\uparrow$) \\
\hline
MLP & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\
EWC & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\
\textbf{MoRE-3} & \textbf{%.4f $\pm$ %.4f} & \textbf{%.4f $\pm$ %.4f} \\
\hline
\end{tabular}
\end{table}
""" % (
        mnist_results['mlp']['acc_mean'], mnist_results['mlp']['acc_std'], mnist_results['mlp']['bwt_mean'], mnist_results['mlp']['bwt_std'],
        mnist_results['ewc']['acc_mean'], mnist_results['ewc']['acc_std'], mnist_results['ewc']['bwt_mean'], mnist_results['ewc']['bwt_std'],
        mnist_results['more']['acc_mean'], mnist_results['more']['acc_std'], mnist_results['more']['bwt_mean'], mnist_results['more']['bwt_std']
    )
    
    with open("latex_results_table.txt", "w") as f:
        f.write(latex_table)
        
    console.print("\n[bold reverse gold1] REPRODUCTION COMPLETE. LATEX TABLE SAVED. [/bold reverse gold1]")

if __name__ == "__main__":
    reproduce_all()
