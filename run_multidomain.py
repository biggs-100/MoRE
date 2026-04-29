"""
Multi-Domain Stress Test: MNIST to FashionMNIST Transfer.
Validates structural mitosis and stability across heterogeneous domains.
"""

import torch
import numpy as np
from benchmark.more_wrapper import MoREBenchmarkWrapper
from benchmark.stream import TaskStream
from benchmark.metrics import MetricEngine
from rich.console import Console
from rich.table import Table

console = Console()

def run_experiment():
    console.rule("[bold magenta]MoRE-3 Multi-Domain Stress Test[/bold magenta]")
    
    # Configuration
    N_TASKS = 2 # 0: MNIST, 1: FashionMNIST
    D_INPUT = 784
    N_INITIAL_EXPERTS = 3
    M_PROTOTYPES = 16
    THETA = 0.5 # Novelty threshold
    MAX_SAMPLES = 1000
    
    stream = TaskStream(n_tasks=N_TASKS, mode='multidomain_mnist_fashion', max_samples_per_task=MAX_SAMPLES)
    # Correct signature: (d_input, n_classes, n_experts, M, theta)
    wrapper = MoREBenchmarkWrapper(D_INPUT, 10, N_INITIAL_EXPERTS, M_PROTOTYPES, theta=THETA)
    metrics = MetricEngine(n_tasks=N_TASKS)
    
    task_accuracies = []
    task_results = []
    
    for task_id, (X, y, _) in enumerate(stream):
        domain_name = "MNIST" if task_id == 0 else "FashionMNIST"
        console.print(f"\n[bold yellow]Task {task_id}: {domain_name}[/bold yellow]")
        console.print(f"Samples: {len(X)} | Experts before: {len(wrapper.model.experts)}")
        
        # Train on task
        wrapper.train_task(X, y, task_id=task_id, epochs=100) # Increased epochs for real data
        
        # Evaluate on all tasks seen so far
        current_accuracies = []
        # Re-iterate stream for evaluation (simple but works for demo)
        eval_stream = TaskStream(n_tasks=task_id+1, mode='multidomain_mnist_fashion', max_samples_per_task=200)
        
        for eval_id, (X_ev, y_ev, _) in enumerate(eval_stream):
            acc = wrapper.evaluate(X_ev, y_ev)
            current_accuracies.append(acc)
            metrics.update(task_id, eval_id, acc)
            
        task_accuracies.append(current_accuracies)
        console.print(f"Experts after: [bold cyan]{len(wrapper.model.experts)}[/bold cyan]")
        console.print(f"Accuracies: {current_accuracies}")

    # 📊 Final Report
    table = Table(title="Multi-Domain Stability Report")
    table.add_column("Task", justify="right", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Final Accuracy", justify="center")
    table.add_column("Forget Rate", justify="center", style="red")
    
    final_accs = task_accuracies[-1]
    for i, acc in enumerate(final_accs):
        domain = "MNIST" if i == 0 else "FashionMNIST"
        # Forget rate = Initial Acc - Final Acc
        initial_acc = task_accuracies[i][i]
        forget = initial_acc - acc
        table.add_row(str(i), domain, f"{acc:.2%}", f"{forget:.2%}")
        
    console.print("\n")
    console.print(table)
    
    bwt = metrics.calculate_bwt()
    console.print(f"\n[bold green]Backward Transfer (BWT): {bwt:.4f}[/bold green]")
    
    if len(wrapper.model.experts) > N_INITIAL_EXPERTS:
        console.log("[bold rainbow]SUCCESS: Multi-domain structural growth detected![/bold rainbow]")
    else:
        console.log("[bold red]FAILED: No structural expansion between domains.[/bold red]")

if __name__ == "__main__":
    run_experiment()
