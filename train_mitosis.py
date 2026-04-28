"""
Mitosis Experiment: Autonomous Growth from 3 to 4 Experts.
"""

import torch
import numpy as np
from more_demo import MoRE
from dataset import generate_clusters
from rich.console import Console
from rich.table import Table

console = Console()

def train_with_mitosis():
    D_INPUT = 128
    N_INITIAL_EXPERTS = 3
    M_PROTOS = 16
    
    # 1. Start with 3 classes
    X_known, y_known, _ = generate_clusters(n_clusters=3, d=D_INPUT, n_samples=1000)
    # Class D (Novelty to be learned)
    X_novel, y_novel, _ = generate_clusters(n_clusters=1, d=D_INPUT, n_samples=500, seed=42)
    y_novel += 0 # We don't actually need the labels for unsupervised, but for reward logic...
    # Let's say class D samples should be handled by a new expert.
    
    model = MoRE(N_INITIAL_EXPERTS, D_INPUT, M_PROTOS, theta=0.5)
    
    console.log("[bold green]Phase 1: Training on A, B, C[/bold green]")
    lr = 0.1
    steps = 400
    
    for step in range(steps):
        idx = torch.randint(0, X_known.size(0), (16,))
        x_b = X_known[idx]
        y_b = y_known[idx]
        
        winners, max_fam, _, _, all_attn = model.forward(x_b)
        
        # Supervised reward for initial training
        reward = torch.where(winners == y_b, 1.0, -1.0)
        
        for s_idx in range(len(x_b)):
            exp_idx = winners[s_idx]
            model.experts[exp_idx].update_local(
                x_b[s_idx:s_idx+1], 
                reward[s_idx:s_idx+1], 
                all_attn[s_idx], 
                lr=lr
            )
                
        if step % 100 == 0:
            acc = (winners == y_b).float().mean()
            console.log(f"Step {step} | Accuracy: {acc:.2%}")

    console.log(f"[bold blue]Experts after Phase 1: {len(model.experts)}[/bold blue]")
    
    # 2. Introduce Class D
    console.log("[bold yellow]Phase 2: Introducing Class D[/bold yellow]")
    
    growth_occurred = False
    for step in range(600):
        # We sample heavily from class D to force the split
        if step % 2 == 0:
            idx = torch.randint(0, X_novel.size(0), (16,))
            x_b = X_novel[idx]
            y_b = torch.full((16,), 3) # Target for new class
        else:
            idx = torch.randint(0, X_known.size(0), (16,))
            x_b = X_known[idx]
            y_b = y_known[idx]
        
        winners, max_fam, _, g, all_attn = model.forward(x_b)
        
        # Self-supervised/Weakly-supervised reward:
        # If novelty gate says "I know this", reward if winner is consistent.
        # Here we use the fact that we know the true labels for the demo.
        # For D, we don't have a 4th expert yet, so winners will be 0, 1, or 2.
        # We reward if the winner matches the true class.
        # This will fail for D at first, creating "error" and lowering familiarity.
        true_labels = y_b
        reward = torch.where(winners == true_labels, 1.0, -1.0)
        
        for s_idx in range(len(x_b)):
            exp_idx = winners[s_idx]
            model.experts[exp_idx].update_local(
                x_b[s_idx:s_idx+1], 
                reward[s_idx:s_idx+1], 
                all_attn[s_idx], 
                lr=lr
            )
        
        # Trigger mitosis check
        if step % 20 == 0:
            splits = model.check_health_and_mitosis()
            if splits:
                console.log(f"[bold reverse green] MITOSIS! [/bold reverse green] Expert {splits} split. Experts: {len(model.experts)}")
                growth_occurred = True
            
        if step % 100 == 0:
            # Accuracy on known classes only for monitoring
            mask = (y_b < 3)
            if mask.any():
                acc = (winners[mask] == true_labels[mask]).float().mean()
                console.log(f"Step {step} | Acc (ABC): {acc:.2%} | Experts: {len(model.experts)}")

    # 3. Final Evaluation
    console.log("[bold green]Final Evaluation[/bold green]")
    # Test on all 4 clusters
    X_test, y_test, _ = generate_clusters(n_clusters=4, d=D_INPUT, n_samples=200)
    winners, _, _, _, _ = model.forward(X_test)
    
    # Since expert indices might have shifted, we check cluster separation
    # (Visual check in logs is often enough for a demo)
    console.log(f"Number of experts at end: [bold cyan]{len(model.experts)}[/bold cyan]")
    
    if len(model.experts) > N_INITIAL_EXPERTS:
        console.log("[bold rainbow]SUCCESS: The model grew autonomously![/bold rainbow]")
    else:
        console.log("[bold red]Growth failed.[/bold red]")

if __name__ == "__main__":
    train_with_mitosis()
