"""
FINAL INTEGRATED DEMO: Real Text + FAISS Scaling + Autonomous Mitosis.
Narrative: Birth (3 classes) -> Discovery (Novelty) -> Evolution (Mitosis) -> Mastery (4 classes).
"""

import torch
import numpy as np
import time
from more_demo import MoRE
from real_dataset import TextDataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

def run_grand_finale():
    console.print(Panel.fit("[bold rainbow]MoRE-3: THE GRAND FINALE[/bold rainbow]\n[italic]Real Text, Scaling & Evolution[/italic]", border_style="bold magenta"))
    
    # 1. Loading Real Text Data
    loader = TextDataset()
    data = loader.get_data()
    X_train, y_train = data['train_x'], data['train_y']
    X_novel = data['novel_x']
    centers = data['centers'] # Mean embeddings for A, B, C
    
    # Data Augmentation
    def augment(X, y, target_n=150):
        repeats = target_n // len(X) + 1
        X_aug = X.repeat(repeats, 1)
        y_aug = y.repeat(repeats)
        X_aug += torch.randn_like(X_aug) * 0.005
        return X_aug[:target_n], y_aug[:target_n]

    X_train, y_train = augment(X_train, y_train, target_n=450) 
    X_novel, y_novel = augment(X_novel, torch.zeros(len(X_novel)), target_n=200)
    y_novel += 3 # Health class
    
    X_all = torch.cat([X_train, X_novel])
    y_all = torch.cat([y_train, y_novel])

    # 2. Model Initialization with Bootstrap
    model = MoRE(n_experts=3, d_input=384, M=32, theta=0.4)
    with torch.no_grad():
        for i in range(3):
            model.experts[i].keys.copy_(centers[i].repeat(32, 1) + torch.randn(32, 384)*0.01)
            model.experts[i]._normalize_keys()

    # --- PHASE 1: REINFORCEMENT ---
    console.print("\n[bold green]PHASE 1: Reinforcing Sports, Tech, and Politics...[/bold green]")
    lr = 0.1
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
        task = progress.add_task("[cyan]Fine-tuning Experts...", total=300)
        for step in range(300):
            idx = torch.randint(0, len(X_train), (16,))
            x_b, y_b = X_train[idx], y_train[idx]
            
            winners, max_fam, _, _, all_attn = model.forward(x_b)
            reward = torch.where(winners == y_b, 1.0, -1.0)
            
            for s_idx in range(len(x_b)):
                exp_idx = winners[s_idx]
                model.experts[exp_idx].update_local(x_b[s_idx:s_idx+1], reward[s_idx:s_idx+1], all_attn[s_idx], lr=lr)
            progress.update(task, advance=1)

    # Initial Validation
    preds, fam = model.predict(X_train, threshold=0.3)
    acc = (preds == y_train).float().mean()
    console.log(f"Phase 1 Mastery: [bold green]{acc:.2%} Accuracy[/bold green] on known classes.")
    
    preds_n, fam_n = model.predict(X_novel, threshold=0.5)
    rejection = (preds_n == -1).float().mean()
    console.log(f"Phase 1 Vigilance: [bold yellow]{rejection:.2%} Rejection[/bold yellow] of 'Health' (Novelty).")

    # --- PHASE 2: DISCOVERY & MITOSIS ---
    console.print("\n[bold yellow]PHASE 2: Introducing 'Health' class. Evolution Triggered...[/bold yellow]")
    
    growth_step = -1
    for step in range(600):
        if step % 2 == 0:
            idx = torch.randint(0, len(X_novel), (16,))
            x_b, y_b = X_novel[idx], y_novel[idx]
        else:
            idx = torch.randint(0, len(X_train), (16,))
            x_b, y_b = X_train[idx], y_train[idx]
        
        winners, max_fam, _, _, all_attn = model.forward(x_b)
        reward = torch.where(winners == y_b, 1.0, -1.0)
        
        for s_idx in range(len(x_b)):
            exp_idx = winners[s_idx]
            if exp_idx < len(model.experts):
                model.experts[exp_idx].update_local(x_b[s_idx:s_idx+1], reward[s_idx:s_idx+1], all_attn[s_idx], lr=lr)
        
        # Trigger mitosis check every 20 steps
        if step % 20 == 0:
            splits = model.check_health_and_mitosis()
            if splits and growth_step == -1:
                growth_step = step
                console.log(f"[bold reverse green] !!! MITOSIS !!! [/bold reverse green] Architecture expanded at step {step}. Experts: [bold cyan]{len(model.experts)}[/bold cyan]")
                lr = 0.05 # Lower LR to stabilize after growth

    # --- FINAL VERIFICATION ---
    console.print("\n[bold rainbow]PHASE 3: Final Mastery Assessment[/bold rainbow]")
    
    preds_all, fam_all = model.predict(X_all, threshold=0.3)
    
    table = Table(title="Integrated Performance Report")
    table.add_column("Category", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Familiarity", justify="right", style="magenta")
    table.add_column("Acc/Gating", justify="right", style="green")
    
    categories = ["Sports", "Tech", "Politics", "Health"]
    for i, name in enumerate(categories):
        mask = (y_all == i)
        avg_f = fam_all[mask].mean().item()
        class_acc = (preds_all[mask] == i).float().mean().item()
        table.add_row(name, str(mask.sum().item()), f"{avg_f:.3f}", f"{class_acc:.2%}")
        
    console.print(table)
    
    console.print(f"\n[bold white]Evolutionary Summary:[/bold white]")
    console.print(f"- Final Expert Count: [bold cyan]{len(model.experts)}[/bold cyan]")
    console.print(f"- Structural Change: {'YES' if growth_step > -1 else 'NO'}")
    
    console.print("\n[bold reverse gold1] RESONANCE, SURPRISE AND GROWTH ARE ALL YOU NEED. [/bold reverse gold1]")

if __name__ == "__main__":
    run_grand_finale()
