
import torch
import numpy as np
import json
from rperceptron import RPerceptron
from real_dataset import TextDataset
from rich.console import Console
from rich.table import Table

console = Console()

def run_robustness_audit():
    console.rule("[bold blue]MoRE-3 Robustness Audit[/bold blue]")
    
    # 1. Load Data
    dataset = TextDataset()
    data = dataset.get_data()
    
    train_x = data['train_x']
    train_y = data['train_y']
    novel_x = data['novel_x']
    
    d_input = train_x.size(1)
    
    # 2. Setup Model
    # We use a threshold of 0.4 as found in the ablation sweep
    model = RPerceptron(d_input=d_input, M=32, theta=0.4, topk=3)
    
    # Manual initialization with cluster centers for mastery simulation
    with torch.no_grad():
        # Inject the "knowledge" of the 3 classes into the first 3 prototypes
        model.keys[:3].copy_(data['centers'])
    
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    audit_results = []
    
    for noise in noise_levels:
        console.print(f"\n[bold yellow]Testing Noise Level: {noise}[/bold yellow]")
        
        # Add noise to testing sets
        noisy_train_x = train_x + torch.randn_like(train_x) * noise
        noisy_novel_x = novel_x + torch.randn_like(novel_x) * noise
        
        # Test Known
        winner_indices, sim, y, g, _ = model(noisy_train_x)
        acc = (winner_indices == train_y).float().mean().item()
        avg_g_known = g.mean().item()
        
        # Test Novel (OOD)
        _, _, _, g_novel, _ = model(noisy_novel_x)
        avg_g_novel = g_novel.mean().item()
        novelty_detection = 1.0 - avg_g_novel # Percentage of rejected novel items
        
        # Delta G
        delta_g = avg_g_known - avg_g_novel
        
        audit_results.append({
            "noise": noise,
            "accuracy": acc,
            "gate_known": avg_g_known,
            "gate_novel": avg_g_novel,
            "novelty_rejection": novelty_detection,
            "delta_g": delta_g
        })
        
    # 3. Display Results
    table = Table(title="Robustness Audit Summary")
    table.add_column("Noise", justify="right")
    table.add_column("Acc (Known)", style="cyan")
    table.add_column("Gate (Known)", style="green")
    table.add_column("Gate (Novel)", style="red")
    table.add_column("Novelty Rejection", style="magenta")
    table.add_column("Delta G", style="bold")
    
    for res in audit_results:
        table.add_row(
            f"{res['noise']:.2f}",
            f"{res['accuracy']*100:.1f}%",
            f"{res['gate_known']:.3f}",
            f"{res['gate_novel']:.3f}",
            f"{res['novelty_rejection']*100:.1f}%",
            f"{res['delta_g']:.3f}"
        )
    
    console.print(table)
    
    # 4. Save results for LaTeX
    with open("robustness_results.json", "w") as f:
        json.dump(audit_results, f, indent=4)
    console.print("\n[bold green]Results saved to robustness_results.json for paper integration.[/bold green]")

if __name__ == "__main__":
    run_robustness_audit()
