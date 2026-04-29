"""
MoRE-3 Brutal Stress Test: "The Chaos Stream"
Testing structural integrity against adversarial noise and rapid task shifts.
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from more_demo import MoRE

console = Console()

def generate_noise_embeddings(n_samples, d=384):
    return torch.randn(n_samples, d)

def train_one_batch(model, x_batch, y_batch, lr=0.01):
    winners, max_scores, all_y, g, all_attn = model.forward(x_batch)
    reward = torch.where(winners == y_batch, 1.0, -1.0)
    
    for e_idx, expert in enumerate(model.experts):
        mask = (winners == e_idx)
        if mask.any():
            x_e = x_batch[mask]
            r_e = reward[mask]
            # In MoRE, all_attn is a list of tensors for each batch element
            # But the forward returns all_attn as a list of Tensors per winner
            # Actually, forward returns all_attn as a list of length batch_size
            # and each element is the scores for that sample.
            # Let's fix the attn retrieval.
            
            # Since forward logic was: all_attn.append(scores)
            # all_attn is a list of length batch_size.
            indices = mask.nonzero().squeeze(1)
            for idx in indices:
                sample_attn = all_attn[idx]
                expert.update_local(x_batch[idx].unsqueeze(0), reward[idx].unsqueeze(0), sample_attn, lr=lr)
                expert.update_voting(y_batch[idx])
    
    # Check for mitosis
    model.check_health_and_mitosis(threshold_h=0.1, threshold_f=0.4)
    return winners, g

def stress_test():
    console.print("[bold red]Starting BRUTAL STRESS TEST: The Chaos Stream[/bold red]")
    
    d_input = 384
    n_classes = 4
    model = MoRE(n_experts=3, d_input=d_input, M=128, n_classes=n_classes, theta=0.4)
    
    # 1. Learn Task 0: Sports (Stability Foundation)
    console.log("Phase 1: Establishing Stability Foundation (Sports)...")
    sports_data = torch.randn(200, d_input) + 3.0 # Distant cluster
    labels_0 = torch.zeros(200, dtype=torch.long)
    
    for i in range(0, 200, 32):
        train_one_batch(model, sports_data[i:i+32], labels_0[i:i+32])
        
    preds, fam = model.predict(sports_data)
    acc_0_initial = (preds == 0).float().mean().item()
    console.log(f"Task 0 Initial Accuracy: [green]{acc_0_initial:.2%}[/green]")
    
    # 2. Injected Chaos: Semantic Noise
    console.log("\n[bold yellow]Phase 2: ADVERSARIAL ATTACK - Injecting 500 Noise Samples...[/bold yellow]")
    noise_data = generate_noise_embeddings(500)
    rejection_count = 0
    
    for i in range(0, 500, 32):
        x_batch = noise_data[i:i+32]
        # Predict full to see rejection
        _, _, _, g, _ = model.forward(x_batch)
        rejection_count += (g < 0.3).sum().item() # g < theta means silent
        
        # Try to force-train with a fake label to see if it resists
        fake_labels = torch.full((x_batch.size(0),), 9, dtype=torch.long)
        train_one_batch(model, x_batch, fake_labels)
        
    rejection_rate = rejection_count / 500
    console.log(f"Novelty Rejection Rate (Noise): [bold cyan]{rejection_rate:.2%}[/bold cyan]")
    console.log(f"Expert Count after Noise: [bold]{len(model.experts)}[/bold] (Target: 3)")

    # 3. Learn Task 1: Tech (Plasticity Test)
    console.log("\nPhase 3: Learning New Manifold (Tech) after Chaos...")
    tech_data = torch.randn(200, d_input) - 3.0 # Distant cluster
    labels_1 = torch.ones(200, dtype=torch.long)
    for i in range(0, 200, 32):
        train_one_batch(model, tech_data[i:i+32], labels_1[i:i+32])
        
    # 4. Final Evaluation
    console.print("\n[bold magenta]FINAL STRESS EVALUATION[/bold magenta]")
    preds_0, _ = model.predict(sports_data)
    preds_1, _ = model.predict(tech_data)
    acc_0_final = (preds_0 == 0).float().mean().item()
    acc_1_final = (preds_1 == 1).float().mean().item()
    
    table = Table(title="MoRE-3 Stress Resilience Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="bold green")
    table.add_row("Task 0 Retention (Sports)", f"{acc_0_final:.2%}")
    table.add_row("Task 1 Mastery (Tech)", f"{acc_1_final:.2%}")
    table.add_row("Noise Immunity (Rejection)", f"{rejection_rate:.2%}")
    table.add_row("Structural Stability", "PASSED" if len(model.experts) < 10 else "FAILED")
    table.add_row("Final Expert Count", str(len(model.experts)))
    
    console.print(table)
    
    if acc_0_final > 0.80 and rejection_rate > 0.80:
        console.print("[bold reverse green] STRESS TEST PASSED: MoRE-3 IS INDESTRUCTIBLE [/bold reverse green]")
    else:
        console.print("[bold reverse red] STRESS TEST FAILED: STRUCTURAL COLLAPSE DETECTED [/bold reverse red]")

if __name__ == "__main__":
    stress_test()
