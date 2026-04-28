"""
Training Demo: Local Hebbian learning for MoRE.
"""

import torch
import torch.nn.functional as F
from more_demo import MoRE
from dataset import generate_clusters, generate_novelty
from rich.console import Console
from rich.table import Table

console = Console()

def train_more(model, X_train, y_train, epochs=20, lr=0.01):
    batch_size = 32
    n_samples = X_train.size(0)
    
    console.log(f"[bold green]Starting Training...[/bold green] ({epochs} epochs)")
    
    for epoch in range(epochs):
        # Shuffle
        idx = torch.randperm(n_samples)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        total_reward = 0
        correct_preds = 0
        
        for i in range(0, n_samples, batch_size):
            x_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            
            # Forward
            winners, max_scores, all_y, all_g, all_attn = model.forward(x_batch)
            
            # Recompensa (+1 si gano el experto correcto, -1 si no)
            reward = torch.where(winners == y_batch, 1.0, -1.0)
            
            # Actualizar expertos
            for e_idx, expert in enumerate(model.experts):
                # Solo el experto que gano se actualiza para cada ejemplo
                # (aunque la regla local Hebbiana podria aplicarse a todos, 
                # aqui forzamos competencia MoE)
                mask = (winners == e_idx)
                if mask.any():
                    x_e = x_batch[mask]
                    r_e = reward[mask]
                    attn_e = all_attn[e_idx][mask]
                    
                    expert.update_local(x_e, r_e, attn_e, lr=lr)
            
            total_reward += reward.sum().item()
            correct_preds += (winners == y_batch).sum().item()
            
        acc = correct_preds / n_samples
        avg_f = total_reward / n_samples # This is wrong in previous code, it should be avg reward
        # Corrected avg reward and added familiarity
        if (epoch + 1) % 5 == 0 or epoch == 0:
            console.log(f"Epoch {epoch+1:2d} | Accuracy: {acc:.2%} | Max Familiarity: {max_scores.mean():.3f}")

if __name__ == "__main__":
    # Params
    D_INPUT = 128
    N_EXPERTS = 3
    M_PROTOS = 20 # Prototipos por experto
    
    # Data
    X, y, centers = generate_clusters(n_clusters=N_EXPERTS, d=D_INPUT, n_samples=200)
    
    # Model
    model = MoRE(N_EXPERTS, D_INPUT, M_PROTOS, topk=5, gamma=0.5, decay=0.995)
    
    # Train
    train_more(model, X, y, epochs=30, lr=0.05)
    
    # Save model
    torch.save(model.state_dict(), "more_model.pt")
    console.log("[bold blue]Model saved to more_model.pt[/bold blue]")
