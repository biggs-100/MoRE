"""
Training with Real Text: Benchmark for MoRE-3.
"""

import torch
from more_demo import MoRE
from real_dataset import TextDataset
from rich.console import Console

console = Console()

def train_real_text():
    # 1. Cargar datos reales
    dataset = TextDataset()
    data = dataset.get_data()
    X_train = data['train_x']
    y_train = data['train_y']
    
    D_INPUT = X_train.shape[1] # Debe ser 384
    N_EXPERTS = 3
    M_PROTOS = 5 # Menos prototipos porque el dataset es pequeño
    
    console.log(f"[bold blue]Real Text Benchmark[/bold blue]")
    console.log(f"Input dimensions: {D_INPUT}")
    console.log(f"Training samples: {X_train.shape[0]}")
    
    # 2. Inicializar MoRE
    # Ajustamos theta ligeramente para 384 dims si es necesario
    model = MoRE(N_EXPERTS, D_INPUT, M_PROTOS, topk=3, gamma=0.5, decay=0.99, theta=0.5)
    
    # 3. Entrenamiento Hebbiano Local
    epochs = 50
    lr = 0.1
    batch_size = 8
    n_samples = X_train.size(0)
    
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)
        X_sh = X_train[idx]
        y_sh = y_train[idx]
        
        correct = 0
        for i in range(0, n_samples, batch_size):
            x_b = X_sh[i : i + batch_size]
            y_b = y_sh[i : i + batch_size]
            
            winners, max_scores, _, _, all_attn = model.forward(x_b)
            reward = torch.where(winners == y_b, 1.0, -1.0)
            
            for e_idx, expert in enumerate(model.experts):
                mask = (winners == e_idx)
                if mask.any():
                    expert.update_local(x_b[mask], reward[mask], all_attn[e_idx][mask], lr=lr)
            
            correct += (winners == y_b).sum().item()
            
        if (epoch + 1) % 10 == 0:
            console.log(f"Epoch {epoch+1:2d} | Accuracy: {correct/n_samples:.2%}")
            
    # Guardar modelo del benchmark
    torch.save(model.state_dict(), "more_real_text.pt")
    console.log("[bold green]Benchmark model saved to more_real_text.pt[/bold green]")

if __name__ == "__main__":
    train_real_text()
