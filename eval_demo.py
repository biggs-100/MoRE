"""
Evaluation Demo: Testing classification and novelty detection.
"""

import torch
from more_demo import MoRE
from dataset import generate_clusters, generate_novelty
from rich.console import Console
from rich.table import Table

console = Console()

def evaluate_novelty(model, centers):
    D_INPUT = centers.shape[1]
    
    # 1. Datos familiares (Clases A, B, C) - Mismo seed que entrenamiento
    X_fam, y_fam, _ = generate_clusters(n_clusters=3, d=D_INPUT, n_samples=50, seed=42)
    
    # 2. Datos nuevos (Clase D) - Seed distinto
    X_nov, _ = generate_novelty(n_samples=50, d=D_INPUT, existing_centers=centers, seed=999)
    
    # Predecir
    preds_fam, scores_fam = model.predict(X_fam, threshold=0.5)
    preds_nov, scores_nov = model.predict(X_nov, threshold=0.5)
    
    # Resultados conocidos
    acc_fam = (preds_fam == y_fam).float().mean().item()
    rejected_fam = (preds_fam == -1).float().mean().item()
    
    # Resultados novedad
    acc_nov_err = (preds_nov != -1).float().mean().item() # Errores (clasifico algo nuevo como viejo)
    rejected_nov = (preds_nov == -1).float().mean().item() # Exito (detecto novedad)
    
    # Tabla de resultados
    table = Table(title="MoRE Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Known Accuracy (A, B, C)", f"{acc_fam:.2%}")
    table.add_row("Known Rejection Rate", f"{rejected_fam:.2%}")
    table.add_row("Novelty Detection Rate (D)", f"{rejected_nov:.2%}")
    table.add_row("Avg Familiarity Known", f"{scores_fam.mean():.3f}")
    table.add_row("Avg Familiarity Novel", f"{scores_nov.mean():.3f}")
    
    console.print(table)

if __name__ == "__main__":
    D_INPUT = 128
    N_EXPERTS = 3
    M_PROTOS = 20
    
    # Re-generar centros para asegurar consistencia
    _, _, centers = generate_clusters(n_clusters=N_EXPERTS, d=D_INPUT)
    
    # Cargar modelo
    model = MoRE(N_EXPERTS, D_INPUT, M_PROTOS)
    try:
        model.load_state_dict(torch.load("more_model.pt"))
        model.eval()
        console.log("[bold green]Model loaded successfully.[/bold green]")
        evaluate_novelty(model, centers)
    except FileNotFoundError:
        console.log("[bold red]Model not found. Run train_demo.py first.[/bold red]")
