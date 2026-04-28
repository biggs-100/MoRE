"""
Evaluation for Real Text Benchmark: Novelty Detection.
"""

import torch
from more_demo import MoRE
from real_dataset import TextDataset
from rich.console import Console
from rich.table import Table

console = Console()

def evaluate_real():
    dataset = TextDataset()
    data = dataset.get_data()
    X_train = data['train_x']
    y_train = data['train_y']
    X_novel = data['novel_x']
    
    D_INPUT = X_train.shape[1]
    N_EXPERTS = 3
    M_PROTOS = 5
    
    # Cargar modelo
    model = MoRE(N_EXPERTS, D_INPUT, M_PROTOS)
    try:
        model.load_state_dict(torch.load("more_real_text.pt"))
        model.eval()
        console.log("[bold green]Benchmark model loaded.[/bold green]")
    except FileNotFoundError:
        console.log("[bold red]Model not found. Run train_real.py first.[/bold red]")
        return

    # 1. Evaluar conocidos
    # Usamos un umbral para detectar "No se" incluso en conocidos si la confianza es baja
    preds_known, scores_known = model.predict(X_train, threshold=0.5)
    acc_known = (preds_known == y_train).float().mean().item()
    
    # 2. Evaluar novedad (Salud)
    preds_novel, scores_novel = model.predict(X_novel, threshold=0.5)
    rejection_novel = (preds_novel == -1).float().mean().item()
    
    # Resultados
    table = Table(title="MoRE Real Text Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Known Accuracy (Sports, Tech, Politics)", f"{acc_known:.2%}")
    table.add_row("Novelty Detection Rate (Health)", f"{rejection_novel:.2%}")
    table.add_row("Avg Familiarity Known", f"{scores_known.mean():.3f}")
    table.add_row("Avg Familiarity Novel", f"{scores_novel.mean():.3f}")
    
    console.print(table)
    
    # Analisis cualitativo
    if rejection_novel > 0.9:
        console.print("[bold green]Success: Novelty detected reliably in real text![/bold green]")
    else:
        console.print("[bold yellow]Warning: Novelty detection could be improved. Consider tuning threshold.[/bold yellow]")

if __name__ == "__main__":
    evaluate_real()
