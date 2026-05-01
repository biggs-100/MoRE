import torch
import torch.nn as nn
from more_tiny_llm import MoREGPT, load_more_gpt
from tiny_stories_loader import TinyStoriesLoader
from rich.console import Console
from rich.text import Text

console = Console()

def analyze_specialization(model_path="more_gpt_organism.pth", n_samples=3):
    console.rule("[bold magenta]MoRE-4 Expert Specialization Audit[/bold magenta]")
    
    # 1. Load evolved organism
    checkpoint = torch.load(model_path)
    model = MoREGPT(checkpoint['vocab_size'], checkpoint['d_model'], checkpoint['n_experts'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    console.print(f"Loaded organism with [bold cyan]{len(model.expert_pool.experts)}[/bold cyan] experts.")
    
    # 2. Setup Data
    loader = TinyStoriesLoader("tinystories_subset.txt", batch_size=1, seq_len=128)
    
    # 3. Colors for experts
    expert_colors = ["cyan", "green", "yellow", "magenta", "red", "blue"]
    
    for s in range(n_samples):
        x, y = loader.get_batch()
        # loader has encode/decode directly
        
        # We need to run token by token to capture the expert choice
        output_text = Text()
        expert_counts = [0] * len(model.expert_pool.experts)
        
        console.print(f"\n[bold]Sample {s+1}:[/bold]")
        
        with torch.no_grad():
            # Process sequence
            for i in range(x.size(1)):
                char_idx = x[:, i]
                # Embed
                emb = model.tok_emb(char_idx)
                
                # Get fidelities directly from pool logic
                fidelities = torch.stack([e.get_rea_fidelity(emb) for e in model.expert_pool.experts], dim=1)
                winner = torch.argmax(fidelities, dim=1).item()
                
                # Run the step to keep state updated
                model(char_idx.unsqueeze(0), reset_state=(i==0))
                
                # Decode single char
                char_str = loader.decode([char_idx.item()])
                color = expert_colors[winner % len(expert_colors)]
                output_text.append(char_str, style=color)
                expert_counts[winner] += 1
                
        console.print(output_text)
        
        # Breakdown
        total = sum(expert_counts)
        breakdown = " | ".join([f"Expert {i}: {c/total:.1%}" for i, c in enumerate(expert_counts)])
        console.print(f"[dim]{breakdown}[/dim]")

    console.print("\n[bold green]Interpretation Guide:[/bold green]")
    console.print("  - If an expert dominates [italic]function words[/italic] (the, a, in, to), it is a [bold cyan]Syntactic Expert[/]")
    console.print("  - If an expert dominates [italic]content words[/italic] (dragon, forest, happy), it is a [bold green]Semantic Expert[/]")
    console.print("  - If an expert handles [italic]punctuation/ends[/italic], it is a [bold yellow]Boundary Expert[/]")

if __name__ == "__main__":
    analyze_specialization()
