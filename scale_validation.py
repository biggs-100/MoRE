import torch
from more_tiny_llm import MoREGPT
from tiny_stories_loader import TinyStoriesLoader
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress

console = Console()

def run_scale_validation():
    console.rule("[bold magenta]MoRE-4 Massive Scaling Validation[/bold magenta]")
    
    # Setup for scaling with high sensitivity
    d_model = 128
    loader = TinyStoriesLoader("tinystories_subset.txt", batch_size=32, seq_len=128)
    # Force high sensitivity to show scaling (threshold=0.75)
    model = MoREGPT(vocab_size=loader.vocab_size, d_model=d_model, n_experts=1)
    model.expert_pool.threshold_mitosis = 0.75 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    max_steps = 2000 # Increased steps
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Scaling Experts...", total=max_steps)
        
        for step in range(max_steps):
            x, y = loader.get_batch()
            optimizer.zero_grad()
            
            logits = model(x, reset_state=True)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            # Monitoring Experts
            n_experts = len(model.expert_pool.experts)
            if step % 50 == 0:
                progress.update(task, advance=50, description=f"[cyan]Step {step} | Experts: {n_experts} | Loss: {loss.item():.4f}")
            
            # Success condition for the paper: reach 4 experts
            if n_experts >= 4:
                console.print(f"\n[bold green]SUCCESS: Reached {n_experts} experts at step {step}[/bold green]")
                break
                
    console.print(f"Final Expert Count: {len(model.expert_pool.experts)}")
    # Save the scaled model
    torch.save(model.state_dict(), "more_scaled_4experts.pt")

if __name__ == "__main__":
    run_scale_validation()
