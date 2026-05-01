import torch
import torch.nn.functional as F
from more_tiny_llm import load_more_gpt
from tiny_stories_loader import TinyStoriesLoader

def audit_generate(model, loader, prompt, expert_idx, max_new_tokens=50):
    """
    Generates text by forcing all tokens through a specific expert.
    """
    model.eval()
    idx = torch.tensor(loader.encode(prompt), dtype=torch.long).unsqueeze(0)
    generated_ids = idx.tolist()[0]
    
    for _ in range(max_new_tokens):
        # Force expert_idx
        logits = model(torch.tensor([generated_ids]), reset_state=True, forced_expert_idx=expert_idx)
        
        next_token_logits = logits[:, -1, :] / 0.8 # Lower temp for clearer patterns
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        generated_ids.append(token_id)
        
        if loader.decode([token_id]) == "\n":
            break
            
    return loader.decode(generated_ids[len(idx[0]):])

def run_expert_audit():
    print("=== MoRE-GPT Expert Coherence Audit ===")
    
    # 1. Load the bifurcated model (which has 5 experts)
    try:
        model = load_more_gpt("more_gpt_bifurcated.pth")
    except FileNotFoundError:
        print("Bifurcated model not found. Running bifurcation_demo.py first...")
        import bifurcation_demo
        bifurcation_demo.run_bifurcation_demo()
        model = load_more_gpt("more_gpt_bifurcated.pth")
        
    loader = TinyStoriesLoader("tinystories_subset.txt")
    
    prompt = "One day, a"
    print(f"\nPrompt: '{prompt}'")
    print("-" * 50)
    
    for i in range(len(model.expert_pool.experts)):
        print(f"Expert {i} Output: ", end="")
        output = audit_generate(model, loader, prompt, i)
        print(f"'{output.strip()}'")
        
    print("-" * 50)
    print("Audit Complete.")

if __name__ == "__main__":
    run_expert_audit()
