import torch
import torch.nn.functional as F
from more_tiny_llm import load_more_gpt
from tiny_stories_loader import TinyStoriesLoader

def audit_generate(model, loader, prompt, expert_idx, max_new_tokens=100):
    model.eval()
    idx = torch.tensor(loader.encode(prompt), dtype=torch.long).unsqueeze(0)
    generated_ids = idx.tolist()[0]
    
    for _ in range(max_new_tokens):
        # Force expert_idx
        logits = model(torch.tensor([generated_ids]), reset_state=True, forced_expert_idx=expert_idx)
        
        next_token_logits = logits[:, -1, :] / 0.7 
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        generated_ids.append(token_id)
        
        if loader.decode([token_id]) == "\n":
            break
            
    return loader.decode(generated_ids[len(idx[0]):])

def run_qualitative_audit():
    print("=== MoRE-GPT: Qualitative Expert Audit ===")
    
    model_path = "more_gpt_massive_final.pth"
    try:
        model = load_more_gpt(model_path)
    except FileNotFoundError:
        print("Final massive model not found.")
        return
        
    loader = TinyStoriesLoader("tinystories_massive.txt")
    
    prompts = [
        "Once upon a time, a small boy",
        "The red car was",
        "The sun was very"
    ]
    
    for p_idx, prompt in enumerate(prompts):
        print(f"\n[PROMPT {p_idx+1}] '{prompt}'")
        print("-" * 50)
        
        for e_idx in range(len(model.expert_pool.experts)):
            output = audit_generate(model, loader, prompt, e_idx)
            print(f"Expert {e_idx}: '{output.strip()}'")
            
    print("\n--- Audit Complete ---")

if __name__ == "__main__":
    run_qualitative_audit()
