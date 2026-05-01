import torch
import torch.nn.functional as F
from more_tiny_llm import load_more_gpt
from tiny_stories_loader import TinyStoriesLoader

def generate(model, loader, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    Generates text from a prompt using MoRE-GPT.
    """
    model.eval()
    
    # Tokenize prompt
    idx = torch.tensor(loader.encode(prompt), dtype=torch.long).unsqueeze(0)
    
    # Initialize hidden state for recurrent model
    # Note: Our MoREGPT forward loop handles step_reset. 
    # For generation, we want to maintain state across steps.
    
    generated_ids = idx.tolist()[0]
    
    print(f"\n[PROMPT]: {prompt}")
    print("[GENERATED]: ", end="", flush=True)
    
    # Current implementation of MoREGPT forward takes the whole sequence
    # For efficient generation, we should ideally have a step-by-step forward
    # but for this prototype, we'll just pass the full prefix (recurrent state handles it)
    
    # Actually, MoREGPT.forward iterates over time steps.
    # To maintain state efficiently, we should manage the expert pool state.
    
    for _ in range(max_new_tokens):
        # We pass the full sequence so far, but we could optimize by only passing the last token
        # if the experts were already warmed up.
        # For now, we'll pass the whole generated sequence.
        # We only reset state at the VERY beginning of the sequence.
        
        logits = model(torch.tensor([generated_ids]), reset_state=True)
        
        # Take the logits for the last token
        next_token_logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        generated_ids.append(token_id)
        
        char = loader.decode([token_id])
        print(char, end="", flush=True)
        
        if char == "\n" or len(generated_ids) > 200:
            break
            
    print("\n")
    return loader.decode(generated_ids)

if __name__ == "__main__":
    # 1. Load model and loader
    try:
        model = load_more_gpt("more_gpt_bifurcated.pth")
    except FileNotFoundError:
        model = load_more_gpt("more_gpt.pth")

        
    loader = TinyStoriesLoader("tinystories_subset.txt", batch_size=1, seq_len=64)
    
    # 2. Generate samples
    prompts = [
        "Once upon a time, there was a little",
        "Lily saw a big",
        "The sun was",
    ]
    
    for p in prompts:
        generate(model, loader, p, max_new_tokens=100, top_k=5)
