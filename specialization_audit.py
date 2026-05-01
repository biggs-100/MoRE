import torch
import torch.nn.functional as F
from more_tiny_llm import load_more_gpt
from tiny_stories_loader import TinyStoriesLoader
import numpy as np

def get_vocab_profile(model, loader, expert_idx, prompt="Once upon a time", length=100):
    """
    Generates text and returns the set of unique tokens used.
    """
    model.eval()
    idx = torch.tensor(loader.encode(prompt), dtype=torch.long).unsqueeze(0)
    generated_ids = idx.tolist()[0]
    
    tokens_used = set()
    
    for _ in range(length):
        logits = model(torch.tensor([generated_ids]), reset_state=True, forced_expert_idx=expert_idx)
        next_token_logits = logits[:, -1, :] / 0.8
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        generated_ids.append(token_id)
        tokens_used.add(token_id)
        
    return tokens_used, loader.decode(generated_ids[len(idx[0]):])

def calculate_jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def run_specialization_audit():
    print("=== MoRE-GPT: Specialization & Speciation Audit ===")
    
    # 1. Load the model with 5 experts
    model_path = "more_gpt_bifurcated.pth"
    try:
        model = load_more_gpt(model_path)
    except FileNotFoundError:
        print("Model not found. Run bifurcation_demo.py first.")
        return
        
    loader = TinyStoriesLoader("tinystories_full.txt")

    
    n_experts = len(model.expert_pool.experts)
    print(f"Auditing {n_experts} experts...")
    
    profiles = []
    texts = []
    
    for i in range(n_experts):
        vocab, sample = get_vocab_profile(model, loader, i)
        profiles.append(vocab)
        texts.append(sample)
        print(f"\n[Expert {i}] Sample: '{sample[:60]}...'")
        print(f"  Vocabulary Size: {len(vocab)} unique tokens")

    # 2. Cross-Expert Vocabulary Overlap (Jaccard Index)
    print("\n--- Semantic Overlap Matrix (Jaccard Index) ---")
    print("Lower values = Higher Specialization")
    
    matrix = np.zeros((n_experts, n_experts))
    for i in range(n_experts):
        for j in range(n_experts):
            matrix[i, j] = calculate_jaccard(profiles[i], profiles[j])
            
    # Print matrix
    header = "      " + " ".join([f"Exp{i}" for i in range(n_experts)])
    print(header)
    for i in range(n_experts):
        row = f"Exp{i}: " + " ".join([f"{val:.3f}" for val in matrix[i]])
        print(row)
        
    avg_overlap = (matrix.sum() - n_experts) / (n_experts * (n_experts - 1))
    print(f"\nAverage Cross-Expert Overlap: {avg_overlap:.4f}")
    
    if avg_overlap < 0.4:
        print("RESULT: HIGH SPECIALIZATION. The mitosis successfully partitioned the semantic manifold.")
    elif avg_overlap < 0.7:
        print("RESULT: MODERATE SPECIALIZATION. Experts share a core vocabulary but diverge on details.")
    else:
        print("RESULT: LOW SPECIALIZATION. Experts are redundant. Consider lowering threshold_mitosis.")

if __name__ == "__main__":
    run_specialization_audit()
