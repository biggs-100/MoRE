import torch
import torch.nn as nn
from more_tiny_llm import MoREGPT, train_more_gpt, save_more_gpt
from tiny_stories_loader import download_tinystories, TinyStoriesLoader
from specialization_audit import get_vocab_profile, calculate_jaccard
import numpy as np

def run_divergence_tuning():
    print("=== MoRE-GPT: Controlled Divergence Tuning ===")
    
    # 1. Setup Data (500k chars)
    data_path = "tinystories_tuning.txt"
    download_tinystories(data_path, limit_chars=500000)
    loader = TinyStoriesLoader(data_path, batch_size=32, seq_len=128)
    
    # 2. Setup Model (Throttled capacity: 64 dims)
    model = MoREGPT(vocab_size=loader.vocab_size, d_model=64, n_experts=1, rank=16)
    
    # 3. Apply Evolutionary Pressure
    model.expert_pool.threshold_mitosis = 0.7  # High sensitivity
    model.expert_pool.routing_penalty = 0.5    # 50% penalty

    
    print(f"\n[CONFIG] Threshold: {model.expert_pool.threshold_mitosis} | Penalty: {model.expert_pool.routing_penalty}")
    
    for epoch in range(1, 4):
        print(f"\n--- EPOCH {epoch} ---")
        train_more_gpt(model, loader, epochs=1)
        
        # Audit after each epoch
        n_experts = len(model.expert_pool.experts)
        print(f"  [STATUS] Experts: {n_experts}")
        
        if n_experts > 1:
            profiles = []
            for i in range(n_experts):
                vocab, _ = get_vocab_profile(model, loader, i, length=50)
                profiles.append(vocab)
            
            # Calculate average Jaccard
            overlaps = []
            for i in range(n_experts):
                for j in range(i + 1, n_experts):
                    overlaps.append(calculate_jaccard(profiles[i], profiles[j]))
            
            avg_jaccard = np.mean(overlaps) if overlaps else 1.0
            print(f"  [AUDIT] Average Jaccard: {avg_jaccard:.4f}")
            
            if avg_jaccard < 0.4:
                print("  [SUCCESS] Target Divergence Reached!")
        else:
            print("  [AUDIT] No mitosis yet.")

    print("\n=== Tuning Complete ===")
    save_more_gpt(model, "more_gpt_tuned.pth")

if __name__ == "__main__":
    run_divergence_tuning()
