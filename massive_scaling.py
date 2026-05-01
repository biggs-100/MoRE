import torch
import torch.nn as nn
from more_tiny_llm import MoREGPT, train_more_gpt, save_more_gpt
from tiny_stories_loader import download_tinystories, TinyStoriesLoader
from specialization_audit import get_vocab_profile, calculate_jaccard
import numpy as np
import time

def run_massive_scaling():
    print("=== MoRE-GPT: MASSIVE SOVEREIGN SCALING ===")
    print("Environment: Full TinyStories (5M+ chars)")
    print("Configuration: d_model=128, threshold=0.7, penalty=0.5")
    
    # 1. Setup Data (5M characters)
    data_path = "tinystories_massive.txt"
    download_tinystories(data_path, limit_chars=5000000)
    loader = TinyStoriesLoader(data_path, batch_size=32, seq_len=128)
    
    # 2. Setup Model
    model = MoREGPT(vocab_size=loader.vocab_size, d_model=128, n_experts=1, rank=32)
    model.expert_pool.threshold_mitosis = 0.7
    model.expert_pool.routing_penalty = 0.5
    
    print(f"\n[INIT] Model initialized with {len(model.expert_pool.experts)} expert.")
    
    start_time = time.time()
    
    for epoch in range(1, 11):
        print(f"\n--- EPOCH {epoch}/10 ---")

        
        # Training loop
        train_more_gpt(model, loader, epochs=1)
        
        # Monitoring Metrics
        n_experts = len(model.expert_pool.experts)
        print(f"\n  [TELEMETRY]")
        print(f"    Current Experts: {n_experts}")
        
        # Calculate Diversity if we have splits
        if n_experts > 1:
            profiles = []
            for i in range(min(n_experts, 5)): # Audit top 5 to save time
                vocab, _ = get_vocab_profile(model, loader, i, length=50)
                profiles.append(vocab)
            
            overlaps = []
            for i in range(len(profiles)):
                for j in range(i + 1, len(profiles)):
                    overlaps.append(calculate_jaccard(profiles[i], profiles[j]))
            
            avg_jaccard = np.mean(overlaps) if overlaps else 1.0
            print(f"    Avg Jaccard (Diversity): {avg_jaccard:.4f}")
        
        # Periodical Save
        save_more_gpt(model, f"more_gpt_massive_epoch_{epoch}.pth")
        
    end_time = time.time()
    print(f"\n[MISSION COMPLETE] Total duration: {(end_time - start_time)/60:.2f} minutes")
    print(f"Final Expert Pool Size: {len(model.expert_pool.experts)}")
    save_more_gpt(model, "more_gpt_massive_final.pth")

if __name__ == "__main__":
    run_massive_scaling()
