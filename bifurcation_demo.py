import torch
from more_tiny_llm import MoREGPT, train_more_gpt, save_more_gpt
from tiny_stories_loader import download_tinystories, TinyStoriesLoader
import os

def run_bifurcation_demo():
    print("=== MoRE-GPT Phase Bifurcation Demo ===")
    
    # 1. Setup Data
    # Download more characters for diversity
    download_tinystories("tinystories_full.txt", limit_chars=500000)
    base_loader = TinyStoriesLoader("tinystories_full.txt", batch_size=16, seq_len=64)
    
    # Create Domain A: Flowers & Nature
    domain_a = base_loader.filter_by_keywords(["Lily", "flower", "garden", "sun", "bird"])
    # Create Domain B: Trucks & Tools
    domain_b = base_loader.filter_by_keywords(["Tom", "truck", "hammer", "tool", "work"])
    
    print(f"Domain A Size: {len(domain_a.data)} chars")
    print(f"Domain B Size: {len(domain_b.data)} chars")
    
    # 2. Setup Model (Start with 1 expert)
    model = MoREGPT(vocab_size=base_loader.vocab_size, d_model=128, n_experts=1)
    
    # 3. Train on Domain A (Homestasis Phase)
    print("\n[PHASE 1] Training on Domain A (Flowers)...")
    train_more_gpt(model, domain_a, epochs=3)
    print(f"Experts after Phase 1: {len(model.expert_pool.experts)}")
    
    # 4. Transition to Domain B (Stress Phase)
    print("\n[PHASE 2] Transitioning to Domain B (Trucks)...")
    # We lower the mitosis threshold temporarily to make it easier to trigger for the demo
    model.expert_pool.threshold_mitosis = 0.7 # Trigger if fidelity drops below 0.7

    
    train_more_gpt(model, domain_b, epochs=5)
    
    print("\n=== Demo Complete ===")
    print(f"Final Expert Count: {len(model.expert_pool.experts)}")
    
    if len(model.expert_pool.experts) > 1:
        print("[SUCCESS] Autonomous mitosis triggered by domain transition!")
    else:
        print("[INFO] Mitosis not triggered. Perhaps the domains were too similar or threshold too low.")

    save_more_gpt(model, "more_gpt_bifurcated.pth")

if __name__ == "__main__":
    run_bifurcation_demo()
