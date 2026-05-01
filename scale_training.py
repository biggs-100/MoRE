import torch
from more_tiny_llm import MoREGPT, train_more_gpt, save_more_gpt
from tiny_stories_loader import download_tinystories, TinyStoriesLoader

def run_scaling_phase():
    print("=== MoRE-GPT Scaling Phase: Full Organism Growth ===")
    
    # 1. Setup Data (500k chars for a solid training session)
    data_path = "tinystories_scaling.txt"
    download_tinystories(data_path, limit_chars=500000)
    loader = TinyStoriesLoader(data_path, batch_size=32, seq_len=128)
    
    # 2. Setup Model (Smaller capacity to force stress)
    model = MoREGPT(vocab_size=loader.vocab_size, d_model=128, n_experts=1, rank=32)
    
    # 3. Training with Active Mitosis
    # threshold_mitosis = 0.5 (Very sensitive)
    model.expert_pool.threshold_mitosis = 0.5

    
    print("\n[GROWTH] Starting large-scale training...")
    # 10 Epochs over 500k chars
    train_more_gpt(model, loader, epochs=10)
    
    print("\n=== Scaling Phase Complete ===")
    print(f"Final Expert Pool Size: {len(model.expert_pool.experts)}")
    
    # 4. Final Save
    save_more_gpt(model, "more_gpt_scaled.pth")
    
    # 5. Quick Test Generation
    from generate import generate
    generate(model, loader, "Once upon a time, a small boy", max_new_tokens=100)

if __name__ == "__main__":
    run_scaling_phase()
