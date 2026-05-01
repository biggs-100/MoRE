import torch
import torch.nn as nn
from more_tiny_llm import MoREGPT, train_more_gpt, save_more_gpt
from tiny_stories_loader import download_tinystories, TinyStoriesLoader
import random

def run_sovereign_organism():
    print("=== MoRE-GPT: Sovereign Organism Simulation ===")
    print("Goal: Observe autonomous growth during environmental shifts (Latent Drift).")
    
    # 1. Setup Data
    data_path = "tinystories_scaling.txt"
    download_tinystories(data_path, limit_chars=500000)
    loader = TinyStoriesLoader(data_path, batch_size=32, seq_len=128)
    
    # 2. Setup Model
    model = MoREGPT(vocab_size=loader.vocab_size, d_model=128, n_experts=1, rank=32)
    model.expert_pool.threshold_mitosis = 0.5 # Sensitive
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\n[LIVE] Starting Life Cycle...")
    
    for cycle in range(1, 6):
        print(f"\n--- CYCLE {cycle} ---")
        
        # A. Normal Training (Homeostasis)
        model.train()
        total_loss = 0
        n_batches = 30
        for _ in range(n_batches):
            x, y = loader.get_batch()
            optimizer.zero_grad()
            with torch.no_grad(): emb = model.tok_emb(x)
            logits = model(x, reset_state=True)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            model.update_anchors(x, emb)
            total_loss += loss.item()
            
        print(f"  [LEARN] Loss: {total_loss/n_batches:.4f} | Experts: {len(model.expert_pool.experts)}")
        
        # B. Environmental Shift (Drift)
        if cycle % 2 == 0:
            print("  [SHIFT] The latent world is drifting! Perturbing embeddings...")
            with torch.no_grad():
                # Systemic drift
                drift = torch.eye(model.d_model) + torch.randn(model.d_model, model.d_model) * 0.2
                model.tok_emb.weight.copy_(torch.matmul(model.tok_emb.weight, drift))
            
            # Observe the crisis (populate fidelity_history)
            with torch.no_grad():
                for _ in range(10): # Observe 10 batches of drift
                    x_c, _ = loader.get_batch()
                    model(x_c, reset_state=True)

            
            # C. Autonomous Response
            print("  [CRISIS] Experts detecting fidelity drop...")
            # We must check for mitosis BEFORE aligning, otherwise alignment hides the stress
            mitosis_happened = model.expert_pool.check_mitosis()
            if mitosis_happened:
                print(f"    [MITOSIS] Organism expanded! New size: {len(model.expert_pool.experts)}")
            
            print("  [REA] Performing homeostasis alignment...")
            for i, expert in enumerate(model.expert_pool.experts):
                expert.align_to_encoder(model.tok_emb, "cpu")
            
            print(f"  [STATUS] Expert Pool Size: {len(model.expert_pool.experts)}")


    print("\n=== Life Cycle Complete ===")
    print(f"Final Organism State: {len(model.expert_pool.experts)} experts.")
    save_more_gpt(model, "more_gpt_organism.pth")

if __name__ == "__main__":
    run_sovereign_organism()
