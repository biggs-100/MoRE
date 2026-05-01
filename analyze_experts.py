import torch
from resonant_mamba3_final import RM3ExpertPool

def analyze_diversity():
    print("=== [Expert Diversity Analysis] ===")
    pool = RM3ExpertPool(d_model=384, rank=32)
    # We need to add the other 4 experts to the pool before loading state_dict
    for _ in range(4):
        pool.perform_mitosis(0)
    
    pool.load_state_dict(torch.load("rm3_mitosis_final.pth", map_location="cpu"))
    
    for i, expert in enumerate(pool.experts):
        a_imag_mean = expert.A_imag.mean().item()
        a_imag_std = expert.A_imag.std().item()
        print(f"Expert {i}: Mean Frequency: {a_imag_mean:.6f}, Std: {a_imag_std:.6f}")
        
    # Check for duplicates
    means = [e.A_imag.mean().item() for e in pool.experts]
    if len(set(means)) == len(means):
        print("\n[VERIFIED] All experts have unique resonance signatures. Symmetry broken successfully.")
    else:
        print("\n[WARNING] Duplicate signatures found. Symmetry breaking may be insufficient.")

if __name__ == "__main__":
    analyze_diversity()
