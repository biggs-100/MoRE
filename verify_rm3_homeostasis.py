import torch
import torch.nn as nn
import torch.nn.functional as F
from alignment_core import GradientsProjector
from mamba3_prototype import ResonantMamba3Layer

def verify_rm3():
    print("=== [RM3 Homeostasis Verification] ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Environment
    # Encoder produces the input for our RM3 layer
    encoder = nn.Linear(128, 384).to(device)
    rm3_layer = ResonantMamba3Layer(d_model=384, rank=8).to(device)
    
    # 2. Baseline Learning (Calibration)
    print("\n[Step 1] Calibrating Layer and Anchors...")
    raw_x = torch.randn(20, 128).to(device) # 20 anchor samples
    with torch.no_grad():
        h_latent = encoder(raw_x)
        # Process and store as anchors
        # In RM3, the latent features are what we align
        for i in range(len(raw_x)):
            rm3_layer.anchor_buffer.add(raw_x[i:i+1], h_latent[i:i+1])
    
    # 3. Simulate Encoder Drift
    print("\n[Step 2] Simulating Encoder Drift (Injecting Noise)...")
    with torch.no_grad():
        # Drastic drift: 30% noise
        encoder.weight.add_(torch.randn_like(encoder.weight) * 0.3)
    
    # Check drift impact
    with torch.no_grad():
        h_drifted = encoder(raw_x)
        mse_drift = F.mse_loss(h_drifted, h_latent).item()
        print(f"  MSE due to Drift: {mse_drift:.4f}")
    
    # 4. Perform REA (Alignment)
    print("\n[Step 3] Triggering REA (Alignment Phase)...")
    success = rm3_layer.align_to_encoder(encoder, device)
    if success:
        with torch.no_grad():
            h_new = encoder(raw_x)
            h_aligned = rm3_layer.apply_alignment(h_new)
            mse_recovered = F.mse_loss(h_aligned, h_latent).item()
            print(f"  MSE after REA: {mse_recovered:.4f}")
            improvement = (1 - mse_recovered/mse_drift) * 100
            print(f"  [Homeostasis] Manifold Recovery: {improvement:.2f}%")
    
    # 5. Verify Resonant OGD
    print("\n[Step 4] Verifying Resonant OGD (Gradient Protection)...")
    ogd = GradientsProjector(encoder)
    
    # Compute basis for the RM3 layer using current anchors
    _, anchors_h = rm3_layer.anchor_buffer.get_anchors()
    ogd.update_basis("RM3_Expert_0", anchors_h.to(device))
    
    # Fake backprop to generate gradients
    loss = encoder(raw_x).sum()
    loss.backward()
    
    grad_norm_before = encoder.weight.grad.norm().item()
    print(f"  Grad Norm before OGD: {grad_norm_before:.4f}")
    
    # Project!
    ogd.project()
    
    grad_norm_after = encoder.weight.grad.norm().item()
    print(f"  Grad Norm after OGD: {grad_norm_after:.4f}")
    
    if grad_norm_after < grad_norm_before:
        print(f"  [OGD Success] Protected Subspace Diminished Gradient by {((1 - grad_norm_after/grad_norm_before)*100):.2f}%")
    
    print("\n=== [Verification Complete] ===")

if __name__ == "__main__":
    verify_rm3()
