import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from mamba3_prototype import ResonantMamba3Layer
from alignment_core import GradientsProjector

def generate_counting_data(n_samples=1000, seq_len=10):
    """
    Generates counting sequences: 16D random values and their cumulative sum.
    """
    X = torch.randn(n_samples, seq_len, 16) * 0.1
    Y = torch.sum(X, dim=-1, keepdim=True) # Target is sum of features
    Y = torch.cumsum(Y, dim=1)
    return X, Y

def run_state_tracking_challenge():
    print("=== [RM3 State-Tracking Challenge: 16D Counting Task] ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Architecture
    encoder = nn.Linear(16, 384).to(device)
    rm3_layer = ResonantMamba3Layer(d_model=384, rank=32).to(device)
    decoder = nn.Linear(384, 1).to(device)
    
    # 2. Initial Training
    print("\n[Phase 1] Training RM3 to solve 16D Cumulative Sum...")
    optimizer = optim.Adam(list(encoder.parameters()) + list(rm3_layer.parameters()) + list(decoder.parameters()), lr=0.005)
    criterion = nn.MSELoss()
    
    X, Y = generate_counting_data(n_samples=3000)
    X, Y = X.to(device), Y.to(device)
    
    for epoch in range(300):
        idx = torch.randint(0, len(X), (64,))
        x_seq, y_seq = X[idx], Y[idx]
        
        optimizer.zero_grad()
        rm3_layer.forward(torch.zeros(64, 384).to(device), reset_state=True)
        
        loss = 0
        for t in range(x_seq.size(1)):
            x_t = x_seq[:, t]
            h_t = encoder(x_t)
            out, _ = rm3_layer(h_t)
            pred = decoder(out).squeeze()
            loss += criterion(pred, y_seq[:, t, 0])
        
        loss /= x_seq.size(1)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")

    # 3. Save Anchors (Homeostatic Memory)
    print("\n[Phase 2] Recording Homeostatic Anchors (1000 samples)...")
    with torch.no_grad():
        anchor_x, _ = generate_counting_data(n_samples=1000)
        anchor_x = anchor_x[:, 0].to(device) # Just the first step for simplicity
        anchor_h = encoder(anchor_x)
        rm3_layer.anchor_buffer.add(anchor_x, anchor_h)

    # 4. The Crisis: Severe Encoder Drift
    print("\n[Phase 3] THE CRISIS: Injecting Severe Encoder Drift...")
    with torch.no_grad():
        encoder.weight.data *= 2.0 # Scale shift
        encoder.weight.add_(torch.randn_like(encoder.weight) * 1.0)
        encoder.bias.add_(torch.randn_like(encoder.bias) * 1.0)

    # Test failure
    with torch.no_grad():
        idx = torch.randint(0, len(X), (64,))
        x_seq, y_seq = X[idx], Y[idx]
        rm3_layer.forward(torch.zeros(64, 384).to(device), reset_state=True)
        total_mse = 0
        for t in range(x_seq.size(1)):
            h_t = encoder(x_seq[:, t])
            out, _ = rm3_layer(h_t)
            pred = decoder(out).squeeze()
            total_mse += criterion(pred, y_seq[:, t, 0]).item()
        print(f"  MSE after Drift: {total_mse/x_seq.size(1):.6f}")

    # 5. The Salvation: REA Alignment
    print("\n[Phase 4] THE SALVATION: Triggering REA Alignment...")
    rm3_layer.align_to_encoder(encoder, device)
    
    # Final Validation
    with torch.no_grad():
        rm3_layer.forward(torch.zeros(64, 384).to(device), reset_state=True)
        total_mse = 0
        for t in range(x_seq.size(1)):
            h_t = encoder(x_seq[:, t])
            out, _ = rm3_layer(h_t)
            pred = decoder(out).squeeze()
            total_mse += criterion(pred, y_seq[:, t, 0]).item()
        print(f"  MSE after REA: {total_mse/x_seq.size(1):.6f}")
        
    print("\n=== [Challenge Complete] ===")

if __name__ == "__main__":
    run_state_tracking_challenge()
