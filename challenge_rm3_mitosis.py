import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from resonant_mamba3_final import RM3ExpertPool

def generate_dual_domain_data(n_samples=2000, seq_len=10):
    """
    Generates two domains: Sine-based counting and Square-based counting.
    """
    # Domain A: Sine
    t = torch.linspace(0, 1, 16)
    x_sine = torch.sin(2 * np.pi * t).repeat(n_samples // 2, seq_len, 1)
    x_sine += torch.randn_like(x_sine) * 0.01
    y_sine = torch.cumsum(x_sine.sum(dim=-1, keepdim=True), dim=1)
    
    # Domain B: Square
    x_square = torch.sign(torch.sin(4 * np.pi * t)).repeat(n_samples // 2, seq_len, 1)
    x_square += torch.randn_like(x_square) * 0.01
    y_square = torch.cumsum(x_square.sum(dim=-1, keepdim=True), dim=1)
    
    return x_sine, y_sine, x_square, y_square

def run_mitosis_challenge():
    print("=== [RM3 Autonomous Mitosis Challenge] ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Architecture
    encoder = nn.Linear(16, 384).to(device)
    pool = RM3ExpertPool(d_model=384, rank=32, threshold_mitosis=0.45).to(device)
    decoder = nn.Linear(384, 1).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(pool.parameters()) + list(decoder.parameters()), lr=0.005)
    criterion = nn.MSELoss()
    
    x_a, y_a, x_b, y_b = generate_dual_domain_data()
    x_a, y_a = x_a.to(device), y_a.to(device)
    x_b, y_b = x_b.to(device), y_b.to(device)

    # Phase 1: Training on Domain A only
    print("\n[Phase 1] Training on Domain A (Sine)...")
    for epoch in range(100):
        idx = torch.randint(0, len(x_a), (32,))
        xs, ys = x_a[idx], y_a[idx]
        
        optimizer.zero_grad()
        loss = 0
        for t in range(xs.size(1)):
            h = encoder(xs[:, t])
            # Reset state only on the first step
            out = pool(h, reset_state=(t == 0))
            pred = decoder(out).squeeze()
            loss += criterion(pred, ys[:, t, 0])
        
        loss /= xs.size(1)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Step {epoch}, Loss: {loss.item():.6f}, Experts: {len(pool.experts)}")

    # Phase 2: Introduce Domain B (The Crisis)
    print("\n[Phase 2] Introducing Domain B (Square) - Expecting Mitosis...")
    for epoch in range(200):
        # Mix A and B
        idx_a = torch.randint(0, len(x_a), (16,))
        idx_b = torch.randint(0, len(x_b), (16,))
        xs = torch.cat([x_a[idx_a], x_b[idx_b]], dim=0)
        ys = torch.cat([y_a[idx_a], y_b[idx_b]], dim=0)
        
        optimizer.zero_grad()
        loss = 0
        for t in range(xs.size(1)):
            h = encoder(xs[:, t])
            # Reset state only on the first step
            out = pool(h, reset_state=(t == 0))
            pred = decoder(out).squeeze()
            loss += criterion(pred, ys[:, t, 0])
        
        loss /= xs.size(1)
        loss.backward()
        optimizer.step()
        
        # Check for Mitosis
        if pool.check_mitosis():
            print(f"  !!! MITOSIS TRIGGERED at step {epoch} !!!")
            # Update optimizer to include new expert
            optimizer = optim.Adam(list(encoder.parameters()) + list(pool.parameters()) + list(decoder.parameters()), lr=0.002)
        
        if epoch % 50 == 0:
            print(f"  Step {epoch}, Loss: {loss.item():.6f}, Experts: {len(pool.experts)}")

    print("\n=== [Challenge Complete] ===")
    print(f"Final Expert Count: {len(pool.experts)}")
    torch.save(pool.state_dict(), "rm3_mitosis_final.pth")
    print("Model saved to rm3_mitosis_final.pth")

if __name__ == "__main__":
    run_mitosis_challenge()
