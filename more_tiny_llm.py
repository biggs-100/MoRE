import torch
import torch.nn as nn
from resonant_mamba3_final import RM3ExpertPool

class MoREGPT(nn.Module):
    """
    Self-evolving GPT-style model using MoRE-4 Experts.
    """
    def __init__(self, vocab_size, d_model=384, n_experts=1, rank=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. Embedding (The 'Encoder' for REA)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # 2. Resonant Expert Pool (The sequence backbone)
        self.expert_pool = RM3ExpertPool(d_model, rank=rank)
        # Initialize with n_experts if requested
        while len(self.expert_pool.experts) < n_experts:
            self.expert_pool.perform_mitosis(0)
            
        # 3. Output Head
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tieing (Optional but standard for small models)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, reset_state=False, forced_expert_idx=None):
        """
        Forward pass for sequences.
        idx: [batch, seq_len]
        """
        b, t = idx.size()
        
        # 1. Embed tokens
        x = self.tok_emb(idx) # [batch, seq_len, d_model]
        
        # 2. Process through MoRE Pool
        outputs = []
        for i in range(t):
            step_reset = reset_state if i == 0 else False
            step_out = self.expert_pool(x[:, i, :], reset_state=step_reset, forced_expert_idx=forced_expert_idx)
            outputs.append(step_out)
            
        x = torch.stack(outputs, dim=1) # [batch, seq_len, d_model]
        
        # 3. Head
        x = self.ln(x)
        logits = self.head(x) # [batch, seq_len, vocab_size]
        
        return logits


    def update_anchors(self, idx, embeddings):
        """
        Populates anchor buffers of winners to enable REA alignment.
        idx: [batch, seq_len]
        embeddings: [batch, seq_len, d_model] (original embeddings)
        """
        with torch.no_grad():
            b, t = idx.size()
            # We take a subset of tokens to avoid bloating the buffer
            for i in range(b):
                x_sample = embeddings[i, 0, :]
                
                # Find winner
                fidelities = torch.stack([e.get_rea_fidelity(x_sample.unsqueeze(0)) for e in self.expert_pool.experts])
                winner = torch.argmax(fidelities)
                
                # Add to winner's buffer
                # x_sample is the embedding at time of learning
                # idx[i, 0] is the raw token index
                self.expert_pool.experts[winner].anchor_buffer.add(idx[i, 0].unsqueeze(0), x_sample)

def train_more_gpt(model, loader, epochs=5, lr=1e-3):
    from rich.progress import Progress
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Training MoRE-GPT...", total=epochs)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 50 # Reduced for demonstration
            
            for b_idx in range(n_batches):
                x, y = loader.get_batch()
                
                optimizer.zero_grad()
                
                # We record embeddings for anchor updates BEFORE training update
                with torch.no_grad():
                    emb = model.tok_emb(x)
                
                logits = model(x, reset_state=True)
                
                # Reshape for loss
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                
                # Update REA anchors
                model.update_anchors(x, emb)
                
                total_loss += loss.item()
                
            avg_loss = total_loss / n_batches
            
            # Check for Mitosis
            model.expert_pool.check_mitosis()
            
            progress.update(task, advance=1, description=f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Experts: {len(model.expert_pool.experts)}")
            
    print("Training complete.")

def save_more_gpt(model, path="more_gpt.pth"):
    torch.save({
        'state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'n_experts': len(model.expert_pool.experts)
    }, path)
    print(f"Model saved to {path}")

def load_more_gpt(path="more_gpt.pth"):
    checkpoint = torch.load(path)
    model = MoREGPT(checkpoint['vocab_size'], checkpoint['d_model'], checkpoint['n_experts'])
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from {path}")
    return model

def drift_stress_test(model, loader):
    """
    Simulates embedding drift and verifies REA recovery.
    """
    print("\n[STRESS TEST] Starting REA Resilience Audit...")
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # 1. Baseline
    x, y = loader.get_batch()
    with torch.no_grad():
        logits = model(x, reset_state=True)
        baseline_loss = criterion(logits.view(-1, model.vocab_size), y.view(-1)).item()
        baseline_fidelity = torch.stack([e.get_rea_fidelity(model.tok_emb(x)[:, 0, :]) for e in model.expert_pool.experts]).mean().item()
    
    print(f"  - Baseline Loss: {baseline_loss:.4f}")
    print(f"  - Baseline Fidelity: {baseline_fidelity:.4f}")
    
    # 2. Inject Drift
    print("  [DRIFT] Injecting systemic representation drift...")
    with torch.no_grad():
        # Apply a systemic linear transformation + small noise
        drift_transform = torch.eye(model.d_model) + torch.randn(model.d_model, model.d_model) * 0.2
        model.tok_emb.weight.copy_(torch.matmul(model.tok_emb.weight, drift_transform))
        # Add a bit of bias too
        model.tok_emb.weight.add_(torch.randn(model.d_model) * 0.1)

        
    with torch.no_grad():
        logits = model(x, reset_state=True)
        drift_loss = criterion(logits.view(-1, model.vocab_size), y.view(-1)).item()
        drift_fidelity = torch.stack([e.get_rea_fidelity(model.tok_emb(x)[:, 0, :]) for e in model.expert_pool.experts]).mean().item()
    
    print(f"  - Post-Drift Loss: {drift_loss:.4f} (Spike!)")
    print(f"  - Post-Drift Fidelity: {drift_fidelity:.4f} (Drop!)")
    
    # 3. REA Recovery
    print("  [REA] Aligning experts to drifted encoder...")
    device = next(model.parameters()).device
    for i, expert in enumerate(model.expert_pool.experts):
        success = expert.align_to_encoder(model.tok_emb, device)
        if success:
            print(f"    Expert {i} aligned.")
            
    with torch.no_grad():
        logits = model(x, reset_state=True)
        recovered_loss = criterion(logits.view(-1, model.vocab_size), y.view(-1)).item()
        recovered_fidelity = torch.stack([e.get_rea_fidelity(model.tok_emb(x)[:, 0, :]) for e in model.expert_pool.experts]).mean().item()
        
    print(f"  - Recovered Loss: {recovered_loss:.4f}")
    print(f"  - Recovered Fidelity: {recovered_fidelity:.4f}")
    
    if recovered_loss < drift_loss or recovered_fidelity > drift_fidelity:
        print("[SUCCESS] REA neutralized representation drift.")
    else:
        print("[WARNING] REA recovery was insufficient.")

if __name__ == "__main__":
    from tiny_stories_loader import download_tinystories, TinyStoriesLoader
    
    # 1. Setup Data
    download_tinystories("tinystories_subset.txt", limit_chars=50000)
    loader = TinyStoriesLoader("tinystories_subset.txt", batch_size=16, seq_len=64)
    
    # 2. Setup Model
    model = MoREGPT(vocab_size=loader.vocab_size, d_model=128, n_experts=1)
    
    # 3. Train
    # Using 5 epochs for the demo
    train_more_gpt(model, loader, epochs=5)
    
    # 4. Stress Test
    drift_stress_test(model, loader)
    
    # 5. Manual Mitosis Test
    print("\n[MITOSIS] Forcing expert mitosis...")
    model.expert_pool.perform_mitosis(0) # Split the first expert
    print(f"  - New Expert Count: {len(model.expert_pool.experts)}")
    
    with torch.no_grad():
        x, y = loader.get_batch()
        logits = model(x, reset_state=True)
        print(f"  - Forward pass successful with {len(model.expert_pool.experts)} experts.")
        print(f"  - Post-Mitosis Logits shape: {logits.shape}")

    # 6. Final Save
    save_more_gpt(model)



