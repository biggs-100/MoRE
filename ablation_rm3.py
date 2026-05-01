import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from resonant_mamba3_final import ResonantMamba3Layer
from alignment_core import AlignableModule

class RealMamba3Layer(AlignableModule):
    """
    Real-valued ablation of ResonantMamba3Layer.
    Uses double the rank (rank * 2) to compensate for the lack of imaginary state,
    matching memory and parameter budgets.
    """
    def __init__(self, d_model=384, rank=64): # Double rank compared to RM3's 32
        super().__init__(d_model)
        self.rank = rank
        
        # Input projections (gate, x_val) -> rank * 2
        self.in_proj = nn.Linear(d_model, rank * 2, bias=False)
        
        # Real decay
        self.A_log = nn.Parameter(torch.log(torch.ones(rank))) 
        
        # Discretization
        self.dt_proj = nn.Linear(d_model, rank, bias=False)
        
        # Output Projection
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        
        self.register_buffer("state", None, persistent=False)

    def reset_state(self, batch_size, device):
        self.state = torch.zeros(batch_size, self.rank, device=device)

    def forward(self, x, reset_state=False):
        # We omit REA alignment here since we are only testing recurrence capability
        batch_size = x.size(0)
        if reset_state or self.state is None or self.state.size(0) != batch_size:
            self.reset_state(batch_size, x.device)
            
        projected = self.in_proj(x)
        gate, x_val = torch.chunk(projected, 2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x))
        decay = torch.exp(-dt * torch.exp(self.A_log))
        
        # Pure real recurrence
        new_state = self.state * decay + torch.sigmoid(gate) * x_val
        self.state = new_state
        
        out = self.out_proj(new_state)
        return out


def generate_sequence_task(seq_len=100, batch_size=32, d_model=64):
    """
    Generates a temporal sequence task where the model must remember a phase-shifted 
    signal. This tests the ability to model continuous rotations and memory.
    """
    # Base signal is a combination of sines
    t = torch.linspace(0, 10, seq_len)
    signal = torch.sin(t) + 0.5 * torch.cos(2*t)
    
    # Expand to d_model and add noise
    X = signal.view(1, seq_len, 1).expand(batch_size, seq_len, d_model).clone()
    X += torch.randn_like(X) * 0.1
    
    # Target is predicting the next step
    y = torch.roll(X, shifts=-1, dims=1)
    return X, y

def train_model(model_name, model, epochs=50, seq_len=100, d_model=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        X, y = generate_sequence_task(seq_len=seq_len, batch_size=16, d_model=d_model)
        
        # Unroll sequence
        outputs = []
        for i in range(seq_len):
            reset = (i == 0)
            out = model(X[:, i, :], reset_state=reset)
            outputs.append(out)
            
        preds = torch.stack(outputs, dim=1)
        loss = criterion(preds, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    duration = time.time() - start_time
    final_loss = sum(losses[-5:]) / 5
    
    print(f"{model_name:15s} | Final Loss: {final_loss:.4f} | Time: {duration:.2f}s")
    return final_loss, duration

if __name__ == "__main__":
    d_model = 64
    seq_len = 200
    
    print("--- RM3 Ablation Study ---")
    print(f"Task: Next-step sequence prediction (Seq Len: {seq_len}, Dim: {d_model})")
    print("Comparing Resonant (Complex) vs Real-Valued Mamba (Double Rank)\n")
    
    # 1. Complex RM3 (Rank 16)
    rm3 = ResonantMamba3Layer(d_model=d_model, rank=16)
    
    # 2. Real Mamba (Rank 32 to match param/memory budget)
    real_mamba = RealMamba3Layer(d_model=d_model, rank=32)
    
    torch.manual_seed(42)
    loss_rm3, t_rm3 = train_model("RM3 (Complex)", rm3, epochs=100, seq_len=seq_len, d_model=d_model)
    
    torch.manual_seed(42)
    loss_real, t_real = train_model("Real Mamba", real_mamba, epochs=100, seq_len=seq_len, d_model=d_model)
    
    print("\n--- Conclusion ---")
    if loss_rm3 < loss_real:
        improvement = ((loss_real - loss_rm3) / loss_real) * 100
        print(f"RM3 outperforms Real Mamba by {improvement:.1f}% in final loss.")
        print("This empirically justifies the use of complex-valued state recurrence.")
    else:
        print("Real Mamba outperformed or matched RM3. Complex values may not be strictly necessary for this task.")
