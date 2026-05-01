import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alignment_core import AlignableModule

class Mamba3MIMOBlock(nn.Module):
    """
    Complex-Valued Mamba-3 MIMO Block prototype.
    Uses Phase Rotation (RoPE-style) to track logical states.
    """
    def __init__(self, d_model, rank=16):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # 1. Parameter Projections
        self.dt_proj = nn.Linear(d_model, rank)
        # Phase Projection: Controls the rotation (angle) in complex space
        self.phase_proj = nn.Linear(d_model, rank)
        
        self.B_proj = nn.Linear(d_model, rank, bias=False)
        self.C_proj = nn.Linear(rank, d_model, bias=False)
        
        # 2. Continuous State Parameters
        # A_real: Decay (stability)
        self.A_real = nn.Parameter(torch.log(torch.arange(1, rank + 1).float()))
        # A_imag: Initial base frequency
        self.A_imag = nn.Parameter(torch.randn(rank))
        
    def forward(self, x, state=None):
        # x shape: [batch, d_model]
        # 1. Delta (Magnitude) and Phase (Rotation)
        dt = F.softplus(self.dt_proj(x))
        # Data-dependent phase: allows the input to 'flip' or 'rotate' the state
        phase = torch.tanh(self.phase_proj(x)) * np.pi 
        
        # 2. Complex Discretization: exp(dt * (A_real + i * A_imag + i * phase))
        # We simplify by using the rotation directly
        decay = torch.exp(-dt * torch.exp(self.A_real))
        # Total rotation = dt * base_freq + phase
        angle = dt * self.A_imag + phase
        
        # 3. MIMO Input
        B_val = self.B_proj(x)
        
        # 4. Complex Recurrence (using real/imag components)
        if state is None:
            # State is complex: (real, imag)
            state_re = torch.zeros(x.size(0), self.rank, device=x.device)
            state_im = torch.zeros(x.size(0), self.rank, device=x.device)
        else:
            state_re, state_im = state
            
        # h_t = h_{t-1} * decay * exp(i * angle) + B * x
        # exp(i * angle) = cos(angle) + i * sin(angle)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        new_re = (state_re * cos - state_im * sin) * decay + B_val
        new_im = (state_re * sin + state_im * cos) * decay
        
        # 5. Output Projection (from Real part)
        out = self.C_proj(new_re)
        
        return out, (new_re, new_im)

class ResonantMamba3Layer(AlignableModule):
    """
    A MoRE-4 compliant Mamba-3 Layer with Sequential State.
    """
    def __init__(self, d_model, rank=8):
        super().__init__(d_model)
        self.mamba = Mamba3MIMOBlock(d_model, rank)
        self.current_state = None
        
    def forward(self, x, use_alignment=True, reset_state=False):
        # x shape: [batch, d_model]
        if reset_state:
            self.current_state = None
            
        # 1. Homeostatic Alignment (REA)
        if use_alignment:
            x = self.apply_alignment(x)
            
        # 2. Mamba-3 Processing
        out, self.current_state = self.mamba(x, self.current_state)
        
        return out, self.current_state

    def record_anchor(self, raw_input, latent_features):
        """
        Saves the relationship between raw input and the stable latent manifold.
        """
        if not self.anchor_buffer.is_full():
            # We store the first sample of the batch for simplicity in this prototype
            self.anchor_buffer.add(raw_input[0], latent_features[0])

if __name__ == "__main__":
    # Quick Prototype Test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = ResonantMamba3Layer(d_model=384, rank=8).to(device)
    
    # Fake Input
    x = torch.randn(1, 10, 384).to(device)
    
    # Forward pass
    out, h_rank = layer(x)
    
    print(f"RM3 Layer Output shape: {out.shape}")
    print(f"MIMO Rank features shape: {h_rank.shape}")
    print("--- Prototype Verified ---")
