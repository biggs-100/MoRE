import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alignment_core import AlignableModule, GradientsProjector

class ResonantMamba3Layer(AlignableModule):
    """
    Final Integrated Resonant Mamba-3 Layer.
    Combines MIMO State-Tracking, Complex Phase Rotations, 
    and Homeostatic Alignment (REA).
    """
    def __init__(self, d_model=384, rank=32):
        super().__init__(d_model)
        self.rank = rank
        
        # 1. Input Projections
        self.in_proj = nn.Linear(d_model, rank * 4, bias=False)
        
        # 2. Resonant Parameters (learned)
        self.A_log = nn.Parameter(torch.log(torch.ones(rank))) # Real part (decay)
        self.A_imag = nn.Parameter(torch.arange(rank).float() * (2 * np.pi / rank)) # Imaginary part (freq)
        
        # 3. Data-Dependent Discretization & Phase
        self.dt_proj = nn.Linear(d_model, rank, bias=False)
        self.phase_proj = nn.Linear(d_model, rank, bias=False)
        
        # 4. Output Projection
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        
        # Recurrent State (Complex) - Non-persistent
        self.register_buffer("state_re", None, persistent=False)
        self.register_buffer("state_im", None, persistent=False)

    def reset_state(self, batch_size, device):
        self.state_re = torch.zeros(batch_size, self.rank, device=device)
        self.state_im = torch.zeros(batch_size, self.rank, device=device)

    def get_rea_fidelity(self, x):
        """
        Calculates how well the input aligns with the expert's manifold.
        x: [batch, d_model]
        Returns: [batch] fidelity scores (0 to 1)
        """
        with torch.no_grad():
            x_aligned = self.apply_alignment(x)
            error = torch.norm(x - x_aligned, dim=-1)
            # Use exponential kernel for fidelity
            sigma = 1.0 # Adaptive sigma could be added later
            fidelity = torch.exp(-error / sigma)
        return fidelity

    def mutate(self, scale=0.1):
        """
        Breaks symmetry for mitosis by perturbing frequencies and projections.
        """
        with torch.no_grad():
            # Perturb resonant frequencies (Imaginary part of A)
            self.A_imag.add_(torch.randn_like(self.A_imag) * scale)
            # Perturb input projections slightly
            self.in_proj.weight.add_(torch.randn_like(self.in_proj.weight) * (scale * 0.1))
            # Clear anchor buffer for the new specialized manifold
            self.anchor_buffer.clear()

    def forward(self, x, reset_state=False):
        """
        Forward pass with automatic REA alignment.
        x: [batch, d_model]
        """
        # Apply Homeostasis (REA)
        x = self.apply_alignment(x)
        
        batch_size = x.size(0)
        if reset_state or self.state_re is None or self.state_re.size(0) != batch_size:
            self.reset_state(batch_size, x.device)
            
        # 1. Project to Internal Rank
        # We split for gate, input, and other params
        projected = self.in_proj(x)
        gate, x_val, _, _ = torch.chunk(projected, 4, dim=-1)
        
        # 2. Learn Discretization (dt) and Phase Rotation
        dt = F.softplus(self.dt_proj(x))
        phase = torch.tanh(self.phase_proj(x)) * np.pi
        
        # 3. Calculate Complex Evolution
        # Continuous-to-Discrete (Euler-like for complex)
        decay = torch.exp(-dt * torch.exp(self.A_log))
        angle = dt * self.A_imag + phase
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        # 4. Complex Recurrence: z_t = (z_{t-1} * exp(i*angle)) * decay + x_t
        new_re = (self.state_re * cos - self.state_im * sin) * decay + torch.sigmoid(gate) * x_val
        new_im = (self.state_re * sin + self.state_im * cos) * decay
        
        self.state_re = new_re
        self.state_im = new_im
        
        # 5. Output Projection
        out = self.out_proj(new_re)
        return out

class RM3ExpertPool(nn.Module):
    """
    Manages a collection of RM3 experts with autonomous mitosis capabilities.
    """
    def __init__(self, d_model=384, rank=32, threshold_mitosis=0.4):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.threshold_mitosis = threshold_mitosis
        self.experts = nn.ModuleList([ResonantMamba3Layer(d_model, rank)])
        
        # Performance trackers
        self.fidelity_history = []
        self.window_size = 100

    def forward(self, x, reset_state=False):
        """
        Routes input to the most resonant expert and monitors for mitosis.
        """
        if reset_state:
            for expert in self.experts:
                expert.reset_state(x.size(0), x.device)

        # 1. Calculate Fidelity for all experts
        fidelities = []
        for expert in self.experts:
            f = expert.get_rea_fidelity(x)
            fidelities.append(f)
        
        fidelities = torch.stack(fidelities, dim=1) # [batch, n_experts]
        
        # 2. Select winner per sample in batch
        winner_indices = torch.argmax(fidelities, dim=1)
        max_fidelities = torch.max(fidelities, dim=1)[0]
        
        # 3. Aggregate output
        outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (winner_indices == i)
            if mask.any():
                outputs[mask] = expert(x[mask], reset_state=reset_state)
        
        # 4. Monitor Health (Homeostasis) - Average across batch
        avg_fidelity = max_fidelities.mean().item()
        # We only record one data point per sequence batch to avoid buffer flooding
        if reset_state:
            self.fidelity_history.append(avg_fidelity)
            if len(self.fidelity_history) > self.window_size:
                self.fidelity_history.pop(0)
            
        return outputs

    def check_mitosis(self, max_experts=8):
        """
        Checks if the current pool is saturated and triggers mitosis if needed.
        """
        if len(self.experts) >= max_experts:
            return False

        if len(self.fidelity_history) < self.window_size // 2:
            return False
            
        recent_avg = sum(self.fidelity_history) / len(self.fidelity_history)
        if recent_avg < self.threshold_mitosis:
            # Split the expert that is currently handling most of the load but failing
            self.perform_mitosis(0)
            self.fidelity_history = [] # Reset history to give time to adapt
            return True
        return False

    def perform_mitosis(self, expert_idx):
        """
        Duplicates an expert and perturbs the daughter.
        """
        import copy
        parent = self.experts[expert_idx]
        # Deepcopy weight state
        daughter = ResonantMamba3Layer(self.d_model, self.rank).to(parent.in_proj.weight.device)
        daughter.load_state_dict(parent.state_dict())
        
        # Symmetry Breaking
        daughter.mutate(scale=0.1)
        
        self.experts.append(daughter)
        print(f"  [MITOSIS] Expert {expert_idx} split. Pool size: {len(self.experts)}")

if __name__ == "__main__":
    print("Resonant Mamba-3 Module with Expert Pool Initialized.")
    pool = RM3ExpertPool(d_model=384, rank=32)
    sample_input = torch.randn(8, 384)
    output = pool(sample_input)
    print(f"Pool size: {len(pool.experts)}")
    print(f"Output shape: {output.shape}")
    
    # Test manual mitosis
    pool.perform_mitosis(0)
    print(f"Pool size after mitosis: {len(pool.experts)}")

