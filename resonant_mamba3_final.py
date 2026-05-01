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
        
        # 2. Physics-Anchored Parameters
        # omega: "Quantum" Resonant frequencies (learned frequencies)
        self.log_omega = nn.Parameter(torch.log(torch.arange(rank).float() + 1.0))
        # T: "Thermal" Semantic Temperature (learned plasticity)
        self.log_temp = nn.Parameter(torch.zeros(rank))
        # nu: Decay parameter (for stability)
        self.nu = nn.Parameter(torch.log(-torch.log(torch.rand(rank) * 0.9 + 0.1))) 
        
        # 3. Hamiltonian / Symplectic Logic (Optional toggle)
        self.use_symplectic = False 
        
        # 4. Data-Dependent Projections
        self.dt_proj = nn.Linear(d_model, rank, bias=False)
        self.phase_proj = nn.Linear(d_model, rank, bias=False) # Modulation of phase
        self.energy_proj = nn.Linear(d_model, rank, bias=False) # "Semantic Energy" input
        
        # 5. Output Projection
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

    def mutate_morse(self, axis, scale=0.1):
        """
        Breaks symmetry for mitosis by perturbing projections along the Morse axis.
        """
        with torch.no_grad():
            # Perturb projections along the axis to differentiate the daughter
            # We treat the axis as a direction in the latent space
            # and shift weights to be more 'sensitive' to that direction
            perturbation = axis.unsqueeze(0) * scale # [1, d_model]
            
            # Affect the input projection (MIMO)
            # perturbation is d_model, in_proj is [rank*4, d_model]
            self.in_proj.weight.add_(torch.randn(self.rank*4, 1, device=axis.device) * perturbation)
            
            # Clear anchor buffer for the new specialized manifold
            self.anchor_buffer.clear()

    def forward(self, x, reset_state=False):
        """
        Forward pass with automatic REA alignment and Physics-Anchored Phase.
        x: [batch, d_model]
        """
        # Apply Homeostasis (REA)
        x = self.apply_alignment(x)
        
        batch_size = x.size(0)
        if reset_state or self.state_re is None or self.state_re.size(0) != batch_size:
            self.reset_state(batch_size, x.device)
            
        # 1. Project to Internal Rank
        projected = self.in_proj(x)
        gate, x_val, _, _ = torch.chunk(projected, 4, dim=-1)
        
        # 2. Physics-Anchored Phase Calculation
        # φ = (hbar * omega) / (k_B * T) -> simplified to learned ratio for ML
        omega = torch.exp(self.log_omega)
        temp = torch.exp(self.log_temp)
        phi_base = omega / (temp + 1e-6) # Anchored base frequency
        
        # 3. Data-Dependent Modulation
        dt = F.softplus(self.dt_proj(x))
        # Input-driven phase shift (perturbation of the physics-based anchor)
        phase_shift = torch.tanh(self.phase_proj(x)) * np.pi
        # Total Angle: anchored rotation + input-driven shift
        angle = dt * (phi_base + phase_shift)
        
        # 4. Calculate Complex Evolution (Hardened Stability)
        decay = torch.exp(-dt * torch.exp(self.nu))
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        # 5. Complex Recurrence: z_t = (z_{t-1} * exp(i*angle)) * decay + x_t
        new_re = (self.state_re * cos - self.state_im * sin) * decay + torch.sigmoid(gate) * x_val
        new_im = (self.state_re * sin + self.state_im * cos) * decay
        
        self.state_re = new_re
        self.state_im = new_im
        
        # 6. Output Projection
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

        self.routing_penalty = 1.0 # Multiplier for unhealthy experts (default 1.0 = no penalty)

    def forward(self, x, reset_state=False, forced_expert_idx=None):
        """
        Routes input to the most resonant expert and monitors for mitosis.
        """
        if reset_state:
            for expert in self.experts:
                expert.reset_state(x.size(0), x.device)

        if forced_expert_idx is not None:
            # Force all samples in batch to use a specific expert
            winner_indices = torch.full((x.size(0),), forced_expert_idx, dtype=torch.long, device=x.device)
            max_fidelities = torch.ones((x.size(0),), device=x.device) # Dummy
        else:
            # 1. Calculate Fidelity for all experts
            fidelities = []
            for expert in self.experts:
                f = expert.get_rea_fidelity(x) # [batch]
                
                # Apply Routing Penalty if expert is drifting
                if expert.needs_alignment and self.routing_penalty != 1.0:
                    f = f * self.routing_penalty
                    
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
        Duplicates an expert and perturbs the daughter along the Morse axis.
        """
        parent = self.experts[expert_idx]
        
        # 1. Compute Morse Axis from parent anchors
        axis = parent.compute_morse_axis()
        
        # 2. Duplication
        daughter = ResonantMamba3Layer(self.d_model, self.rank).to(parent.in_proj.weight.device)
        daughter.load_state_dict(parent.state_dict())
        
        # 3. Symmetry Breaking (Morse-directed)
        daughter.mutate_morse(axis, scale=0.1)
        
        self.experts.append(daughter)
        print(f"  [MITOSIS] Expert {expert_idx} bifurcated along Morse Axis. Pool size: {len(self.experts)}")

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

