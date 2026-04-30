import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

class AnchorBuffer:
    """
    Circular buffer to store anchors (raw data and their original embeddings).
    Used for Anchor Projection (REA).
    """
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.raw_data = deque(maxlen=max_size)
        self.embeddings = deque(maxlen=max_size)

    def add(self, raw_x, h):
        """
        raw_x: Raw data sample (e.g. image)
        h: Embedding at the time of learning
        """
        self.raw_data.append(raw_x.detach().cpu())
        self.embeddings.append(h.detach().cpu())

    def get_anchors(self):
        if len(self.raw_data) == 0:
            return None, None
        return torch.stack(list(self.raw_data)), torch.stack(list(self.embeddings))

    def is_full(self):
        return len(self.raw_data) >= self.max_size

class AlignableModule(nn.Module):
    """
    Base class for modules that can recover from representation drift.
    Includes an AnchorBuffer and a W_align layer.
    """
    def __init__(self, d_latent):
        super().__init__()
        self.d_latent = d_latent
        self.anchor_buffer = AnchorBuffer()
        
        # Alignment Layer: W_align (no bias as we assume normalized space)
        self.W_align = nn.Linear(d_latent, d_latent, bias=False)
        with torch.no_grad():
            # Start with identity
            self.W_align.weight.copy_(torch.eye(d_latent))
        
        self.needs_alignment = False
        self.drift_threshold = 0.2 # 20% drop triggers alignment

    def align_to_encoder(self, encoder, device):
        """
        Learns the W_align matrix to map encoder_new(anchors) -> h_old.
        """
        x_anchor, h_old = self.anchor_buffer.get_anchors()
        if x_anchor is None:
            return False
        
        x_anchor = x_anchor.to(device)
        h_old = h_old.to(device)
        
        # Reset W_align to identity before starting? 
        # Or keep current to refine? Let's reset for stability.
        with torch.no_grad():
            self.W_align.weight.copy_(torch.eye(self.d_latent).to(device))
            
        self.W_align.train()
        encoder.eval()
        
        optimizer = torch.optim.Adam(self.W_align.parameters(), lr=0.01)
        
        for _ in range(100):
            optimizer.zero_grad()
            with torch.no_grad():
                h_new = encoder(x_anchor)
            
            h_aligned = self.W_align(h_new)
            loss = F.mse_loss(h_aligned, h_old)
            loss.backward()
            optimizer.step()
            
        self.W_align.eval()
        self.needs_alignment = False
        return True

    def apply_alignment(self, h):
        """Applies the learned alignment to the input embeddings."""
        return self.W_align(h)

class GradientsProjector:
    """
    Resonant OGD (Orthogonal Gradient Descent) implementation.
    Projects encoder gradients to the null space of anchor features to minimize drift.
    """
    def __init__(self, encoder, threshold=0.99):
        self.encoder = encoder
        self.threshold = threshold
        self.bases = {} # expert_id -> basis matrix

    def update_basis(self, expert_id, raw_anchors):
        """
        Computes the orthogonal basis of the subspace spanned by anchors using SVD.
        """
        with torch.no_grad():
            # x: [N, D] where D is flattened raw input dim
            x = raw_anchors.view(raw_anchors.size(0), -1)
            
            # SVD to find principal components
            U, S, V = torch.svd(x)
            
            # Keep components that explain 'threshold' of variance
            energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
            k = torch.where(energy > self.threshold)[0][0].item() + 1
            
            # Basis is the first k columns of V
            self.bases[expert_id] = V[:, :k].detach() # [D, k]

    def project(self):
        """
        Projects current gradients of the encoder to the null space of all bases.
        Simplification: We project the gradients of the first layer (which receives raw input).
        """
        if not self.bases:
            return

        with torch.no_grad():
            # Find the first linear layer in the encoder
            target_param = None
            for name, param in self.encoder.named_parameters():
                if "weight" in name and param.grad is not None:
                    target_param = param
                    break # Usually the first layer is the most critical for OGD
            
            if target_param is None:
                return

            for expert_id in self.bases:
                B = self.bases[expert_id] # [D, k]
                # Grad g: [Out, D]
                g = target_param.grad
                
                # Projection: g_proj = g - (g @ B) @ B.T
                # g @ B: [Out, k]
                # (g @ B) @ B.T: [Out, D]
                proj = torch.matmul(torch.matmul(g, B), B.t())
                target_param.grad.sub_(proj)
