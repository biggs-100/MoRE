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

    def clear(self):
        self.raw_data.clear()
        self.embeddings.clear()

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
        Supports MIMO anchors (batch, rank, dim).
        """
        x_anchor, h_old = self.anchor_buffer.get_anchors()
        if x_anchor is None:
            return False
        
        x_anchor = x_anchor.to(device)
        h_old = h_old.to(device)
        
        with torch.no_grad():
            self.W_align.weight.copy_(torch.eye(self.d_latent).to(device))
            
        self.W_align.train()
        encoder.eval()
        
        optimizer = torch.optim.Adam(self.W_align.parameters(), lr=0.05)
        
        for _ in range(500):
            optimizer.zero_grad()
            with torch.no_grad():
                h_new = encoder(x_anchor)
            
            # Ensure shapes match for MSE loss
            h_new = h_new.view(h_old.shape)
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

    def compute_morse_axis(self):
        """
        Computes the Morse Bifurcation Axis using PCA on the anchor buffer.
        Corresponds to the direction of maximum local variance in the latent manifold.
        """
        raw_x, h_old = self.anchor_buffer.get_anchors()
        if h_old is None or h_old.size(0) < 5:
            # Fallback to random if not enough data
            return torch.randn(self.d_latent, device=self.W_align.weight.device)
            
        # h_old shape might be [N, Dim] or [N, Rank, Dim]
        data = h_old.view(-1, h_old.size(-1))
        
        # PCA via SVD
        mean = data.mean(dim=0)
        data_centered = data - mean
        U, S, V = torch.pca_lowrank(data_centered, q=1)
        
        return V[:, 0]

class GradientsProjector:
    """
    Resonant OGD (Orthogonal Gradient Descent) implementation.
    Projects encoder gradients to the null space of anchor features to minimize drift.
    """
    def __init__(self, encoder, threshold=0.99):
        self.encoder = encoder
        self.threshold = threshold
        self.bases = {} # expert_id -> basis matrix

    def update_basis(self, expert_id, anchor_features):
        """
        Computes the orthogonal basis of the subspace spanned by anchor features using SVD.
        Mamba-3: "calculates an orthonormal basis Vi of its anchor features h_anc".
        Supports MIMO shapes by flattening [Batch, Rank, Dim] -> [Batch * Rank, Dim].
        """
        with torch.no_grad():
            # Flatten batch and MIMO rank dimensions
            # x shape: [N * R, d_latent]
            x = anchor_features.view(-1, anchor_features.size(-1))
            
            # Center the data for more robust SVD
            x_mean = torch.mean(x, dim=0)
            x = x - x_mean
            
            # SVD to find principal components
            # We use full_matrices=False for efficiency
            U, S, V = torch.svd(x)
            
            # Keep components that explain 'threshold' of variance
            energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
            k = torch.where(energy > self.threshold)[0][0].item() + 1
            
            # Basis is the first k columns of V
            self.bases[expert_id] = V[:, :k].detach() # [d_latent, k]

    def project(self):
        """
        Projects current gradients of the encoder to the null space of all bases.
        Simplification: We project the gradients of the first layer (which receives raw input).
        """
        if not self.bases:
            return

        # Project the gradients of all layers that produce the latent space
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                # g shape: [Out, In] or [Out]
                g = param.grad
                
                for expert_id in self.bases:
                    B = self.bases[expert_id] # [d_latent, k]
                    
                    # Projecting the output dimension (rows) of the weight matrix
                    if g.size(0) == B.size(0):
                        # proj = B @ (B.T @ g)
                        proj = torch.matmul(B, torch.matmul(B.t(), g))
                        param.grad.sub_(proj)
                    # Projecting the input dimension (cols) if it matches (less common but possible)
                    elif g.size(-1) == B.size(0):
                        # proj = (g @ B) @ B.T
                        proj = torch.matmul(torch.matmul(g, B), B.t())
                        param.grad.sub_(proj)
