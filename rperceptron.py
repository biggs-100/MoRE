import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class RPerceptron(nn.Module):
    """
    R-Perceptron: A resonant unit with lateral inhibition (WTA),
    diversity bias, importance decay, and optional FAISS scaling.
    """
    def __init__(self, d_input, M, n_classes=10, topk=3, gamma=0.1, decay=0.999, theta=0.5, beta=10.0, use_faiss=True, faiss_threshold=1024):
        super().__init__()
        self.d_input = d_input
        self.M = M
        self.n_classes = n_classes
        self.topk = min(topk, M)
        self.gamma = gamma
        self.decay = decay
        self.theta = theta # Novelty threshold
        self.beta = beta   # Gate steepness
        
        # Associative memory: M prototypes (keys)
        self.keys = nn.Parameter(torch.randn(M, d_input))
        self._normalize_keys()
        
        # Diversity bias: avoid expert collapse
        self.register_buffer('usage', torch.zeros(M))
        
        # Importance: weight of each prototype
        self.register_buffer('s', torch.ones(M))

        # Stable Voting Head: frequency-based class mapping V
        # Immunity to catastrophic forgetting by avoiding gradient-based classification
        self.register_buffer('v', torch.zeros(n_classes))
        
        # FAISS scaling
        self.index = None
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_threshold = faiss_threshold
        if self.use_faiss and self.M >= self.faiss_threshold:
            self._rebuild_index()

    def _normalize_keys(self):
        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, p=2, dim=1))

    def _rebuild_index(self):
        """Builds/Rebuilds the FAISS index for fast Inner Product search."""
        if not FAISS_AVAILABLE: return
        self.index = faiss.IndexFlatIP(self.d_input)
        keys_np = self.keys.detach().cpu().numpy().astype('float32')
        self.index.add(keys_np)

    def forward(self, x):
        """
        Forward pass with WTA inhibition and novelty gating.
        Hybrid mode: uses FAISS for large M, dense dot product for small M.
        """
        x = F.normalize(x, p=2, dim=1)
        batch_size = x.size(0)
        
        # 1. Calculate similarity scores
        if self.use_faiss and self.M >= self.faiss_threshold:
            # FAISS path
            x_np = x.detach().cpu().numpy().astype('float32')
            raw_scores_np, indices_np = self.index.search(x_np, self.topk)
            indices = torch.from_numpy(indices_np).to(x.device)
            raw_top_scores = torch.from_numpy(raw_scores_np).to(x.device)
            
            # Biased scores for winner selection
            bias = -self.gamma * self.usage[indices] + torch.log(self.s[indices] + 1e-6)
            biased_top_scores = raw_top_scores + bias
            
            # Winner selection from biased scores
            idx_in_topk = biased_top_scores.argmax(dim=1)
            winner_indices = indices[torch.arange(batch_size), idx_in_topk]
            
            # Familiarity score MUST be unbiased pure similarity
            max_similarity = raw_top_scores[torch.arange(batch_size), idx_in_topk]
            
            # For local update tracking
            scores = (indices, biased_top_scores)
        else:
            # Dense path
            all_scores = torch.matmul(x, self.keys.T) # Pure similarity
            
            # Biased scores for winner selection
            bias = -self.gamma * self.usage + torch.log(self.s + 1e-6)
            biased_scores = all_scores + bias
            
            # 2. Lateral Inhibition (WTA)
            top_val, top_idx = torch.topk(biased_scores, self.topk, dim=1)
            mask = torch.full_like(biased_scores, float('-inf'))
            mask.scatter_(1, top_idx, 0)
            inhibited_scores = biased_scores + mask
            
            # 3. Winning and Novelty
            _, winner_indices = inhibited_scores.max(dim=1)
            
            # Pure similarity for gating
            max_similarity = all_scores[torch.arange(batch_size), winner_indices]
            
            scores = inhibited_scores

        # Novelty Gate: g = sigma(beta * (f - theta))
        # This provides a smooth, differentiable awareness of the unknown.
        g = torch.sigmoid(self.beta * (max_similarity - self.theta))
        
        # Final output: gating the similarity score
        y = max_similarity * g
        
        return winner_indices, max_similarity, y, g, scores

    def update_local(self, x, reward, all_attn, lr=0.01):
        """
        Reward-modulated Hebbian update.
        """
        x = F.normalize(x, p=2, dim=1)
        
        with torch.no_grad():
            for i in range(x.size(0)):
                # Find winner for this sample i
                if isinstance(all_attn, tuple):
                    indices, top_scores = all_attn
                    w_idx = indices[i, top_scores[i].argmax()]
                else:
                    w_idx = all_attn[i].argmax()
                
                # Hebbian rule
                delta = lr * reward[i] * (x[i] - self.keys[w_idx])
                self.keys[w_idx] += delta
                
                # Update usage and importance
                if reward[i] > 0:
                    self.usage[w_idx] += 1
                    self.s[w_idx] *= 1.01
                else:
                    self.s[w_idx] *= 0.95
            
            self._normalize_keys()
            self.s *= self.decay
            
            # Sync FAISS index if modified
            if self.use_faiss and self.M >= self.faiss_threshold:
                self._rebuild_index()

    def update_voting(self, true_label):
        """
        Updates the frequency counts for the voting head.
        Non-gradient update that maps experts to classes via evidence.
        """
        with torch.no_grad():
            if isinstance(true_label, torch.Tensor):
                if true_label.dim() == 0:
                    label = int(true_label.item())
                    if 0 <= label < self.n_classes:
                        self.v[label] += 1
                else:
                    for label in true_label:
                        l_idx = int(label.item())
                        if 0 <= l_idx < self.n_classes:
                            self.v[l_idx] += 1
            else:
                l_idx = int(true_label)
                if 0 <= l_idx < self.n_classes:
                    self.v[l_idx] += 1
