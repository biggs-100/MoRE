import torch
import torch.nn as nn
from rperceptron import RPerceptron
from collections import deque
import numpy as np
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MoRE(nn.Module):
    """
    MoRE: Mixture of R-Experts with Autonomous Mitosis.
    """
    def __init__(self, n_experts, d_input, M, n_classes=10, topk=3, gamma=0.1, decay=0.999, theta=0.5):
        super().__init__()
        self.d_input = d_input
        self.M = M
        self.n_classes = n_classes
        self.experts = nn.ModuleList([
            RPerceptron(d_input, M, n_classes, topk, gamma, decay, theta) 
            for _ in range(n_experts)
        ])
        
        # Mitosis Buffers and Health Monitoring
        self.buffers = [deque(maxlen=256) for _ in range(n_experts)]
        self.health_f = [deque(maxlen=100) for _ in range(n_experts)] # Recent familiarity
        self.health_h = [deque(maxlen=100) for _ in range(n_experts)] # Recent entropy
        self.min_samples_mitosis = 128
        
    def forward(self, x):
        batch_size = x.size(0)
        n_exp = len(self.experts)
        
        f_scores = []
        all_results = []
        for expert in self.experts:
            res = expert.forward(x) # (win_idx, max_sim, y, g, scores)
            all_results.append(res)
            f_scores.append(res[1])
            
        f_matrix = torch.stack(f_scores, dim=1)
        max_fam, winner_experts = f_matrix.max(dim=1)
        
        y = torch.zeros(batch_size, device=x.device)
        g = torch.zeros(batch_size, device=x.device)
        all_attn = []
        
        for i in range(batch_size):
            exp_idx = winner_experts[i]
            res = all_results[exp_idx]
            y[i] = res[2][i]
            g[i] = res[3][i]
            
            # Scores from RPerceptron (top_scores if FAISS, else inhibited_scores)
            scores = res[4]
            all_attn.append(scores)
            
            # Monitoring
            self.buffers[exp_idx].append(x[i].detach().cpu().numpy())
            self.health_f[exp_idx].append(max_fam[i].item())
            
            # Calculate Entropy of attention for this sample
            if isinstance(scores, tuple):
                # FAISS mode: (indices, top_scores)
                p = torch.softmax(scores[1][i], dim=0)
            else:
                # Dense mode
                # Filter out -inf
                valid_scores = scores[i][scores[i] > float('-inf')]
                p = torch.softmax(valid_scores, dim=0)
            
            h = -torch.sum(p * torch.log(p + 1e-9)).item()
            self.health_h[exp_idx].append(h)
            
        return winner_experts, max_fam, y, g, all_attn

    def predict(self, x, threshold=0.5):
        """
        Uses the Stable Voting Head (argmax of frequency counts) for classification.
        Immune to catastrophic forgetting.
        """
        winner_experts, max_fam, _, g, _ = self.forward(x)
        batch_size = x.size(0)
        
        preds = torch.full((batch_size,), -1, dtype=torch.long, device=x.device)
        
        for i in range(batch_size):
            if g[i] > 0 and max_fam[i] > threshold:
                exp_idx = winner_experts[i]
                # argmax of the frequency counts V in the winning expert
                # If counts are all zero (new expert), it returns 0 or -1
                counts = self.experts[exp_idx].v
                if counts.sum() > 0:
                    preds[i] = counts.argmax()
                else:
                    # Optional: fallback to expert index or -1
                    preds[i] = -1 
                    
        return preds, max_fam

    def perform_mitosis(self, expert_idx):
        if not SKLEARN_AVAILABLE: return False
        buffer = self.buffers[expert_idx]
        if len(buffer) < self.min_samples_mitosis: return False
        
        X = np.stack(list(buffer))
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        
        old_expert = self.experts[expert_idx]
        e1 = RPerceptron(self.d_input, self.M, self.n_classes, old_expert.topk, 
                         old_expert.gamma, old_expert.decay, old_expert.theta)
        e2 = RPerceptron(self.d_input, self.M, self.n_classes, old_expert.topk, 
                         old_expert.gamma, old_expert.decay, old_expert.theta)
        
        # Inherit voting state (daughter experts share parent's expertise initially)
        with torch.no_grad():
            e1.v.copy_(old_expert.v)
            e2.v.copy_(old_expert.v)
            
            for i, centroid in enumerate(centroids):
                new_keys = torch.from_numpy(centroid).float().repeat(self.M, 1)
                new_keys += torch.randn_like(new_keys) * 0.05 # Break symmetry
                if i == 0: e1.keys.copy_(new_keys)
                else: e2.keys.copy_(new_keys)
            e1._normalize_keys()
            e2._normalize_keys()
            
        self.experts[expert_idx] = e1
        self.experts.insert(expert_idx + 1, e2)
        
        # Update trackers
        self.buffers.pop(expert_idx)
        self.buffers.insert(expert_idx, deque(maxlen=256))
        self.buffers.insert(expert_idx + 1, deque(maxlen=256))
        
        self.health_f.pop(expert_idx)
        self.health_f.insert(expert_idx, deque(maxlen=100))
        self.health_f.insert(expert_idx + 1, deque(maxlen=100))
        
        self.health_h.pop(expert_idx)
        self.health_h.insert(expert_idx, deque(maxlen=100))
        self.health_h.insert(expert_idx + 1, deque(maxlen=100))
        
        return True

    def check_health_and_mitosis(self, threshold_h=0.1, threshold_f=0.5):
        split_indices = []
        for i in range(len(self.experts) - 1, -1, -1):
            if len(self.health_f[i]) >= 64:
                avg_f = np.mean(self.health_f[i])
                avg_h = np.mean(self.health_h[i])
                
                # MITOSIS TRIGGER: Surprise (low fam) AND moderate dispersion
                if avg_f < threshold_f and avg_h > threshold_h:
                    if self.perform_mitosis(i):
                        split_indices.append(i)
                        
        return split_indices

    def reset_health(self):
        """Clears health monitoring deques for all experts."""
        for i in range(len(self.experts)):
            self.health_f[i].clear()
            self.health_h[i].clear()
            self.buffers[i].clear()
