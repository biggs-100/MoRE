"""
MoRE Manager: Mixture of R-Experts.
Orchestrates multiple experts and routes inputs based on familiarity.
"""

import torch
import torch.nn as nn
from rperceptron import RPerceptron

class MoRE(nn.Module):
    def __init__(self, n_experts, d_input, M, **kwargs):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            RPerceptron(d_input, M, **kwargs) for _ in range(n_experts)
        ])

    def forward(self, x):
        # x: (batch, d_input)
        all_y = []
        all_f = []
        all_g = []
        all_attn = []
        
        for expert in self.experts:
            y, f, g, attn = expert(x)
            all_y.append(y)
            all_f.append(f)
            all_g.append(g)
            all_attn.append(attn)
            
        # Stack scores: (batch, n_experts)
        scores = torch.stack(all_f, dim=-1)
        winners = scores.argmax(dim=-1)
        max_scores = scores.max(dim=-1).values
        
        return winners, max_scores, all_y, all_g, all_attn

    def predict(self, x, threshold=0.5):
        """
        Clasifica o retorna -1 (No se) si la familiaridad es baja.
        """
        winners, max_scores, _, gates, _ = self.forward(x)
        
        # Si la familiaridad maxima es menor al umbral -> No se
        no_se = max_scores < threshold
        
        final_preds = winners.clone()
        final_preds[no_se] = -1
        
        return final_preds, max_scores
