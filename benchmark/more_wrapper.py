import torch
import torch.nn as nn
from more_demo import MoRE

class MoREBenchmarkWrapper:
    def __init__(self, d_input, n_classes, n_experts=10, M=64, theta=0.7, lr=0.1):
        self.model = MoRE(n_experts=n_experts, d_input=d_input, M=M, theta=theta, n_classes=n_classes)
        self.n_classes = n_classes
        self.lr = lr
        
    def train_task(self, X, y, task_id, epochs=10, batch_size=16):
        for epoch in range(epochs):
            win_stats = torch.zeros(len(self.model.experts))
            dataset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch_X, batch_y in loader:
                # 1. MoRE Forward
                winners, max_fam, _, g, all_attn = self.model.forward(batch_X)
                
                # 2. Update Experts (Hebbian + Voting)
                for exp_idx in range(len(self.model.experts)):
                    mask = (winners == exp_idx)
                    if not mask.any(): continue
                    
                    expert = self.model.experts[exp_idx]
                    win_stats[exp_idx] += mask.sum()
                    
                    # Update Voting Head (v) for all samples where this expert won
                    target_labels = batch_y[mask]
                    expert.update_voting(target_labels)
                    
                    # Update Prototypes (keys) only for resonant samples
                    resonant_mask = mask & (g > 0.5)
                    if resonant_mask.any():
                        first_idx = torch.where(mask)[0][0].item()
                        full_scores = all_attn[first_idx]
                        
                        if isinstance(full_scores, tuple):
                            sub_scores = (full_scores[0][resonant_mask], full_scores[1][resonant_mask])
                        else:
                            sub_scores = full_scores[resonant_mask]
                            
                        expert.update_local(
                            batch_X[resonant_mask],
                            torch.ones(resonant_mask.sum()), 
                            sub_scores,
                            lr=self.lr
                        )
                
            self.model.check_health_and_mitosis(threshold_f=self.model.experts[0].theta)
            
    def predict(self, X):
        winner_experts, max_fam, _, g, _ = self.model.forward(X)
        batch_size = X.size(0)
        
        # Recolectamos los logits (v) de cada experto ganador
        # v es de forma (n_classes,)
        logits = torch.stack([self.model.experts[exp_idx].v for exp_idx in winner_experts])
        return logits
        
    def evaluate(self, X, y):
        logits = self.predict(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return acc.item()
