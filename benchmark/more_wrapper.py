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
            
            for b_idx, (batch_X, batch_y) in enumerate(loader):
                # 1. MoRE Forward
                winners, max_fam, _, g, all_attn, proto_winners = self.model.forward(batch_X)
                
                # 2. Update Experts (Refine if familiar, trigger mitosis if novel)
                novelty_count = 0
                for exp_idx in range(len(self.model.experts)):
                    mask = (winners == exp_idx)
                    if not mask.any(): continue
                    
                    if exp_idx >= len(win_stats):
                        new_stats = torch.zeros(len(self.model.experts))
                        new_stats[:len(win_stats)] = win_stats
                        win_stats = new_stats
                    
                    expert = self.model.experts[exp_idx]
                    win_stats[exp_idx] += mask.sum()
                    
                    # Familiar samples (g < 0.5): Refine internal prototypes and voting
                    familiar_mask = mask & (g < 0.5)
                    if familiar_mask.any():
                        target_labels = batch_y[familiar_mask]
                        target_protos = proto_winners[familiar_mask]
                        expert.update_voting(target_labels, target_protos)
                        
                        # Use local update for the benchmark
                        # reward is 1 for all samples as we are training on a task
                        reward = torch.ones(familiar_mask.sum(), device=batch_X.device)
                        sub_scores = all_attn[familiar_mask]
                            
                        expert.update_local(batch_X[familiar_mask], reward, sub_scores, lr=self.lr)

                    # Novel samples (g > 0.5): Track to trigger mitosis
                    novel_mask = mask & (g > 0.5)
                    if novel_mask.any():
                        novelty_count += novel_mask.sum().item()
                
                # 3. Periodically check for Mitosis (Reactive: every 20 batches OR on high novelty)
                if b_idx % 20 == 0 or novelty_count > (batch_size // 4):
                    self.model.check_health_and_mitosis(threshold_f=self.model.experts[0].theta)
            
    def predict(self, X):
        winner_experts, max_fam, _, g, _, proto_winners = self.model.forward(X)
        batch_size = X.size(0)
        
        # Recolectamos los logits (v) de cada experto ganador y su prototipo
        # v es de forma (M, n_classes)
        logits = torch.stack([self.model.experts[winner_experts[i]].v[proto_winners[i]] for i in range(batch_size)])
        return logits
        
    def evaluate(self, X, y):
        logits = self.predict(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return acc.item()
