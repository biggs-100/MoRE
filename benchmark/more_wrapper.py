import torch
import torch.nn as nn
from more_demo import MoRE

class MoREBenchmarkWrapper:
    def __init__(self, d_input, n_classes, n_experts=10, M=64, theta=0.3, lr=0.1):
        self.model = MoRE(n_experts=n_experts, d_input=d_input, M=M, theta=theta)
        self.n_classes = n_classes
        self.lr = lr
        # Mapeo estable: experto -> conteo de clases vistas
        self.expert_counts = torch.zeros(n_experts, n_classes)
        
    def _sync_counts(self):
        n_curr = len(self.model.experts)
        if n_curr > self.expert_counts.shape[0]:
            new_counts = torch.zeros(n_curr, self.n_classes)
            new_counts[:self.expert_counts.shape[0]] = self.expert_counts
            self.expert_counts = new_counts

    def train_task(self, X, y, task_id, epochs=10, batch_size=16):
        for epoch in range(epochs):
            self._sync_counts()
            dataset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch_X, batch_y in loader:
                # 1. MoRE Forward
                winners, max_fam, _, g, all_attn = self.model.forward(batch_X)
                
                # Reward: Siempre positivo para que el experto aprenda el cluster (Hebbiano)
                reward = torch.ones(len(batch_X))
                
                for s_idx in range(len(batch_X)):
                    exp_idx = winners[s_idx]
                    if exp_idx < len(self.model.experts):
                        # Siempre actualizamos el mapeo estable (Supervisado)
                        self.expert_counts[exp_idx, batch_y[s_idx]] += 1
                        
                        if g[s_idx] == 0: # Solo actualizamos el prototipo si hay resonancia (Hebbiano)
                            self.model.experts[exp_idx].update_local(
                                batch_X[s_idx:s_idx+1], 
                                reward[s_idx:s_idx+1], 
                                all_attn[s_idx], 
                                lr=self.lr
                            )
                
            self.model.check_health_and_mitosis(threshold_f=self.model.experts[0].theta)
            
    def predict(self, X):
        self._sync_counts()
        winner_experts, max_fam, _, g, _ = self.model.forward(X)
        
        # Predicción basada en el voto histórico del experto
        # Usamos los conteos como 'logits' para que el MetricEngine sea feliz
        logits = self.expert_counts[winner_experts]
        return logits
        
    def evaluate(self, X, y):
        logits = self.predict(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return acc.item()
