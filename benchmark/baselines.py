import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class MLPModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class MLPBaseline:
    def __init__(self, input_dim, n_classes, lr=1e-3):
        self.model = MLPModel(input_dim, n_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_task(self, X, y, task_id, epochs=5, batch_size=64):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

class EWCBaseline(MLPBaseline):
    def __init__(self, input_dim, n_classes, lr=1e-3, lambda_ewc=100.0):
        super().__init__(input_dim, n_classes, lr)
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.params = {}
        
    def consolidate(self, X, y):
        self.model.train()
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        # Calcular Fisher (usando el gradiente de la log-probabilidad)
        output = self.model(X)
        loss = nn.functional.log_softmax(output, dim=1).gather(1, y.unsqueeze(1)).mean()
        self.model.zero_grad()
        loss.backward()
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = (p.grad ** 2).detach()
                
    def train_task(self, X, y, task_id, epochs=5, batch_size=64):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                
                # Loss estándar
                loss = self.criterion(output, batch_y)
                
                # Penalización EWC
                ewc_loss = 0
                for n, p in self.model.named_parameters():
                    if n in self.fisher:
                        ewc_loss += (self.fisher[n] * (p - self.params[n])**2).sum()
                
                (loss + self.lambda_ewc * ewc_loss).backward()
                self.optimizer.step()
