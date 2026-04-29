import torch
import numpy as np
from torchvision import datasets, transforms
from dataset import generate_clusters

class TaskStream:
    def __init__(self, n_tasks=5, mode='synthetic', d=128, max_samples_per_task=1000):
        self.n_tasks = n_tasks
        self.mode = mode
        self.d = d
        self.max_samples_per_task = max_samples_per_task
        self.current_task = 0
        
        if mode in ['permuted_mnist', 'split_mnist']:
            self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
            if mode == 'permuted_mnist':
                self.permutations = [torch.randperm(784) for _ in range(n_tasks)]
            else:
                self.classes_per_task = 10 // n_tasks
        elif mode == 'split_cifar':
            self.dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
            self.classes_per_task = 100 // n_tasks
        elif mode == 'multidomain_mnist_fashion':
            self.mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
            self.fashion = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
            self.classes_per_task = 10 // n_tasks
            
    def __iter__(self):
        self.current_task = 0
        return self
        
    def __next__(self):
        if self.current_task >= self.n_tasks:
            raise StopIteration
            
        task_id = self.current_task
        self.current_task += 1
        
        if self.mode == 'synthetic':
            X, y, _ = generate_clusters(n_clusters=3, d=self.d, seed=42 + task_id)
            return X, y, task_id
            
        if self.mode == 'permuted_mnist':
            X = self.dataset.data.view(-1, 784).float() / 255.0
            y = self.dataset.targets
            
            # Subsampling
            if self.max_samples_per_task:
                idx = torch.randperm(len(X))[:self.max_samples_per_task]
                X, y = X[idx], y[idx]
                
            perm = self.permutations[task_id]
            X_perm = X[:, perm]
            return X_perm, y, task_id

        if self.mode == 'split_mnist':
            start_cls = task_id * self.classes_per_task
            end_cls = start_cls + self.classes_per_task
            
            mask = (self.dataset.targets >= start_cls) & (self.dataset.targets < end_cls)
            X = self.dataset.data[mask].view(-1, 784).float() / 255.0
            y = self.dataset.targets[mask]

            if self.max_samples_per_task and len(X) > self.max_samples_per_task:
                idx = torch.randperm(len(X))[:self.max_samples_per_task]
                X, y = X[idx], y[idx]
            
            return X, y, task_id
            
        if self.mode == 'split_cifar':
            # Filtrar por clases
            start_cls = task_id * self.classes_per_task
            end_cls = start_cls + self.classes_per_task
            
            targets = torch.tensor(self.dataset.targets)
            mask = (targets >= start_cls) & (targets < end_cls)
            
            # Nota: CIFAR100.data es un numpy array (N, 32, 32, 3)
            X = torch.from_numpy(self.dataset.data[mask]).float().permute(0, 3, 1, 2) / 255.0
            y = targets[mask]

            if self.max_samples_per_task and len(X) > self.max_samples_per_task:
                idx = torch.randperm(len(X))[:self.max_samples_per_task]
                X, y = X[idx], y[idx]
            
            return X, y, task_id
            
        if self.mode == 'multidomain_mnist_fashion':
            # Task 0: MNIST, Task 1: FashionMNIST
            dataset = self.mnist if task_id == 0 else self.fashion
            
            # Simple binary split for demo: 0-4 or 5-9
            # But let's just take all 10 classes and filter by max_samples
            X = dataset.data.view(-1, 784).float() / 255.0
            y = dataset.targets
            
            if self.max_samples_per_task and len(X) > self.max_samples_per_task:
                idx = torch.randperm(len(X))[:self.max_samples_per_task]
                X, y = X[idx], y[idx]
                
            return X, y, task_id
            
        raise ValueError(f"Unknown mode: {self.mode}")
