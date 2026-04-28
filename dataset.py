"""
Dataset generator for MoRE demo.
Generates clusters in high-dimensional space.
"""

import torch

def generate_clusters(n_clusters=3, d=128, n_samples=100, std=0.1, seed=42):
    torch.manual_seed(seed)
    
    # Generar centros aleatorios
    centers = torch.randn(n_clusters, d)
    centers = torch.nn.functional.normalize(centers, dim=-1)
    
    X = []
    y = []
    
    for i in range(n_clusters):
        # Muestras alrededor del centro
        samples = centers[i].unsqueeze(0) + std * torch.randn(n_samples, d)
        X.append(samples)
        y.append(torch.full((n_samples,), i, dtype=torch.long))
        
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    
    # Shuffle
    idx = torch.randperm(X.size(0))
    return X[idx], y[idx], centers

def generate_novelty(n_samples=50, d=128, existing_centers=None, std=0.1, seed=99):
    torch.manual_seed(seed)
    
    # Generar un centro nuevo que este lejos de los existentes
    while True:
        new_center = torch.randn(1, d)
        new_center = torch.nn.functional.normalize(new_center, dim=-1)
        
        if existing_centers is not None:
            # Similitud maxima con existentes
            sim = torch.matmul(new_center, existing_centers.T)
            if sim.max().item() < 0.3: # Que sea distinto
                break
        else:
            break
            
    X_novel = new_center + std * torch.randn(n_samples, d)
    return X_novel, new_center
