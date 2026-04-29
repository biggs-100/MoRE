import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor:
    def __init__(self):
        # Usamos ResNet18 pre-entrenado
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Eliminamos la ultima capa de clasificacion para obtener embeddings de 512
        self.model.fc = nn.Identity()
        
        # Congelamos todos los pesos
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
    def extract(self, x):
        """
        Input: (N, 3, H, W) or (N, D)
        Output: (N, 512) or (N, D)
        """
        # Si ya es un vector (Synthetic mode), devolvemos tal cual
        if len(x.shape) == 2:
            return x
            
        # Si el input es escala de grises (1, 28, 28) como MNIST, repetimos canales
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Si el input no es 224x224, ResNet aun funciona pero los features pueden variar.
        # Para el benchmark, redimensionar a 224 es ideal pero pesado.
        # Por ahora lo dejamos tal cual (CIFAR es 32x32).
        
        with torch.no_grad():
            features = self.model(x)
            
        return features
