"""
R-Perceptron: Improved version for MoRE-3 Demo.
Includes WTA inhibition, diversity bias, importance decay, and dual output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RPerceptron(nn.Module):
    def __init__(
        self, 
        d_input: int, 
        M: int, 
        d_output: int | None = None,
        tau: float = 0.1, 
        beta: float = 10.0,
        theta: float = 0.5,
        topk: int = 3,
        gamma: float = 0.5,
        decay: float = 0.99
    ):
        super().__init__()
        self.d_input = d_input
        self.M = M
        self.d_output = d_output or d_input
        
        # Prototipos K (Claves) y V (Valores)
        self.K = nn.Parameter(torch.randn(M, d_input))
        self.V = nn.Parameter(torch.zeros(M, self.d_output))
        
        # Importancia s y Umbral theta
        self.register_buffer('s', torch.ones(M))
        self.register_buffer('theta', torch.tensor(theta))
        self.register_buffer('winner_counts', torch.zeros(M))
        
        self.tau = tau
        self.beta = beta
        self.topk = topk
        self.gamma = gamma
        self.decay = decay

        # Normalizar K inicial
        with torch.no_grad():
            self.K.data = F.normalize(self.K.data, dim=-1)

    def forward(self, x: torch.Tensor):
        # x: (batch, d_input)
        batch_size = x.shape[0]
        
        # Resonancia (similitud coseno)
        x_norm = F.normalize(x, dim=-1)
        K_norm = F.normalize(self.K, dim=-1)
        S = torch.matmul(x_norm, K_norm.T)  # (batch, M)
        
        # 1. Inhibicion Lateral (WTA)
        k = min(self.topk, self.M)
        topk_vals, topk_idx = S.topk(k, dim=-1)
        mask = torch.zeros_like(S).scatter_(-1, topk_idx, 1.0)
        S_masked = S.masked_fill(mask == 0, float('-inf'))
        
        # 2. Sesgo de Diversidad
        # winner_counts se actualiza en el paso de entrenamiento, 
        # pero aqui usamos un sesgo basado en lo que va del batch
        winners_batch = S.argmax(dim=-1)
        counts = torch.bincount(winners_batch, minlength=self.M).float()
        diversity_bias = -self.gamma * (counts.unsqueeze(0) / batch_size)
        
        # 3. Atencion con Importancia s
        log_s = torch.log(self.s.unsqueeze(0) + 1e-8)
        S_final = (S_masked / self.tau) + log_s + diversity_bias
        attn = F.softmax(S_final, dim=-1)
        
        # Recuperacion
        v_ret = torch.matmul(attn, self.V)
        
        # Familiaridad f (coseno maximo original)
        f, _ = S.max(dim=-1)
        
        # 4. Compuerta de Novedad g
        g = 1.0 - torch.sigmoid(self.beta * (f - self.theta))
        
        # Salida dual
        y = g.unsqueeze(-1) * v_ret
        
        return y, f, g, attn

    def update_local(self, x: torch.Tensor, reward: torch.Tensor, attn: torch.Tensor, lr: float = 0.01):
        """
        Aprendizaje Hebbiano local modulado por recompensa.
        x: (batch, d_input)
        reward: (batch,) -> +1 (acierto), -1 (error)
        attn: (batch, M)
        """
        with torch.no_grad():
            # Encontrar el prototipo mas atendido (el ganador)
            best_proto_idx = attn.argmax(dim=-1)  # (batch,)
            
            for i in range(x.size(0)):
                p_idx = best_proto_idx[i].item()
                r = reward[i].item()
                
                # Diferencia entre entrada y prototipo
                diff = x[i] - self.K[p_idx]
                
                if r > 0:
                    # REFORZAR: mover hacia x y aumentar s
                    self.K.data[p_idx] += lr * diff
                    self.s.data[p_idx] += 0.01
                    self.winner_counts[p_idx] += 1
                else:
                    # DEBILITAR: alejar de x y reducir s mas rapido
                    self.K.data[p_idx] -= 0.5 * lr * diff
                    self.s.data[p_idx] -= 0.02
                
                # Clampear y Normalizar
                self.s.data[p_idx].clamp_(0.01, 10.0)
            
            self.K.data = F.normalize(self.K.data, dim=-1)
            
            # Decaimiento por desuso (cada batch)
            self.s.data *= self.decay
            self.s.data.clamp_(min=0.01)
