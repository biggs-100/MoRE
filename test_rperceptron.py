import torch
import pytest
from rperceptron import RPerceptron

def test_rperceptron_forward_shape():
    d_in, M, batch = 128, 10, 5
    rp = RPerceptron(d_in, M)
    x = torch.randn(batch, d_in)
    y, f, g, attn = rp(x)
    
    assert y.shape == (batch, d_in)
    assert f.shape == (batch,)
    assert g.shape == (batch,)
    assert attn.shape == (batch, M)

def test_rperceptron_wta_inhibition():
    d_in, M, topk = 10, 5, 2
    rp = RPerceptron(d_in, M, topk=topk)
    x = torch.randn(1, d_in)
    _, _, _, attn = rp(x)
    
    # Solo top-k deben tener atencion significativa (softmax)
    # Los demas deben ser casi 0
    non_zero_attn = (attn > 1e-5).sum().item()
    assert non_zero_attn == topk

def test_rperceptron_local_update_positive():
    d_in, M = 10, 1
    rp = RPerceptron(d_in, M)
    x = torch.randn(1, d_in)
    x = torch.nn.functional.normalize(x, dim=-1)
    
    # Inicialmente s es 1.0
    initial_s = rp.s[0].item()
    
    # Forzar forward para obtener attn
    _, _, _, attn = rp(x)
    
    # Recompensa positiva
    reward = torch.tensor([1.0])
    rp.update_local(x, reward, attn, lr=0.1)
    
    # s debe haber aumentado (menos el decay)
    # decay=0.99, s = (1.0 + 0.01) * 0.99 = 0.9999
    assert rp.s[0].item() > 0.99 # Depende del decay, pero s_increment lo compensa
    
    # K debe haberse movido hacia x
    # Calculamos similitud coseno
    cos_sim = torch.nn.functional.cosine_similarity(x, rp.K, dim=-1)
    assert cos_sim.item() > 0.0 # Debe ser algo positivo si se movio hacia x

def test_rperceptron_novelty_gate():
    d_in, M = 10, 2
    rp = RPerceptron(d_in, M, theta=0.8, beta=100.0)
    
    # Punto muy cercano a un prototipo
    with torch.no_grad():
        rp.K[0] = torch.tensor([1.0] + [0.0]*9)
        rp.K[1] = torch.tensor([0.0, 0.0, 1.0] + [0.0]*7)
    
    x_familiar = torch.tensor([[1.0] + [0.0]*9])
    _, f_fam, g_fam, _ = rp(x_familiar)
    
    # f debe ser ~1, g debe ser ~0 (no novedad)
    assert f_fam.item() > 0.9
    assert g_fam.item() < 0.1
    
    # Punto muy lejano (ortogonal)
    x_novel = torch.tensor([[0.0, 1.0] + [0.0]*8])
    _, f_nov, g_nov, _ = rp(x_novel)
    
    # f debe ser ~0, g debe ser ~1 (novedad)
    assert f_nov.item() < 0.2
    assert g_nov.item() > 0.9
