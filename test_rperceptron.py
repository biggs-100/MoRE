import torch
import pytest
from rperceptron import RPerceptron

def test_rperceptron_forward_shape():
    d_in, M, batch = 128, 10, 5
    rp = RPerceptron(d_in, M)
    x = torch.randn(batch, d_in)
    winners, f, y, g, scores = rp(x)
    
    assert y.shape == (batch,)
    assert f.shape == (batch,)
    assert g.shape == (batch,)
    # scores is the inhibited_scores tensor or (indices, top_scores)
    if isinstance(scores, tuple):
        assert scores[1].shape == (batch, rp.topk)
    else:
        assert scores.shape == (batch, M)

def test_rperceptron_wta_inhibition():
    d_in, M, topk = 10, 5, 2
    rp = RPerceptron(d_in, M, topk=topk)
    x = torch.randn(1, d_in)
    _, _, _, _, scores = rp(x)
    
    # Solo top-k deben tener atencion significativa (softmax)
    if not isinstance(scores, tuple):
        non_zero_attn = (scores > float('-inf')).sum().item()
        assert non_zero_attn == topk

def test_rperceptron_local_update_positive():
    d_in, M = 10, 1
    rp = RPerceptron(d_in, M)
    x = torch.randn(1, d_in)
    x = torch.nn.functional.normalize(x, dim=-1)
    
    # Inicialmente s es 1.0
    initial_s = rp.s[0].item()
    
    # Forzar forward para obtener attn
    _, _, _, _, scores = rp(x)
    
    # Recompensa positiva
    reward = torch.tensor([1.0])
    rp.update_local(x, reward, scores, lr=0.1)
    
    # s debe haber aumentado (s * 1.01 * decay)
    # decay=0.999, s = (1.0 * 1.01) * 0.999 = 1.00899
    # Antes del update
    cos_sim_before = torch.nn.functional.cosine_similarity(x, rp.keys, dim=-1).item()
    
    # Recompensa positiva
    reward = torch.tensor([1.0])
    rp.update_local(x, reward, scores, lr=0.1)
    
    # keys debe haberse movido hacia x
    cos_sim_after = torch.nn.functional.cosine_similarity(x, rp.keys, dim=-1).item()
    assert cos_sim_after > cos_sim_before

def test_rperceptron_novelty_gate():
    d_in, M = 10, 2
    rp = RPerceptron(d_in, M, theta=0.8, beta=100.0)
    
    # Punto muy cercano a un prototipo
    with torch.no_grad():
        rp.keys[0] = torch.tensor([1.0] + [0.0]*9)
        rp.keys[1] = torch.tensor([0.0, 0.0, 1.0] + [0.0]*7)
    
    x_familiar = torch.tensor([[1.0] + [0.0]*9])
    _, f_fam, _, g_fam, _ = rp(x_familiar)
    
    # Activity Gate: f > theta => g ~ 1
    assert f_fam.item() > 0.9
    assert g_fam.item() > 0.9
    
    # Punto muy lejano (ortogonal)
    x_novel = torch.tensor([[0.0, 1.0] + [0.0]*8])
    _, f_nov, _, g_nov, _ = rp(x_novel)
    
    # Activity Gate: f < theta => g ~ 0
    assert f_nov.item() < 0.2
    assert g_nov.item() < 0.1
