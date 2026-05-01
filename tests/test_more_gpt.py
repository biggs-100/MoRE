import pytest
import torch
from more_tiny_llm import MoREGPT

def test_gpt_forward_shape():
    vocab_size = 50
    d_model = 64
    n_experts = 2
    batch_size = 4
    seq_len = 16
    
    model = MoREGPT(vocab_size, d_model, n_experts)
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(idx, reset_state=True)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)

def test_gpt_state_reset():
    vocab_size = 10
    d_model = 16
    model = MoREGPT(vocab_size, d_model, n_experts=1)
    
    idx = torch.randint(0, vocab_size, (2, 5))
    
    # First pass
    _ = model(idx, reset_state=True)
    s1_re = model.expert_pool.experts[0].state_re.clone()
    
    # Second pass without reset should change state
    _ = model(idx, reset_state=False)
    s2_re = model.expert_pool.experts[0].state_re
    assert not torch.equal(s1_re, s2_re)
    
    # Third pass with reset should clear state to zeros then compute
    _ = model(idx, reset_state=True)
    # The state after the pass won't be zero, but we can check if it's consistent
    # if the input is the same and we reset.
    pass1_output = model(idx, reset_state=True)
    pass2_output = model(idx, reset_state=True)
    assert torch.allclose(pass1_output, pass2_output)
