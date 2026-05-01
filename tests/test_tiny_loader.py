import pytest
import torch
import os
from tiny_stories_loader import TinyStoriesLoader

def test_loader_initialization():
    # Test with dummy text if no file exists
    dummy_text = "Once upon a time. The end."
    with open("dummy_stories.txt", "w") as f:
        f.write(dummy_text)
    
    loader = TinyStoriesLoader(file_path="dummy_stories.txt", batch_size=2, seq_len=5)
    assert loader.vocab_size > 0
    assert len(loader.tokens) > 0
    
    os.remove("dummy_stories.txt")

def test_batch_generation():
    dummy_text = "abcdefghijklmnopqrstuvwxyz" * 10
    with open("dummy_stories.txt", "w") as f:
        f.write(dummy_text)
    
    batch_size = 4
    seq_len = 8
    loader = TinyStoriesLoader(file_path="dummy_stories.txt", batch_size=batch_size, seq_len=seq_len)
    
    x, y = loader.get_batch()
    
    assert x.shape == (batch_size, seq_len)
    assert y.shape == (batch_size, seq_len)
    # y should be x shifted by 1
    assert torch.equal(x[:, 1:], y[:, :-1])
    
    os.remove("dummy_stories.txt")

def test_tokenization_cycle():
    loader = TinyStoriesLoader(batch_size=1, seq_len=1, data_str="Hello World")
    encoded = loader.encode("Hello")
    decoded = loader.decode(encoded)
    assert decoded == "Hello"
