import torch
import pytest
from benchmark.features import FeatureExtractor

def test_feature_extractor_output_shape():
    # GIVEN: A FeatureExtractor
    extractor = FeatureExtractor()
    
    # WHEN: We pass a batch of images (N, 3, 32, 32)
    x = torch.randn(4, 3, 32, 32)
    features = extractor.extract(x)
    
    # THEN: Output should be (4, 512) for ResNet-18
    assert features.shape == (4, 512)
    assert not features.requires_grad # Should be frozen
