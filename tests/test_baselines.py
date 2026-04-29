import torch
import pytest
from benchmark.baselines import MLPBaseline, EWCBaseline

def test_mlp_baseline_training():
    # GIVEN: A simple classification task
    X = torch.randn(100, 512)
    y = torch.randint(0, 10, (100,))
    model = MLPBaseline(input_dim=512, n_classes=10)
    
    # WHEN: We train
    loss_before = torch.nn.functional.cross_entropy(model.predict(X), y)
    model.train_task(X, y, task_id=0, epochs=5)
    loss_after = torch.nn.functional.cross_entropy(model.predict(X), y)
    
    # THEN: Loss should decrease
    assert loss_after < loss_before

def test_ewc_baseline_consolidation():
    # GIVEN: An EWC model
    X = torch.randn(100, 512)
    y = torch.randint(0, 10, (100,))
    model = EWCBaseline(input_dim=512, n_classes=10, lambda_ewc=100.0)
    
    # WHEN: We train on task 0 and consolidate
    model.train_task(X, y, task_id=0, epochs=2)
    model.consolidate(X, y)
    
    # THEN: Fisher information should be computed for parameters
    assert len(model.fisher) > 0
    for p_name, f in model.fisher.items():
        assert f.shape == model.model.state_dict()[p_name].shape
