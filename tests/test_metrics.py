import pytest
import numpy as np
from benchmark.metrics import MetricEngine

def test_acc_calculation():
    # GIVEN: A 2x2 accuracy matrix
    # R[i, j] is accuracy on task j after training on task i
    R = np.array([
        [0.8, 0.0],
        [0.7, 0.9]
    ])
    engine = MetricEngine(n_tasks=2)
    engine.R = R
    
    # WHEN: We calculate ACC for the final state (i=1)
    acc = engine.calculate_acc()
    
    # THEN: ACC = (R[1,0] + R[1,1]) / 2 = (0.7 + 0.9) / 2 = 0.8
    assert acc == pytest.approx(0.8)

def test_bwt_calculation():
    # GIVEN: A 2x2 accuracy matrix
    R = np.array([
        [0.8, 0.0],
        [0.7, 0.9]
    ])
    engine = MetricEngine(n_tasks=2)
    engine.R = R
    
    # WHEN: We calculate BWT
    # BWT = (R[1,0] - R[0,0]) / 1 = (0.7 - 0.8) = -0.1
    bwt = engine.calculate_bwt()
    
    # THEN: BWT should be -0.1
    assert bwt == pytest.approx(-0.1)
