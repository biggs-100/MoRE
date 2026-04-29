import pytest
import numpy as np
from unittest.mock import MagicMock
from benchmark.stream import TaskStream
from benchmark.metrics import MetricEngine

def test_benchmark_loop_structure():
    # GIVEN: A mock stream, metric engine and model
    n_tasks = 2
    stream = TaskStream(n_tasks=n_tasks, mode='synthetic')
    engine = MetricEngine(n_tasks=n_tasks)
    model = MagicMock()
    model.predict.return_value = torch.zeros(100, 10) # 100 samples, 10 classes
    
    # Simular que el modelo entrena y predice
    # (En la práctica, el loop real llamará a train_task y evaluará)
    
    # WHEN: We run the loop (we'll implement this in benchmark/runner.py)
    # Por ahora, verificamos que las dimensiones coincidan
    assert engine.R.shape == (2, 2)
