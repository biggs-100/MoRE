import torch
import pytest
from benchmark.stream import TaskStream

def test_task_stream_length():
    # GIVEN: A TaskStream with 5 tasks
    stream = TaskStream(n_tasks=5, mode='synthetic')
    
    # WHEN: We iterate over it
    tasks = list(stream)
    
    # THEN: We should get exactly 5 tasks
    assert len(tasks) == 5

def test_permuted_mnist_uniqueness():
    # GIVEN: A TaskStream for Permuted MNIST
    stream = TaskStream(n_tasks=3, mode='permuted_mnist')
    
    # WHEN: We get two different tasks
    tasks = list(stream)
    X1, y1, tid1 = tasks[0]
    X2, y2, tid2 = tasks[1]
    
    # THEN: The permutations should be different (pixels shifted differently)
    # We compare the first samples. They should not be identical.
    assert not torch.equal(X1[0], X2[0])
    assert tid1 == 0
    assert tid2 == 1

def test_split_cifar_disjointness():
    # GIVEN: A TaskStream for Split CIFAR-100
    stream = TaskStream(n_tasks=2, mode='split_cifar')
    
    # WHEN: We get two different tasks
    tasks = list(stream)
    X1, y1, tid1 = tasks[0]
    X2, y2, tid2 = tasks[1]
    
    # THEN: Labels should be disjoint
    labels1 = set(y1.tolist())
    labels2 = set(y2.tolist())
    assert labels1.isdisjoint(labels2)
