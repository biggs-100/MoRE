# Task Breakdown: MoRE-3 Demo

## Phase 1: Infrastructure & Model (rperceptron.py)
- [x] 1.1 Implement `RPerceptron` class with improved `forward` (WTA, diversity, dual output).
- [x] 1.2 Add `decay` logic to importance `s`.
- [x] 1.3 Implement `update_local` method for Hebbian/contrastive learning.
- [x] 1.4 Write unit tests for `RPerceptron` (Scenario 4).

## Phase 2: MoRE Architecture (more_demo.py)
- [x] 2.1 Implement `MoRE` class with similarity-based router.
- [x] 2.2 Add `predict` method with novelty threshold.
- [x] 2.3 Write unit tests for routing logic (Scenario 1 & 2).

## Phase 3: Training & Data (train_demo.py, dataset.py)
- [x] 3.1 Implement synthetic dataset generator with 4 clusters (A, B, C, D).
- [x] 3.2 Implement local training loop with reward modulation (Scenario 3).
- [x] 3.3 Add logging for training progress (familiarity, rewards).

## Phase 4: Evaluation & Demo (eval_demo.py)
- [x] 4.1 Implement evaluation script for classification and novelty.
- [x] 4.2 Create visualization for novelty gate behavior.
- [x] 4.3 Verify all scenarios from `spec.md`.

## Phase 5: Documentation
- [x] 5.1 Create `README.md` with execution instructions.
