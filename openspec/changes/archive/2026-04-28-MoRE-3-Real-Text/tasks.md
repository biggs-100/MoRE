# Task Breakdown: MoRE-3 Real Text Benchmark

## Phase 1: Data Engineering
- [x] 1.1 Create `real_dataset.py` with categorized headlines.
- [x] 1.2 Implement `TextEmbedder` class using `SentenceTransformer`.
- [x] 1.3 Add caching for embeddings to avoid re-encoding every run.

## Phase 2: Training & Benchmarking
- [x] 2.1 Implement `train_real.py` with 384-dimensional MoRE.
- [x] 2.2 Implement local Hebbian training loop for text clusters.
- [x] 2.3 Verify convergence (Accuracy > 90%).

## Phase 3: Novelty Validation
- [x] 3.1 Implement `eval_real.py` with Healthheadlines as novelty.
- [x] 3.2 Tune `theta` and `prediction_threshold` for optimal rejection.
- [x] 3.3 Generate comparison table between Synthetic vs Real performance.

## Phase 4: Archiving
- [x] 4.1 Update `README.md` with real benchmark results.
- [x] 4.2 Archive change in `openspec`.
