# Task Breakdown: MoRE Scale with FAISS

## Phase 1: R-Perceptron Refactor
- [x] 1.1 Add `faiss` import and conditional initialization.
- [x] 1.2 Implement `_rebuild_index` helper.
- [x] 1.3 Update `forward` to support hybrid FAISS/Dense paths.
- [x] 1.4 Update `update_local` to sync the index.

## Phase 2: Benchmarking & Optimization
- [x] 2.1 Create `faiss_benchmark.py`.
- [x] 2.2 Run accuracy parity tests for $M=500$.
- [x] 2.3 Run speed benchmarks for $M=10^2, 10^3, 10^4, 10^5$.
- [x] 2.4 Identify the optimal `faiss_threshold`.

## Phase 3: Integration & Archive
- [x] 3.1 Verify real-text benchmark results with `use_faiss=True`.
- [x] 3.2 Update `README.md` with scalability metrics.
- [x] 3.3 Archive change in `openspec`.
