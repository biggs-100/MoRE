# Tasks: MoRE-3 Stress Tests

## Phase 1: Infrastructure Expansion
- [ ] 1.1 Implement `split_mnist` support in `benchmark/stream.py`.
- [ ] 1.2 Update `run_benchmark.py` to accept `--theta` and `--n_experts`.
- [ ] 1.3 Ensure `results/` naming convention includes theta value.

## Phase 2: Orchestration & Viz
- [ ] 2.1 Create `run_theta_sweep.py` for automated ablation.
- [ ] 2.2 Update `benchmark/visualizer.py` to generate the ACC vs BWT Pareto curve.

## Phase 3: Stress Test Execution
- [ ] 3.1 Execute Split MNIST (5 tasks, theta=0.3, n_experts=10).
- [ ] 3.2 Execute Theta Sweep on Permuted MNIST (4 runs).

## Phase 4: Final Reporting
- [ ] 4.1 Consolidate all plots in `results/plots/`.
- [ ] 4.2 Validate that Split MNIST achieves > 90% ACC.
