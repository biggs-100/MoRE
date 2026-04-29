# Proposal: MoRE-3 Stress Tests

## Intent
Validate the structural stability and scalability of MoRE-3 through specialized benchmarking scenarios that isolate the effects of structural expansion, class separation, and novelty detection sensitivity.

## Scope
1. **MoRE-5 Experiment**: Evaluate 5-task Permuted MNIST with a fixed 1:1 Expert-to-Task mapping (or automated growth to 5 experts) to demonstrate near-zero forgetting.
2. **Split MNIST Experiment**: Transition to disjoint class streams (5 tasks, 2 classes each) to validate the "Jury Test" findings in a standardized lifelong learning benchmark.
3. **Ablation Study (Theta Sweep)**: Execute a parametric sweep of the novelty threshold $\theta$ to generate an ACC vs BWT Pareto curve.

## Approach
- Extend `benchmark/stream.py` to support the `split_mnist` mode.
- Update `run_benchmark.py` to allow overriding `theta` and `n_experts` via CLI.
- Implement a sweep script `run_theta_sweep.py` that orchestrates multiple benchmark runs.
- Generate comparative plots integrating these new dimensions.

## Risks
- **Resource Exhaustion**: Large expert pools or many sweep iterations could slow down the benchmark. Subsampling (1000/task) will remain the default.
- **Overfitting**: High $\theta$ might lead to too much expansion (one expert per sample in extreme cases). We will monitor the `n_experts` growth.

## Rollback Plan
- The existing benchmark results are stored in `results/`. New results will use distinct filenames (e.g., `results_sweep_*.json`).
