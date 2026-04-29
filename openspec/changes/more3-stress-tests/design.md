# Technical Design: MoRE-3 Stress Tests

## 1. Stream Architecture Expansion
We will modify `TaskStream` in `benchmark/stream.py` to include a logic for splitting classes by indices.

```python
def _get_split_mnist(self, n_tasks, samples_per_task):
    # Logic to filter MNIST by labels [(0,1), (2,3), ...]
```

## 2. CLI and Wrapper Integration
The `run_benchmark.py` will be updated to handle dynamic initialization of the wrapper.

```python
# run_benchmark.py
parser.add_argument('--theta', type=float, default=0.3)
parser.add_argument('--n_experts', type=int, default=10)

# Pass to MoREBenchmarkWrapper
wrapper = MoREBenchmarkWrapper(..., theta=args.theta, n_experts=args.n_experts)
```

## 3. Automated Sweep Script (`run_theta_sweep.py`)
A thin orchestration script that calls `run_benchmark.py` via `subprocess`.

## 4. Visualization
`visualizer.py` will be updated to scan the `results/` directory for files matching `results_more_*_theta*.json` and aggregate them into a Pareto plot.

## 5. Security & Privacy
No changes to FAISS or model privacy. Data remains local.
