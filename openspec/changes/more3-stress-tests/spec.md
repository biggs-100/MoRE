# Specification: MoRE-3 Stress Tests

## Requirements

### 1. Split MNIST Data Stream
- **Scenario**: 5 sequential tasks.
- **Mapping**: 
  - Task 0: Classes [0, 1]
  - Task 1: Classes [2, 3]
  - Task 2: Classes [4, 5]
  - Task 3: Classes [6, 7]
  - Task 4: Classes [8, 9]
- **Validation**: Test cases must verify that each task contains ONLY its assigned classes and no overlap with previous tasks.

### 2. Parametric Sweep Infrastructure
- **CLI Support**: `run_benchmark.py` MUST accept `--theta` and `--n_experts` arguments.
- **Persistence**: Results MUST be saved using a unique naming convention `results_{model}_{mode}_theta{val}.json`.
- **Ablation Script**: A new script `run_theta_sweep.py` will automate 4 benchmark runs with $\theta \in \{0.3, 0.5, 0.7, 0.9\}$.

### 3. Metric & Visualization Goals
- **Pareto Curve**: Plot Average Accuracy (Y-axis) vs Backward Transfer (X-axis) across different $\theta$ values.
- **Structural Scaling**: Track and plot the final number of experts created vs $\theta$.

## Scenarios

### Scenario 1: Split MNIST Validation
- **Given**: A MoRE-3 model with 10 initial experts.
- **When**: Trained on Split MNIST (5 tasks).
- **Then**: Accuracy should exceed 90% and BWT should be $\ge -0.01$.

### Scenario 2: Theta Sweep Effect
- **Given**: Increasing values of $\theta$.
- **When**: Running Permuted MNIST benchmark.
- **Then**: BWT should improve (become less negative) as $\theta$ increases, while number of experts should grow.
