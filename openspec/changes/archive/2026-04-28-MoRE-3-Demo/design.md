# Technical Design: MoRE-3 Demo

## Architecture Overview
The system consists of a `MoRE` manager that orchestrates multiple `RPerceptron` experts. Each expert represents a semantic cluster (class).

## Component Design

### 1. RPerceptron (Core Unit)
- **Parameters**:
  - `K`: Prototypical keys (normalized).
  - `V`: Prototypical values.
  - `s`: Importance weights.
- **Logic Improvements**:
  - **Inhibition**: `S_masked = S.masked_fill(mask == 0, float('-inf'))` where mask is top-k.
  - **Diversity**: `diversity_bias = -gamma * winner_counts`.
  - **Gate g**: `1 - sigmoid(beta * (f - theta))`.
  - **Dual Output**: Returns `(y, f, g, attn)`.

### 2. MoRE Manager
- **Expert List**: `[Expert_A, Expert_B, Expert_C]`.
- **Router**:
  ```python
  scores = [expert(x)[1] for expert in experts]
  winner = argmax(scores)
  ```
- **Inference**: Returns winning expert index and max familiarity.

### 3. Local Hebbian Learning
Modulated rule based on reward $R \in \{1, -1\}$:
- $\Delta K_{winner} = R \cdot \eta \cdot (x - K_{winner})$
- $\Delta s_{winner} = R \cdot 0.01$ (with thresholding)
- **Contrastive Note**: If $R=-1$, the update pushes $K$ away from $x$ and reduces $s$ faster.

## Data Flow
1. Input $x$ is normalized.
2. Each Expert $i$ computes familiarity $f_i$ and gate $g_i$.
3. Router picks $i = \text{argmax}(f)$.
4. If $f_i < \text{threshold}$, output is "Novelty".
5. During training, $R$ is computed based on true label and $i$.
6. Winning Expert $i$ updates its internal state.

## Aesthetic Considerations
- Clear logging with `rich`.
- Visualizing cluster centroids in 2D (if possible).
- Modular and well-documented Python code.
