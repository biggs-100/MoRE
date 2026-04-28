# Technical Design: MoRE Expert Mitosis

## Data Structures
- **Experience Buffer**: `self.buffers = [deque(maxlen=256) for _ in range(len(self.experts))]` inside `MoRE` class.
- **Health Metrics**:
  - `H_i = -sum(p * log(p + eps))` where $p$ is attention weights from `RPerceptron`.

## Mitosis Logic: more_demo.py
- **Monitor Step**: 
  - Store winning inputs in corresponding `self.buffers[winner_idx]`.
  - Calculate $H$ and $f$ for the batch.
- **Trigger**:
  - Check if `len(self.buffers[i]) >= 128` (min samples).
  - Check if `mean(H_i) > 0.8 * log(M)` and `mean(f_i) < 0.4`.
- **The Split**:
  ```python
  X = np.stack(list(self.buffers[expert_idx]))
  kmeans = KMeans(n_clusters=2).fit(X)
  c1, c2 = kmeans.cluster_centers_
  
  # Create new experts
  e1 = RPerceptron(..., d_input=d, M=M)
  e2 = RPerceptron(..., d_input=d, M=M)
  
  # Initialize with centroids + noise
  e1.keys.data = torch.from_numpy(c1).repeat(M, 1) + torch.randn_like(e1.keys)*0.01
  e2.keys.data = torch.from_numpy(c2).repeat(M, 1) + torch.randn_like(e2.keys)*0.01
  
  # Update expert list and reset buffer
  self.experts[expert_idx] = e1
  self.experts.insert(expert_idx + 1, e2)
  self.buffers.pop(expert_idx)
  self.buffers.insert(expert_idx, deque(maxlen=256))
  self.buffers.insert(expert_idx + 1, deque(maxlen=256))
  ```

## Validation Experiment: train_mitosis.py
- **Dataset**: A, B, C (known), D (introduced mid-training).
- **Phases**:
  - Phase 1: Train on A, B, C (300 steps).
  - Phase 2: Inject D (300 steps).
  - Expected: Mitosis occurs in Phase 2.
