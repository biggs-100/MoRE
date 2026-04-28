# Technical Design: MoRE Scale with FAISS

## R-Perceptron Modification
- **Index Lifecycle**: 
  - `self.index = None`
  - `self.use_faiss = True`
  - `self.faiss_threshold = 1024` (Configurable)
- **Index Construction**:
  ```python
  def _rebuild_index(self):
      self.index = faiss.IndexFlatIP(self.d_input)
      # FAISS expects float32 numpy arrays
      keys_np = self.keys.detach().cpu().numpy().astype('float32')
      self.index.add(keys_np)
  ```
- **Forward Logic**:
  ```python
  if self.use_faiss and self.M >= self.faiss_threshold:
      # Search top-k
      scores, indices = self.index.search(x_np, self.topk)
      # Construct masking for WTA
  else:
      # Standard dot product
      scores = x @ self.keys.T
  ```

## Benchmark Utility: faiss_benchmark.py
- **Dataset**: Synthetic high-dimensional vectors ($d=384$).
- **Scale Test**: $M \in [100, 1000, 10000, 100000]$.
- **Metrics**:
  - **Match Rate**: % of winners that are identical.
  - **Latency**: Mean time per inference.
  - **Familiarity Error**: Mean absolute difference between scores.

## Prototype Updates
Every time `update_local` is called and `self.use_faiss` is active, `_rebuild_index` must be triggered to reflect the weight changes in the associative memory.
