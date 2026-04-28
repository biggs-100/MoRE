# Exploration: MoRE Scale with FAISS

## Objective
Integrate FAISS for approximate nearest neighbor search (ANN) in R-Perceptron to achieve log-scale complexity for large prototype sets.

## Research Findings
- **FAISS Index**: `IndexFlatIP` is the correct choice for normalized vectors (Inner Product is equivalent to Cosine similarity).
- **Hybrid Strategy**:
  - `M < 1000`: Use PyTorch dense matrix multiplication (`x @ K.T`). Faster due to low kernel launch overhead and GPU/CPU SIMD efficiency for small matrices.
  - `M >= 1000`: Use FAISS search. Faster for high-M as it avoids the full scan (if IVF is used) or uses highly optimized CPU kernels.
- **Novelty Integrity**: FAISS `IndexFlatIP` is exact, so it should not degrade novelty detection.
- **Integration Points**: 
  - `RPerceptron.forward` needs a logic switch based on `use_faiss` flag and $M$.
  - `RPerceptron.update_local` needs to rebuild or update the FAISS index after prototypes are updated.

## Prototype Update Strategy
Since Hebbian learning updates prototypes frequently, we have two options:
1. **Rebuild Index**: Fast for `IndexFlatIP` with small-medium $M$.
2. **Incremental Update**: If using more complex indices (like HNSW), we might need `index.add` and `index.remove`. For `IndexFlatIP`, rebuilding is likely simpler and sufficiently fast.
