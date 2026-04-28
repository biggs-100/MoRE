# Proposal: MoRE Scale with FAISS

## Intent
Enable logarithmic scaling for MoRE experts by replacing dense matrix multiplication with optimized FAISS indexing.

## Scope
- `rperceptron.py`: Update `RPerceptron` to include optional FAISS indexing.
- `faiss_benchmark.py`: New benchmark to compare speed and precision of Exact vs FAISS.
- `more_demo.py`: (Optional) Update to propagate `use_faiss` flag.

## Technical Approach
1. **Hybrid Inference**: Implement `if self.use_faiss and self.M > self.faiss_threshold` in `forward()`.
2. **Index Management**: 
   - Maintain `self.index = faiss.IndexFlatIP(d_input)`.
   - Update index in `update_local()` after prototype modification.
3. **WTA Compatibility**: Ensure FAISS `topk` search aligns with existing WTA logic.
4. **Familiarity Parity**: Verify that `scores` returned by FAISS are identical to dot product results.

## Success Criteria
- **Accuracy Parity**: >99.9% match between Exact and FAISS results.
- **Performance**: Significant speedup (at least 2x) for $M > 10,000$ prototypes.
- **Novelty Integrity**: Rejection rates must remain unchanged.
