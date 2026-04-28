# Proposal: MoRE-3 Real Text Benchmark

## Intent
Demonstrate the robustness of the R-Perceptron's novelty detection and local learning using real semantic clusters from language models.

## Scope
- `real_dataset.py`: New module to handle text category generation and embedding.
- `train_real.py`: Training script adapted for text embeddings.
- `eval_real.py`: Benchmark script to measure accuracy and novelty rejection.
- No changes to `rperceptron.py` or `more_demo.py` (architectural stability validation).

## Technical Approach
1. **Embedder**: Use `SentenceTransformer('all-MiniLM-L6-v2')`.
2. **Training**: Train on 3 classes (Sports, Tech, Politics).
3. **Evaluation**: Test on training classes + Health (Novelty).
4. **Metrics**:
   - Accuracy on Known Categories.
   - Rejection Rate on Novel Category.
   - Average Familiarity distribution.

## Success Criteria
- >90% Accuracy on Known Classes.
- >90% Rejection Rate on Health category (threshold-based).
