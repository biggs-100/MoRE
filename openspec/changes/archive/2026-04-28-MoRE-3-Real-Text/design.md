# Technical Design: MoRE-3 Real Text Benchmark

## Data Engineering: real_dataset.py
- **Library**: `sentence-transformers`
- **Model**: `all-MiniLM-L6-v2`
- **Data Source**: Hardcoded representative headline lists for:
  - `Sports` (Known)
  - `Technology` (Known)
  - `Politics` (Known)
  - `Health` (Novelty)
- **Vectorization**:
  ```python
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model.encode(sentences)
  ```

## Training: train_real.py
- **Initialization**: `MoRE(n_experts=3, d_input=384, ...)`
- **Loop**: Similar to `train_demo.py` but uses the `real_dataset` loader.
- **Normalization**: Explicitly normalize embeddings if `SentenceTransformer` doesn't do it (it usually does, but we'll check).

## Evaluation: eval_real.py
- Measures classification accuracy on known text.
- Measures rejection rate on novel text.
- Visualizes the "Semantic Distance" between experts and novelty.

## Threshold Adjustment
In 384-dimensional space, cosine similarities tend to be tighter. We might need to adjust `theta` in `RPerceptron` or the `threshold` in `MoRE.predict` to correctly separate Health from the rest.
