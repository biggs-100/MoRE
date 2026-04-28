# Exploration: MoRE-3 Real Text Benchmark

## Objective
Validate the MoRE architecture using real-world text embeddings instead of synthetic clusters.

## Research Findings
- **Model**: `all-MiniLM-L6-v2` from `sentence-transformers` is ideal. It produces 384-dimensional vectors.
- **Dataset Strategy**: To avoid large downloads, we will use a curated list of ~50-100 highly representative sentences per category.
- **Compatibility**: The current `RPerceptron` and `MoRE` classes are agnostic to the input dimension `d`, so no changes are needed to the core architecture.
- **Novelty Expected Behavior**: Health/Medicine headlines should have lower cosine similarity with Sports, Tech, and Politics prototypes, triggering the novelty gate.

## Categories Curated
- **Sports**: Focus on match results, athletes, and leagues.
- **Tech**: Focus on software, AI, hardware, and gadgets.
- **Politics**: Focus on government, elections, and international relations.
- **Health (Novelty)**: Focus on medicine, diseases, and wellness.

## Risks
- **Semantic Overlap**: Tech and Politics sometimes overlap (e.g., regulation).
- **Prototype Saturation**: With 384 dims, we might need to adjust the `theta` threshold for novelty (expected similarity might be higher than in 128-dim random space).
