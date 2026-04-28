# Exploration: MoRE-3 Demo Implementation

## Objective
Implement a Mixture of R-Experts (MoRE) with 3 experts for classes A, B, C, featuring local learning, novelty detection, and inhibition.

## Research Findings
- **Base Implementation**: The existing `rperceptron.py` already includes WTA and diversity bias, but needs refinement for "contrastive Hebbian" and "dual output" as per the plan.
- **Novelty Detection**: The gate `g` calculation `1 - sigmoid(beta * (f - theta))` is standard but requires careful thresholding (`theta`).
- **Diversity Bias**: Currently implemented using `winner_counts / batch_size`. The plan suggests `gamma * winner_counts.unsqueeze(0)` which is simpler for local batches.
- **FAISS Integration**: Not strictly necessary for a 3-expert demo but can be implemented as an optional layer.
- **Local Learning**: The "reward modulated Hebbian" is the core. We must ensure the reward signal (+1 or -1) correctly updates only the winning expert.

## Proposed Strategy
1. **Refine R-Perceptron**:
   - Add `decay` to importance `s`.
   - Ensure `forward` returns `y, f, g, attn`.
   - Implement contrastive update in a dedicated method or training step.
2. **Implement MoRE Class**:
   - Simple list of experts.
   - Router based on `max(f)`.
3. **Training Loop**:
   - Batch processing with local updates.
   - No `loss.backward()`.
4. **Synthetic Dataset**:
   - 3 Clusters (A, B, C) + 1 Novelty (D).
   - Use fixed random centers for simplicity.

## Risks
- **Threshold Tuning**: If `theta` is too high, everything is novelty; too low, nothing is.
- **Expert Collapse**: If one expert is too strong, it might win all samples. Diversity bias is critical.
