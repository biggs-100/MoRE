# Exploration: MoRE Expert Mitosis

## Objective
Implement autonomous expert splitting (Mitosis) in the MoRE architecture based on experience saturation and entropy.

## Research Findings
- **Trigger Conditions**:
  - **Assignment Entropy ($H$)**: Measures the spread of attention among prototypes within an expert. High entropy indicates the expert is trying to represent too many distinct patterns.
  - **Mean Familiarity ($f$)**: Low familiarity for winning samples indicates poor representation.
- **Experience Buffer**: A `collections.deque` per expert with a fixed capacity (e.g., 256 samples) will store recent inputs that the expert won.
- **K-Means Splitting**:
  - `sklearn.cluster.KMeans(n_clusters=2)` is efficient for dividing the buffer.
  - Initialization of new experts with centroids + small noise ($\epsilon$) prevents identical weight collapse.
- **Dynamic Registry**: The `MoRE.experts` list must be dynamic, and the `MoRE.forward` loop must handle varying number of experts.

## Implementation Risks
- **Catastrophic Forgetting**: When an expert splits, we must ensure the new experts don't immediately "forget" the old patterns while specializing. Centroid initialization helps.
- **Thrashing**: Rapid splitting and merging could occur if thresholds are too sensitive. We need a "refractory period" or conservative thresholds.
