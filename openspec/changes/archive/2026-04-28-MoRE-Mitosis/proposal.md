# Proposal: MoRE Expert Mitosis

## Intent
Allow the MoRE architecture to grow autonomously by splitting saturated experts into specialized units, enabling true lifelong learning.

## Scope
- `more_demo.py`: Update `MoRE` class to include:
  - Experience buffers per expert.
  - Health monitoring (Entropy + Familiarity).
  - `mitosis_step()` method to handle the split.
- `train_mitosis_demo.py`: New script to demonstrate autonomous growth from 3 to 4 experts.

## Technical Approach
1. **Buffer Integration**: Add `self.buffers = [deque(maxlen=256) for _ in range(n_experts)]`.
2. **Health Check**: In `predict()` or `forward()`, track:
   - `H = -mean(p * log(p))` where $p$ is the normalized attention within the winner expert.
   - Trigger mitosis if $H > H_{threshold}$ and $f_{mean} < f_{threshold}$.
3. **The Split**:
   - Run `KMeans(n_clusters=2)` on the winner expert's buffer.
   - Create two `RPerceptron` instances.
   - Initialize their keys with `centroids + noise`.
   - Swap the old expert with the two new ones.
4. **Validation**: Train on 3 classes, then introduce a 4th class and observe the split.

## Success Criteria
- **Autonomous Growth**: Model expands from 3 to 4 experts without manual intervention.
- **Accuracy Parity**: The final 4-expert model achieves $>95\%$ accuracy on all 4 classes.
- **Structural Integrity**: The routing remains robust after the split.
