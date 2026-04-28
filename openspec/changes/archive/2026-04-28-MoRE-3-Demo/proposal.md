# Proposal: MoRE-3 Demo Implementation

## Intent
Build a functional demonstration of the Mixture of R-Experts architecture with 3 experts, showcasing local learning and novelty detection.

## Scope
- `rperceptron.py`: Core R-Perceptron unit with WTA, diversity bias, and novelty gate.
- `more_demo.py`: MoRE manager and similarity router.
- `train_demo.py`: Local Hebbian training implementation.
- `eval_demo.py`: Novelty detection and classification metrics.
- `dataset.py`: Synthetic cluster generation.

## Technical Approach
- **Experts**: Each class (A, B, C) gets one `RPerceptron` expert.
- **Inference**: Each expert computes familiarity `f`. The one with `max(f)` is chosen. If `max(f) < threshold`, response is "I don't know" (or gate `g` is open).
- **Learning**: Modulated Hebbian rule:
  - If Correct: `K[winner] += lr * (x - K[winner])`, `s[winner] += 0.01`
  - If Incorrect: `K[winner] -= 0.5 * lr * (x - K[winner])`, `s[winner] -= 0.02`
- **Novelty**: Class D will be used only during evaluation to verify that experts reject it (high `g`, low `f`).

## Rollback Plan
- Revert to base `RPerceptron` implementation if contrastive learning fails to converge.
- Use simpler backprop-based routing if similarity routing is unstable.

## Stakeholders
- User (Reviewer/Developer)
