# Specification: RM3 Autonomous Mitosis

## Overview
This specification defines the behavior for the autonomous structural expansion of Resonant Mamba-3 (RM3) experts. The system must detect when its current capacity is insufficient for new data and create a new specialized expert.

## Requirements

### 1. Resonance Detection (Routing)
- **Input**: Latent features $h \in \mathbb{R}^{d_{model}}$.
- **Process**: Each expert $E_i$ attempts to align $h$ using its internal REA mechanism.
- **Metric**: $Fidelity_i = \exp(-\|h - h_{aligned, i}\|^2 / \sigma^2)$.
- **Output**: The expert with the highest $Fidelity$ is selected for processing.

### 2. Mitosis Trigger
- **Condition**: If the maximum $Fidelity$ across all experts drops below a threshold $\tau_{novelty} = 0.4$ for $N=100$ consecutive samples.
- **Action**: Trigger `perform_mitosis()` on the most "stressed" expert (the one that received the most data but with the highest average error).

### 3. Bifurcation Mechanism (The Split)
- **Inheritance**: The daughter expert $E_{new}$ inherits all weights from the parent $E_{parent}$ (projections, A parameters, etc.).
- **Symmetry Breaking**: 
    - Add Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.05)$ to the daughter's `A_imag` (resonant frequencies).
    - Add Gaussian noise to the daughter's `dt_proj` weights.
- **Buffer Management**: The parent's `AnchorBuffer` is NOT copied; the daughter starts with an empty buffer to learn the new manifold from scratch.

### 4. Integration
- The `MoRE` class must be updated to support `ResonantMamba3Layer` as a valid expert type.
- The `GradientsProjector` (OGD) must be initialized for the new expert to protect the shared encoder if necessary.

## Scenarios

### Scenario: Novel Domain Detection
- **Given** a model trained on task A (e.g., Sine wave counting).
- **When** task B is introduced (e.g., Square wave counting with different frequencies).
- **Then** the REA fidelity should drop.
- **And** mitosis should trigger after 100 samples.
- **And** a second expert should emerge and specialize in task B.
- **And** performance on task A should remain stable.
