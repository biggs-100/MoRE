# Specification: MoRE Scale with FAISS

## Overview
Implement an optimized search path for MoRE experts to handle large prototype sets ($M \gg 1000$) using FAISS, while maintaining exact parity with dense computation for smaller sets.

## Scenarios

### Scenario 1: Exact Match Parity
**Given** an R-Perceptron with $M=500$ prototypes
**When** inference is performed using FAISS vs Dense dot product
**Then** the resulting winner indices and familiarity scores MUST match exactly.

### Scenario 2: High-Scale Efficiency
**Given** an R-Perceptron with $M=100,000$ prototypes
**When** inference is performed
**Then** the FAISS path MUST be utilized
**And** the latency MUST be significantly lower than the dense path.

### Scenario 3: Novelty Detection under ANN
**Given** an R-Perceptron using FAISS
**When** presented with a novel sample (similarity below `theta`)
**Then** the novelty gate MUST correctly identify it
**And** the familiarity score MUST be consistent with the exact distance.
