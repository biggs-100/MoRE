# Specification: MoRE-3 Real Text Benchmark Core

## Overview
Validate that the R-Perceptron can distinguish between real semantic categories and identify an "out-of-distribution" category using language model embeddings.

## Scenarios

### Scenario 1: Real-World Categorization
**Given** a MoRE model trained on embeddings of Sports, Tech, and Politics headlines
**When** presented with a new headline about "Computing performance"
**Then** the model MUST classify it as Tech
**And** the familiarity score MUST be high (>0.6)

### Scenario 2: Semantic Novelty Detection
**Given** a MoRE model trained on Sports, Tech, and Politics
**When** presented with a headline about "Medical breakthrough in cancer treatment" (Health)
**Then** the model MUST flag it as novelty
**And** the prediction MUST be "I don't know" (-1)
**And** the familiarity score MUST be significantly lower than the training classes.

### Scenario 3: Architectural Stability
**Given** a 384-dimensional embedding space
**When** the current `RPerceptron` is initialized with `d_input=384`
**Then** it MUST perform inference and local updates without modification to its internal logic.
