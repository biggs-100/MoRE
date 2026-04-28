# Specification: MoRE-3 Demo Core

## Overview
This specification defines the functional requirements for the MoRE-3 demonstration, focusing on classification accuracy for known classes and novelty detection for unknown classes.

## Scenarios

### Scenario 1: Successful Classification of Known Classes
**Given** a MoRE model trained on clusters A, B, and C
**And** an input sample from cluster A
**When** the model performs inference
**Then** the winning expert MUST be Expert A
**And** the familiarity score `f` MUST be greater than 0.7

### Scenario 2: Novelty Detection of Unknown Classes
**Given** a MoRE model trained on clusters A, B, and C
**And** an input sample from unknown cluster D
**When** the model performs inference
**Then** the familiarity scores for all experts MUST be less than 0.5
**And** the novelty gate `g` MUST be greater than 0.8 (indicating novelty)
**And** the prediction MUST return "I don't know" (index -1)

### Scenario 3: Local Learning Reinforcement
**Given** an untrained MoRE model
**When** presented with a batch of samples from cluster B with labels
**Then** the winning expert (Expert B) MUST update its prototypes `K` towards the samples
**And** the importance `s` for the winning prototypes MUST increase

### Scenario 4: Expert Inhibition (WTA)
**Given** an R-Perceptron with multiple prototypes
**When** an input activates multiple prototypes
**Then** only the top-k prototypes MUST receive significant attention
**And** all other prototypes MUST be masked with `-inf` before softmax
