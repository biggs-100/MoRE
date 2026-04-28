# Task Breakdown: MoRE Expert Mitosis

## Phase 1: Core Architecture Update
- [x] 1.1 Implement experience buffers in `MoRE.__init__`.
- [x] 1.2 Update `MoRE.forward` to store inputs in winner buffers.
- [x] 1.3 Add entropy and familiarity monitoring logic.

## Phase 2: Mitosis Implementation
- [x] 2.1 Implement `MoRE.perform_mitosis(expert_idx)`.
- [x] 2.2 Integrate `KMeans` for centroid-based initialization.
- [x] 2.3 Implement replacement logic and expert list management.

## Phase 3: Autonomous Growth Experiment
- [x] 3.1 Create `train_mitosis.py` with 2-phase training (ABC -> ABCD).
- [x] 3.2 Add "Novelty injection" trigger to simulate new class introduction.
- [x] 3.3 Verify 4th expert creation and accuracy (>90%).

## Phase 4: Integration & Archive
- [x] 4.1 Update `README.md` with mitosis autonomous growth results.
- [x] 4.2 Archive change in `openspec`.
