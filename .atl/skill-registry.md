# Skill Registry

## Compact Rules

### python-research
- Follow PEP8.
- Use `torch` and `numpy` for all numerical operations.
- Always implement `__repr__` or logging for structural changes (mitosis).
- Use `Rich` for console outputs.

### sdd-research
- Every code change must have a corresponding test.
- Benchmarks must save JSON results in `results/`.
- Visualizations must use the `Agg` backend for Matplotlib.

## User Skills

| Skill | Trigger | Context |
|-------|---------|---------|
| sdd-apply | Launch implementation | Implementation phase |
| sdd-verify | Validate changes | Verification phase |
| sdd-tasks | Create breakdown | Planning phase |
| sdd-spec | Write specs | Specification phase |
| sdd-design | Write technical design | Design phase |
| sdd-propose | Create proposal | Initiation phase |
| sdd-explore | Investigate ideas | Exploration phase |
| skill-creator | Create new skills | Documentation phase |
| judgment-day | Review adversarial | Critical review phase |
