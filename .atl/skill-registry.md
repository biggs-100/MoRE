# Skill Registry - MoRE

## User Skills
| Skill | Trigger | Description |
|-------|---------|-------------|
| branch-pr | PR creation, opening a PR | PR workflow |
| issue-creation | GitHub issue, bug report | Issue workflow |
| judgment-day | review adversarial, dual review | Parallel review |
| skill-creator | create new skill, agent instructions | Skill creation |

## Project Conventions
- **Language**: Python
- **Testing**: pytest
- **Workflow**: SDD (Spec-Driven Development)
- **Local Learning**: Hebbian rules for expert adaptation.

## Compact Rules
### Python Standards
- Use type hints for all function signatures.
- Follow PEP 8 style guide.
- Use `torch` for tensor operations.
- Avoid global state; use classes for model encapsulation.

### MoRE Architecture
- Experts must inherit from `nn.Module`.
- Use `RPerceptron` as the base expert unit.
- Familiarity gate must be used for novelty detection.
