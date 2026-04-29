# Implementation Progress: Catastrophic Forgetting Benchmark

**Change**: catastrophic-forgetting
**Mode**: Strict TDD

### Completed Tasks
- [x] 1.1 Crear `benchmark/stream.py` con la clase `TaskStream` (soporte para Permuted MNIST y Split CIFAR).
- [x] 1.2 Crear `benchmark/metrics.py` con `MetricEngine` para calcular Matriz $R$, ACC y BWT.
- [x] 1.3 Implementar extractor de características en `benchmark/features.py` usando ResNet-18 (congelado).

### Files Changed
| File | Action | What Was Done |
|------|--------|---------------|
| `benchmark/stream.py` | Created | Implementación de `TaskStream` con soporte para Permuted MNIST y Split CIFAR. |
| `benchmark/metrics.py` | Created | Motor de métricas para cálculo de $R_{i,j}$, ACC y BWT. |
| `benchmark/features.py` | Created | Extractor de características basado en ResNet-18 congelado. |
| `tests/test_benchmark_stream.py` | Created | Suite de pruebas para el stream de tareas. |
| `tests/test_metrics.py` | Created | Suite de pruebas para el motor de métricas. |
| `tests/test_features.py` | Created | Suite de pruebas para el extractor de características. |

### TDD Cycle Evidence
| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 1.1 | `tests/test_benchmark_stream.py` | Unit | N/A | ✅ Written | ✅ Passed | ✅ 3 modes | ✅ Clean |
| 1.2 | `tests/test_metrics.py` | Unit | N/A | ✅ Written | ✅ Passed | ✅ ACC/BWT | ✅ Clean |
| 1.3 | `tests/test_features.py` | Unit | N/A | ✅ Written | ✅ Passed | ➖ Single | ✅ Clean |

### Deviations from Design
None — implementation matches design.

### Issues Found
- La descarga de datasets (CIFAR-100) y pesos de modelos (ResNet) puede ser lenta en la primera ejecución de los tests.

### Status
3/11 tasks complete. Ready for Batch 2 (Baselines & Main Loop).
