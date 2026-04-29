# Tasks: Catastrophic Forgetting Benchmark

## Phase 1: Infraestructura de Datos y Métricas
- [x] 1.1 Crear `benchmark/stream.py` con la clase `TaskStream` (soporte para Permuted MNIST y Split CIFAR).
- [x] 1.2 Crear `benchmark/metrics.py` con `MetricEngine` para calcular Matriz $R$, ACC y BWT.
- [x] 1.3 Implementar extractor de características en `benchmark/features.py` usando ResNet-18 (congelado).

## Phase 2: Modelos y Baselines
- [ ] 2.1 Implementar `MLPBaseline` y `EWCBaseline` en `benchmark/baselines.py`.
- [ ] 2.2 Modificar `rperceptron.py` para añadir flag `learning_enabled` (desactivar Hebbian en inferencia).
- [ ] 2.3 Crear `run_benchmark.py` con el bucle principal de entrenamiento/evaluación continua.

## Phase 3: Validación de Infraestructura (Tests)
- [ ] 3.1 Crear `tests/test_benchmark_stream.py` para validar que las tareas son disjuntas y las permutaciones únicas.
- [ ] 3.2 Crear `tests/test_metrics.py` para validar los cálculos de BWT con datos sintéticos.

## Phase 4: Ejecución y Reporte
- [ ] 4.1 Ejecutar experimento Permuted MNIST (10 tareas) y guardar `mnist_results.json`.
- [ ] 4.2 Ejecutar experimento Split CIFAR-100 (5 tareas) y guardar `cifar_results.json`.
- [ ] 4.3 Generar `benchmark/report.md` con tablas comparativas y análisis de hipótesis.
