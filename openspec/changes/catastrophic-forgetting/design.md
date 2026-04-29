# Design: Catastrophic Forgetting Benchmark

## Technical Approach
Implementaremos un framework de evaluación desacoplado que permita inyectar diferentes modelos (MoRE-3, MLP, EWC) en el mismo flujo de datos no estacionario. La clave es el uso de un **Feature Extractor congelado** para garantizar que comparamos la gestión del conocimiento y no la capacidad de extracción de características.

## Architecture Decisions

### Decision: Feature Extractor
**Choice**: Frozen ResNet-18 (pre-trained on ImageNet).
**Alternatives considered**: Trainable CNN, raw pixel input.
**Rationale**: Entrenar una CNN desde cero en Split CIFAR-100 introduciría variabilidad por la optimización de pesos convolucionales. Al congelar ResNet, nos enfocamos puramente en cómo el clasificador (MoRE vs MLP) maneja el drift de la tarea.

### Decision: Metric Storage
**Choice**: Matrix $R \in \mathbb{R}^{T \times T}$ almacenada en JSON/CSV.
**Alternatives considered**: Solo promedios finales.
**Rationale**: Necesitamos la matriz completa para visualizar el mapa de calor de interferencia y calcular BWT con precisión.

## Data Flow
```text
Raw Data (CIFAR/MNIST) ──→ Feature Extractor (ResNet) ──→ Embeddings (d=512)
                                                               │
Metrics Tracker ←── Evaluation (all seen tasks) ←── TaskStream (T_i) ──→ Model
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `benchmark/stream.py` | Create | Generadores de tareas (Permuted MNIST, Split CIFAR-100). |
| `benchmark/metrics.py` | Create | Cálculo de ACC, BWT y utilidades de plotting (heatmaps). |
| `benchmark/baselines.py` | Create | Implementación de MLP Estándar y EWC (Elastic Weight Consolidation). |
| `run_benchmark.py` | Create | Script principal para ejecutar el stream y guardar resultados. |
| `rperceptron.py` | Modify | Añadir modo `eval` para desactivar el aprendizaje Hebbiano durante el test. |

## Interfaces / Contracts
```python
class TaskStream:
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, int]: # X, y, task_id
        ...

class ContinualModel(Protocol):
    def train_on_batch(self, X: torch.Tensor, y: torch.Tensor):
        ...
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        ...
```

## Testing Strategy
| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | `TaskStream` | Verificar que cada tarea tiene etiquetas disjuntas (Split CIFAR). |
| Unit | `Metrics` | Validar cálculo de BWT con una matriz $R$ conocida. |
| Integration | `run_benchmark.py` | Ejecución de un mini-stream (2 tareas) para validar el flujo. |
