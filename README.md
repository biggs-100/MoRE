# MoRE-3 Demo (Mixture of R-Experts)

Este proyecto es una demostración funcional de la arquitectura MoRE (Mixture of R-Experts), diseñada para clasificación con aprendizaje local y detección de novedad.

## Características
- **Experts**: Basados en R-Perceptron con inhibición lateral (WTA).
- **Aprendizaje Local**: Regla Hebbiana modulada por recompensa global, sin backpropagation.
- **Detección de Novedad**: Compuerta de novedad basada en familiaridad coseno.
- **Diversidad**: Sesgo para evitar el colapso de expertos.

## Instalación
```bash
py -m pip install torch numpy scikit-learn sentence-transformers rich pytest
```

## Ejecución
1. **Entrenamiento**:
   ```bash
   py train_demo.py
   ```
   Esto generará 3 clusters (A, B, C) y entrenará a los expertos localmente. El modelo se guardará en `more_model.pt`.

2. **Evaluación**:
   ```bash
   py eval_demo.py
   ```
   Evalúa la precisión en las clases conocidas y la capacidad de detectar una clase nueva (D) como "no sé".

## Estructura
- `rperceptron.py`: Implementación de la unidad R-Perceptron.
- `more_demo.py`: Clase MoRE que gestiona los expertos.
- `train_demo.py`: Bucle de entrenamiento Hebbiano.
- `eval_demo.py`: Script de métricas y detección de novedad.
- `dataset.py`: Generador de clusters sintéticos.
- `test_rperceptron.py`: Tests unitarios.
