# Spec: Catastrophic Forgetting Benchmark (MoRE-3 vs Baselines)

## 1. Contexto y Objetivos
El objetivo es cuantificar la capacidad de MoRE-3 para aprender tareas secuenciales sin sufrir olvido catastrófico. Se evaluará contra un MLP estándar y un MLP con EWC.

## 2. Requisitos Funcionales (REQ)
- **REQ-1 (Task Stream)**: El sistema DEBE proporcionar un flujo de tareas $T_1, T_2, ..., T_n$.
- **REQ-2 (Permuted MNIST)**: Cada tarea $T_i$ DEBE consistir en el dataset MNIST con una permutación aleatoria de píxeles única para esa tarea.
- **REQ-3 (Split CIFAR-100)**: Cada tarea $T_i$ DEBE consistir en 10 clases mutuamente excluyentes de CIFAR-100.
- **REQ-4 (Metrics Engine)**: El sistema DEBE calcular la Matriz de Precisión $R_{i,j}$ donde $i$ es el entrenamiento actual y $j$ es la tarea evaluada.
- **REQ-5 (Evaluation)**: El sistema DEBE calcular ACC (Average Accuracy) y BWT (Backward Transfer) al final del stream.

## 3. Escenarios de Validación

### Escenario 1: Estabilidad en Permuted MNIST
- **GIVEN**: Un modelo MoRE-3 con 3 expertos iniciales.
- **WHEN**: Se entrena secuencialmente en 10 tareas de Permuted MNIST (1 epoch por tarea).
- **THEN**: La precisión final en la Tarea 1 ($R_{10,1}$) NO DEBE caer más de un 5% respecto a su precisión inicial ($R_{1,1}$).

### Escenario 2: Mitosis por Especialización (CIFAR-100)
- **GIVEN**: Un stream de Split CIFAR-100 con 5 tareas.
- **WHEN**: El modelo detecta una caída de resonancia significativa al cambiar de tarea.
- **THEN**: El mecanismo de mitosis DEBE crear un nuevo experto especializado para la nueva distribución de características.

### Escenario 3: Comparativa de Interferencia
- **GIVEN**: MoRE-3 y un MLP estándar de capacidad similar.
- **WHEN**: Se completa el stream de tareas.
- **THEN**: El BWT de MoRE-3 DEBE ser superior al del MLP en al menos 0.30 puntos (ej. -0.05 vs -0.35).

## 4. Restricciones Técnicas
- **Single Pass**: No se permite el re-entrenamiento (re-run) sobre tareas pasadas.
- **No Replay**: No se permite el uso de buffers de memoria para almacenar ejemplos de tareas previas.
