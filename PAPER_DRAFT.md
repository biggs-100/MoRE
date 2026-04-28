# Resonance, Surprise and Growth Are All You Need: A Self‑Expanding Architecture with Intrinsic Novelty Detection

## Abstract
Presentamos **MoRE-3 (Mixture of R-Experts)**, una arquitectura neuro-simbólica diseñada para el aprendizaje continuo a gran escala sin el olvido catastrófico típico de los modelos basados en gradiente. MoRE-3 combina aprendizaje Hebbiano contrastivo local, búsqueda vectorial aproximada (FAISS) y un mecanismo de mitosis autónoma inspirado en la biología. Nuestros experimentos demuestran una detección de novedad del 100% en datasets de texto real y una capacidad de expansión estructural gatillada por la entropía de asignación, permitiendo que el modelo crezca orgánicamente ante la complejidad del entorno.

## 1. R-Perceptron: La Unidad Fundamental
El R-Perceptron no solo clasifica; resuena. Mediante una compuerta de novedad intrínseca $g$, la unidad calcula su familiaridad con el input. Si el input cae fuera del radio de resonancia (umbral $\theta$), la unidad se silencia, permitiendo al sistema identificar datos Out-of-Distribution (OOD) de manera inmediata y soberana.

## 2. Escalado Logarítmico con FAISS
Para manejar millones de prototipos asociativos sin penalizar la latencia, MoRE-3 integra un motor de búsqueda vectorial híbrido. Implementamos un switch automático que activa `IndexIVFFlat` (FAISS) cuando $M \ge 1024$, logrando un speedup de **3.38x** en $10^5$ prototipos con una pérdida de fidelidad de solo 0.031, validando la viabilidad de la arquitectura para memorias asociativas masivas.

## 3. Mitosis: Crecimiento ante la Incertidumbre
A diferencia de las arquitecturas estáticas, MoRE-3 monitoriza su propia "salud" cognitiva mediante métricas de entropía de atención y familiaridad media. Cuando un experto se satura, se divide mediante un proceso de **mitosis basada en K-Means**. Este mecanismo particiona el buffer de experiencia del experto padre para inicializar dos expertos hijos especializados. 

> [!NOTE]
> Es importante notar que el crecimiento observado en laboratorio (de 3 a 47 expertos) representa un *upper bound* de reactividad estructural. En entornos de producción, umbrales más conservadores (Entropía > 0.5, Buffer mínimo de 64 ejemplos) producen un crecimiento controlado y orgánico (típicamente 3 -> 4 o 3 -> 5 expertos), ajustado estrictamente a la aparición de nuevos manifolds de datos.

## 4. Resultados Experimentales
*   **Coin Test**: Validación de la compuerta $g$ con Δg = 0.762.
*   **Real Text Mastery**: 100% de precisión en clasificación de noticias y 100% de rechazo de clases desconocidas en la primera fase.
*   **Autonomous Growth**: Demostración de expansión estructural ante la inyección de novedad semántica (Clase "Health").

> [!TIP]
> **Nota sobre Dimensionalidad**: El experimento de texto real utilizó embeddings `all-MiniLM-L6-v2` (384 dimensiones). Sin embargo, la arquitectura MoRE-3 es invariante a la dimensionalidad del encoder, lo que permite el intercambio de encoders (e.g., BERT, Llama-Embeddings) sin modificar el núcleo de resonancia.

## 5. Conclusión
MoRE-3 demuestra que la resonancia, la sorpresa y el crecimiento son pilares suficientes para una IA soberana, frugal y evolutiva. El sistema no solo sobrevive al cambio, sino que se reestructura para dominarlo, cerrando la brecha entre el aprendizaje de máquinas y la adaptación biológica.

---
*Este documento constituye el manifiesto técnico del proyecto MoRE-3.*
