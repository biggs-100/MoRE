# MoRE-3: Mixture of Resonance-Experts (v3.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

**MoRE-3** is a neuro-symbolic architecture designed for autonomous, lifelong learning. Unlike traditional gradient-based models, MoRE-3 uses **Resonant Associative Memory**, **Novelty-Aware Gating**, and **Biological Mitosis** to grow its knowledge base without catastrophic forgetting.

## 🚀 Key Features

- **Intrinsic Novelty Detection**: Each expert unit (R-Perceptron) calculates a familiarity score ($f$) to identify and reject Out-of-Distribution (OOD) data.
- **Scalable Vector Search**: Integrated with **FAISS-IVF** for $O(\log M)$ retrieval, enabling millions of prototypes with sub-millisecond latency.
- **Autonomous Mitosis**: Experts monitor their own "cognitive health" (entropy/familiarity) and split into specialized units when overloaded.
- **Local Hebbian Learning**: Contrastive learning rule that strengthens winning prototypes and repels errors without global backpropagation.
- **Encoder Agnostic**: Compatible with any embedding dimension (validated on 384d `all-MiniLM-L6-v2`).

## 🧠 Core Philosophy

> "Resonance, Surprise and Growth Are All You Need."

MoRE-3 treats learning as a resonance problem. If the system "surprises" itself (low familiarity), it gates the input as novelty and triggers structural evolution (growth) instead of overwriting existing knowledge.

## 🛠 Installation

```bash
pip install torch numpy faiss-cpu sentence-transformers scikit-learn rich
```

## 📊 Benchmarks & Demos

### 1. Final Integrated Demo (The Graduation)
Runs the full evolutionary narrative using real news headlines (Sports, Tech, Politics, Health).
```bash
python final_integrated_demo.py
```
- **Phase 1**: Mastery of known classes (100% Acc).
- **Phase 2**: Autonomous Mitosis triggered by the introduction of a novel class.

### 2. FAISS Scaling Benchmark
Validates the efficiency of the hybrid search engine.
```bash
python faiss_benchmark.py
```
- Demonstrated **3.38x speedup** on $10^5$ prototypes.

### 3. Mitosis Validation
Visualizes the structural growth of the network.
```bash
python train_mitosis.py
```

### 4. Robustness Audit
Measures system resilience under varying levels of semantic noise.
```bash
python robustness_audit.py
```
- Validated **94%+ Novelty Rejection** even under high noise ($\sigma = 0.20$).

## 📜 Research
For a deep dive into the mathematical foundations and experimental results, see the [PAPER_DRAFT.md](PAPER_DRAFT.md) included in this repository.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed as part of the Sovereign AI initiative for Frugal LLMs.*
