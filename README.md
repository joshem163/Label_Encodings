# Label Encodings: Supervision as an Input Channel for Graph Learning

📄 **Paper**: Label Encodings: Supervision as an Input Channel for Graph Learning  
🔗 **Repository**: https://github.com/joshem163/Label_Encodings  

---

## 🧠 Overview

Graph learning models typically rely on **node features** and **graph structure**, while **labels are only used in the loss function**.

In this work, we introduce a new paradigm:

> **Supervision as an input channel**

We propose **Label Encodings**, a lightweight and leakage-safe representation that transforms training labels into node-level descriptors. These encodings can be directly used with:

- MLPs  
- Graph Neural Networks (GCN, GraphSAGE, GAT)  
- Graph Transformers  

without modifying backbone architectures.

---

## 🚀 Key Contributions

- Introduce **Label Encodings** as a reusable supervision-derived input  
- Two complementary components:
  - **Proto-Embeddings** (feature-side class geometry)  
  - **Hop-wise Label Descriptors (HLD)** (graph-side label accessibility)  
- Show that **integration strategy matters**:
  - Concatenation works for MLPs & Transformers  
  - **Late fusion is crucial for GNNs**  
- Propose **FLA** and **GLS** diagnostics to predict effectiveness  
- Demonstrate consistent improvements on **21 node classification benchmarks**

---

## 🏗️ Method

### 1. Proto-Embeddings

Maps each node to distances from class prototypes:

- Captures **class-relative structure**
- Computed using training labels only
- Independent of graph structure

---

### 2. Hop-wise Label Descriptors (HLD)

For each node:

- Collect labeled neighbors at hops \( k = 1,2,3 \)
- Compute:
  - Label distributions
  - Coverage statistics  

Captures **local supervision signal in the graph**

---

### 3. Final Encoding

\[
\psi(v) = \alpha(v) \;\|\; \beta(v)
\]

- \( \alpha(v) \): Proto-embedding  
- \( \beta(v) \): HLD descriptor  

---

## ⚙️ Integration Strategy

| Model | Strategy |
|------|--------|
| MLP | Concatenation |
| Graph Transformers | Concatenation |
| GNNs (GCN, SAGE, GAT) | ⚠️ Late Fusion |

**Why late fusion?**

Early concatenation in GNNs causes **message passing to wash out supervision signals**.

---

## 📊 Results

- Improvements across **21 datasets**
- Strong gains for:
  - MLPs (+6.26 avg)
  - GCN (+7.66 avg with late fusion)
- Stable improvements for Graph Transformers

Performance depends on:

- **FLA (Feature Label Accessibility)**
- **GLS (Graph Label Separability)**

---
## 🚀 Running

```bash
python train_fusion.py --dataset cora --model_type gcn
