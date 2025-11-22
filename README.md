# A Comparative Analysis of Transformer-Based Topic Modeling Pipelines for Scientific Literature

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org/try)
[![BERTopic](https://img.shields.io/badge/Model-BERTopic-red?style=for-the-badge)](https://maartengr.github.io/BERTopic/)
[![Transformers](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=for-the-badge)](https://huggingface.co/)

## 📄 Abstract

The exponential growth of scientific literature necessitates automated methods to identify thematic trends. This project conducts a comparative analysis of three distinct topic modeling pipelines to determine the optimal configuration for maximizing topic coherence ($C_v$) in scientific abstracts. 

We evaluate a **Custom Modular Pipeline** (SBERT + UMAP + HDBSCAN), a **Benchmark Pipeline** (RoBERTa + PCA + K-Means), and an **Integrated Framework** (BERTopic). The study utilizes a corpus of **20,972 scientific articles** and demonstrates that integrated models leveraging class-based TF-IDF (c-TF-IDF) significantly outperform traditional clustering combinations.

---

## 🛠️ Methodology & Pipelines

This project implements and benchmarks three distinct architectural philosophies for topic modeling.

### 1. Pipeline 1: The Modular Approach (SBERT-UMAP-HDBSCAN)
A custom-built pipeline designed to mimic modern topic modeling flows using separate, high-performance components.
*   **Embedding:** `all-MiniLM-L6-v2` (Sentence-BERT). Generates 384-d dense vectors.
*   **Dimensionality Reduction:** UMAP (Uniform Manifold Approximation and Projection) reduces vectors to 5 components for density preservation.
*   **Clustering:** HDBSCAN (Hierarchical Density-Based Spatial Clustering). Auto-detects cluster counts and handles noise.
*   **Topic Extraction:** Custom c-TF-IDF implementation.

### 2. Pipeline 2: The Baseline (RoBERTa-PCA-KMeans)
A replication of standard benchmark architectures found in previous literature (Wijanto et al.).
*   **Embedding:** `stsb-roberta-base-v2`. Generates 768-d robust embeddings.
*   **Dimensionality Reduction:** PCA (Principal Component Analysis). Linear reduction to 50 components (capturing ~71% variance).
*   **Clustering:** K-Means ($K=6$). Forces data into spherical, pre-defined clusters.
*   **Constraint:** Poor handling of non-linear semantic relationships and noise.

### 3. Pipeline 3: The Integrated Model (BERTopic)
An advanced, end-to-end framework that integrates transformer embeddings with c-TF-IDF topic representation.
*   **Embedding:** `all-mpnet-base-v2` (Default BERTopic model).
*   **Process:** Internally handles UMAP and HDBSCAN with optimized hyperparameters.
*   **Representation:** Uses class-based TF-IDF to extract highly coherent topic labels, treating clusters as single documents.
*   **Optimization:** The notebook further explores n-gram ranges (1,2) and (1,3) to maximize coherence.

---

## 📊 Dataset

The study utilizes a publicly available dataset of research articles (sourced from Kaggle).
*   **Size:** 20,972 Documents.
*   **Fields:** Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, Quantitative Finance.
*   **Preprocessing:**
    *   Concatenation of `TITLE` and `ABSTRACT`.
    *   Lowercase conversion & Regex cleaning (URL/Email removal).
    *   Tokenization & Stopword removal (NLTK).
    *   Lemmatization (WordNet).

---

## 🚀 Installation & Usage

### Prerequisites
The project requires Python 3.x and GPU acceleration (recommended for Transformer embeddings).

### Installation
Clone the repository and install the dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
pip install sentence-transformers bertopic umap-learn hdbscan
pip install nltk gensim
```

### Running the Project
1.  Ensure the dataset file (`dataset.csv`) is in the root directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook topic_modelling_pipelines_project.ipynb
    ```
3.  Run the cells sequentially. The notebook includes:
    *   Data Preprocessing.
    *   Execution of all three pipelines.
    *   Automated $C_v$ Coherence calculation using Gensim.
    *   Hyperparameter tuning for the BERTopic pipeline.

---

## 📈 Results & Findings

The performance was quantitatively benchmarked using the **$C_v$ Coherence Score**, which measures the semantic similarity of the top words within a topic.

| Pipeline | Configuration | Coherence Score ($C_v$) | Topics Found | Comparison |
| :--- | :--- | :---: | :---: | :--- |
| **1. Modular** | SBERT + UMAP + HDBSCAN | **0.6079** | 175 | Strong baseline, high granularity. |
| **2. Baseline** | RoBERTa + PCA + K-Means | **0.4756** | 6 | Lowest score. Rigid clustering fails to capture nuance. |
| **3. Integrated** | **BERTopic (Original)** | **0.7012** | 69 | **State-of-the-Art performance.** |
| *3.1 Optimized* | *BERTopic (n-gram 1,2)* | *0.7089* | *70* | *Further improvement via hyperparameter tuning.* |

### Key Insights
1.  **Integration Wins:** BERTopic's integrated architecture significantly outperforms ad-hoc combinations. The synergy between density-based clustering and c-TF-IDF allows for cleaner topic descriptions.
2.  **Density vs. Centroid:** HDBSCAN (Pipelines 1 & 3) is far superior to K-Means (Pipeline 2) for text data, as semantic clusters are rarely spherical or equal in size.
3.  **Embedding Quality:** While RoBERTa is a powerful model, PCA's linear reduction destroys the semantic manifold structure, leading to lower coherence compared to UMAP-based approaches.

---

## 📂 Project Structure

```text
.
├── dataset.csv                          # The input dataset (20k abstracts)
├── topic_modelling_pipelines_project.ipynb  # Main execution notebook
├── README.md                            # Comparison documentation
└── requirements.txt                     # Python dependencies
```

---

## 👥 Authors

*   **Farriel Arrianta Akbar Pratama**
*   **Muhammad Eka Nur Arief**

*Informatics Engineering, University of Muhammadiyah Malang, Indonesia*

---
