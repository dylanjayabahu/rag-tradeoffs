# Failure Modes and Tradeoffs in Retrieval-Augmented Generation (RAG)

This repository contains a framework for quantifying the architectural tradeoffs between retrieval strategies and LLM performance. By stress-testing 10+ models under varying noise conditions, this project maps the **Pareto Frontier** of inference latency versus accuracy and identifies critical "failure cliffs" in long-context retrieval.

## 🚀 Key Achievements
* **Characterized RAG Failure Modes:** Identified performance degradation patterns (e.g., "Lost-in-the-Middle") by varying chunk sizes (128–1024) and Top-K retrieval density.
* **Pareto Frontier Mapping:** Quantified the optimal balance between inference speed and model grounding using **PyTorch**, **FAISS**, and **HuggingFace Transformers**.
* **Quantified Architectural Tradeoffs:** Conducted systematic ablations to determine how model size (1B vs 3B) affects reasoning robustness under retrieval noise.
* **Responsible AI & Telemetry:** Implemented structured evaluation loops to identify hallucinations and context-window truncation.

---

## 📊 Experimental Results & Analysis

### 1. The Pareto Frontier: Accuracy vs. Latency
Our benchmarking reveals that model parameter count does not linearly correlate with RAG efficiency. 

* **StableLM-Zephyr-3B** emerged as the **Pareto Optimal** choice, delivering the highest accuracy (~59%) with the lowest relative latency.
* Smaller models like **TinyLlama-1.1B** exhibited dominated behavior—higher latency with lower accuracy—likely due to less efficient attention mechanisms during long-prompt processing on CPU/MPS backends.

### 2. Failure Mode: The "Context Cliff"
By varying the document length from 512 to 16,000+ tokens, we identified a critical failure mode:

* **Chunking Granularity:** Smaller chunk sizes (128 tokens) demonstrated significantly higher resilience to retrieval noise, maintaining ~80% accuracy across long contexts.
* **The Truncation Cliff:** Larger chunk sizes (1024 tokens) suffered a massive performance collapse beyond 2,000 tokens. This quantifies the **Positional Embedding Limit**, where retrieved context exceeds the model's native context window (2048 tokens), leading to catastrophic forgetting of the "needle."

---

## 🛠 Tech Stack
- **Inference:** PyTorch, HuggingFace Transformers
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Data Science:** Pandas, Seaborn, Matplotlib
- **Experiment Tracking:** Custom Checkpointing & Telemetry System

---

## 📂 Project Structure
```text
├── src/
│   ├── models.py          # Unified LLM Interface
│   ├── retrieval.py       # FAISS indexing and search logic
│   └── data_generator.py  # Synthetic "Needle-in-a-Haystack" generator
├── notebooks/
│   └── analysis.py        # Pareto and Failure Mode visualization
├── experiments/           # Raw CSV logs and generated plots
├── main.py                # Grid search & experiment orchestrator
└── environment.yml        # Reproducible Conda environment
```

Gemini said
Here is a comprehensive, professional README.md designed specifically to match the high-level bullet points on your resume. It translates your experimental charts into the technical narrative recruiters look for.

Markdown
# Failure Modes and Tradeoffs in Retrieval-Augmented Generation (RAG)

This repository contains a framework for quantifying the architectural tradeoffs between retrieval strategies and LLM performance. By stress-testing 10+ models under varying noise conditions, this project maps the **Pareto Frontier** of inference latency versus accuracy and identifies critical "failure cliffs" in long-context retrieval.

## 🚀 Key Achievements
* **Characterized RAG Failure Modes:** Identified performance degradation patterns (e.g., "Lost-in-the-Middle") by varying chunk sizes (128–1024) and Top-K retrieval density.
* **Pareto Frontier Mapping:** Quantified the optimal balance between inference speed and model grounding using **PyTorch**, **FAISS**, and **HuggingFace Transformers**.
* **Quantified Architectural Tradeoffs:** Conducted systematic ablations to determine how model size (1B vs 3B) affects reasoning robustness under retrieval noise.
* **Responsible AI & Telemetry:** Implemented structured evaluation loops to identify hallucinations and context-window truncation.

---

## 📊 Experimental Results & Analysis

### 1. The Pareto Frontier: Accuracy vs. Latency
Our benchmarking reveals that model parameter count does not linearly correlate with RAG efficiency. 

* **StableLM-Zephyr-3B** emerged as the **Pareto Optimal** choice, delivering the highest accuracy (~59%) with the lowest relative latency.
* Smaller models like **TinyLlama-1.1B** exhibited "dominated" behavior—higher latency with lower accuracy—likely due to less efficient attention mechanisms during long-prompt processing on CPU/MPS backends.

### 2. Failure Mode: The "Context Cliff"
By varying the document length from 512 to 16,000+ tokens, we identified a critical failure mode:

* **Chunking Granularity:** Smaller chunk sizes (128 tokens) demonstrated significantly higher resilience to retrieval noise, maintaining ~80% accuracy across long contexts.
* **The Truncation Cliff:** Larger chunk sizes (1024 tokens) suffered a massive performance collapse beyond 2,000 tokens. This quantifies the **Positional Embedding Limit**, where retrieved context exceeds the model's native context window (2048 tokens), leading to catastrophic forgetting of the "needle."

---

## 🛠 Tech Stack
- **Inference:** PyTorch, HuggingFace Transformers
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Data Science:** Pandas, Seaborn, Matplotlib
- **Experiment Tracking:** Custom Checkpointing & Telemetry System

---

## 📂 Project Structure
```text
├── src/
│   ├── models.py          # Unified LLM Interface (MPS/CPU optimized)
│   ├── retrieval.py       # FAISS indexing and search logic
│   └── data_generator.py  # Synthetic "Needle-in-a-Haystack" generator
├── notebooks/
│   └── analysis.py        # Pareto and Failure Mode visualization
├── experiments/           # Raw CSV logs and generated plots
├── main.py                # Grid search & experiment orchestrator
└── environment.yml        # Reproducible Conda environment
```

## 🔧 Setup & Reproducibility

### Clone and Install
```bash
git clone https://github.com/yourusername/rag-tradeoffs.git
cd rag-tradeoffs
conda env create -f environment.yml
conda activate rag
```

### Generate Synthetic Stress-Test Data
```bash
python src/data_generator.py
```

### Run the Benchmark Sweep

The orchestrator includes built-in checkpointing. If the process is interrupted, it will resume from the last saved unique ID.
```bash
python main.py
```

### Visualize Results
```bash
python notebooks/analysis.py
```
