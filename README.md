# Failure Modes and Tradeoffs in Retrieval-Augmented Generation (RAG)

This framework quantifies architectural tradeoffs between retrieval strategies and LLM performance by stress-testing 10+ models under varying context density. The project identifies the Pareto Frontier of inference latency versus accuracy and maps performance degradation across long-context retrieval windows.

<img width="1422" height="550" alt="image" src="https://github.com/user-attachments/assets/8d5d54c6-c473-4535-ad86-3322137a5aa3" />

## 📊 Experimental Results & Analysis

### 1. Model Tradeoffs: Accuracy vs. Latency
Benchmarking reveals that modern distilled and small-parameter models redefine RAG efficiency:

* **Top Performers:** **google/gemma-3-1b-it** and **meta-llama/Llama-3.2-1B-Instruct** occupy the high-efficiency frontier, delivering near-perfect accuracy with minimal latency.
* **Reasoning Value:** **microsoft/Phi-4-mini-instruct** and **meta-llama/Llama-3.2-3B-Instruct** provide the highest accuracy ceiling (~78-79%) for complex reasoning, albeit with a 4-5x latency penalty compared to 1B models.
* **Legacy Comparison:** Older architectures like **TinyLlama-1.1B** and **phi-2** are now dominated, showing significantly lower accuracy despite similar or higher latency profiles.

### 2. Retrieval Sensitivity: Positional Bias
Analysis of retrieval accuracy by information position (Needle-in-a-Haystack) highlights architectural recall limits:

* **Primacy Bias:** Nearly all models achieve 90-100% accuracy when the relevant information is located in the Top (0-33%) of the context.
* **The Lost-in-the-Middle Phenomenon:** Accuracy drops by up to 40% for models like **HuggingFaceTB/SmolLM-135M** and **EleutherAI/pythia-1.4b** when the answer is buried in the middle (33-66%) of the prompt.
* **Context Robustness:** **Gemma-3** and **Llama-3.2** variants maintain the most consistent recall across all positions, demonstrating superior long-context attention training.

### 3. Failure Mode: The Context Cliff
Systematic testing of document lengths from 512 to 16,000+ tokens identified critical scaling limits:

* **Chunking Resilience:** Small chunk sizes (128 tokens) maintain high accuracy (~80%+) even as total document length scales, effectively filtering noise.
* **Performance Collapse:** Larger chunk sizes (1024 tokens) experience a sharp accuracy decline beyond 2,000 tokens, quantifying the impact of context window saturation and positional embedding drift.

## 🚀 Key Achievements
* **Characterized RAG Failure Modes:** Quantified performance decay patterns by isolating chunk size (128–1024) and Top-K retrieval density variables.
* **Pareto Frontier Mapping:** Established optimal balance between inference speed and model grounding using **PyTorch**, **FAISS**, and **HuggingFace Transformers**.
* **Quantified Architectural Tradeoffs:** Conducted systematic ablations to determine how model size (1B vs 3B) and architecture (Gemma vs Llama vs Phi) affect reasoning robustness.
* **Telemetry & Observability:** Implemented structured evaluation loops to detect hallucinations, context-window truncation, and latency scaling per 1k tokens.

## 🛠 Tech Stack
- **Inference:** PyTorch, HuggingFace Transformers (Accelerated with Apple Metal/MPS)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Data Science:** Pandas, Seaborn, Matplotlib
- **Automation:** Custom experiment orchestrator with built-in checkpointing and resume logic

## 📂 Project Structure
```text
├── src/
│   ├── models.py        # Unified LLM Interface (MPS/CPU optimized)
│   ├── retrieval.py     # FAISS indexing and chunking logic
│   └── data_generator.py # Synthetic Needle-in-a-Haystack generator
├── notebooks/
│   └── analysis.py      # Pareto, Heatmap, and Failure Mode visualization
├── experiments/         # Raw CSV logs and diagnostic plots
└── main.py              # Grid search & experiment orchestrator
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
