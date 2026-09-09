# Failure Modes and Tradeoffs in Retrieval-Augmented Generation (RAG)

A synthetic passkey-retrieval benchmark comparing accuracy and latency across 10+ models, document lengths, needle positions, and chunk sizes. The plots summarize the tested configurations; they do not isolate the effects of model architecture or training.

<img width="1422" height="550" alt="image" src="https://github.com/user-attachments/assets/8d5d54c6-c473-4535-ad86-3322137a5aa3" />

## 📊 Experimental Results & Analysis

### 1. Model Tradeoffs: Accuracy vs. Latency
The reported runs show the following accuracy–latency tradeoffs:

* **Top Performers:** **google/gemma-3-1b-it** and **meta-llama/Llama-3.2-1B-Instruct** combine high retrieval accuracy with low latency among the tested configurations.
* **Retrieval Results:** **microsoft/Phi-4-mini-instruct** and **meta-llama/Llama-3.2-3B-Instruct** reach approximately 78–79% retrieval accuracy in the plotted comparison, with 4–5× the latency of the compared 1B models.
* **Legacy Comparison:** **TinyLlama-1.1B** and **phi-2** show lower accuracy at similar or higher latency in these runs.

### 2. Retrieval Sensitivity: Positional Bias
The needle-in-a-haystack task measures retrieval accuracy by information position:

* **Primacy Bias:** Nearly all models achieve 90-100% accuracy when the relevant information is located in the Top (0-33%) of the context.
* **The Lost-in-the-Middle Phenomenon:** Accuracy drops by up to 40% for models like **HuggingFaceTB/SmolLM-135M** and **EleutherAI/pythia-1.4b** when the answer is buried in the middle (33-66%) of the prompt.
* **Context Robustness:** **Gemma-3** and **Llama-3.2** variants show comparatively consistent recall across the tested positions. This benchmark does not establish why their results differ.

### 3. Failure Mode: The Context Cliff
The generator varies target document length from 512 to 16,384 using an approximate word-based construction, not exact token counts:

* **Chunking Resilience:** The 128-word chunk configuration maintains approximately 80%+ accuracy across the plotted document lengths.
* **Performance Collapse:** The 1024-word chunk configuration shows declining accuracy as document length increases. The results do not isolate positional embedding drift or another causal mechanism.

## Measurement

Accuracy is scored by checking whether the expected passkey appears in the generated answer. The runner records latency and truncation flags. This is a retrieval test, not a general reasoning or hallucination benchmark.

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
├── experiments/         # Diagnostic plots; the runner writes CSV results here
└── main.py              # Grid search & experiment orchestrator
```

## 🔧 Setup & Reproducibility

### Clone and Install
```bash
git clone https://github.com/dylanjayabahu/rag-tradeoffs.git
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
