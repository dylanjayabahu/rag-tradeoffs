import json
import pandas as pd
import os
import torch
from src.retrieval import RAGRetriever
from src.models import LLMInterface
from tqdm import tqdm

def run_professional_benchmarks(num_samples=100):
    os.makedirs("experiments", exist_ok=True)
    results_path = "experiments/pareto_data.csv"
    
    # Expand to your 10 models here
    models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "google/gemma-2b",
        "stabilityai/stablelm-zephyr-3b"
    ]
    chunk_sizes = [128, 512, 1024] 
    top_k_values = [1, 5]

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        print(f"Resuming from {len(existing_df)} existing records...")
    else:
        existing_df = pd.DataFrame()

    with open("data/test_set.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]

    retriever = RAGRetriever()

    for model_id in models:
        model_name_safe = model_id.replace("/", "_")
        try:
            llm = LLMInterface(model_id)
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            continue
        
        for c_size in chunk_sizes:
            for k in top_k_values:
                print(f"\n>> Model: {model_id} | Chunk: {c_size} | Top-K: {k}")
                
                for item in tqdm(dataset[:num_samples]):
                    case_id = f"{model_name_safe}_{c_size}_{k}_{item['id']}"
                    
                    if not existing_df.empty and case_id in existing_df['unique_id'].values:
                        continue
                    
                    # Retrieval
                    retriever.build_index(item['context'], chunk_size=c_size)
                    hits = retriever.search(item['query'], top_k=k)
                    context_text = " ".join([h['text'] for h in hits])
                    
                    # Inference with Truncation Check
                    output = llm.generate_answer(item['query'], context_text)
                    
                    # Scoring (Handle truncation as a failure mode)
                    is_correct = item['answer'].lower() in output['answer'].lower()
                    
                    res = {
                        "unique_id": case_id,
                        "model": model_id,
                        "chunk_size": c_size,
                        "top_k": k,
                        "latency": output['latency'],
                        "accuracy": int(is_correct),
                        "token_count": output['token_count'],
                        "is_truncated": int(output['is_truncated']),
                        "context_len": item['metadata']['target_length'],
                        "needle_pos": item['metadata']['needle_pos']
                    }
                    
                    pd.DataFrame([res]).to_csv(
                        results_path, 
                        mode='a', 
                        header=not os.path.exists(results_path), 
                        index=False
                    )
                    
                    # Update checkpoint tracker
                    new_row = pd.DataFrame([res])
                    existing_df = pd.concat([existing_df, new_row], ignore_index=True)

        del llm
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    print(f"\n[SUCCESS] Grid search complete. Results in {results_path}")

if __name__ == "__main__":
    run_professional_benchmarks(num_samples=50)