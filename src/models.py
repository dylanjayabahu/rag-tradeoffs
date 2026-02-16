import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

class LLMInterface:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)
        
        # Get the strict limit from the config
        self.max_tokens = getattr(self.tokenizer, "model_max_length", 2048)
        if self.max_tokens > 1e6: # Handle cases where it defaults to a huge number
            self.max_tokens = 2048

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate_answer(self, query: str, retrieved_context: str) -> dict:
        prompt = f"Context: {retrieved_context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Check for truncation
        token_count = self.count_tokens(prompt)
        is_truncated = token_count > self.max_tokens

        start_time = time.perf_counter()
        outputs = self.pipe(prompt, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        latency = time.perf_counter() - start_time
        
        response = outputs[0]['generated_text'].split("Answer:")[-1].strip()
        
        return {
            "answer": response,
            "latency": latency,
            "model_id": self.model_id,
            "token_count": token_count,
            "is_truncated": is_truncated
        }


if __name__ == "__main__":
    try:
        tester = LLMInterface("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        res = tester.generate_answer("What is the secret?", "The secret is BANANA.")
        print("\n--- Result ---")
        print(res)
    except Exception as e:
        print(f"Error occurred: {e}")