import json
import random
import uuid
import os

class RAGDataGenerator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        #facts to prevent memorization
        self.fact_pool = [
            ("The secret passkey for the vault is: {code}.", "What is the secret passkey for the vault?"),
            ("The project code name is {code}.", "What is the project code name?"),
            ("The server access token is {code}.", "What is the server access token?"),
            ("The emergency override protocol is {code}.", "What is the emergency override protocol?")
        ]
        
        # Distractors (Technical/Wiki-style)
        self.distractors = [
            "The transformer architecture relies on multi-head self-attention mechanisms.",
            "Photosynthesis converts light energy into chemical energy in chloroplasts.",
            "The Byzantine Fault Tolerance algorithm ensures consensus in distributed systems.",
            "Standard deviation is a measure of the amount of variation in a set of values.",
            "The Magna Carta was signed in 1215 at Runnymede.",
            "Python's Global Interpreter Lock (GIL) prevents multi-threaded CPU execution.",
            "Quantum entanglement occurs when particles remain connected regardless of distance."
        ]

    def generate_synthetic_test_case(self, context_length: int, needle_position: str):
        fact_template, query = random.choice(self.fact_pool)
        passkey = f"MONKEY-{random.choice(['ALPHA', 'BETA', 'GAMMA'])}-{random.randint(100, 999)}-X"
        fact = fact_template.format(code=passkey)
        
        #find num sentences needed for approx context_length (assumes 12 words per sentence)
        num_sentences = context_length // 12
        noise = [random.choice(self.distractors) for _ in range(num_sentences)]
        
        if needle_position == "top":
            insertion_point = 0
        elif needle_position == "bottom":
            insertion_point = len(noise)
        elif needle_position == "middle":
            insertion_point = len(noise) // 2
        else:
            raise ValueError("Invalid needle_position. Choose from 'top', 'middle', 'bottom'.")
            

        noise.insert(insertion_point, fact)
        context = " ".join(noise)

        return {
            "id": str(uuid.uuid4()),
            "query": query,
            "context": context,
            "answer": passkey,
            "metadata": {
                "target_length": context_length,
                "actual_word_count": len(context.split()),
                "needle_pos": needle_position,
                "noise_depth_pct": round(insertion_point / len(noise), 2) if noise else 0
            }
        }

    def save_dataset(self, num_samples: int, filename: str = "test_set.jsonl"):
        path = os.path.join(self.data_dir, filename)
        lengths = [512, 1024, 2048, 4096, 8192, 16384] # Testing specific power-of-2 boundaries
        positions = ["top", "middle", "bottom"]
        
        with open(path, 'w') as f:
            for _ in range(num_samples):
                case = self.generate_synthetic_test_case(
                    context_length=random.choice(lengths),
                    needle_position=random.choice(positions)
                )
                f.write(json.dumps(case) + "\n")
        
        print(f"Successfully generated {num_samples} samples at {path}")

if __name__ == "__main__":
    gen = RAGDataGenerator()
    gen.save_dataset(200)