import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class RAGRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load a standard embedding model from HuggingFace
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 20) -> List[str]:
        words = text.split()

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        stride = chunk_size - overlap

        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), stride)
        ]


    def build_index(self, text: str, chunk_size: int = 256, overlap: int = 20):
        self.chunks = self.chunk_text(text, chunk_size, overlap)
        embeddings = self.encoder.encode(self.chunks)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "score": float(distances[0][i])
                })
        return results
    
def test_retriever():
    test_text = "The password is APPLE. " * 50 + "The secret is BANANA. " + "The end."
    retriever = RAGRetriever()
    retriever.build_index(test_text, chunk_size=20, overlap=5)
    hits = retriever.search("What is the secret?")
    
    for h in hits:
        print(f"Score: {h['score']:.4f} | Content: {h['text']}")

if __name__ == "__main__":
    test_retriever()