from typing import List
import numpy as np
from openai import OpenAI
from src.config import Config

class LegalEmbedder:
    """
    Handles embedding generation using OpenAI's API.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model_name

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of strings using OpenAI.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean texts: replace newlines with spaces as recommended by OpenAI
        texts = [t.replace("\n", " ") for t in texts]
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings).astype('float32')

if __name__ == "__main__":
    # Quick test
    try:
        embedder = LegalEmbedder()
        test_texts = ["Section 1. Confidentiality", "Termination notice period is 30 days."]
        embeddings = embedder.embed_text(test_texts)
        print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    except Exception as e:
        print(f"Test failed: {e}")
