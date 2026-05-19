from typing import List
import numpy as np
import google.generativeai as genai
from src.config import Config

class LegalEmbedder:
    """
    Handles embedding generation using Gemini's API.
    """
    
    def __init__(self, model_name: str = "models/gemini-embedding-2"):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = model_name

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of strings using Gemini.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean texts: replace newlines with spaces
        texts = [t.replace("\n", " ") for t in texts]
        
        response = genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_document"
        )
        
        embeddings = response['embedding']
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
