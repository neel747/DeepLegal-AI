import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
from src.config import Config

class VectorStore:
    """
    Manages FAISS index and associated metadata for dense retrieval.
    """
    
    def __init__(self, index_name: str = "legal_index"):
        self.index_path = os.path.join(Config.VECTOR_STORE_DIR, f"{index_name}.faiss")
        self.metadata_path = os.path.join(Config.VECTOR_STORE_DIR, f"{index_name}.pkl")
        self.index = None
        self.metadata = [] # List of metadata dicts corresponding to index rows

        if not os.path.exists(Config.VECTOR_STORE_DIR):
            os.makedirs(Config.VECTOR_STORE_DIR)

    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Adds embeddings and their metadata to the store.
        """
        dim = embeddings.shape[1]
        if self.index is None:
            # Initialize IndexFlatL2 for simplicity (exact search)
            self.index = faiss.IndexFlatL2(dim)
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def save(self):
        """
        Saves index and metadata to disk.
        """
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def load(self):
        """
        Loads index and metadata from disk.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs similarity search and returns metadata for top-k results.
        """
        if self.index is None:
            if not self.load():
                return []
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                result = self.metadata[idx].copy()
                result["score"] = float(distances[0][i])
                results.append(result)
        
        return results
