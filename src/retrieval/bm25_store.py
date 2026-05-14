import os
import pickle
import nltk
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.config import Config

class BM25Store:
    """
    Manages a BM25 index for sparse retrieval.
    """
    
    def __init__(self, index_name: str = "legal_bm25"):
        self.index_path = os.path.join(Config.BM25_STORE_DIR, f"{index_name}.pkl")
        self.bm25 = None
        self.metadata = []
        
        if not os.path.exists(Config.BM25_STORE_DIR):
            os.makedirs(Config.BM25_STORE_DIR)
            
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')

    def tokenize(self, text: str) -> List[str]:
        """
        Cleans and tokenizes text.
        """
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        # Filter non-alphanumeric and stopwords
        return [t for t in tokens if t.isalnum() and t not in stop_words]

    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]):
        """
        Tokenizes chunks and builds the BM25 index.
        """
        tokenized_corpus = [self.tokenize(doc) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.metadata = metadata

    def save(self):
        """
        Saves index and metadata to disk.
        """
        if self.bm25:
            data = {
                "bm25": self.bm25,
                "metadata": self.metadata
            }
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)

    def load(self) -> bool:
        """
        Loads index and metadata from disk.
        """
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.metadata = data["metadata"]
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs BM25 search and returns metadata for top-k results.
        """
        if self.bm25 is None:
            if not self.load():
                return []
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only return results with some match
                result = self.metadata[idx].copy()
                result["score"] = float(scores[idx])
                results.append(result)
        
        return results
