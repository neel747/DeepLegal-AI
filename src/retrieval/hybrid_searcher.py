from typing import List, Dict, Any
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import LegalEmbedder

class HybridSearcher:
    """
    Combines Dense (Vector) and Sparse (BM25) search using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store, embedder: LegalEmbedder):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Performs hybrid search using RRF.
        """
        # 1. Get Dense results
        query_emb = self.embedder.embed_text([query])
        dense_results = self.vector_store.search(query_emb, top_k=top_k * 2)
        
        # 2. Get Sparse results
        sparse_results = self.bm25_store.search(query, top_k=top_k * 2)
        
        # 3. Apply RRF
        # We use 'source' + 'content' as a unique key for chunks
        scores = {}
        metadata_map = {}
        
        # Helper to process results
        def process_results(results: List[Dict[str, Any]], weight: float = 1.0):
            for rank, res in enumerate(results):
                key = f"{res['source']}_{res['content']}"
                if key not in scores:
                    scores[key] = 0
                    metadata_map[key] = res
                
                scores[key] += weight * (1.0 / (rrf_k + rank + 1))

        process_results(dense_results)
        process_results(sparse_results)
        
        # 4. Sort and return top_k
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        final_results = []
        for key in sorted_keys[:top_k]:
            res = metadata_map[key].copy()
            res["hybrid_score"] = scores[key]
            final_results.append(res)
            
        return final_results
