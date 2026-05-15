import logging
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Mocking the components for demonstration
class MockHybridSearcher:
    def search(self, query: str, top_k: int = 5):
        return [
            {"source": "sample_sow.docx", "content": "This SOW is governed by the Master Services Agreement.", "metadata": {"section": "Introduction"}},
            {"source": "sample_sow.docx", "content": "Project completion by December 31, 2024.", "metadata": {"section": "Timeline"}}
        ]

class MockGraphQueries:
    def expand_context(self, filenames: List[str]):
        if "sample_sow.docx" in filenames:
            return [
                {"source": "sample_msa.docx", "section": "Section 3", "content": "The total liability shall not exceed $1,000,000.", "page": 5}
            ]
        return []

class MockReranker:
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5):
        # In a real scenario, this would use a model to score relevance
        # Here we just sort by whether "liability" is in the content for demonstration
        sorted_chunks = sorted(chunks, key=lambda x: "liability" in x["content"].lower(), reverse=True)
        return sorted_chunks[:top_k]

# Import the real pipeline logic but use mocks
from src.retrieval.pipeline import LegalRetriever

def run_phase5_demo():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize mocks
    hs = MockHybridSearcher()
    gq = MockGraphQueries()
    rr = MockReranker()
    
    # Initialize real retriever with mocks
    retriever = LegalRetriever(hs, gq, rr)
    
    query = "What is the liability cap in the MSA that governs SOW-003?"
    print(f"\n--- Testing Phase 5 Pipeline ---")
    print(f"Query: {query}")
    
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\n--- Final Retrieved Results (Top 3) ---")
    for i, res in enumerate(results):
        is_expanded = res.get("metadata", {}).get("expanded", False)
        tag = "[GRAPH EXPANDED]" if is_expanded else "[HYBRID]"
        print(f"\nResult {i+1} {tag}")
        print(f"Source: {res['source']}")
        print(f"Section: {res.get('metadata', {}).get('section', 'N/A')}")
        print(f"Content: {res['content']}")

if __name__ == "__main__":
    run_phase5_demo()
