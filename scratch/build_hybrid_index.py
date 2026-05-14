import os
import json
import logging
from src.retrieval.embedder import LegalEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_searcher import HybridSearcher
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_hybrid_index():
    embedder = LegalEmbedder()
    v_store = VectorStore()
    b_store = BM25Store()
    
    output_dir = Config.OUTPUT_DIR
    files = [f for f in os.listdir(output_dir) if f.endswith('_processed.json')]
    
    all_chunks = []
    all_metadata = []
    
    for filename in files:
        with open(os.path.join(output_dir, filename), 'r') as f:
            data = json.load(f)
            chunks = data.get("chunks", [])
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                meta = chunk["metadata"].copy()
                meta["content"] = chunk["content"]
                all_metadata.append(meta)
    
    if not all_chunks:
        logger.warning("No chunks found to index.")
        return

    # 1. Build Vector Index
    logger.info(f"Embedding {len(all_chunks)} chunks for Vector Index...")
    embeddings = embedder.embed_text(all_chunks)
    v_store.add_documents(embeddings, all_metadata)
    v_store.save()
    
    # 2. Build BM25 Index
    logger.info(f"Indexing {len(all_chunks)} chunks for BM25 Store...")
    b_store.add_documents(all_chunks, all_metadata)
    b_store.save()
    
    logger.info("Hybrid indices built successfully.")

def test_hybrid_search(query: str):
    embedder = LegalEmbedder()
    v_store = VectorStore()
    b_store = BM25Store()
    searcher = HybridSearcher(v_store, b_store, embedder)
    
    logger.info(f"\n--- Hybrid Search Test: '{query}' ---")
    results = searcher.search(query, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1} (Score: {res['hybrid_score']:.4f})")
        print(f"Source: {res['source']}, Section: {res['section']}")
        print(f"Content: {res['content'][:150]}...")

if __name__ == "__main__":
    build_hybrid_index()
    # Test queries
    test_hybrid_search("Section 1. Confidential Information") # Exact match
    test_hybrid_search("ending the agreement") # Semantic match
    test_hybrid_search("liability cap of $1,000,000") # Keyword + Semantic
