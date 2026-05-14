import os
import json
import logging
from src.retrieval.embedder import LegalEmbedder
from src.retrieval.vector_store import VectorStore
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_index():
    embedder = LegalEmbedder()
    store = VectorStore()
    
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
                # Include content in metadata for easy verification in search results
                meta = chunk["metadata"].copy()
                meta["content"] = chunk["content"]
                all_metadata.append(meta)
    
    if not all_chunks:
        logger.warning("No chunks found to index.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.embed_text(all_chunks)
    
    logger.info("Adding to vector store...")
    store.add_documents(embeddings, all_metadata)
    store.save()
    logger.info("Vector index built successfully.")

def test_search(query: str):
    embedder = LegalEmbedder()
    store = VectorStore()
    
    logger.info(f"Searching for: '{query}'")
    query_emb = embedder.embed_text([query])
    results = store.search(query_emb, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {res['score']:.4f}) ---")
        print(f"Source: {res['source']}, Section: {res['section']}")
        print(f"Content: {res['content'][:200]}...")

if __name__ == "__main__":
    build_index()
    test_search("What is the confidentiality period?")
    test_search("How much is the liability cap?")
