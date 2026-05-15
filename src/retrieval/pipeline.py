from typing import List, Dict, Any
import logging
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import LegalReranker
from src.ingestion.graph_queries import GraphQueries

logger = logging.getLogger(__name__)

class LegalRetriever:
    """
    Three-stage retrieval pipeline:
    1. Hybrid Search (FAISS + BM25)
    2. Graph Expansion (Neo4j)
    3. Cross-Encoder Reranking (GPT-4o-mini)
    """
    
    def __init__(self, hybrid_searcher: HybridSearcher, graph_queries: GraphQueries, reranker: LegalReranker):
        self.hybrid_searcher = hybrid_searcher
        self.graph_queries = graph_queries
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes the full three-stage retrieval pipeline.
        """
        logger.info(f"Retrieving for query: {query}")
        
        # Stage 1: Hybrid Search (Retrieve more than top_k to allow for expansion and reranking)
        initial_chunks = self.hybrid_searcher.search(query, top_k=top_k * 4)
        logger.info(f"Stage 1: Found {len(initial_chunks)} chunks via Hybrid Search.")
        
        # Stage 2: Graph Context Expansion
        # Extract unique source filenames
        filenames = list(set([c["source"] for c in initial_chunks]))
        expanded_context = self.graph_queries.expand_context(filenames)
        
        if expanded_context:
            logger.info(f"Stage 2: Expanded context with {len(expanded_context)} related clauses from Neo4j.")
            # Merge and avoid duplicates
            seen_content = set([c["content"] for c in initial_chunks])
            for ctx in expanded_context:
                if ctx["content"] not in seen_content:
                    # Format expanded context to match chunk structure
                    initial_chunks.append({
                        "content": ctx["content"],
                        "source": ctx["source"],
                        "metadata": {
                            "section": ctx.get("section"),
                            "page_number": ctx.get("page"),
                            "expanded": True
                        }
                    })
                    seen_content.add(ctx["content"])
        else:
            logger.info("Stage 2: No related context found in Graph.")

        # Stage 3: Reranking
        logger.info(f"Stage 3: Reranking {len(initial_chunks)} chunks...")
        final_chunks = self.reranker.rerank(query, initial_chunks, top_k=top_k)
        
        return final_chunks

if __name__ == "__main__":
    # Example setup (requires initialized stores and graph)
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_store import BM25Store
    from src.retrieval.embedder import LegalEmbedder
    from src.config import Config
    
    # This is just a skeletal main for local testing if needed
    try:
        vs = VectorStore(Config.VECTOR_STORE_DIR)
        bs = BM25Store(Config.BM25_STORE_DIR)
        emb = LegalEmbedder()
        hs = HybridSearcher(vs, bs, emb)
        gq = GraphQueries()
        rr = LegalReranker()
        
        retriever = LegalRetriever(hs, gq, rr)
        results = retriever.retrieve("What is the liability cap in the MSA?")
        
        for i, res in enumerate(results):
            print(f"\n[{i+1}] Source: {res['source']}")
            print(f"Content: {res['content'][:200]}...")
    except Exception as e:
        print(f"Skipping main execution: {e}")
