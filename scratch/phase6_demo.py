import logging
from src.agents.basic_rag import LegalRAG
from scratch.phase5_demo import MockHybridSearcher, MockGraphQueries, MockReranker
from src.retrieval.pipeline import LegalRetriever

def run_phase6_demo():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize mocks from Phase 5 to simulate retrieval
    hs = MockHybridSearcher()
    gq = MockGraphQueries()
    rr = MockReranker()
    
    # Initialize retriever
    retriever = LegalRetriever(hs, gq, rr)
    
    # Initialize RAG agent
    rag_agent = LegalRAG(retriever=retriever)
    
    queries = [
        "What is the liability cap in the MSA that governs SOW-003?",
        "When is the project completion date according to the timeline?",
        "Tell me about the termination clause."
    ]
    
    print("\n" + "="*50)
    print("PHASE 6: SINGLE-AGENT RAG DEMO")
    print("="*50)
    
    for query in queries:
        print(f"\n[QUERY]: {query}")
        print("-" * 50)
        
        result = rag_agent.answer_query(query)
        
        print("[ANSWER]:")
        print(result["answer"])
        print("-" * 50)
        print(f"Sources cited: {len(result['sources'])} chunks")
        print("="*50)

if __name__ == "__main__":
    run_phase6_demo()
