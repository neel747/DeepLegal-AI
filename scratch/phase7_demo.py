import logging
from scratch.phase5_demo import MockHybridSearcher, MockGraphQueries, MockReranker
from src.retrieval.pipeline import LegalRetriever
from src.agents.multi_agent import MultiAgentRAG

def run_phase7_demo():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize mocks from Phase 5
    hs = MockHybridSearcher()
    gq = MockGraphQueries()
    rr = MockReranker()
    
    # Initialize retriever
    retriever = LegalRetriever(hs, gq, rr)
    
    # Initialize MultiAgentRAG
    agent_system = MultiAgentRAG(retriever=retriever)
    app = agent_system.compile_graph()
    
    query = "What is the liability cap in the MSA that governs SOW-003?"
    
    print("\n" + "="*50)
    print("PHASE 7: MULTI-AGENT LANGGRAPH DEMO")
    print("="*50)
    
    initial_state = {
        "query": query,
        "sub_queries": [],
        "retrieved_chunks": [],
        "draft_answer": None,
        "final_answer": None,
        "validation_feedback": None,
        "retry_count": 0
    }
    
    # Run the graph
    print(f"\n[USER QUERY]: {query}\n")
    
    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"--- Completed Node: [{node_name.upper()}] ---")
            if "sub_queries" in state_update:
                print(f"  > Sub Queries: {state_update['sub_queries']}")
            if "draft_answer" in state_update:
                print(f"  > Draft Answer: {str(state_update['draft_answer'])[:100]}...")
            if "validation_feedback" in state_update:
                print(f"  > Feedback: {state_update['validation_feedback']}")
            if "final_answer" in state_update:
                print(f"\n[FINAL APPROVED ANSWER]:\n{state_update['final_answer']}")
            print()
            
    print("="*50)

if __name__ == "__main__":
    run_phase7_demo()
