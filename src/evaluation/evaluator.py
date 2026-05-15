import json
import os
import logging
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.agents.multi_agent import MultiAgentRAG
from src.retrieval.pipeline import LegalRetriever
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import LegalEmbedder
from src.ingestion.graph_queries import GraphQueries
from src.retrieval.reranker import LegalReranker
from src.config import Config

logger = logging.getLogger(__name__)

class LegalEvaluator:
    """
    Evaluates the Legal RAG pipeline using Ragas metrics.
    """
    
    def __init__(self, agent_system: MultiAgentRAG):
        self.agent_system = agent_system
        self.app = agent_system.compile_graph()

    def run_evaluation(self, dataset_path: str):
        """
        Runs the full evaluation on the provided dataset.
        """
        with open(dataset_path, 'r') as f:
            golden_data = json.load(f)
            
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for item in golden_data:
            query = item["question"]
            logger.info(f"Evaluating: {query}")
            
            # Run the multi-agent graph
            inputs = {
                "query": query,
                "sub_queries": [],
                "retrieved_chunks": [],
                "draft_answer": None,
                "final_answer": None,
                "validation_feedback": None,
                "retry_count": 0
            }
            
            # Capture the result from the graph stream or invoke
            result_state = self.app.invoke(inputs)
            
            questions.append(query)
            answers.append(result_state.get("final_answer", ""))
            
            # Format contexts as list of strings
            chunk_texts = [c["content"] for c in result_state.get("retrieved_chunks", [])]
            contexts.append(chunk_texts)
            
            ground_truths.append(item["ground_truth"])
            
        # Create Dataset
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data_dict)
        
        # Run Ragas
        # Note: Ragas metrics might require an LLM (defaults to OpenAI)
        # If API key is missing, this might fail unless we mock the metrics or provide a custom LLM
        try:
            logger.info("Running Ragas metrics computation...")
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
            )
            return result
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            # Return a mock result if API fails
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Setup the system (using mocks if necessary)
    # For the real implementation, we'd use the actual retriever components
    # Here we'll just show the structure
    print("Evaluator structure ready. Run with actual system components for full metrics.")
