import logging
import json
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from src.config import Config
from src.retrieval.pipeline import LegalRetriever

logger = logging.getLogger(__name__)

# --- State Schema ---
class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    retrieved_chunks: Annotated[List[Dict[str, Any]], operator.add]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    validation_feedback: Optional[str]
    retry_count: int

class MultiAgentRAG:
    """
    LangGraph-based multi-agent system for self-correcting legal retrieval.
    """
    def __init__(self, retriever: LegalRetriever, model_name: str = "gemini-1.5-flash"):
        self.retriever = retriever
        if Config.GEMINI_API_KEY:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(model_name)
        else:
            logger.warning("GEMINI_API_KEY missing. Agents will use fallback mocks.")
            self.model = None

    # --- Nodes ---
    def planner_node(self, state: AgentState) -> dict:
        """Decomposes the main query into sub-queries if necessary."""
        query = state["query"]
        logger.info(f"[Planner] Analyzing query: {query}")
        
        if not self.model:
            # Fallback
            return {"sub_queries": [query]}

        prompt = f"""
        You are a legal search planner. If the following query requires multiple pieces of distinct information, break it down into a JSON list of sub-queries. 
        If it is a single simple question, just return the original query in the list.
        Return ONLY a valid JSON list of strings.
        
        Query: {query}
        """
        try:
            response = self.model.generate_content(prompt)
            # Clean up potential markdown formatting
            text = response.text.replace("```json", "").replace("```", "").strip()
            sub_queries = json.loads(text)
            if not isinstance(sub_queries, list):
                sub_queries = [query]
        except Exception as e:
            logger.error(f"[Planner] Failed to parse sub-queries: {e}")
            sub_queries = [query]
            
        logger.info(f"[Planner] Generated sub-queries: {sub_queries}")
        return {"sub_queries": sub_queries}

    def retriever_node(self, state: AgentState) -> dict:
        """Executes retrieval for all pending sub-queries."""
        queries_to_run = state.get("sub_queries", [])
        if not queries_to_run:
            queries_to_run = [state["query"]]
            
        logger.info(f"[Retriever] Fetching context for {len(queries_to_run)} queries.")
        
        all_chunks = []
        for q in queries_to_run:
            chunks = self.retriever.retrieve(q, top_k=3)
            all_chunks.extend(chunks)
            
        return {"retrieved_chunks": all_chunks}

    def analyzer_node(self, state: AgentState) -> dict:
        """Synthesizes context into a drafted answer."""
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        logger.info(f"[Analyzer] Drafting answer using {len(chunks)} chunks.")
        
        if not chunks:
            return {"draft_answer": "No relevant context found."}
            
        if not self.model:
            return {"draft_answer": "[MOCK] Draft answer based on context."}

        # Format context
        context_str = ""
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            sec = chunk.get("metadata", {}).get("section", "Unknown")
            context_str += f"[Source: {source} | Section: {sec}]\n{chunk.get('content', '')}\n\n"

        prompt = f"""
        You are a legal assistant. Answer the query based ONLY on the context.
        Cite sources inline (e.g., "According to Section X of Y...").
        If the context lacks information, state what is missing.
        
        Context:
        {context_str}
        
        Query: {query}
        """
        try:
            response = self.model.generate_content(prompt)
            draft = response.text
        except Exception as e:
            logger.error(f"[Analyzer] Generation failed: {e}")
            draft = "[Error generating draft]"
            
        return {"draft_answer": draft}

    def validator_node(self, state: AgentState) -> dict:
        """Evaluates the draft and decides if self-correction is needed."""
        draft = state.get("draft_answer", "")
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        
        logger.info(f"[Validator] Checking draft (Attempt {retry_count + 1}).")
        
        if not self.model:
            # Fallback for demo without API
            if retry_count == 0:
                logger.warning(f"[Validator] [MOCK] Draft rejected: FAIL: Missing section details.")
                return {"validation_feedback": "FAIL: Missing section details.", "retry_count": retry_count + 1}
            logger.info("[Validator] [MOCK] Draft approved.")
            return {"final_answer": draft, "validation_feedback": "PASS"}

        prompt = f"""
        You are a validator. Review the following draft answer to the query.
        Does the draft explicitly state that it is missing information? 
        If it is missing info, reply with 'FAIL: [reason]'.
        If the answer is complete and cited, reply with 'PASS'.
        
        Query: {query}
        Draft: {draft}
        """
        try:
            response = self.model.generate_content(prompt)
            evaluation = response.text.strip().upper()
        except Exception:
            evaluation = "PASS"

        if evaluation.startswith("FAIL"):
            logger.warning(f"[Validator] Draft rejected: {evaluation}")
            # If rejected, we ask the planner to try a different query next time
            # For simplicity, we just pass the feedback back
            return {
                "validation_feedback": evaluation,
                "retry_count": retry_count + 1
            }
        else:
            logger.info("[Validator] Draft approved.")
            return {"final_answer": draft, "validation_feedback": "PASS"}

    # --- Routing ---
    def should_continue(self, state: AgentState) -> str:
        """Determines whether to end or loop back to retrieval."""
        feedback = state.get("validation_feedback", "")
        retries = state.get("retry_count", 0)
        
        if feedback == "PASS" or retries >= 3:
            return "end"
        return "retry"

    # --- Graph Compilation ---
    def compile_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("analyzer", self.analyzer_node)
        workflow.add_node("validator", self.validator_node)
        
        # Set entry
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "analyzer")
        workflow.add_edge("analyzer", "validator")
        
        # Conditional logic
        workflow.add_conditional_edges(
            "validator",
            self.should_continue,
            {
                "end": END,
                "retry": "planner" # Alternatively, could go straight to retriever if we update sub_queries in validator
            }
        )
        
        return workflow.compile()
