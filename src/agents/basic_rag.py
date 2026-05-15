import logging
from typing import List, Dict, Any
import google.generativeai as genai
from src.config import Config
from src.retrieval.pipeline import LegalRetriever

logger = logging.getLogger(__name__)

class LegalRAG:
    """
    Basic Single-Agent RAG using Gemini 1.5 Flash.
    Takes a LegalRetriever to get context and generates a cited legal analysis.
    """
    def __init__(self, retriever: LegalRetriever, model_name: str = "gemini-1.5-flash"):
        self.retriever = retriever
        
        # Configure Gemini
        if Config.GEMINI_API_KEY:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(model_name)
        else:
            logger.warning("GEMINI_API_KEY not found in config. RAG generation will use fallback mock.")
            self.model = None

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a structured context string for the prompt.
        """
        context_str = ""
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "Unknown Document")
            meta = chunk.get("metadata", {})
            section = meta.get("section", "Unknown Section")
            page = meta.get("page_number", "Unknown Page")
            
            context_str += f"[Document: {source} | Section: {section} | Page: {page}]\n"
            context_str += f"{chunk.get('content', '')}\n\n"
        return context_str

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Retrieves context and generates an answer with citations.
        """
        # 1. Retrieve context
        logger.info(f"Retrieving context for query: {query}")
        retrieved_chunks = self.retriever.retrieve(query, top_k=5)
        
        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant information in the provided contracts to answer your query.",
                "sources": []
            }

        # 2. Format Context
        context_text = self._format_context(retrieved_chunks)
        
        # 3. Construct Prompt
        system_instructions = """
        You are an expert legal assistant. Your task is to answer user queries based STRICTLY on the provided contract excerpts.
        
        RULES:
        1. CITE YOUR SOURCES: Every factual claim must be cited inline using the format: "According to [Section X] of [Document Name] (p.[Y]), ...". 
           Example: "According to Section 4.2 of MSA-001 (p.12), the liability cap is $1,000,000."
        2. BE PRECISE: Do not hallucinate or assume legal standards outside the provided text.
        3. ACKNOWLEDGE UNCERTAINTY: If the context does not fully answer the question, explicitly state what is missing.
        """
        
        prompt = f"{system_instructions}\n\nCONTEXT:\n{context_text}\n\nQUERY:\n{query}\n\nANSWER:"

        # 4. Generate Answer
        if self.model:
            try:
                response = self.model.generate_content(prompt)
                answer_text = response.text
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                answer_text = f"[API Error: Unable to generate response. Retrieved {len(retrieved_chunks)} chunks of context.]"
        else:
            # Fallback for testing when no API key is available
            answer_text = (
                f"[MOCK RESPONSE - API Key Missing]\n"
                f"Based on the retrieved context, I can answer your query.\n"
                f"According to {retrieved_chunks[0].get('metadata', {}).get('section', 'Unknown Section')} of {retrieved_chunks[0].get('source', 'Unknown Document')}, "
                f"here is the relevant information: {retrieved_chunks[0].get('content', '')[:100]}...\n"
                f"(Please add GEMINI_API_KEY to .env for real generations.)"
            )

        return {
            "answer": answer_text,
            "sources": retrieved_chunks
        }
