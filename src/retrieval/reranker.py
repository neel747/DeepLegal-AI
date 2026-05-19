from typing import List, Dict, Any
import logging
import google.generativeai as genai
from src.config import Config

logger = logging.getLogger(__name__)

class LegalReranker:
    """
    Reranks chunks using Gemini 1.5 Flash to ensure high precision.
    Local sentence-transformers cross-encoders are avoided due to Python 3.13 stability issues.
    """
    
    def __init__(self):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks based on their relevance to the query.
        """
        if not chunks:
            return []

        # Prepare the prompt for reranking
        chunk_texts = [f"[{i}] {c['content']}" for i, c in enumerate(chunks)]
        prompt = f"""
        Query: {query}
        
        Below are several snippets from legal documents. Rank them by their relevance to the query.
        Return ONLY a comma-separated list of indices in order of relevance, from most to least relevant.
        Example: 3, 0, 2, 1
        
        Snippets:
        {chr(10).join(chunk_texts)}
        """

        try:
            response = self.model.generate_content(prompt)
            order_str = response.text.strip()
            
            # Clean up the output if it's not just indices
            import re
            indices = [int(idx) for idx in re.findall(r'\d+', order_str)]
            
            # Filter and sort chunks
            reranked_chunks = []
            seen_indices = set()
            for idx in indices:
                if 0 <= idx < len(chunks) and idx not in seen_indices:
                    reranked_chunks.append(chunks[idx])
                    seen_indices.add(idx)
            
            # Add any missing chunks at the end (just in case)
            for i in range(len(chunks)):
                if i not in seen_indices:
                    reranked_chunks.append(chunks[i])

            return reranked_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original order
            return chunks[:top_k]

class LocalReranker:
    """
    Placeholder for Local Cross-Encoder Reranker.
    Note: Currently disabled due to segmentation faults on Python 3.13.
    """
    def __init__(self):
        logger.warning("LocalReranker is currently disabled. Using LegalReranker (Gemini) instead.")
        self.model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        return chunks[:top_k]
