import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LegalChunker:
    """
    Chunks legal documents with awareness of sections and clauses.
    """
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Patterns for legal sections
        self.section_patterns = [
            r'^(Section|Article|Clause|Paragraph|Item)\s+([0-9A-Z\.]+)',
            r'^[0-9]+\.\s+[A-Z][a-z]+' # e.g. "1. Introduction"
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.section_patterns]

    def is_section_header(self, element: Dict[str, Any]) -> bool:
        text = element.get("text", "").strip()
        el_type = element.get("type", "")
        
        if el_type in ["Title", "Heading"]:
            return True
            
        for pattern in self.compiled_patterns:
            if pattern.match(text):
                return True
        return False

    def chunk(self, elements: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk_text = ""
        current_section = "Preamble"
        
        for el in elements:
            text = el.get("text", "").strip()
            if not text:
                continue
                
            metadata = el.get("metadata", {})
            page_number = metadata.get("page_number", 1)
            
            if self.is_section_header(el):
                # New section detected. If we have text, flush it as a chunk.
                if current_chunk_text:
                    chunks.append(self._create_chunk(current_chunk_text, source_name, current_section, page_number))
                    current_chunk_text = ""
                
                # Update current section name
                current_section = text
                # We also include the header in the new chunk
                current_chunk_text = text
            else:
                # Normal text. Check if adding it exceeds chunk size.
                if len(current_chunk_text) + len(text) > self.chunk_size and current_chunk_text:
                    chunks.append(self._create_chunk(current_chunk_text, source_name, current_section, page_number))
                    # Simplified overlap: just start fresh for now
                    # (In a more advanced version, we'd take the last N words)
                    current_chunk_text = text
                else:
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + text
                    else:
                        current_chunk_text = text
                        
        # Final flush
        if current_chunk_text:
            chunks.append(self._create_chunk(current_chunk_text, source_name, current_section, 1))
            
        return chunks

    def _create_chunk(self, text: str, source: str, section: str, page: int) -> Dict[str, Any]:
        return {
            "content": text,
            "metadata": {
                "source": source,
                "section": section,
                "page_number": page,
                "doc_type": "legal_contract" # Can be refined later
            }
        }

if __name__ == "__main__":
    # Test chunking
    from src.ingestion.parser import DocumentParser
    parser = DocumentParser()
    chunker = LegalChunker(chunk_size=500) # Small chunk size for testing
    
    file_path = "data/contracts/sample_nda.docx"
    elements = parser.parse(file_path)
    chunks = chunker.chunk(elements, "sample_nda.docx")
    
    print(f"Generated {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({chunk['metadata']['section']}) ---")
        print(chunk['content'][:200] + "...")
