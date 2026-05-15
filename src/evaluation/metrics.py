import re
from typing import List, Dict, Any

def calculate_citation_accuracy(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Checks if citations in the answer (e.g., 'Section 4.2 of MSA') correspond 
    to the metadata in the retrieved chunks.
    """
    # Regex to find citations like "Section 3 of sample_msa.docx"
    # Or "According to Section 3 of sample_msa.docx"
    citation_pattern = r"(?:Section|Article|Clause)\s+([\d\.]+)\s+of\s+([^\s\.\)]+)"
    citations = re.findall(citation_pattern, answer)
    
    if not citations:
        return 0.0
    
    valid_citations = 0
    
    # Extract valid section/source pairs from retrieved chunks
    available_sources = []
    for chunk in retrieved_chunks:
        source = chunk.get("source", "").lower()
        section = str(chunk.get("metadata", {}).get("section", "")).lower()
        available_sources.append((section, source))
    
    for section_cite, source_cite in citations:
        section_cite = section_cite.lower()
        source_cite = source_cite.lower()
        
        # Check if this cited pair exists in our retrieved context
        match_found = False
        for sec, src in available_sources:
            if section_cite in sec and source_cite in src:
                match_found = True
                break
        
        if match_found:
            valid_citations += 1
            
    return valid_citations / len(citations)

def evaluate_cross_doc_consistency(answer: str, category: str, retrieved_chunks: List[Dict[str, Any]]) -> bool:
    """
    For comparative queries, check if chunks from more than one document are present.
    """
    if "comparison" not in category.lower() and "cross-doc" not in category.lower():
        return True
        
    sources = set([c.get("source") for c in retrieved_chunks])
    return len(sources) > 1
