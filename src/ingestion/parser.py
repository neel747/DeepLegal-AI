import os
import logging
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Parses PDF and DOCX files using unstructured.
    """
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing document: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # partition() automatically handles file types
            elements = partition(filename=file_path)
            
            results = []
            for i, el in enumerate(elements):
                el_dict = el.to_dict()
                el_dict["element_index"] = i
                results.append(el_dict)
            
            logger.info(f"Successfully parsed {len(results)} elements from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise

if __name__ == "__main__":
    # Quick test
    parser = DocumentParser()
    sample_file = "data/contracts/sample_nda.docx"
    if os.path.exists(sample_file):
        elems = parser.parse(sample_file)
        print(f"Parsed {len(elems)} elements.")
        for e in elems[:5]:
            print(f"Type: {e.get('type')}, Text: {e.get('text')[:50]}...")
