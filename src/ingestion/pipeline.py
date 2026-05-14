import os
import json
import logging
from typing import List, Dict, Any
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import LegalChunker
from src.ingestion.extractor import EntityExtractor
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = LegalChunker()
        # Initialize extractor only if API key is present
        try:
            self.extractor = EntityExtractor()
        except Exception:
            logger.warning("OPENAI_API_KEY not found. Entity extraction will be skipped.")
            self.extractor = None

    def run(self, input_dir: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = [f for f in os.listdir(input_dir) if f.endswith(('.pdf', '.docx'))]
        logger.info(f"Found {len(files)} files to process.")

        for filename in files:
            file_path = os.path.join(input_dir, filename)
            try:
                # 1. Parse
                elements = self.parser.parse(file_path)
                full_text = "\n".join([el.get("text", "") for el in elements])
                
                # 2. Chunk
                chunks = self.chunker.chunk(elements, filename)
                
                # 3. Extract Entities
                entities = {}
                if self.extractor:
                    entities = self.extractor.extract(full_text)
                
                # 4. Save Results
                result = {
                    "filename": filename,
                    "entities": entities,
                    "chunks": chunks
                }
                
                output_filename = f"{os.path.splitext(filename)[0]}_processed.json"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Successfully processed and saved {filename} to {output_path}")

            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run(Config.CONTRACTS_DIR, Config.OUTPUT_DIR)
