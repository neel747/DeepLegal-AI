import os
import json
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from src.config import Config

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generates a synthetic Q&A dataset (Golden Dataset) for RAG evaluation.
    """
    
    def __init__(self):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_qa_pairs(self, json_data: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generates Q&A pairs from a single processed contract JSON.
        """
        filename = json_data.get("filename", "Unknown")
        chunks = json_data.get("chunks", [])
        entities = json_data.get("entities", {})
        
        # Combine some chunks to provide context for generation
        # We'll take the first few chunks and any with high information density
        context_text = "\n".join([c["content"] for c in chunks[:10]])
        
        prompt = f"""
        You are a legal expert. Based on the following contract excerpts, generate {num_questions} diverse question-answer pairs.
        
        Guidelines:
        1. Each question must be factual and answerable from the text.
        2. Provide a "ground_truth" answer that is precise.
        3. Include a "source" citation field (e.g., "Section 4.2 of {filename}").
        4. Categorize each question as: "Single-doc factual", "Multi-hop reasoning", or "Temporal".
        
        Return ONLY a JSON list of objects:
        [
          {{"question": "...", "ground_truth": "...", "source": "...", "category": "..."}}
        ]
        
        Contract Context:
        {context_text}
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # The model might return a wrapper object like {"questions": [...]}
            raw_output = json.loads(response.text)
            if isinstance(raw_output, dict):
                # Try to find the list inside
                for key, val in raw_output.items():
                    if isinstance(val, list):
                        return val
            return raw_output if isinstance(raw_output, list) else []
            
        except Exception as e:
            logger.error(f"Failed to generate QA pairs for {filename}: {e}")
            return []

    def process_all_files(self, output_dir: str, save_path: str, max_per_file: int = 5):
        """
        Iterates through all processed JSONs and builds the full dataset.
        """
        files = [f for f in os.listdir(output_dir) if f.endswith('_processed.json')]
        all_qa_pairs = []
        
        for filename in files:
            logger.info(f"Generating questions for {filename}...")
            with open(os.path.join(output_dir, filename), 'r') as f:
                data = json.load(f)
            
            pairs = self.generate_qa_pairs(data, num_questions=max_per_file)
            all_qa_pairs.extend(pairs)
            
        # Save the golden dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_qa_pairs, f, indent=2)
            
        logger.info(f"Generated {len(all_qa_pairs)} QA pairs and saved to {save_path}")
        return all_qa_pairs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = DatasetGenerator()
    dataset_path = os.path.join("data", "evaluation", "golden_dataset.json")
    generator.process_all_files(Config.OUTPUT_DIR, dataset_path)
