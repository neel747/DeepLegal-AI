import json
import logging
from typing import Dict, Any, List
import google.generativeai as genai
from src.config import Config

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts structured entities from legal documents using Gemini 1.5 Flash.
    """
    
    def __init__(self):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities from the full document text.
        """
        prompt = f"""
        You are a legal expert assistant. Extract the following entities from the provided contract text and return them in a valid JSON format.
        
        Entities to extract:
        1. Parties: Identify the Client and Vendor (or similar roles like Disclosing/Receiving Party).
        2. Contract Type: e.g., MSA, SOW, NDA, Amendment.
        3. Key Dates: Effective date, expiration date, renewal date.
        4. Monetary Values: Liability caps, payment terms, total value.
        5. Key Obligations: Top 3-5 critical obligations.
        6. Termination Clauses: Summary of how the contract can be terminated.

        Return ONLY the JSON object. No extra text.
        
        Contract Text:
        {text}
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            extracted_data = json.loads(response.text)
            logger.info("Successfully extracted entities from document.")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error during entity extraction: {str(e)}")
            return {}

if __name__ == "__main__":
    try:
        extractor = EntityExtractor()
        test_text = "This NDA is between ACME Corp and Globex. Effective May 1, 2024. Total liability is capped at $500k."
        entities = extractor.extract(test_text)
        print(json.dumps(entities, indent=2))
    except Exception as e:
        print(f"Skipping live test: {e}")
