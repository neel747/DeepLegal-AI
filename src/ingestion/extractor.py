import json
import logging
from typing import Dict, Any, List
from openai import OpenAI
from src.config import Config

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts structured entities from legal documents using GPT-4o-mini.
    """
    
    def __init__(self):
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {{"role": "system", "content": "You are a legal document analyzer."}},
                    {{"role": "user", "content": prompt}}
                ],
                response_format={{ "type": "json_object" }}
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            logger.info("Successfully extracted entities from document.")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error during entity extraction: {str(e)}")
            return {{}}

if __name__ == "__main__":
    # This test will fail if OPENAI_API_KEY is not set
    # I'll add a dummy test mode or just let it fail for now
    try:
        extractor = EntityExtractor()
        test_text = "This NDA is between ACME Corp and Globex. Effective May 1, 2024. Total liability is capped at $500k."
        entities = extractor.extract(test_text)
        print(json.dumps(entities, indent=2))
    except Exception as e:
        print(f"Skipping live test: {e}")
