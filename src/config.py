import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CONTRACTS_DIR = os.getenv("CONTRACTS_DIR", "data/contracts")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env file")
