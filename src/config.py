import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    CONTRACTS_DIR = os.getenv("CONTRACTS_DIR", "data/contracts")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    BM25_STORE_DIR = os.getenv("BM25_STORE_DIR", "data/bm25_store")
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env file")
        if not cls.NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD is not set in .env file")
