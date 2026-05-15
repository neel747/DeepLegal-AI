
from sentence_transformers import SentenceTransformer
import sys

try:
    print("Loading SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
    
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
