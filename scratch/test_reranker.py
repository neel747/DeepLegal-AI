
from sentence_transformers import CrossEncoder
import sys

try:
    print("Loading CrossEncoder...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("Model loaded successfully.")
    
    query = "What is the liability cap?"
    passages = ["The liability cap is $1M.", "The weather is nice today."]
    
    print("Reranking...")
    scores = model.predict([(query, p) for p in passages])
    print(f"Scores: {scores}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
