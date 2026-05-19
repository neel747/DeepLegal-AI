import os
import shutil
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal imports
from src.config import Config
from src.retrieval.pipeline import LegalRetriever
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import LegalEmbedder
from src.ingestion.graph_queries import GraphQueries
from src.retrieval.reranker import LegalReranker
from src.agents.multi_agent import MultiAgentRAG

# For demonstration, we'll import the ingestion pipeline logic here
# In a real app, this would be a more complex orchestration
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import LegalChunker
from src.ingestion.extractor import EntityExtractor
from src.ingestion.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepLegal AI API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State (Initialized on startup)
class State:
    retriever: LegalRetriever = None
    agent_system: MultiAgentRAG = None
    app_graph = None

state = State()

@app.on_event("startup")
def startup_event():
    """Initializes the RAG system on startup."""
    try:
        # 1. Setup Retrieval Components
        embedder = LegalEmbedder()
        vector_store = VectorStore()
        bm25_store = BM25Store()
        
        # Load indices
        vector_store.load()
        bm25_store.load()
        
        searcher = HybridSearcher(vector_store, bm25_store, embedder)
        graph_queries = GraphQueries()
        reranker = LegalReranker()
        
        state.retriever = LegalRetriever(searcher, graph_queries, reranker)
        state.agent_system = MultiAgentRAG(state.retriever)
        state.app_graph = state.agent_system.compile_graph()
        
        logger.info("DeepLegal AI System Initialized Successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        # Note: In a production app, we'd handle this more gracefully

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retry_count: int

@app.post("/query", response_model=QueryResponse)
async def query_legal_agent(request: QueryRequest):
    """Processes a legal query through the Multi-Agent RAG pipeline."""
    if not state.app_graph:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        initial_state = {
            "query": request.query,
            "sub_queries": [],
            "retrieved_chunks": [],
            "draft_answer": None,
            "final_answer": None,
            "validation_feedback": None,
            "retry_count": 0
        }
        
        # Execute the graph
        result = state.app_graph.invoke(initial_state)
        
        return QueryResponse(
            answer=result.get("final_answer", "No answer generated."),
            sources=result.get("retrieved_chunks", []),
            retry_count=result.get("retry_count", 0)
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a document and starts the ingestion pipeline in the background."""
    os.makedirs(Config.CONTRACTS_DIR, exist_ok=True)
    file_path = os.path.join(Config.CONTRACTS_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger background ingestion
    background_tasks.add_task(run_ingestion_pipeline, file_path)
    
    return {"message": f"Successfully uploaded {file.filename}. Ingestion started in background."}

def run_ingestion_pipeline(file_path: str):
    """Placeholder for the full ingestion pipeline logic."""
    logger.info(f"Starting background ingestion for: {file_path}")
    # In a real app, you'd call the classes from Phase 1-4 here
    # 1. Parse -> 2. Chunk -> 3. Extract -> 4. Graph Build -> 5. Index
    pass

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
