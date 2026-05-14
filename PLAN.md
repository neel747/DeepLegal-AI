# DeepLegal AI — Master Implementation Plan

> **Goal:** Build a multi-agent contract intelligence system with GraphRAG, hybrid retrieval, and self-correcting reasoning — one phase at a time.

> **Rule:** No phase starts until the previous one is understood AND working.

---

## Phase 0: Project Setup & Foundation
**What you learn:** Project structure, virtual env, dependency management
- [x] Create project folder structure
- [x] Set up Python venv + requirements.txt
- [x] Create `.env` for API keys (Google Gemini API - Free Tier, Neo4j)
- [x] Create a basic `main.py` that loads env vars and prints "DeepLegal AI ready"
- [x] Set up Git repo with `.gitignore`

**Key files created:**
```
DeepLegal-AI/
├── src/
│   ├── __init__.py
│   ├── config.py          # All config & env loading
│   ├── ingestion/         # Phase 1-2
│   ├── retrieval/         # Phase 3-5
│   ├── agents/            # Phase 6-7
│   ├── evaluation/        # Phase 8
│   └── api/               # Phase 9
├── data/
│   └── contracts/         # Sample legal docs
├── tests/
├── requirements.txt
├── .env
├── .gitignore
├── PLAN.md
├── CHANGELOG.md
└── README.md
```

---

## Phase 1: Document Ingestion & Legal-Aware Chunking
**What you learn:** How to parse PDFs/DOCX, why naive chunking fails for legal text, legal-aware splitting strategies

### 1A: Basic PDF/DOCX Parsing
- [x] Install `unstructured`, `python-docx`, `pymupdf`
- [x] Write a parser that extracts raw text from PDF and DOCX files
- [x] Test with 2-3 sample contracts
- [x] Understand: What gets lost? (headers, tables, formatting)

### 1B: Legal-Aware Chunking
- [x] Understand why 500-char naive chunks break legal context
- [x] Implement section-aware chunking (split by clause/section headers like "Section 4.2")
- [x] Preserve metadata per chunk: source file, section number, page number, doc type
- [x] Test: Print chunks and verify they are semantically complete

### 1C: Structured Entity Extraction
- [x] Use GPT-4o-mini to extract key entities from each contract:
  - Parties (Client, Vendor)
  - Contract type (MSA, SOW, NDA, Amendment)
  - Key dates (effective, expiration, renewal)
  - Monetary values (liability caps, payment terms)
  - Key obligations and termination clauses
- [x] Store extracted entities as structured JSON alongside chunks

**Deliverable:** A pipeline that takes a folder of contracts → outputs clean chunks + extracted entities as JSON files.

---

## Phase 2: Knowledge Graph Construction (Neo4j)
**What you learn:** What a knowledge graph is, why it matters for legal docs, Neo4j basics, entity-relationship modeling

### 2A: Neo4j Setup & Schema Design
- [x] Install Neo4j Desktop (or use Docker `neo4j` image)
- [x] Learn Cypher basics (CREATE, MATCH, MERGE, relationships)
- [x] Design the graph schema:
  - Nodes: `Contract`, `Clause`, `Party`, `Obligation`, `Date`, `Amount`
  - Relationships: `HAS_CLAUSE`, `REFERENCES`, `AMENDS`, `BETWEEN_PARTIES`, `GOVERNED_BY`

### 2B: Graph Population Pipeline
- [x] Write code Python, LangChain, LangGraph, Gemini 1.5 Flash (Free Tier), 
- [x] Write code to create relationships (MSA → SOW → Amendment chains)
- [x] Test: Visualize the graph in Neo4j Browser — can you "see" contract relationships?

### 2C: Graph Querying
- [x] Write Cypher queries for common patterns:
  - "Find all SOWs under MSA-001"
  - "Find all contracts with Party X"
  - "Find all amendments to Contract Y"
- [x] Wrap these in Python helper functions

**Deliverable:** A populated Neo4j graph you can visually explore, with Python query functions.

---

## Phase 3: Dense Vector Retrieval (FAISS + Voyage AI)
**What you learn:** Embeddings, vector similarity, FAISS indexing, why dense retrieval alone isn't enough

### 3A: Embedding Generation
- [x] Install `sentence-transformers` (Local) or `google-generativeai`
- [x] Generate embeddings for all chunks using a free model (e.g., `all-MiniLM-L6-v2` or Gemini)
- [x] Understand: What does an embedding represent? Why cosine similarity works?

### 3B: FAISS Index Construction
- [x] Build a FAISS index from chunk embeddings
- [x] Implement `dense_search(query, top_k)` function
- [x] Test with 5 sample queries — examine retrieved chunks manually
- [x] Understand: When does dense retrieval fail? (exact legal terms, section numbers, rare entities)

**Deliverable:** A working `dense_search()` that returns top-k chunks for any query.

---

## Phase 4: Sparse Retrieval (BM25) + Hybrid Fusion
**What you learn:** TF-IDF / BM25, why keyword search matters for legal text, Reciprocal Rank Fusion

### 4A: BM25 Index
- [ ] Install `rank_bm25`
- [ ] Build a BM25 index over all chunk texts
- [ ] Implement `sparse_search(query, top_k)` function
- [ ] Test same 5 queries from Phase 3 — compare results with dense search
- [ ] Understand: What does BM25 catch that FAISS misses? (exact terms, IDs, section numbers)

### 4B: Hybrid Fusion (Reciprocal Rank Fusion)
- [ ] Implement RRF: combine dense + sparse rankings into a single ranked list
- [ ] Implement `hybrid_search(query, top_k)` function
- [ ] Test same queries — is hybrid better than either alone?
- [ ] Add metadata filtering (filter by contract type, date range, party)

**Deliverable:** A working `hybrid_search()` that combines BM25 + FAISS.

---

## Phase 5: Graph-Enhanced Retrieval + Cross-Encoder Reranking
**What you learn:** Three-stage retrieval, graph traversal for context expansion, cross-encoder reranking

### 5A: Graph Context Expansion
- [ ] After hybrid retrieval, take top-k chunks → find their source contracts in Neo4j
- [ ] Expand context: pull related clauses (e.g., if chunk is from SOW, also pull parent MSA clauses)
- [ ] Implement `graph_expand(chunks)` function

### 5B: Cross-Encoder Reranking
- [ ] Install `sentence-transformers` (Local Reranker)
- [ ] Take expanded chunk set → rerank with a local cross-encoder model
- [ ] Implement `rerank(query, chunks, top_k)` function
- [ ] Compare final retrieval quality vs Phase 3 (dense-only) and Phase 4 (hybrid)

### 5C: Full Three-Stage Pipeline
- [ ] Chain: `hybrid_search()` → `graph_expand()` → `rerank()` = `retrieve(query)`
- [ ] Test with complex multi-hop queries:
  - "What is the liability cap in the MSA that governs SOW-003?"
  - "Which contracts with Vendor X expire before 2025?"

**Deliverable:** Complete `retrieve(query)` pipeline — three-stage hybrid + graph + rerank.

---

## Phase 6: Single-Agent RAG (Basic Generation)
**What you learn:** Prompt engineering for legal QA, citation formatting, answer quality

### 6A: Basic RAG Chain
- [ ] Build a basic chain: `retrieve(query)` → stuff context into prompt → Gemini 1.5 Flash generates answer
- [ ] Engineer the system prompt for legal analysis (cite sources, be precise, acknowledge uncertainty)
- [ ] Test with 10 queries — manually evaluate answer quality

### 6B: Citation & Source Tracking
- [ ] Every answer must cite: document name, section number, page
- [ ] Format: "According to Section 4.2 of MSA-001 (p.12), the liability cap is..."
- [ ] Implement structured output with citations as metadata

**Deliverable:** A working RAG system that answers legal queries with cited sources.

---

## Phase 7: Multi-Agent Architecture (LangGraph)
**What you learn:** Agent orchestration, state machines, specialized agents, self-correction

### 7A: LangGraph Basics
- [ ] Install `langgraph`
- [ ] Learn: State graphs, nodes, edges, conditional routing
- [ ] Build a minimal 2-node graph (query → answer) to understand the pattern

### 7B: Specialized Agents
- [ ] **Planner Agent** — Decomposes complex queries into sub-questions
- [ ] **Retriever Agent** — Executes the Phase 5 three-stage retrieval for each sub-question
- [ ] **Analyzer Agent** — Synthesizes retrieved context into a reasoned answer
- [ ] **Validator Agent** — Fact-checks the answer against source chunks, checks for hallucinations

### 7C: Self-Correcting Loop
- [ ] If Validator detects insufficient evidence or contradiction → route BACK to Retriever with refined query
- [ ] Implement max-retry limit (3 loops) to prevent infinite cycles
- [ ] Test with deliberately tricky queries that require self-correction

### 7D: Full Orchestration
- [ ] Wire all agents into a single LangGraph state machine
- [ ] Add proper state schema (query, sub_queries, retrieved_chunks, answer, validation_result, retry_count)
- [ ] Test end-to-end with 20 queries

**Deliverable:** Full multi-agent pipeline with self-correcting retrieval loops.

---

## Phase 8: Evaluation Framework
**What you learn:** RAG evaluation metrics, domain-specific eval, benchmarking methodology

### 8A: Golden Dataset Creation
- [ ] Create 200+ question-answer pairs across categories:
  - Single-doc factual (50 queries)
  - Cross-doc comparison (50 queries)
  - Multi-hop reasoning (50 queries)
  - Temporal queries (25 queries)
  - Aggregation queries (25 queries)
- [ ] Include ground truth answers with source citations

### 8B: RAGAS Evaluation
- [ ] Install `ragas`
- [ ] Run evaluation: Context Recall, Context Precision, Answer Relevancy, Faithfulness
- [ ] Target: 97%+ Context Recall, 95%+ Answer Relevancy

### 8C: Custom Legal Metrics
- [ ] **Clause Extraction F1** — How accurately entities are extracted
- [ ] **Citation Accuracy** — Do cited sources actually support the answer?
- [ ] **Cross-Doc Consistency** — For comparative queries, are all relevant docs considered?

### 8D: Ablation Study
- [ ] Compare: Dense-only vs Hybrid vs Hybrid+Graph vs Full pipeline
- [ ] Document the improvement at each stage (this is resume gold)

**Deliverable:** Full evaluation report with metrics, ablation study, and improvement story.

---

## Phase 9: Production API & Frontend
**What you learn:** FastAPI, async processing, basic React frontend, Docker

### 9A: FastAPI Backend
- [ ] Create REST API: `POST /query`, `POST /upload`, `GET /contracts`, `GET /graph`
- [ ] Add async processing for document ingestion
- [ ] Add request validation, error handling, CORS

### 9B: Streamlit / React Frontend
- [ ] Build a clean UI with:
  - Query input + response display with citations
  - Document upload interface
  - Knowledge graph visualization
  - Evaluation metrics dashboard

### 9C: Docker & Deployment
- [ ] Create `Dockerfile` and `docker-compose.yml` (app + Neo4j + Redis)
- [ ] Test full stack locally with Docker Compose
- [ ] Deploy to Railway / Render

**Deliverable:** Deployed, production-ready system with API + frontend.

---

## Phase 10: Polish & README
**What you learn:** Technical writing, project presentation, recruiter optimization

- [ ] Write a comprehensive README with architecture diagram, setup instructions, demo GIF
- [ ] Record a demo video showing complex multi-hop queries
- [ ] Write resume bullet points with real metrics from Phase 8
- [ ] Add LangSmith observability tracing screenshots
- [ ] Final code cleanup, docstrings, type hints

**Deliverable:** A GitHub-ready project that will make recruiters stop scrolling.

---

## Summary: Build Order

```
Phase 0  → Setup                        (30 min)
Phase 1  → Ingestion + Chunking         (1-2 days)
Phase 2  → Knowledge Graph              (1-2 days)
Phase 3  → Dense Retrieval (FAISS)      (1 day)
Phase 4  → Sparse + Hybrid              (1 day)
Phase 5  → Graph Expansion + Rerank     (1-2 days)
Phase 6  → Basic RAG Generation         (1 day)
Phase 7  → Multi-Agent (LangGraph)      (2-3 days)
Phase 8  → Evaluation                   (1-2 days)
Phase 9  → API + Frontend + Deploy      (2-3 days)
Phase 10 → Polish + README              (1 day)
─────────────────────────────────────────────────
Total                                   ~3-4 weeks
```
