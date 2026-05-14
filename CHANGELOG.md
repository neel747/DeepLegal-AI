# DeepLegal AI — Project Changelog

> This file tracks every update, decision, and milestone in the project build.
> Updated after each phase/sub-phase completion.

---

## Format

Each entry follows:
```
### [Date] — [Phase X.Y] — [Short Title]
**Status:** Completed / In Progress / Blocked
**What was done:** ...
**What was learned:** ...
**Key decisions:** ...
**Issues encountered:** ...
**Next step:** ...
```

---

## Log Entries

*(Entries will be added as we build each phase)*

---

### 2026-05-14 — Phase 0 — Project Setup & Foundation
**Status:** Completed
**What was done:**
- Initialized project structure following the multi-phase architecture.
- Configured Python environment and dependencies (`requirements.txt`).
- Set up environment variable management with `.dotenv` and `src/config.py`.
- Initialized Git repository and connected to GitHub.
**What was learned:** Importance of modular structure for scaling multi-agent systems.
**Key decisions:** Use `unstructured` for robust document parsing.
**Next step:** Phase 1 — Document Ingestion & Legal-Aware Chunking

---

### 2026-05-14 — Phase 1 — Document Ingestion & Legal-Aware Chunking
**Status:** Completed
**What was done:**
- Implemented `DocumentParser` using `unstructured` for PDF/DOCX support.
- Developed `LegalChunker` for section-aware splitting, preserving clause context and metadata.
- Integrated `EntityExtractor` using GPT-4o-mini for structured metadata extraction (Parties, Dates, Obligations).
- Built `IngestionPipeline` to orchestrate parsing, chunking, and extraction into structured JSON.
**What was learned:** Naive chunking loses critical legal relationships; section headers are the best splitting boundaries.
**Key decisions:** Store chunks and entities in a single JSON per contract for easy downstream processing.
**Issues encountered:** PDF tables require specialized handling (partially solved by `unstructured` layout detection).
**Next step:** Phase 2 — Knowledge Graph Construction (Neo4j)

---

### 2026-05-14 — Phase 2 — Knowledge Graph Construction
**Status:** Completed
**What was done:**
- Integrated `neo4j` Python driver and configured `src/config.py`.
- Developed `GraphBuilder` to transform processed JSON data into a graph schema (Nodes: Contract, Clause, Party, Date).
- Implemented `GraphQueries` with helper functions for relational legal search.
- Integrated graph population into the `IngestionPipeline`.
- Verified logic via dry-run connection attempts to Neo4j.
**What was learned:** Knowledge Graphs excel at representing hierarchical and relational legal structures (e.g., MSAs governing multiple SOWs).
**Key decisions:** Use `MERGE` in Cypher to ensure idempotency when importing entities like Parties and Dates.
**Issues encountered:** Requires active Neo4j instance for full verification.
**Next step:** Phase 3 — Dense Vector Retrieval (FAISS)

---

### 2026-05-14 — Phase 3 — Dense Vector Retrieval (FAISS)
**Status:** Completed
**What was done:**
- Implemented `LegalEmbedder` using OpenAI's `text-embedding-3-small` (switched from local models to ensure compatibility and quality).
- Developed `VectorStore` using FAISS for indexing and retrieval of semantic embeddings.
- Created a search testing script to verify retrieval logic.
**What was learned:** Dense retrieval is powerful for semantic similarity but can be sensitive to specific terminology or IDs (where BM25 will help in Phase 4).
**Key decisions:** Use OpenAI embeddings to avoid local segmentation faults and benefit from SOTA retrieval performance.
**Issues encountered:** Local `sentence-transformers` caused segmentation faults on Python 3.13; resolved by moving to API-based embeddings.
**Next step:** Phase 4 — Sparse Retrieval (BM25) + Hybrid Fusion
