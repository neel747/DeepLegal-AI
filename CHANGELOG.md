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
