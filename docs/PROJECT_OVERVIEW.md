# Project Overview

## What it is

**EL-MOSTASHAR** is a FastAPI-based AI backend that processes Egyptian criminal case documents and produces a simulated legal verdict. It uses a LangGraph pipeline of 10 specialized agents, each powered by a configurable LLM.

---

## Core components

### 1. FastAPI application (`main.py`)

The entry point. On startup it pre-warms the Neo4j KG connection and the KG retriever. It mounts 7 routers.

### 2. Agent pipeline (`src/Graph/`)

A LangGraph `StateGraph` with 10 nodes sharing a single `AgentState` Pydantic model. The pipeline processes case documents end-to-end from raw text to a final verdict.

**Agent execution order:**

1. `DataIngestionAgent` — reads uploaded `.txt` files and extracts structured entities (defendants, charges, evidence, witnesses, confessions, lab reports, criminal records, criminal proceedings, defense documents, case metadata)
2. `ProceduralAuditorAgent` — detects procedural violations and critical nullities; can short-circuit the pipeline directly to the judge
3. `LegalResearcherAgent` — retrieves applicable law articles and Cassation Court principles using KG + FAISS hybrid retrieval
4. `EvidenceAnalystAgent` *(parallel)* — scores each evidence item and identifies chain-of-custody issues
5. `ConfessionValidityAgent` *(parallel)* — assesses whether each confession was obtained legally
6. `WitnessCredibilityAgent` *(parallel)* — assesses the credibility of each witness
7. `ProsecutionAnalystAgent` — builds the prosecution theory and arguments
8. `DefenseAnalystAgent` — analyses defense memoranda
9. `SentencingAgent` — determines aggravating/mitigating factors, article 17 applicability, and suggested sentencing per charge
10. `JudgeAgent` — produces the final verdict: preamble, facts, reasoning, and operative text

### 3. Knowledge Graph (`src/Graphstore/`)

A Neo4j graph built from 9 Egyptian law corpora. Stores `Law`, `Article`, `Penalty`, `Definition`, `Topic`, `Amendment`, `Table`, and `TableChunk` nodes with typed relationships.

### 4. KG Retrieval (`src/retriever/kg_retriever/`)

`LegalRetriever` combines:
- Neo4j vector ANN search (cosine similarity over HuggingFace embeddings)
- In-memory BM25 (`rank_bm25`) over all article nodes
- RRF (Reciprocal Rank Fusion) merge
- Graph expansion (penalties, amendments, definitions, cross-references, topics)
- Budget-aware context assembly (max 9,000 characters)

### 5. Vector Store (`src/Vectorstore/`, `src/retriever/vs_retriever/`)

A FAISS index built from the Na2d corpus (Cassation Court rulings + legal principle books). Supports dense, sparse (BM25), and hybrid retrieval.

### 6. Corpus Chunking (`src/Chunking/`)

`CorpusChunker` converts raw `.txt` files into LangChain `Document` objects:
- **Law articles**: split at article boundaries using Arabic regex
- **Amendment articles**: split and tagged with amendment metadata
- **Tables**: split at table-header markers
- **Na2d rulings**: word-based chunking (300 words, 40 overlap)
- **Principle books**: word-based chunking (400 words, 50 overlap)

### 7. AgentBase (`src/agents/agent_base/`)

All agents extend `AgentBase` which provides:
- LLM factory with support for 7 providers (OpenRouter, Groq, Mistral, Google, Vertex, Cloudflare, Cerebras)
- Retry logic with exponential back-off and rate-limit handling
- JSON parsing with `json-repair` and optional entity validation
- Hybrid retrieval over the Na2d corpus
- State update helpers

---

## Laws covered

| Law ID | Arabic name |
|---|---|
| `penal_code` | قانون العقوبات |
| `criminal_procedure` | قانون الإجراءات الجنائية |
| `anti_drugs` | قانون مكافحة المخدرات |
| `anti_terror` | قانون مكافحة الإرهاب |
| `money_laundering` | قانون مكافحة غسل الأموال |
| `weapons_ammunition` | قانون الأسلحة والذخيرة |
| `criminal_constitution` | الدستور الجنائي |
| `emergency_law` | قانون الطوارئ |
| `cybercrime` | قانون مكافحة جرائم تقنية المعلومات |

---

## Key design decisions

**Graceful degradation:** If Neo4j is unavailable, the KG and BM25 index are disabled. Agents that need them fall back to FAISS-only retrieval. The server still starts and processes cases.

**Parallel fan-out/fan-in:** `EvidenceAnalystAgent`, `ConfessionValidityAgent`, and `WitnessCredibilityAgent` run in parallel between `legal_researcher` and `prosecution_analyst`. State fields they write to use `Annotated[List[...], add]` to merge rather than overwrite.

**Error isolation:** Every agent node is wrapped in `_safe_run()`. A failing agent records its error in `state.errors` and execution continues.

**Singleton resources:** Neo4j connection, FAISS index, LegalRetriever, BM25 retriever, and Na2d chunks are all initialized once and cached for the application lifetime.

**Conditional short-circuit:** If `ProceduralAuditorAgent` detects a fatal nullity (statute of limitations, absolute nullity, lack of jurisdiction, case extinguishment, defendant death), the graph skips directly to `JudgeAgent`.

---

## Technology stack

| Component | Technology |
|---|---|
| API framework | FastAPI 0.129 + Uvicorn 0.41 |
| Agent orchestration | LangGraph 1.0 |
| LLM integration | LangChain 1.2 |
| Knowledge Graph | Neo4j 6.1 + neo4j Python driver |
| Vector store | FAISS (faiss-cpu 1.13) |
| Embeddings | HuggingFace (sentence-transformers 5.4) |
| Sparse retrieval | rank-bm25 0.2 |
| Data validation | Pydantic v2 + pydantic-settings |
| JSON repair | json-repair 0.44 |
| Caching | cachetools 7.1 (TTLCache) |
| Observability | LangSmith (optional) |
| Containerization | Docker + Docker Compose |
