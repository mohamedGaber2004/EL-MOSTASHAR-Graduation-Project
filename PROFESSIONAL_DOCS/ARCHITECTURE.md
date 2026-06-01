# 🏗️ System Architecture: Egyptian Legal Multi-Agent System

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [System Layers](#system-layers)
3. [Multi-Agent Pipeline](#multi-agent-pipeline)
4. [Data Flow](#data-flow)
5. [Retrieval Architecture](#retrieval-architecture)
6. [Storage & Databases](#storage--databases)
7. [API Architecture](#api-architecture)
8. [Component Interactions](#component-interactions)

---

## Architecture Overview

### Core Design Principles

The system follows these architectural principles:

1. **Modular Agent Architecture**: 10 independent, specialized agents working in a coordinated pipeline
2. **Hybrid Retrieval**: Multiple retrieval strategies (semantic, lexical, graph-based)
3. **FastAPI REST Interface**: Stateless, scalable HTTP endpoints
4. **Lazy Initialization**: Resources (graph, vector store, retrievers) initialized on-demand
5. **Thread-Safe Singletons**: Global instances with locking mechanisms
6. **Arabic-First**: Native support for Arabic language processing

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST Layer                        │
│  (/cases, /kg/retriever, /vs/retriever, /kg, /vs, /agents) │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼───────────┐ ┌──▼──────────┐ ┌──▼──────────┐
    │ Case Router   │ │ Retrieval   │ │ Management  │
    │ invoke_case   │ │ Routers     │ │ Routers     │
    │               │ │ (KG, VS)    │ │ (KG, VS)    │
    └───┬───────────┘ └──┬──────────┘ └──┬──────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  LangGraph Multi-Agent Engine   │
        │  (src/Graph/graph_builder.py)   │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  10 Specialized Legal Agents    │
        │  (src/agents/*/agent.py)        │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
    ┌───▼────────────┐          ┌────────▼──────┐
    │ Knowledge Graph│          │ Vector Store  │
    │ (Neo4j)        │          │ (FAISS)       │
    │ - Articles     │          │ - Rulings     │
    │ - Amendments   │          │ - Principles  │
    └────────────────┘          └───────────────┘
```

---

## System Layers

### Layer 1: API Gateway Layer (FastAPI)

**Location**: `main.py`, `src/routers/*.py`

#### Core Routers
- **Case Router** (`case_router.py`)
  - `POST /cases/invoke_case` - Main entry point for case processing
  - `GET /cases/get_case_result` - Retrieve cached results

- **KG Retriever Router** (`kg_retriever_router.py`)
  - `POST /kg/retriever/retrieve` - Hybrid legal document retrieval
  - `POST /kg/retriever/index/setup` - Create vector indexes
  - `POST /kg/retriever/index/embed` - Embed nodes
  - `POST /kg/retriever/index/reindex` - Re-embed articles
  - `POST /kg/retriever/index/rebuild` - Full index rebuild
  - `GET /kg/retriever/status` - Check retriever status

- **VS Retriever Router** (`vs_retriever_router.py`)
  - `POST /vs/retriever/dense` - FAISS semantic search
  - `POST /vs/retriever/sparse` - BM25 lexical search
  - `POST /vs/retriever/hybrid` - RRF fusion (dense + sparse)

- **Knowledge Graph Router** (`kg_router.py`)
  - `GET /kg/statistics` - Node and relationship counts
  - `GET /kg/amendments` - Query amendments
  - `GET /kg/{law_id}/{article_number}` - Fetch article text
  - `POST /kg/build` - Rebuild knowledge graph
  - `DELETE /kg/cache` - Invalidate caches

- **Vector Store Router** (`vs_router.py`)
  - `GET /vs/status` - Check if vector store loaded
  - `POST /vs/build` - Build FAISS index
  - `POST /vs/load` - Load FAISS index
  - `POST /vs/search` - Similarity search
  - `POST /vs/search/scored` - Search with L2 scores

- **Data Ingestion Router** (`data_ingestion_router.py`)
  - `POST /agents/ingest_data` - Ingest documents

- **Chunking Router** (`chunking_router.py`)
  - `GET /Services/get_chunked_articles` - Get chunked legal articles
  - `GET /Services/get_chunked_principles` - Get chunked court principles

---

### Layer 2: Multi-Agent Orchestration Layer

**Location**: `src/Graph/graph_builder.py`, `src/Graph/state.py`

#### LangGraph Pipeline

The system uses **LangGraph** for stateful multi-agent workflow:

```
START
  ↓
DATA_INGESTION (Extract case metadata)
  ↓
PROCEDURAL_AUDITOR (Check for fatal procedural issues)
  ├─ If fatal issue found → JUDGE
  ├─ Otherwise ↓
LEGAL_RESEARCHER (Research applicable laws)
  ↓
┌─────────────────────────────────┐
│ PARALLEL ANALYSIS PHASE         │
├─────────────────────────────────┤
│ 1. EVIDENCE_ANALYST             │
│ 2. DEFENSE_ANALYST              │
│ 3. CONFESSION_VALIDITY          │
│ 4. WITNESS_CREDIBILITY          │
└─────────────────────────────────┘
  ↓
PROSECUTION_ANALYST (Analyze prosecution case)
  ↓
SENTENCING_AGENT (Recommend sentencing)
  ↓
JUDGE_AGENT (Generate final judgment)
  ↓
END
```

#### 10 Specialized Agents

1. **Data Ingestion Agent**
   - Extracts case metadata from documents
   - Normalizes document format
   - Output: Case ID, court info, filing date, prosecutor, charges

2. **Procedural Auditor Agent**
   - Identifies fatal procedural violations
   - Checks statute of limitations, jurisdiction
   - May bypass entire pipeline if critical issues found

3. **Legal Researcher Agent**
   - Queries knowledge graph for applicable laws
   - Retrieves relevant articles and amendments
   - Performs hybrid search across legal corpus

4. **Evidence Analyst Agent**
   - Analyzes quality and relevance of evidence
   - Scores evidence strength
   - Identifies gaps in evidence chain

5. **Defense Analyst Agent**
   - Analyzes defense arguments
   - Evaluates defense viability
   - Identifies counter-evidence

6. **Confession Validity Agent**
   - Evaluates confession validity
   - Checks for coercion, Miranda-equivalent rights
   - Assesses psychological factors

7. **Witness Credibility Agent**
   - Evaluates witness testimony reliability
   - Analyzes inconsistencies
   - Assesses bias potential

8. **Prosecution Analyst Agent**
   - Synthesizes prosecution case
   - Evaluates prosecution arguments
   - Recommendations for prosecution strength

9. **Sentencing Agent**
   - Recommends appropriate sentencing
   - Considers precedents
   - Applies sentencing guidelines

10. **Judge Agent**
    - Synthesizes all analyses
    - Makes final judgment call
    - Issues recommendations

#### Agent State Management

**AgentState** tracks shared context:
```python
- case_id: str
- case_number: str
- source_documents: List[str]
- court: str
- jurisdiction: str
- filing_date: str
- prosecutor_name: str
- completed_agents: List[str]
- errors: List[str]
- procedural_audit: ProcedureAnalysisOutput
- legal_research: LegalResearchOutput
- evidence_analysis: EvidenceAnalysisOutput
- defense_analysis: DefenseAnalysisOutput
- confession_validity: ConfessionValidityOutput
- witness_credibility: WitnessCredibilityOutput
- prosecution_analysis: ProsecutionAnalysisOutput
- sentencing: SentencingOutput
- judge_decision: JudgeOutput
```

---

## Data Flow

### Case Invocation Flow

```
1. CLIENT REQUEST
   POST /cases/invoke_case
   ├─ case_id: "CASE_2024_001"
   └─ files: [document.pdf, ...]

2. CASE ROUTER (case_router.py)
   ├─ Validate case_id
   ├─ Create case directory
   ├─ Save uploaded files
   └─ Create AgentState

3. LANGGRAPH PIPELINE EXECUTION (graph_builder.py)
   ├─ Data Ingestion → Extract metadata
   ├─ Procedural Auditor → Check validity
   ├─ Legal Researcher → Query KG & VS
   ├─ Parallel Analysis (Evidence, Defense, Confession, Witness)
   ├─ Prosecution Analysis → Synthesize
   ├─ Sentencing → Recommendations
   └─ Judge → Final output

4. RETRIEVAL SUBSYSTEM
   ├─ Query Knowledge Graph (Neo4j)
   │  └─ Hybrid: Vector ANN + BM25 + graph expansion
   ├─ Query Vector Store (FAISS)
   │  └─ Hybrid: Dense embedding + BM25 + RRF
   └─ Return context to agents

5. OUTPUT GENERATION
   ├─ Collect all agent outputs
   ├─ Serialize to JSON
   └─ Return in response

6. CLIENT RESPONSE
   {
     "success": true,
     "case_id": "CASE_2024_001",
     "files_received": 3,
     "result": { ... agent outputs ... }
   }
```

---

## Retrieval Architecture

### Knowledge Graph Retrieval (Neo4j)

```
Query: "What laws apply to theft cases?"
  ↓
HybridRetriever (kg_retriever.py)
  ├─ Step 1: Vector ANN Search
  │  └─ Embed query → Search Neo4j vector index
  ├─ Step 2: BM25 Ranking
  │  └─ Full-text search on article text
  ├─ Step 3: RRF (Reciprocal Rank Fusion)
  │  └─ Combine rankings: score = 1/(rank_vector + 1) + 1/(rank_bm25 + 1)
  ├─ Step 4: Graph Expansion
  │  └─ Follow relationships to related articles/amendments
  └─ Threshold filtering (default: 0.5)
  ↓
Return: Ranked list of articles with sources
```

### Vector Store Retrieval (FAISS + BM25)

```
Query: "Court ruling on contract disputes"
  ↓
Hybrid Retriever (vs_retriever.py)
  ├─ Dense Search (FAISS)
  │  └─ Embed query → FAISS similarity search
  ├─ Sparse Search (BM25)
  │  └─ Lexical ranking on Okapi BM25
  ├─ RRF Fusion
  │  └─ Combine rankings with configurable weights:
  │     score = (dense_weight * dense_rank) + (sparse_weight * sparse_rank)
  └─ Return top-k results
  ↓
Return: Hybrid-ranked documents (rulings + principles)
```

#### Retrieval Strategies

1. **Dense (Vector Search)**
   - Method: FAISS L2 distance
   - Use case: Semantic similarity
   - Example: "Liability in contract disputes" finds similar concepts

2. **Sparse (BM25)**
   - Method: Okapi BM25 ranking
   - Use case: Keyword matching
   - Example: "contract" keyword search

3. **Hybrid (RRF)**
   - Method: Reciprocal Rank Fusion
   - Combines dense + sparse
   - Configurable weights (default: dense=0.6, sparse=0.4)
   - Use case: Best of both worlds

---

## Storage & Databases

### Neo4j Knowledge Graph

**Purpose**: Store legal documents, articles, amendments, relationships

**Node Types**:
- **Article**: Legal articles with full text and metadata
- **Law**: Parent law documents
- **Amendment**: Amendments to articles
- **TableChunk**: Table extracts from documents

**Properties**:
- `law_id`: Unique law identifier
- `article_number`: Article numbering
- `text`: Full article text
- `embedding`: Vector embedding (1024-dim by default)
- `date_created`: Creation timestamp
- `category`: Legal category

**Relationships**:
- `HAS_ARTICLE`: Law → Article
- `HAS_AMENDMENT`: Article → Amendment
- `REFERENCES`: Article → Article (cross-references)
- `AMENDED_BY`: Article → Amendment
- `CONTAINS_TABLE`: Document → TableChunk

### FAISS Vector Store

**Purpose**: Fast semantic search over rulings and principles

**Content**:
- Court rulings (from Na2d corpus)
- Legal principles
- Combined ~10,000+ documents

**Vector Model**:
- Arabic-native embeddings (specified in config)
- Default: AraBERT or similar Arabic SBERT
- Dimension: 768 or 1024 (configurable)

**Index Structure**:
```
├── FAISS Index
│   ├── Vector data (documents × dimensions)
│   ├── ID mapping (doc_id → vector)
│   └── Metadata (source, type, etc.)
└── BM25 Index (for sparse search)
    └── Inverted index on document terms
```

---

## API Architecture

### Request/Response Pattern

**Request**:
```json
{
  "query": "What is the penalty for theft?",
  "k": 5,
  "threshold": 0.5
}
```

**Response**:
```json
{
  "query": "What is the penalty for theft?",
  "context_text": "Combined relevant excerpts...",
  "sources": [
    {
      "type": "article",
      "law_id": "law_001",
      "article_number": "155",
      "law_title": "Penal Code",
      "score": 0.92,
      "text": "Article text..."
    }
  ],
  "article_count": 5,
  "table_count": 2
}
```

### Error Handling

- **400**: Invalid parameters
- **404**: Resource not found (e.g., no cached result)
- **409**: Conflict (e.g., KG already building)
- **422**: Unprocessable (validation error)
- **503**: Service unavailable (vector store not loaded)
- **500**: Internal server error

### Caching Strategy

**TTLCache Implementation**:
```
Statistics Cache:    1 minute   (1 entry max)
Amendments Cache:    5 minutes  (256 entries max)
History Cache:       5 minutes  (512 entries max)
Case Results Cache:  In-memory  (per-process)
```

---

## Component Interactions

### On Server Startup

```
main.py startup (lifespan context manager)
  ├─ Initialize LangSmith tracing (if enabled)
  ├─ Get KG singleton
  │  └─ Connect to Neo4j
  ├─ Get KG Retriever singleton
  │  └─ Initialize BM25 index
  └─ Log readiness
```

### On Case Invocation

```
POST /cases/invoke_case
  ├─ Validate input
  ├─ Save files to disk
  ├─ Create AgentState
  ├─ Run LangGraph pipeline
  │  ├─ Each agent runs in order/parallel
  │  ├─ Agents may query KG/VS for context
  │  ├─ State accumulates results
  │  └─ Error handling per agent
  ├─ Collect final output
  └─ Return to client
```

### Dependency Injection Pattern

```python
FastAPI Dependencies:
├─ get_graph() → Neo4j singleton
├─ get_kg_retriever() → LegalRetriever singleton
├─ get_vs() → FAISS singleton
├─ get_embeddings() → HuggingFaceEmbeddings
└─ get_cached_sparse_retriever() → BM25 singleton
```

---

## Design Patterns

### 1. Singleton Pattern
- **GraphDB**: Single Neo4j connection per process
- **KGRetriever**: Single LegalRetriever instance
- **VectorStore**: Single FAISS index in memory
- **Thread-safety**: Double-checked locking with threading.Lock()

### 2. Dependency Injection
- FastAPI `Depends()` for resource injection
- Lazy initialization on first use
- Testable and loosely coupled

### 3. State Machine (LangGraph)
- AgentState passed through pipeline
- Each agent reads/updates state
- Conditional routing based on state

### 4. TTL Caching
- Dynamic cache invalidation
- Automatic expiration
- Manual cache clear endpoints

### 5. Error Wrapping
- `_safe_run()` wrapper for agents
- Graceful degradation on agent failure
- Error collection in state

---

## Key Technologies Used

### LangChain Stack
- **LangChain**: LLM framework and chains
- **LangGraph**: Multi-agent orchestration
- **Embeddings**: HuggingFace Transformers (Arabic)

### Data Storage
- **Neo4j**: Graph database (legal knowledge base)
- **FAISS**: Vector indexing (semantic search)
- **BM25**: Lexical ranking (hybrid search)

### Backend
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### LLM Providers
- OpenAI (GPT-4, GPT-3.5)
- Google (Gemini)
- Groq (LPU)
- Cerebras
- Mistral
- Cohere

---

**Last Updated**: June 2024  
**Version**: 1.0.0  
**Architecture Status**: Production Ready ✅
