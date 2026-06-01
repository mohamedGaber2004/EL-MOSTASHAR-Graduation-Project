# 🏗️ System Architecture: Egyptian Legal Multi-Agent System

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [System Layers](#system-layers)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Agent Architecture](#agent-architecture)
6. [Storage Architecture](#storage-architecture)
7. [API Architecture](#api-architecture)
8. [Deployment Architecture](#deployment-architecture)
9. [Design Patterns](#design-patterns)
10. [Scalability Considerations](#scalability-considerations)

---

## Architecture Overview

### Core Principles

The system architecture is built on the following principles:

1. **Modularity**: Independent, reusable components
2. **Scalability**: Horizontal and vertical scaling capability
3. **Resilience**: Graceful error handling and recovery
4. **Extensibility**: Easy to add new agents and features
5. **Observability**: Comprehensive logging and monitoring

### Architectural Pattern: Microservices-Ready Monolith

The current implementation is a **monolithic architecture** with microservices design principles:

```
┌────────────────────────────────────────────────────────────┐
│                API Gateway Layer (FastAPI)                │
│              Request validation & routing                 │
└────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│         Multi-Agent Orchestration (LangGraph)           │
│  Coordinates specialized agents for different tasks    │
└─────────────────────────────────────────────────────────┘
      ↓           ↓            ↓             ↓
┌──────────┐  ┌────────┐  ┌────────┐  ┌─────────┐
│ Ingest   │  │Analyze │  │ Query  │  │ Audit   │
│ Agent    │  │ Agent  │  │ Agent  │  │ Agent   │
└──────────┘  └────────┘  └────────┘  └─────────┘
      ↓           ↓            ↓             ↓
┌──────────────────────────────────────────────────────────┐
│              Data Processing & Retrieval                 │
│  Chunking │ Embedding │ Indexing │ Searching            │
└──────────────────────────────────────────────────────────┘
      ↓                    ↓
┌──────────────────┐  ┌─────────────────┐
│  Neo4j Graph DB  │  │  FAISS Vectors  │
│  Knowledge Base  │  │  Semantic Index │
└──────────────────┘  └─────────────────┘
```

---

## System Layers

### Layer 1: API & Request Handling Layer

**File**: `main.py`, `src/routers/*.py`

```python
# FastAPI Application Architecture
FastAPI Application
├── Root Endpoint "/"
├── Case Analysis Endpoints
│   └── POST /case/analyze
├── Data Ingestion Endpoints
│   ├── POST /ingest/documents
│   └── POST /ingest/batch
├── Text Chunking Endpoints
│   └── POST /chunk/process
├── Knowledge Graph Endpoints
│   ├── POST /kg/query
│   └── POST /kg/build
├── Retrieval Endpoints
│   ├── POST /retrieval/hybrid
│   ├── POST /retrieval/vector
│   └── POST /retrieval/kg
└── Vector Store Endpoints
    ├── POST /vector/search
    └── POST /vector/index
```

### Layer 2: Multi-Agent Orchestration Layer

**File**: `src/Graph/graph_builder.py`, `src/agents/`

Manages workflow execution and agent coordination:

```
Agent Orchestrator (LangGraph)
├── State Management
│   ├── Current agent state
│   ├── Workflow history
│   └── Context persistence
├── Workflow Execution
│   ├── Agent routing
│   ├── Conditional branching
│   └── Error handling
└── Agent Communication
    ├── Message passing
    ├── State updates
    └── Result aggregation
```

### Layer 3: Processing & Retrieval Layer

**Files**: `src/Chunking/`, `src/retriever/`, `src/LLMs/`

Handles data transformation and information retrieval:

```
Processing Pipeline
├── Document Chunking
│   ├── Split logic
│   ├── Overlap handling
│   └── Metadata preservation
├── Embedding Generation
│   ├── Sentence-Transformers
│   ├── Batch processing
│   └── Caching
├── Retrieval Systems
│   ├── Vector Search (FAISS)
│   ├── Graph Search (Neo4j)
│   └── Hybrid Ranking
└── LLM Integration
    ├── Provider management
    ├── Prompt engineering
    └── Response parsing
```

### Layer 4: Data Storage Layer

**Files**: `src/Graphstore/`, `src/Vectorstore/`

Persistent storage of processed data:

```
Storage Systems
├── Neo4j Graph Database
│   ├── Case nodes & relationships
│   ├── Entity graph
│   └── Precedent network
├── FAISS Vector Store
│   ├── Document embeddings
│   ├── Similarity indices
│   └── Metadata mapping
└── File System
    ├── Original documents
    ├── Processed chunks
    └── Index files
```

---

## Component Architecture

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **FastAPI App** | `main.py` | REST API server, lifespan management |
| **Routers** | `src/routers/` | API endpoint handlers |
| **Graph Builder** | `src/Graph/` | Graph construction, state management |
| **Agents** | `src/agents/` | Task-specific AI agents |
| **Chunking** | `src/Chunking/` | Document preprocessing |
| **KG Builder** | `src/Graphstore/` | Knowledge graph construction |
| **Retrievers** | `src/retriever/` | Vector & graph search |
| **LLM Manager** | `src/LLMs/` | LLM provider integration |
| **Configuration** | `src/Config/` | Settings and logging |
| **Utilities** | `src/Utils/` | Helper functions |

### Component Interaction Map

```
HTTP Request
    ↓
[Router Layer]
    ├─→ Input validation
    ├─→ Parameter extraction
    └─→ Route handling
    ↓
[Business Logic]
    ├─→ Agent selection
    ├─→ Workflow coordination
    └─→ State management
    ↓
[Processing Layer]
    ├─→ Data transformation
    ├─→ Retrieval execution
    └─→ Result compilation
    ↓
[Storage Layer]
    ├─→ Database queries
    ├─→ Index updates
    └─→ Cache operations
    ↓
[Response Builder]
    ├─→ Result formatting
    ├─→ Error handling
    └─→ Status codes
    ↓
HTTP Response
```

---

## Data Flow

### End-to-End Data Processing Flow

```
1. INGESTION PHASE
┌─────────────────────────────────────┐
│ User uploads legal document         │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ Validation & Format Detection       │
│ - Check document integrity          │
│ - Extract metadata                  │
│ - Normalize format                  │
└────────────┬────────────────────────┘
             ↓
2. PREPROCESSING PHASE
┌─────────────────────────────────────┐
│ Text Chunking & Segmentation        │
│ - Identify logical sections         │
│ - Split into chunks                 │
│ - Preserve context/overlap          │
└────────────┬────────────────────────┘
             ↓
3. EMBEDDING PHASE
┌─────────────────────────────────────┐
│ Generate Vector Embeddings          │
│ - Sentence-Transformers model       │
│ - Batch processing                  │
│ - Store embeddings                  │
└────────────┬────────────────────────┘
             ↓
4. KNOWLEDGE EXTRACTION PHASE
┌──────────┬──────────────────────────┐
│          │                          │
Entity   Relationship               Metadata
Extract   Extraction                Extraction
│          │                          │
└──────────┴──────────────────────────┘
             ↓
5. STORAGE PHASE
┌──────────┬────────────────────┬──────┐
│          │                    │      │
Neo4j    FAISS               File
Graph    Vectors            System
└──────────┴────────────────────┴──────┘
             ↓
6. INDEXING PHASE
┌──────────────────────────────────────┐
│ Create/Update Search Indices         │
│ - Graph indices                      │
│ - Vector indices                     │
│ - Full-text indices                  │
└──────────────────────────────────────┘
```

### Query Processing Flow

```
User Query
    ↓
┌──────────────────────────────┐
│ Query Parsing & Expansion    │
│ - Extract key terms          │
│ - Generate related queries   │
│ - Determine intent           │
└────────┬─────────────────────┘
         ↓
    Retrieval Phase
    ┌────────────────┬──────────────┐
    ↓                ↓              ↓
Vector Search   Graph Query    Metadata
(FAISS)         (Neo4j)        Lookup
    │                │              │
    └────────────────┴──────────────┘
             ↓
    ┌──────────────────────────────┐
    │ Hybrid Ranking & Scoring     │
    │ - BM25 scoring               │
    │ - Semantic similarity        │
    │ - Graph relevance            │
    └────────┬─────────────────────┘
             ↓
    ┌──────────────────────────────┐
    │ Multi-Agent Processing       │
    │ - Select relevant agents     │
    │ - Execute analysis tasks     │
    │ - Aggregate results          │
    └────────┬─────────────────────┘
             ↓
    ┌──────────────────────────────┐
    │ Response Generation          │
    │ - Format results             │
    │ - Add explanations           │
    │ - Include citations          │
    └────────┬─────────────────────┘
             ↓
User Response
```

---

## Agent Architecture

### Agent Types & Responsibilities

```
┌─────────────────────────────────────────┐
│      Multi-Agent System                 │
│      (LangGraph Orchestration)          │
└──────────────┬────────────────────────── ┘
               │
    ┌──────────┼──────────┬──────────┐
    ↓          ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Data   │ │Analysis│ │  KG    │ │ Audit  │
│ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │
└────────┘ └────────┘ └────────┘ └────────┘
```

#### 1. Data Ingestion Agent
- **Responsibility**: Handle document lifecycle
- **Tasks**:
  - Parse and validate documents
  - Extract structured metadata
  - Perform quality assurance
  - Trigger chunking pipeline
- **Inputs**: Raw documents
- **Outputs**: Validated chunks with metadata

#### 2. Analysis Agent
- **Responsibility**: Perform legal analysis
- **Tasks**:
  - Analyze case relationships
  - Extract legal principles
  - Assess evidence
  - Generate insights
- **Inputs**: Chunks, retrieved context
- **Outputs**: Analysis results

#### 3. Knowledge Graph Agent
- **Responsibility**: Maintain knowledge base
- **Tasks**:
  - Extract entities and relationships
  - Update graph structure
  - Manage graph queries
  - Handle deduplication
- **Inputs**: Processed documents
- **Outputs**: Graph updates, query results

#### 4. Audit Agent
- **Responsibility**: Verify and validate
- **Tasks**:
  - Audit processing steps
  - Verify accuracy
  - Check consistency
  - Generate audit trails
- **Inputs**: Analysis results
- **Outputs**: Audit reports, flags

### Agent Communication Protocol

```
Agent A                          Agent B
    │                              │
    ├──────── Query Request ──────→│
    │                              │
    │←────── Processing ───────────│
    │                              │
    │←────── Result Response ──────│
    │                              │
    ├──────── State Update ───────→│
    │                              │
```

### Workflow Execution Model

```
Sequential Workflow:
Data Agent → Analysis Agent → KG Agent → Audit Agent

Parallel Workflow:
Data Agent ───┬──→ Analysis Agent
              └──→ Vector Indexing

Conditional Workflow:
Analysis Agent ─┬─→ [Valid] ─→ KG Agent
                └─→ [Invalid] ─→ Manual Review Agent
```

---

## Storage Architecture

### Neo4j Knowledge Graph Schema

```
Node Types:
├── Case
│   ├─ caseId (PRIMARY KEY)
│   ├─ caseNumber
│   ├─ title, date, court
│   ├─ summary, status
│   └─ confidence: 0.0-1.0
├── Judge
│   ├─ judgeId, name
│   ├─ court, yearsExperience
│   └─ specialization
├── Party
│   ├─ partyId, name
│   ├─ role: PLAINTIFF|DEFENDANT
│   └─ type: INDIVIDUAL|CORPORATION
├── Law
│   ├─ lawId, code, title
│   ├─ article, category
│   └─ effectiveDate
└── Precedent
    ├─ precedentId, caseId
    ├─ principle
    └─ applicability: 0.0-1.0

Relationships:
├─ [PRESIDED_BY]: Case → Judge
├─ [INVOLVED]: Case → Party
├─ [CITES]: Case → Law
├─ [CITED_BY]: Case → Case
├─ [OVERRULES]: Case → Case
├─ [RELATED_TO]: Case → Case
└─ [FOLLOWS]: Case → Precedent
```

### FAISS Vector Index Organization

```
Vector Store Structure
├─ Metadata Index
│  └─ Maps vector IDs to document references
├─ Embedding Vectors
│  ├─ Document chunks (d-dimensional vectors)
│  ├─ Semantic space representation
│  └─ Normalized L2 vectors
├─ Index Types
│  ├─ Flat Index: Brute force exact search
│  ├─ IVF: Inverted file quantization
│  └─ HNSW: Hierarchical navigable world
└─ Search Interface
   ├─ similarity_search(query_vector, k)
   ├─ range_search(query_vector, radius)
   └─ batch_search(queries, k)
```

---

## API Architecture

### RESTful Endpoint Design

```
/api/v1
├─ /health
│  └─ GET: System health check
│
├─ /case
│  ├─ POST /analyze
│  │  └─ Analyze legal case
│  ├─ GET /{caseId}
│  │  └─ Retrieve case details
│  └─ PUT /{caseId}
│     └─ Update case information
│
├─ /ingestion
│  ├─ POST /upload
│  │  └─ Upload single document
│  ├─ POST /batch
│  │  └─ Batch upload documents
│  └─ GET /status/{jobId}
│     └─ Get job status
│
├─ /chunking
│  ├─ POST /process
│  │  └─ Process and chunk document
│  └─ GET /chunks/{docId}
│     └─ Retrieve document chunks
│
├─ /kg
│  ├─ POST /query
│  │  └─ Execute graph query
│  ├─ POST /build
│  │  └─ Build KG from documents
│  └─ GET /entities/{type}
│     └─ List entities by type
│
├─ /retrieval
│  ├─ POST /hybrid
│  │  └─ Hybrid search (vector + graph)
│  ├─ POST /vector
│  │  └─ Vector similarity search
│  └─ POST /kg
│     └─ Knowledge graph search
│
└─ /admin
   ├─ GET /stats
   │  └─ System statistics
   └─ POST /rebuild
      └─ Rebuild all indices
```

### Response Format Standard

```json
{
  "status": "success|error|partial",
  "code": 200,
  "message": "Operation successful",
  "timestamp": "2024-06-01T13:25:50Z",
  "data": {
    "results": [],
    "metadata": {}
  },
  "errors": [
    {
      "code": "ERR_001",
      "message": "Error description",
      "field": "fieldName"
    }
  ],
  "pagination": {
    "page": 1,
    "pageSize": 20,
    "totalItems": 100,
    "totalPages": 5
  }
}
```

---

## Deployment Architecture

### Container Architecture

```
Docker Compose Stack
├── Application Service
│   ├─ Image: custom/el-mostashar:latest
│   ├─ Ports: 8000:8000
│   ├─ Environment: .env loaded
│   ├─ Volumes:
│   │  ├─ /data → persistent data
│   │  └─ /logs → application logs
│   └─ Dependencies: neo4j, vector-service
│
├── Neo4j Service
│   ├─ Image: neo4j:6.1.0
│   ├─ Ports: 7474:7474 (HTTP), 7687:7687 (Bolt)
│   ├─ Environment: NEO4J_AUTH
│   └─ Volumes: neo4j-data:/var/lib/neo4j/data
│
└── (Optional) Vector Service
    ├─ Image: custom/vector-service:latest
    ├─ Ports: 5000:5000
    └─ Volumes: /vector-indices
```

### Production Architecture

```
Internet
    ↓
┌─────────────────────────────────┐
│   Load Balancer / Reverse Proxy │
│   (Nginx / HAProxy)             │
└────────────┬────────────────────┘
             │
    ┌────────┼────────┐
    ↓        ↓        ↓
┌──────┐ ┌──────┐ ┌──────┐
│ App  │ │ App  │ │ App  │  (Kubernetes Replicas)
│ Pod1 │ │ Pod2 │ │ Pod3 │
└──────┘ └──────┘ └──────┘
    ↓        ↓        ↓
    └────────┼────────┘
             ↓
┌──────────────────────────────────┐
│     Service Mesh (Istio)         │
│   - Traffic routing              │
│   - Circuit breaking             │
│   - Distributed tracing          │
└────────────┬─────────────────────┘
             │
    ┌────────┼──────────┐
    ↓        ↓          ↓
┌────────┐┌─────┐┌──────────┐
│Neo4j   ││FAISS││Redis     │
│Cluster ││Dist ││Cache     │
└────────┘└─────┘└──────────┘
```

---

## Design Patterns

### Patterns Implemented

| Pattern | Usage | Benefit |
|---------|-------|---------|
| **Repository** | Data access abstraction | Testability, flexibility |
| **Factory** | Agent/Service creation | Decoupling, extensibility |
| **Strategy** | Multiple retrieval methods | Flexibility, swappability |
| **Observer** | Event-driven updates | Loose coupling |
| **Singleton** | Shared resources (DB connections) | Resource efficiency |
| **Chain of Responsibility** | Multi-agent workflows | Modular processing |
| **Pipeline** | Sequential transformations | Organized processing |

### Error Handling Strategy

```
Error Handling Flow
    ↓
┌──────────────────────────────┐
│ Error Detection              │
│ - Try/catch blocks           │
│ - Validation checks          │
│ - Type checking              │
└────────┬─────────────────────┘
         ↓
┌──────────────────────────────┐
│ Error Classification         │
│ - System errors (5xx)        │
│ - Client errors (4xx)        │
│ - Business logic errors      │
└────────┬─────────────────────┘
         ↓
┌──────────────────────────────┐
│ Recovery Strategy            │
│ - Retry (exponential backoff)│
│ - Fallback (alternative path)│
│ - Graceful degradation       │
└────────┬─────────────────────┘
         ↓
┌──────────────────────────────┐
│ Error Reporting              │
│ - Logging                    │
│ - Monitoring                 │
│ - User notification          │
└──────────────────────────────┘
```

---

## Scalability Considerations

### Horizontal Scaling

**Stateless API Design**:
- No session affinity required
- Easy load balancing
- Replicate instances freely

**Database Scaling**:
- Neo4j read replicas
- FAISS distributed indices
- Cache layer distribution

### Vertical Scaling

**Performance Optimization**:
- Connection pooling
- Query optimization
- Index strategies
- Caching layers

### Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| API Latency (p95) | <500ms | Caching, indexing |
| Document Processing | <100ms/page | Batch processing |
| Search Response | <500ms | Vector index optimization |
| Throughput | 100+ req/s | Load balancing |

---

**This architecture provides a solid foundation for a scalable, maintainable, and extensible legal AI system.**
