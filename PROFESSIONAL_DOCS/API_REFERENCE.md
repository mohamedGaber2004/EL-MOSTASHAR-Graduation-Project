# 🔌 API Reference: Egyptian Legal Multi-Agent System

## Table of Contents
1. [API Overview](#api-overview)
2. [Case Endpoints (invoke_case)](#case-endpoints)
3. [Knowledge Graph Retrieval APIs](#knowledge-graph-retrieval-apis)
4. [Vector Store Retrieval APIs](#vector-store-retrieval-apis)
5. [Knowledge Graph Management APIs](#knowledge-graph-management-apis)
6. [Vector Store Management APIs](#vector-store-management-apis)
7. [Data Processing APIs](#data-processing-apis)
8. [Response Format](#response-format)
9. [Error Handling](#error-handling)

---

## API Overview

### Base URL
```
http://localhost:8000
```

### API Version
- **Current Version**: 1.0.0
- **Released**: June 2024

### Core APIs
As per the system design, the **core APIs** are:
1. **Retrieval APIs** - All retrieval endpoints (KG and Vector Store)
2. **invoke_case** - Case invocation endpoint

---

## Case Endpoints

### Main Entry Point: Invoke Case

**Endpoint**: `POST /cases/invoke_case`

**Description**: Submit a legal case with documents and receive comprehensive AI analysis from all 10 agents.

**Request**:
```
Content-Type: multipart/form-data

Parameters:
- case_id (string, required): Unique case identifier
- files (file[], required): Legal document files
```

**Example**:
```bash
curl -X POST http://localhost:8000/cases/invoke_case \
  -F "case_id=CASE_2024_001" \
  -F "files=@case_document.pdf" \
  -F "files=@witness_statement.pdf"
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Case processed successfully",
  "case_id": "CASE_2024_001",
  "files_received": 2,
  "result": {
    "case_id": "CASE_2024_001",
    "case_number": "123/2024",
    "court": "Cairo Court of First Instance",
    "jurisdiction": "Cairo",
    "filing_date": "2024-01-15",
    "procedural_audit": { ... },
    "legal_research": { ... },
    "evidence_analysis": { ... },
    "defense_analysis": { ... },
    "confession_validity": { ... },
    "witness_credibility": { ... },
    "prosecution_analysis": { ... },
    "sentencing": { ... },
    "judge_decision": { ... }
  }
}
```

**Error Responses**:
- `400`: Missing case_id or no valid files
- `500`: Internal server error during processing

---

## Knowledge Graph Retrieval APIs

### Hybrid Legal Document Retrieval

**Endpoint**: `POST /kg/retriever/retrieve`

**Description**: Query the knowledge graph using hybrid retrieval (vector + BM25 + graph expansion).

**Request**:
```json
{
  "question": "What are the penalties for theft in Egyptian law?",
  "k": 15,
  "threshold": 0.5
}
```

**Parameters**:
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| question | string | required | - | Natural-language legal question (Arabic) |
| k | integer | 15 | 1-50 | Number of candidate articles to retrieve |
| threshold | float | 0.5 | 0.0-1.0 | Minimum normalized RRF score |

**Example**:
```bash
curl -X POST http://localhost:8000/kg/retriever/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي عقوبة السرقة؟",
    "k": 10,
    "threshold": 0.6
  }'
```

**Response** (200 OK):
```json
{
  "query": "ما هي عقوبة السرقة؟",
  "context_text": "مجموعة من النصوص القانونية ذات الصلة...",
  "sources": [
    {
      "type": "article",
      "law_id": "penal_code_001",
      "article_number": "311",
      "law_title": "قانون العقوبات",
      "score": "0.92",
      "text": "يعاقب بالحبس..."
    }
  ],
  "article_count": 5,
  "table_count": 2
}
```

**Error Responses**:
- `404`: No relevant articles found above threshold
- `500`: Internal server error

### KG Retriever Status

**Endpoint**: `GET /kg/retriever/status`

**Description**: Check if the KG retriever singleton is initialized.

**Response** (200 OK):
```json
{
  "retriever_ready": true,
  "bm25_article_count": 3245
}
```

### Setup Vector Indexes

**Endpoint**: `POST /kg/retriever/index/setup`

**Description**: Create Neo4j vector indexes for Article and TableChunk nodes.

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| dimensions | integer | Optional: Override embedding dimensions |

**Response** (200 OK):
```json
{
  "message": "Vector indexes created (or already existed).",
  "dimensions": 1024
}
```

**Error Responses**:
- `409`: Indexes already exist
- `500`: Internal error

### Embed Nodes

**Endpoint**: `POST /kg/retriever/index/embed`

**Description**: Embed all un-embedded Article and TableChunk nodes.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | integer | 128 | Embedding batch size |
| max_workers | integer | 4 | Parallel encoding threads |

**Response** (200 OK):
```json
{
  "message": "Embedding complete.",
  "nodes_written": 1250
}
```

### Re-index Articles

**Endpoint**: `POST /kg/retriever/index/reindex`

**Description**: Force re-embed all Article nodes (overwrites existing embeddings).

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | integer | 128 | Embedding batch size |
| max_workers | integer | 4 | Parallel threads |

**Response** (200 OK):
```json
{
  "message": "Article re-indexing complete.",
  "nodes_written": 3245
}
```

### Rebuild Vector Index

**Endpoint**: `POST /kg/retriever/index/rebuild`

**Description**: Full rebuild - drop, recreate, and fully re-embed both Neo4j vector indexes.

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| batch_size | integer | Batch size |
| max_workers | integer | Worker threads |
| dimensions | integer | Optional embedding dimensions |

**Response** (200 OK):
```json
{
  "message": "Full vector index rebuild complete.",
  "nodes_written": 3245
}
```

---

## Vector Store Retrieval APIs

### Dense Vector Search (FAISS)

**Endpoint**: `POST /vs/retriever/dense`

**Description**: Semantic retrieval via FAISS dense vectors (court rulings + principles).

**Request**:
```json
{
  "query": "العقود والتزاماتها",
  "k": 5
}
```

**Parameters**:
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| query | string | required | - | Free-text query (Arabic) |
| k | integer | 5 | 1-50 | Number of documents to retrieve |

**Response** (200 OK):
```json
{
  "query": "العقود والتزاماتها",
  "strategy": "dense",
  "hits": [
    {
      "page_content": "حكم محكمة النقض رقم 123...",
      "metadata": {
        "source": "na2d_rulings",
        "type": "court_ruling",
        "date": "2023-06-15"
      }
    }
  ]
}
```

**Error Responses**:
- `503`: Vector store not loaded
- `500`: Internal error

### Sparse BM25 Search

**Endpoint**: `POST /vs/retriever/sparse`

**Description**: Lexical retrieval via cached BM25 (Okapi) index.

**Request**:
```json
{
  "query": "العقود والتزاماتها",
  "k": 5
}
```

**Response** (200 OK):
```json
{
  "query": "العقود والتزاماتها",
  "strategy": "sparse",
  "hits": [
    {
      "page_content": "حكم محكمة النقض...",
      "metadata": { ... }
    }
  ]
}
```

**Error Responses**:
- `422`: Invalid query or empty corpus
- `500`: Internal error

### Hybrid Search (Dense + Sparse + RRF)

**Endpoint**: `POST /vs/retriever/hybrid`

**Description**: Hybrid retrieval using RRF fusion of FAISS and BM25 results.

**Request**:
```json
{
  "query": "العقود والتزاماتها",
  "k": 5,
  "dense_weight": 0.6,
  "sparse_weight": 0.4
}
```

**Parameters**:
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| query | string | required | - | Free-text query |
| k | integer | 5 | 1-50 | Number of results |
| dense_weight | float | 0.6 | 0.0-1.0 | Weight for FAISS results |
| sparse_weight | float | 0.4 | 0.0-1.0 | Weight for BM25 results |

**Response** (200 OK):
```json
{
  "query": "العقود والتزاماتها",
  "strategy": "hybrid(dense=0.60, sparse=0.40)",
  "hits": [
    {
      "page_content": "...",
      "metadata": { ... }
    }
  ]
}
```

**Error Responses**:
- `422`: Both weights are 0, or invalid parameters
- `500`: Internal error

---

## Knowledge Graph Management APIs

### Get KG Statistics

**Endpoint**: `GET /kg/statistics`

**Description**: Get node and relationship counts for the knowledge graph (cached 1 minute).

**Response** (200 OK):
```json
{
  "nodes": {
    "Article": 3245,
    "Law": 42,
    "Amendment": 523,
    "TableChunk": 1850
  },
  "relationships": {
    "HAS_ARTICLE": 3200,
    "HAS_AMENDMENT": 500,
    "REFERENCES": 2100,
    "AMENDED_BY": 450,
    "CONTAINS_TABLE": 1850
  }
}
```

### Query Amendments

**Endpoint**: `GET /kg/amendments`

**Description**: List amendments, optionally filtered by law (cached 5 minutes).

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| law_id | string | Optional filter by law ID |

**Response** (200 OK):
```json
[
  {
    "amendment_id": "amend_001",
    "amendment_law_number": "Law 25/2024",
    "amendment_law_title": "قانون تعديل قانون العقوبات",
    "amendment_date": "2024-01-15",
    "amendment_type": "clarification",
    "description": "توضيح لنص المادة 311...",
    "effective_date": "2024-02-01",
    "affected_articles": ["311", "312", "313"],
    "law_id": "penal_code_001",
    "law_title": "قانون العقوبات"
  }
]
```

### Fetch Article Text

**Endpoint**: `GET /kg/{law_id}/{article_number}`

**Description**: Fetch the latest text for a specific article (cached 5 minutes).

**URL Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| law_id | string | Unique law identifier |
| article_number | string | Article number within law |

**Example**:
```bash
curl http://localhost:8000/kg/penal_code_001/311
```

**Response** (200 OK):
```json
{
  "law_id": "penal_code_001",
  "article_number": "311",
  "text": "يعاقب بالحبس مدة لا تقل عن ستة أشهر ولا تزيد على ثلاث سنوات كل من اختلس منقولا..."
}
```

**Error Responses**:
- `404`: Article not found
- `500`: Internal error

### Build Knowledge Graph

**Endpoint**: `POST /kg/build`

**Description**: Rebuild the knowledge graph in the background (asynchronous).

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| drop_existing | boolean | true | Drop existing data before rebuilding |

**Response** (200 OK):
```json
{
  "message": "Knowledge Graph build started in background."
}
```

**Error Responses**:
- `409`: Build already in progress

### Invalidate Cache

**Endpoint**: `DELETE /kg/cache`

**Description**: Invalidate one or all cache buckets.

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| name | string | Cache name: statistics \| amendments \| history (omit for all) |

**Response** (200 OK):
```json
{
  "cleared": ["statistics", "amendments", "history"],
  "message": "Cleared 3 cache bucket(s) successfully."
}
```

**Error Responses**:
- `400`: Unknown cache name

---

## Vector Store Management APIs

### Check Vector Store Status

**Endpoint**: `GET /vs/status`

**Description**: Check whether a vector store is currently loaded.

**Response** (200 OK):
```json
{
  "loaded": true
}
```

### Build Vector Store

**Endpoint**: `POST /vs/build`

**Description**: Build a FAISS index from the Na2d corpus (rulings + principles).

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| index_path | string | na2d_faiss_index | Directory to save the index |

**Response** (200 OK):
```json
{
  "message": "FAISS Vector store constructed and synchronized successfully.",
  "rulings_count": 5000,
  "principles_count": 1240,
  "total_count": 6240,
  "index_path": "na2d_faiss_index"
}
```

**Error Responses**:
- `422`: Cannot build with zero documents
- `500`: Internal error

### Load Vector Store

**Endpoint**: `POST /vs/load`

**Description**: Load a previously persisted FAISS index into memory.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| index_path | string | na2d_faiss_index | Directory of saved index |

**Response** (200 OK):
```json
{
  "message": "Vector store loaded from 'na2d_faiss_index'.",
  "index_path": "na2d_faiss_index"
}
```

### Vector Store Search

**Endpoint**: `POST /vs/search`

**Description**: Similarity search over rulings and principles.

**Request**:
```json
{
  "query": "العقود والالتزامات",
  "k": 5
}
```

**Response** (200 OK):
```json
{
  "query": "العقود والالتزامات",
  "hits": [
    {
      "page_content": "محتوى الحكم...",
      "metadata": { ... }
    }
  ]
}
```

### Vector Store Search with Scores

**Endpoint**: `POST /vs/search/scored`

**Description**: Similarity search returning L2 distance scores.

**Request**:
```json
{
  "query": "العقود والالتزامات",
  "k": 5
}
```

**Response** (200 OK):
```json
{
  "query": "العقود والالتزامات",
  "hits": [
    {
      "page_content": "محتوى الحكم...",
      "metadata": { ... },
      "score": 0.125
    }
  ]
}
```

---

## Data Processing APIs

### Ingest Data

**Endpoint**: `POST /agents/ingest_data`

**Description**: Ingest documents for processing through data ingestion agent.

**Request**:
```json
{
  "case_id": "CASE_2024_001",
  "source_documents": ["/path/to/case/dir"]
}
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| case_id | string | Unique case identifier |
| source_documents | string[] | Optional paths to documents |

**Response** (200 OK):
```json
{
  "case_id": "CASE_2024_001",
  "case_number": "123/2024",
  "extracted_data": { ... }
}
```

### Get Chunked Articles

**Endpoint**: `GET /Services/get_chunked_articles`

**Description**: Get chunked legal articles from the knowledge base.

**Response** (200 OK):
```json
{
  "status_code": 200,
  "Length of articles": 3245,
  "content": [
    {
      "page_content": "نص المادة...",
      "metadata": {
        "law_id": "penal_code_001",
        "article_number": "311"
      }
    }
  ]
}
```

### Get Chunked Principles

**Endpoint**: `GET /Services/get_chunked_principles`

**Description**: Get chunked court principles from Na2d corpus.

**Response** (200 OK):
```json
{
  "status_code": 200,
  "Length of principles": 1240,
  "content": [
    {
      "page_content": "نص المبدأ...",
      "metadata": {
        "court": "Supreme Court",
        "date": "2023-06-15"
      }
    }
  ]
}
```

---

## Response Format

### Success Response
```json
{
  "success": true,
  "message": "Operation successful",
  "data": { ... },
  "timestamp": "2024-06-01T13:25:50Z"
}
```

### Error Response
```json
{
  "detail": "Detailed error message"
}
```

---

## Error Handling

| Status Code | Error Type | Description |
|-------------|-----------|-------------|
| **400** | Bad Request | Invalid parameters or missing required fields |
| **404** | Not Found | Resource not found |
| **409** | Conflict | Operation conflict (e.g., build already in progress) |
| **422** | Unprocessable Entity | Validation error |
| **500** | Internal Server Error | Unexpected server error |
| **503** | Service Unavailable | Required service not initialized (e.g., vector store) |

---

## Rate Limiting

Currently, no rate limiting is enforced. For production deployment, implement:
- Per-IP rate limiting
- Per-endpoint rate limiting
- Burst allowances

---

## Authentication

Currently, no authentication is required. For production deployment:
- Implement API key authentication
- Add JWT bearer token support
- Add role-based access control (RBAC)

---

**Last Updated**: June 2024  
**API Version**: 1.0.0  
**Status**: Production Ready ✅
