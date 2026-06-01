# 🔌 API Reference: Egyptian Legal Multi-Agent System

## Table of Contents
1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Case Endpoints](#case-endpoints)
4. [Ingestion Endpoints](#ingestion-endpoints)
5. [Chunking Endpoints](#chunking-endpoints)
6. [Knowledge Graph Endpoints](#knowledge-graph-endpoints)
7. [Retrieval Endpoints](#retrieval-endpoints)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)

---

## API Overview

### Base URL
```
http://localhost:8000/api/v1
```

### API Version
- Current Version: **1.0.0**
- Release Date: June 2024

### Response Format

All endpoints return responses in this standardized format:

```json
{
  "status": "success",
  "code": 200,
  "message": "Operation successful",
  "timestamp": "2024-06-01T13:25:50Z",
  "data": {},
  "errors": [],
  "pagination": {}
}
```

---

## Authentication

### API Key Authentication (Future)

Currently, the system runs without authentication. For production deployment:

```bash
# Add API key to request headers
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/case/analyze
```

### Bearer Token (Future)

```bash
# Get token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/case/analyze
```

---

## Case Endpoints

### 1. Analyze Case

**Endpoint**: `POST /case/analyze`

**Description**: Analyze a legal case and retrieve comprehensive insights

**Request Body**:
```json
{
  "case_id": "CASE_2024_001",
  "analysis_type": "comprehensive",
  "include_precedents": true,
  "include_evidence_analysis": true,
  "include_citations": true,
  "confidence_threshold": 0.75
}
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | Yes | Unique case identifier |
| analysis_type | string | No | "quick" or "comprehensive" (default: "comprehensive") |
| include_precedents | boolean | No | Include precedent analysis (default: true) |
| include_evidence_analysis | boolean | No | Analyze evidence strength (default: true) |
| include_citations | boolean | No | Include law citations (default: true) |
| confidence_threshold | float | No | Minimum confidence (0.0-1.0, default: 0.75) |

**Response**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "case_id": "CASE_2024_001",
    "case_number": "2024-1234",
    "title": "Case Title",
    "case_summary": "...",
    "key_legal_principles": ["Principle 1", "Principle 2"],
    "precedents_found": 15,
    "evidence_strength": 0.85,
    "related_cases": ["CASE_002", "CASE_003"],
    "analysis_confidence": 0.88,
    "processing_time_ms": 1200
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/case/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "CASE_2024_001",
    "analysis_type": "comprehensive"
  }'
```

---

### 2. Get Case Details

**Endpoint**: `GET /case/{caseId}`

**Description**: Retrieve detailed information about a specific case

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| caseId | string | Case identifier |

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| include_relationships | boolean | Include related cases (default: true) |
| include_entities | boolean | Include entity information (default: true) |

**Response**:
```json
{
  "status": "success",
  "data": {
    "case_id": "CASE_001",
    "case_number": "2024-1234",
    "title": "Case Title",
    "date": "2024-01-15",
    "court": "Cairo Court",
    "status": "ACTIVE",
    "parties": [...],
    "judges": [...],
    "related_laws": [...],
    "precedents": [...]
  }
}
```

**Example**:
```bash
curl http://localhost:8000/case/CASE_2024_001?include_relationships=true
```

---

### 3. Update Case

**Endpoint**: `PUT /case/{caseId}`

**Description**: Update case information

**Request Body**:
```json
{
  "status": "CLOSED",
  "summary": "Updated case summary",
  "outcome": "DISMISSED",
  "metadata": {}
}
```

**Response**: Updated case object

---

### 4. Compare Cases

**Endpoint**: `POST /case/compare`

**Description**: Compare two cases for similarity and differences

**Request Body**:
```json
{
  "case_id_1": "CASE_001",
  "case_id_2": "CASE_002",
  "comparison_aspects": ["legal_principles", "outcome", "evidence"],
  "return_format": "detailed"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "case_1": {...},
    "case_2": {...},
    "similarity_score": 0.78,
    "differences": [...],
    "shared_elements": [...]
  }
}
```

---

## Ingestion Endpoints

### 1. Upload Single Document

**Endpoint**: `POST /ingest/documents`

**Description**: Upload and process a single legal document

**Request**:
```
Content-Type: multipart/form-data

document: [binary file]
case_id: "CASE_2024_001"
metadata: {"court": "Cairo Court", "date": "2024-01-15"}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "document_id": "doc_12345",
    "case_id": "CASE_2024_001",
    "chunks_created": 25,
    "file_size": 1024000,
    "processing_time_ms": 2500,
    "status": "completed"
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/ingest/documents \
  -F "document=@case_document.pdf" \
  -F "case_id=CASE_2024_001" \
  -F "metadata={\"court\": \"Cairo Court\"}"
```

---

### 2. Batch Upload

**Endpoint**: `POST /ingest/batch`

**Description**: Upload multiple documents in one request

**Request**:
```
Content-Type: multipart/form-data

documents: [binary file 1]
documents: [binary file 2]
documents: [binary file 3]
batch_id: "BATCH_001"
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "batch_id": "BATCH_001",
    "total_documents": 3,
    "successfully_processed": 3,
    "failed": 0,
    "total_chunks": 75,
    "processing_time_ms": 7500
  }
}
```

---

### 3. Check Ingestion Status

**Endpoint**: `GET /ingest/status/{jobId}`

**Description**: Check the status of an ingestion job

**Response**:
```json
{
  "status": "success",
  "data": {
    "job_id": "job_12345",
    "status": "completed",
    "progress": 100,
    "documents_processed": 3,
    "total_chunks": 75,
    "completion_time": "2024-06-01T13:25:50Z"
  }
}
```

---

## Chunking Endpoints

### 1. Process Document Chunks

**Endpoint**: `POST /chunk/process`

**Description**: Process and chunk a document

**Request Body**:
```json
{
  "document_id": "doc_12345",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "preserve_paragraphs": true
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "document_id": "doc_12345",
    "chunks_created": 25,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "metadata": {...}
  }
}
```

---

### 2. Retrieve Document Chunks

**Endpoint**: `GET /chunk/chunks/{docId}`

**Description**: Retrieve chunks for a specific document

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| page | integer | Page number (default: 1) |
| page_size | integer | Results per page (default: 20) |

**Response**:
```json
{
  "status": "success",
  "data": {
    "document_id": "doc_12345",
    "total_chunks": 100,
    "chunks": [
      {
        "chunk_id": "chunk_1",
        "content": "...",
        "tokens": 256,
        "embedding": [...]
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_pages": 5
    }
  }
}
```

---

## Knowledge Graph Endpoints

### 1. Query Knowledge Graph

**Endpoint**: `POST /kg/query`

**Description**: Execute a Cypher query against the knowledge graph

**Request Body**:
```json
{
  "query": "MATCH (c:Case) RETURN c LIMIT 10",
  "parameters": {},
  "return_format": "graph"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "entities": [...],
    "relationships": [...],
    "query_time_ms": 145
  }
}
```

---

### 2. List Entities by Type

**Endpoint**: `GET /kg/entities/{type}`

**Description**: List all entities of a specific type

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| type | string | Entity type: Case, Judge, Party, Law, Precedent |

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| limit | integer | Maximum results (default: 100) |
| offset | integer | Results offset (default: 0) |

**Response**:
```json
{
  "status": "success",
  "data": {
    "entity_type": "Judge",
    "total_count": 150,
    "entities": [...]
  }
}
```

---

### 3. Build Knowledge Graph

**Endpoint**: `POST /kg/build`

**Description**: Build knowledge graph from processed documents

**Request Body**:
```json
{
  "document_ids": ["doc_001", "doc_002"],
  "extraction_model": "advanced",
  "confidence_threshold": 0.75,
  "force_rebuild": false
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "job_id": "job_12345",
    "documents_processed": 2,
    "entities_extracted": 45,
    "relationships_created": 78,
    "status": "completed"
  }
}
```

---

## Retrieval Endpoints

### 1. Vector Search

**Endpoint**: `POST /retrieval/vector`

**Description**: Perform semantic similarity search

**Request Body**:
```json
{
  "query": "ownership rights in property disputes",
  "k": 5,
  "threshold": 0.7
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "query": "ownership rights in property disputes",
    "results": [
      {
        "document_id": "doc_12345",
        "chunk_id": "chunk_1",
        "content": "...",
        "score": 0.92,
        "metadata": {...}
      }
    ],
    "total_results": 5,
    "query_time_ms": 245
  }
}
```

---

### 2. Knowledge Graph Search

**Endpoint**: `POST /retrieval/kg`

**Description**: Search using knowledge graph relationships

**Request Body**:
```json
{
  "entity_type": "Case",
  "filters": {
    "court": "Cairo Court",
    "status": "ACTIVE"
  },
  "limit": 10
}
```

**Response**: List of entities matching filters

---

### 3. Hybrid Search

**Endpoint**: `POST /retrieval/hybrid`

**Description**: Combined vector and graph search

**Request Body**:
```json
{
  "query": "Find cases involving contract disputes",
  "k": 10,
  "vector_weight": 0.6,
  "graph_weight": 0.4,
  "min_score": 0.65
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "results": [
      {
        "document_id": "doc_12345",
        "vector_score": 0.89,
        "graph_score": 0.85,
        "combined_score": 0.875,
        "retrieval_type": "hybrid"
      }
    ],
    "total_results": 8,
    "query_time_ms": 320
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "code": 400,
  "message": "Invalid request",
  "errors": [
    {
      "field": "case_id",
      "code": "ERR_INVALID_FORMAT",
      "message": "Case ID must be alphanumeric"
    }
  ],
  "timestamp": "2024-06-01T13:25:50Z"
}
```

### Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 400 | Bad Request | Invalid input parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource already exists |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Scenarios

#### Invalid Request
```json
{
  "status": "error",
  "code": 400,
  "message": "Validation failed",
  "errors": [
    {"field": "case_id", "message": "Required field"}
  ]
}
```

#### Not Found
```json
{
  "status": "error",
  "code": 404,
  "message": "Case not found",
  "data": {"case_id": "CASE_999"}
}
```

#### Service Error
```json
{
  "status": "error",
  "code": 500,
  "message": "Database connection failed",
  "request_id": "req_12345"
}
```

---

## Rate Limiting

### Current Limits (Production)
- 100 requests per minute per IP
- 10 concurrent requests per IP

### Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1624080000
```

### Rate Limit Exceeded Response

```json
{
  "status": "error",
  "code": 429,
  "message": "Rate limit exceeded",
  "retry_after": 60
}
```

---

## SDK Examples

### Python SDK

```python
import requests

class ELMostasharClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_case(self, case_id):
        response = requests.post(
            f"{self.base_url}/case/analyze",
            json={"case_id": case_id}
        )
        return response.json()
    
    def search(self, query, k=5):
        response = requests.post(
            f"{self.base_url}/retrieval/hybrid",
            json={"query": query, "k": k}
        )
        return response.json()

# Usage
client = ELMostasharClient()
results = client.analyze_case("CASE_2024_001")
```

### JavaScript SDK

```javascript
const client = new ELMostasharClient('http://localhost:8000');

async function analyzeCase(caseId) {
    const response = await fetch(`${client.baseUrl}/case/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId })
    });
    return await response.json();
}

// Usage
const results = await analyzeCase('CASE_2024_001');
```

---

**For complete interactive API documentation, visit http://localhost:8000/docs when the server is running.**
