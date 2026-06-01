# 📖 Usage Guide: Egyptian Legal Multi-Agent System

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Workflows](#core-workflows)
3. [Document Ingestion](#document-ingestion)
4. [Searching & Retrieval](#searching--retrieval)
5. [Case Analysis](#case-analysis)
6. [Knowledge Graph Operations](#knowledge-graph-operations)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)

---

## Quick Start

### Starting the System

```bash
# Using Docker Compose (Recommended)
docker-compose up -d

# Or manually with Python
python main.py

# Access API Documentation
# Open browser: http://localhost:8000/docs
```

### First API Call

```bash
# Test the system with a simple health check
curl http://localhost:8000/

# Response:
# {"message":"Welcome to the Graduation Project API!"}
```

---

## Core Workflows

### Workflow 1: Upload and Process Documents

```
1. Upload Document
   ↓
2. System Processes & Chunks
   ↓
3. Generate Embeddings
   ↓
4. Extract Entities & Relationships
   ↓
5. Store in Knowledge Graph & Vector Store
   ↓
6. Ready for Querying
```

### Workflow 2: Search and Retrieve Information

```
1. User Query
   ↓
2. Vector Search (Semantic Similarity)
   ↓
3. Graph Search (Structured Relationships)
   ↓
4. Hybrid Ranking (Combine Results)
   ↓
5. Multi-Agent Analysis
   ↓
6. Return Results
```

### Workflow 3: Analyze Legal Cases

```
1. Input Case Documents
   ↓
2. Extract Key Information
   ↓
3. Find Relevant Precedents
   ↓
4. Analyze Legal Relationships
   ↓
5. Generate Analysis Report
   ↓
6. Return Insights
```

---

## Document Ingestion

### Upload Single Document

#### Using cURL

```bash
# Upload a PDF document
curl -X POST http://localhost:8000/ingest/documents \
  -F "document=@/path/to/document.pdf" \
  -F "case_id=CASE_001" \
  -F "metadata={\"court\": \"Cairo Court\"}"

# Response:
{
  "status": "success",
  "data": {
    "document_id": "doc_12345",
    "case_id": "CASE_001",
    "chunks_created": 25,
    "processing_time": 2.5
  }
}
```

#### Using Python SDK

```python
import requests

def upload_document(file_path, case_id):
    url = "http://localhost:8000/ingest/documents"
    
    with open(file_path, 'rb') as f:
        files = {'document': f}
        data = {
            'case_id': case_id,
            'metadata': '{"court": "Cairo Court"}'
        }
        
        response = requests.post(url, files=files, data=data)
        return response.json()

# Usage
result = upload_document("case_judgment.pdf", "CASE_2024_001")
print(f"Document ID: {result['data']['document_id']}")
print(f"Chunks: {result['data']['chunks_created']}")
```

### Batch Upload Multiple Documents

```bash
# Upload multiple documents at once
curl -X POST http://localhost:8000/ingest/batch \
  -F "documents=@file1.pdf" \
  -F "documents=@file2.pdf" \
  -F "documents=@file3.pdf" \
  -F "batch_id=BATCH_001"

# Response:
{
  "status": "success",
  "data": {
    "batch_id": "BATCH_001",
    "total_documents": 3,
    "successfully_processed": 3,
    "failed": 0,
    "processing_time": 7.5
  }
}
```

### Monitor Ingestion Progress

```bash
# Check status of ingestion job
curl http://localhost:8000/ingest/status/BATCH_001

# Response:
{
  "status": "success",
  "data": {
    "batch_id": "BATCH_001",
    "status": "completed",
    "progress": 100,
    "documents_processed": 3,
    "total_chunks": 75
  }
}
```

---

## Searching & Retrieval

### Vector Search (Semantic Similarity)

Vector search finds documents based on semantic meaning, not exact keywords.

```bash
# Perform vector search
curl -X POST http://localhost:8000/retrieval/vector \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ownership rights in property disputes",
    "k": 5,
    "threshold": 0.7
  }'

# Response:
{
  "status": "success",
  "data": {
    "results": [
      {
        "document_id": "doc_12345",
        "chunk_id": "chunk_1",
        "content": "...",
        "score": 0.92,
        "metadata": {"case_id": "CASE_001", "date": "2024-01-15"}
      },
      // ... more results
    ],
    "query_time_ms": 245
  }
}
```

### Knowledge Graph Search

Search using structured relationships and entity relationships.

```bash
# Query knowledge graph for related cases
curl -X POST http://localhost:8000/kg/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (c1:Case)-[:CITED_BY]-(c2:Case) WHERE c1.caseNumber = \"2023-1234\" RETURN c2 LIMIT 10",
    "return_format": "graph"
  }'

# Response:
{
  "status": "success",
  "data": {
    "entities": [
      {
        "type": "Case",
        "id": "CASE_002",
        "properties": {
          "caseNumber": "2023-5678",
          "title": "Case Title",
          "court": "Cairo Court"
        }
      }
    ],
    "relationships": [
      {
        "from": "CASE_001",
        "type": "CITED_BY",
        "to": "CASE_002"
      }
    ]
  }
}
```

### Hybrid Search (Combined Vector + Graph)

Combines both semantic similarity and structural relationships for best results.

```bash
# Perform hybrid search
curl -X POST http://localhost:8000/retrieval/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find cases involving contract disputes with judicial precedent analysis",
    "k": 10,
    "vector_weight": 0.6,
    "graph_weight": 0.4,
    "min_score": 0.65
  }'

# Response:
{
  "status": "success",
  "data": {
    "results": [
      {
        "document_id": "doc_12345",
        "content_snippet": "...",
        "vector_score": 0.89,
        "graph_score": 0.85,
        "combined_score": 0.875,
        "retrieval_type": "hybrid",
        "context": {
          "related_entities": ["Judge1", "Law_Article_123"],
          "related_cases": ["CASE_002", "CASE_003"]
        }
      }
    ],
    "total_results": 8,
    "query_time_ms": 320
  }
}
```

---

## Case Analysis

### Analyze a Single Case

```bash
# Analyze case and retrieve insights
curl -X POST http://localhost:8000/case/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "CASE_001",
    "analysis_type": "comprehensive",
    "include_precedents": true,
    "include_evidence_analysis": true
  }'

# Response:
{
  "status": "success",
  "data": {
    "case_id": "CASE_001",
    "case_summary": "...",
    "key_legal_principles": [...],
    "precedents_found": 15,
    "evidence_strength": 0.85,
    "related_cases": [...],
    "analysis_confidence": 0.88,
    "processing_time_ms": 1200
  }
}
```

### Get Detailed Case Information

```bash
# Retrieve comprehensive case details
curl http://localhost:8000/case/CASE_001

# Response:
{
  "status": "success",
  "data": {
    "case_id": "CASE_001",
    "case_number": "2024-1234",
    "title": "Case Title",
    "date": "2024-01-15",
    "court": "Cairo Court",
    "status": "ACTIVE",
    "parties": [
      {"name": "Party A", "role": "PLAINTIFF"},
      {"name": "Party B", "role": "DEFENDANT"}
    ],
    "judges": [{"name": "Judge Name", "years_experience": 20}],
    "related_laws": ["Article 123", "Article 456"],
    "precedents": [...],
    "summary": "..."
  }
}
```

### Compare Multiple Cases

```bash
# Compare two cases for similarity and differences
curl -X POST http://localhost:8000/case/compare \
  -H "Content-Type: application/json" \
  -d '{
    "case_id_1": "CASE_001",
    "case_id_2": "CASE_002",
    "comparison_aspects": ["legal_principles", "outcome", "evidence"]
  }'

# Response:
{
  "status": "success",
  "data": {
    "case_1": {"title": "...", "outcome": "..."},
    "case_2": {"title": "...", "outcome": "..."},
    "similarity_score": 0.78,
    "differences": [...],
    "shared_elements": [...]
  }
}
```

---

## Knowledge Graph Operations

### Query Entities by Type

```bash
# Retrieve all judges
curl http://localhost:8000/kg/entities/Judge

# Response:
{
  "status": "success",
  "data": {
    "entity_type": "Judge",
    "total_count": 150,
    "entities": [
      {
        "id": "JUDGE_001",
        "name": "Judge Ahmed Hassan",
        "court": "Cairo Court",
        "years_experience": 25
      }
    ]
  }
}
```

### Build Knowledge Graph from Documents

```bash
# Trigger knowledge graph construction
curl -X POST http://localhost:8000/kg/build \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc_001", "doc_002"],
    "extraction_model": "advanced",
    "confidence_threshold": 0.75
  }'

# Response:
{
  "status": "success",
  "data": {
    "job_id": "job_12345",
    "documents_processed": 2,
    "entities_extracted": 45,
    "relationships_created": 78,
    "processing_status": "completed"
  }
}
```

### Execute Custom Cypher Query

```bash
# Execute Neo4j Cypher query
curl -X POST http://localhost:8000/kg/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (c:Case)-[:CITES]->(l:Law) WHERE l.category = \"PROPERTY\" RETURN c.title, COUNT(*) as citation_count ORDER BY citation_count DESC LIMIT 20"
  }'

# Response includes query results and execution metrics
```

---

## Advanced Features

### Batch Processing

```bash
# Submit batch job for processing multiple items
curl -X POST http://localhost:8000/batch/process \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "analyze_cases",
    "items": ["CASE_001", "CASE_002", "CASE_003"],
    "priority": "high",
    "notify_url": "https://your-domain.com/webhook"
  }'
```

### Async Operations

```bash
# Start async operation
curl -X POST http://localhost:8000/async/start \
  -H "Content-Type: application/json" \
  -d '{"operation": "bulk_analyze"}'

# Get async job status
curl http://localhost:8000/async/status/{job_id}

# Retrieve async results
curl http://localhost:8000/async/results/{job_id}
```

### Export Results

```bash
# Export analysis results
curl -X POST http://localhost:8000/export \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "CASE_001",
    "format": "pdf",  # pdf, json, csv
    "include": ["summary", "analysis", "references"]
  }'
```

---

## Best Practices

### 1. Document Organization

```
✓ DO:
- Organize documents by case
- Use consistent metadata
- Include relevant dates and parties

✗ DON'T:
- Upload corrupted files
- Mix multiple cases in one document
- Omit important metadata
```

### 2. Search Query Optimization

```python
# Good query (specific and clear)
"patent infringement in pharmaceutical industry"

# Better query (with context)
{
  "query": "patent infringement in pharmaceutical industry",
  "domain": "intellectual_property",
  "date_range": {"from": "2023-01-01", "to": "2024-12-31"}
}

# Best query (with preferences)
{
  "query": "patent infringement in pharmaceutical industry",
  "k": 5,
  "min_score": 0.75,
  "include_precedents": true,
  "sort_by": "relevance"
}
```

### 3. Error Handling

```python
import requests

def search_with_retry(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:8000/retrieval/vector',
                json={"query": query},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
```

### 4. Performance Tips

```bash
# Tip 1: Use appropriate k value for retrieval
# k=5 for quick checks, k=20 for comprehensive analysis

# Tip 2: Set confidence thresholds
# Higher threshold = fewer but more confident results

# Tip 3: Batch operations when possible
# 10x faster than individual requests

# Tip 4: Cache frequently used queries
# Reduces database load
```

### 5. Data Quality

```python
# Validate documents before upload
def validate_document(file_path):
    # Check file size (< 50MB recommended)
    # Check format (PDF, TXT, JSON)
    # Check encoding (UTF-8)
    # Extract and validate metadata
    pass

# Ensure consistent metadata
metadata = {
    "case_number": "2024-1234",
    "court": "Cairo Court",
    "date": "2024-01-15",
    "parties": ["Party A", "Party B"]
}
```

---

## Workflow Examples

### Example 1: Complete Case Analysis Workflow

```python
import requests

# Step 1: Upload document
upload_resp = requests.post(
    'http://localhost:8000/ingest/documents',
    files={'document': open('case.pdf', 'rb')},
    data={'case_id': 'CASE_2024_001'}
)
doc_id = upload_resp.json()['data']['document_id']

# Step 2: Wait for processing
import time
time.sleep(2)

# Step 3: Analyze case
analysis_resp = requests.post(
    'http://localhost:8000/case/analyze',
    json={
        'case_id': 'CASE_2024_001',
        'analysis_type': 'comprehensive'
    }
)

# Step 4: Get results
analysis = analysis_resp.json()['data']
print(f"Summary: {analysis['case_summary']}")
print(f"Key Principles: {analysis['key_legal_principles']}")
print(f"Related Cases: {analysis['precedents_found']}")
```

### Example 2: Find Similar Cases

```python
# Find cases similar to a reference case
similar_resp = requests.post(
    'http://localhost:8000/retrieval/hybrid',
    json={
        'query': 'Find cases similar to CASE_001',
        'k': 10,
        'vector_weight': 0.7,
        'graph_weight': 0.3
    }
)

similar_cases = similar_resp.json()['data']['results']
for case in similar_cases:
    print(f"Score: {case['combined_score']:.2f} - {case['content_snippet']}")
```

---

## Support & Resources

- **API Documentation**: Visit http://localhost:8000/docs
- **System Logs**: Check `logs/` directory
- **Database Explorer**: Visit http://localhost:7474/browser (Neo4j)

---

**Happy analyzing! For more information, refer to the other documentation files.**
