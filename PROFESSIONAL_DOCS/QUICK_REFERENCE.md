# ⚡ Quick Reference

## Core Endpoints

### Case Processing
```bash
POST /cases/invoke_case
```

### Knowledge Graph Retrieval
```bash
POST /kg/retriever/retrieve
```

### Vector Store Retrieval (Hybrid)
```bash
POST /vs/retriever/dense      # Semantic
POST /vs/retriever/sparse     # Keyword
POST /vs/retriever/hybrid     # Best results
```

---

## Quick Examples

### Submit Case
```bash
curl -X POST http://localhost:8000/cases/invoke_case \
  -F "case_id=CASE_001" \
  -F "files=@document.pdf"
```

### Search Legal Articles
```bash
curl -X POST http://localhost:8000/kg/retriever/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي العقوبة؟", "k": 10}'
```

### Find Similar Rulings
```bash
curl -X POST http://localhost:8000/vs/retriever/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "جريمة", "k": 5, "dense_weight": 0.6, "sparse_weight": 0.4}'
```

### Get Article Text
```bash
curl http://localhost:8000/kg/penal_code_001/311
```

### Check Status
```bash
curl http://localhost:8000/kg/retriever/status
curl http://localhost:8000/vs/status
```

---

## Key Parameters

### Case Invocation
- `case_id` (string, required) - Case identifier
- `files` (file[], required) - Document files

### KG Retrieval
- `question` (string) - Legal question
- `k` (integer, 1-50) - Result count
- `threshold` (float, 0.0-1.0) - Min score

### Vector Retrieval
- `query` (string) - Search query
- `k` (integer, 1-50) - Result count
- `dense_weight` (float, 0.0-1.0) - FAISS weight
- `sparse_weight` (float, 0.0-1.0) - BM25 weight

---

## Common Tasks

### Setup
```bash
# Build vector store
POST /vs/build

# Load vector store
POST /vs/load

# Build knowledge graph
POST /kg/build

# Embed nodes
POST /kg/retriever/index/embed
```

### Operations
```bash
# Get statistics
GET /kg/statistics

# Query amendments
GET /kg/amendments

# Clear cache
DELETE /kg/cache
```

### Troubleshooting
```bash
# Check KG retriever
GET /kg/retriever/status

# Check vector store
GET /vs/status

# Rebuild indexes
POST /kg/retriever/index/rebuild
```

---

## Response Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request |
| 404 | Not found |
| 409 | Conflict |
| 422 | Invalid data |
| 500 | Server error |
| 503 | Service unavailable |

---

## 10 Agents Pipeline

1. Data Ingestion
2. Procedural Auditor
3. Legal Researcher
4. Evidence Analyst (parallel)
5. Defense Analyst (parallel)
6. Confession Validity (parallel)
7. Witness Credibility (parallel)
8. Prosecution Analyst
9. Sentencing
10. Judge

---

## Default Values

- K (results): 5-15
- Threshold (KG): 0.5
- Dense weight: 0.6
- Sparse weight: 0.4
- Batch size: 128
- Max workers: 4

---

**Base URL**: http://localhost:8000  
**API Docs**: http://localhost:8000/docs  
**ReDoc**: http://localhost:8000/redoc

---

**Last Updated**: June 2024
