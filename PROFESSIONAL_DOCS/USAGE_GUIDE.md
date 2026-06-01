# 📖 Usage Guide: Egyptian Legal Multi-Agent System

## Overview

This guide explains how to use the EL-MOSTASHAR system to process legal cases and retrieve legal information.

---

## Main Use Case: Case Invocation

### Endpoint: `POST /cases/invoke_case`

**Purpose**: Submit a legal case with documents and receive comprehensive AI analysis.

### Basic Usage

```bash
curl -X POST http://localhost:8000/cases/invoke_case \
  -F "case_id=CASE_2024_001" \
  -F "files=@case_document.pdf" \
  -F "files=@witness_statement.pdf"
```

### Response Structure

```json
{
  "success": true,
  "case_id": "CASE_2024_001",
  "files_received": 2,
  "result": {
    "case_number": "123/2024",
    "court": "Cairo Court",
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

---

## Retrieval APIs

### Knowledge Graph Retrieval

**Find legal articles related to a question:**

```bash
curl -X POST http://localhost:8000/kg/retriever/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي عقوبة السرقة؟",
    "k": 10,
    "threshold": 0.6
  }'
```

**Response**: List of relevant legal articles with sources and scores.

### Vector Store Retrieval

**Find similar court rulings (semantic search):**

```bash
curl -X POST http://localhost:8000/vs/retriever/dense \
  -H "Content-Type: application/json" \
  -d '{
    "query": "العقود والالتزامات",
    "k": 5
  }'
```

**Find similar rulings by keyword (lexical search):**

```bash
curl -X POST http://localhost:8000/vs/retriever/sparse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "العقود والالتزامات",
    "k": 5
  }'
```

**Find best results (hybrid search):**

```bash
curl -X POST http://localhost:8000/vs/retriever/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "العقود والالتزامات",
    "k": 5,
    "dense_weight": 0.6,
    "sparse_weight": 0.4
  }'
```

---

## Knowledge Graph Operations

### Check System Status

```bash
curl http://localhost:8000/kg/retriever/status
```

### Get Statistics

```bash
curl http://localhost:8000/kg/statistics
```

Response includes node counts, relationship counts, etc.

### Fetch Specific Article

```bash
curl http://localhost:8000/kg/penal_code_001/311
```

### Query Amendments

```bash
curl "http://localhost:8000/kg/amendments?law_id=penal_code_001"
```

### Clear Cache

```bash
curl -X DELETE http://localhost:8000/kg/cache?name=statistics
```

---

## Vector Store Operations

### Build Vector Store

```bash
curl -X POST "http://localhost:8000/vs/build?index_path=na2d_faiss_index"
```

### Load Vector Store

```bash
curl -X POST "http://localhost:8000/vs/load?index_path=na2d_faiss_index"
```

### Check Status

```bash
curl http://localhost:8000/vs/status
```

---

## Workflow Examples

### Example 1: Complete Case Analysis

```bash
# Submit case
RESPONSE=$(curl -s -X POST http://localhost:8000/cases/invoke_case \
  -F "case_id=CASE_2024_001" \
  -F "files=@case.pdf")

# Extract results
echo $RESPONSE | jq '.result.judge_decision'
```

### Example 2: Research Legal Question

```bash
# Search knowledge graph
curl -X POST http://localhost:8000/kg/retriever/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما الفرق بين السرقة والاختلاس؟",
    "k": 15
  }' | jq '.sources[] | {article_number, score, text}'
```

### Example 3: Find Precedents

```bash
# Search for similar court rulings
curl -X POST http://localhost:8000/vs/retriever/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "جريمة اختلاس أموال عامة",
    "k": 10,
    "dense_weight": 0.7,
    "sparse_weight": 0.3
  }' | jq '.hits[] | {page_content}'
```

---

## Best Practices

### Case Invocation
- ✅ Use clear case IDs (e.g., "CASE_2024_001")
- ✅ Include all relevant documents (case, evidence, witness statements)
- ✅ Ensure documents are text or PDF format
- ❌ Don't submit cases > 100MB
- ❌ Avoid special characters in case_id

### Retrieval Queries
- ✅ Write queries in Arabic for better results
- ✅ Use 5-15 keywords for best results
- ✅ Adjust k parameter based on needs (5-50)
- ✅ Use hybrid search for balanced results
- ❌ Don't set threshold too high (< 0.7)
- ❌ Avoid single-word queries

### Performance Optimization
- ✅ Cache results when possible
- ✅ Use indexed queries
- ✅ Batch similar queries
- ✅ Monitor system load
- ❌ Don't run multiple large cases simultaneously
- ❌ Avoid rebuilding indexes frequently

---

## Troubleshooting

### Case Returns No Results
- Check if documents are properly formatted
- Verify case_id is unique
- Ensure Neo4j and Vector Store are initialized

### Retrieval Returns Irrelevant Results
- Increase k parameter
- Lower threshold value
- Use more specific keywords
- Try hybrid search instead of dense

### API Responds with 503
- Vector store not loaded: `curl -X POST http://localhost:8000/vs/load`
- Knowledge graph not built: `curl -X POST http://localhost:8000/kg/build`

---

## Advanced Configuration

### Custom Embedding Batch Size

When embedding nodes:
```bash
curl -X POST "http://localhost:8000/kg/retriever/index/embed?batch_size=256&max_workers=8"
```

### Custom Vector Index Dimensions

```bash
curl -X POST "http://localhost:8000/kg/retriever/index/rebuild?dimensions=768&batch_size=128"
```

### Custom Retrieval Weights

Adjust dense vs sparse weights:
```json
{
  "query": "question",
  "k": 10,
  "dense_weight": 0.7,
  "sparse_weight": 0.3
}
```

---

## Performance Metrics

### Expected Response Times
- Case invocation: 2-5 minutes (depends on document size)
- KG retrieval: 500ms - 2s
- Vector search: 100ms - 1s
- Hybrid search: 1s - 3s

### Throughput
- Concurrent case invocations: 1-3 (per 8GB RAM)
- Retrieval queries: 10-20 per second
- Case throughput: 20-40 cases per day

---

## Support

- Email: mg7357432@gmail.com
- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- Review [API_REFERENCE.md](./API_REFERENCE.md)

---

**Last Updated**: June 2024  
**Version**: 1.0.0
