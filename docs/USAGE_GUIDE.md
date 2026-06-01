# Usage Guide

## Running a Case Through the Full Pipeline

### 1. Upload files and invoke the pipeline

```bash
curl -X POST "http://localhost:8000/cases/invoke_case" \
  -F "case_id=case_001" \
  -F "files=@amr_ihala.txt" \
  -F "files=@mahdar_dabt.txt" \
  -F "files=@mahdar_istijwab.txt"
```

Files are saved to `Datasets/User_Cases/case_001/` and the full agent pipeline runs synchronously.

**Expected response:**
```json
{
  "success": true,
  "message": "Case processed successfully",
  "case_id": "case_001",
  "files_received": 3,
  "result": {
    "case_id": "case_001",
    "case_number": "...",
    "defendants": [...],
    "charges": [...],
    "suggested_verdict": {...},
    "verdict_operative": "...",
    ...
  }
}
```

---

### 2. Supported document file names

Each uploaded file is matched by its filename to a document type:

| Filename key | Type |
|---|---|
| `amr_ihala` | Referral order (أمر الإحالة) |
| `mahdar_dabt` | Arrest report (محضر الضبط) |
| `mahdar_istijwab` | Interrogation report (محضر الاستجواب) |
| `aqual_elshuhud` | Witness statements (أقوال الشهود) |
| `taqrir_tibbi` | Medical/lab report (تقرير طبي) |
| `mozakeret_difa` | Defense memorandum (مذكرة الدفاع) |
| `sgl_genaey` | Criminal record (السجل الجنائي) |

Files that do not match any known key are still saved but may not be processed.

---

## Running Only the Data Ingestion Agent

Useful for testing extraction without running the full pipeline:

```bash
curl -X POST "http://localhost:8000/agents/ingest_data" \
  -H "Content-Type: application/json" \
  -d '{"case_id": "case_001"}'
```

If `source_documents` is omitted, the agent looks in `Datasets/User_Cases/{case_id}/`.

---

## Inspecting Chunked Data

### Law article chunks
```bash
curl http://localhost:8000/Services/get_chunked_articles
```

### Na2d ruling and principle chunks
```bash
curl http://localhost:8000/Services/get_chunked_principles
```

---

## Knowledge Graph Operations

### Check graph statistics
```bash
curl http://localhost:8000/kg/statistics
```

### Query amendments for a specific law
```bash
curl "http://localhost:8000/kg/amendments?law_id=penal_code"
```

### Fetch a specific article
```bash
curl "http://localhost:8000/kg/penal_code/230"
```

### Rebuild the graph
```bash
curl -X POST "http://localhost:8000/kg/build?drop_existing=true"
```

### Invalidate all caches
```bash
curl -X DELETE "http://localhost:8000/kg/cache"
```

---

## KG Hybrid Retrieval

```bash
curl -X POST "http://localhost:8000/kg/retriever/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي عقوبة السرقة بالإكراه؟",
    "k": 10,
    "threshold": 0.4
  }'
```

---

## FAISS Vector Store Operations

### Build the index
```bash
curl -X POST "http://localhost:8000/vs/build"
```

### Load a previously saved index
```bash
curl -X POST "http://localhost:8000/vs/load?index_path=Datasets/na2d_faiss_index"
```

### Dense search
```bash
curl -X POST "http://localhost:8000/vs/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "عقوبة السرقة", "k": 5}'
```

### Hybrid retrieval (FAISS + BM25)
```bash
curl -X POST "http://localhost:8000/vs/retriever/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "عقوبة السرقة",
    "k": 5,
    "dense_weight": 0.6,
    "sparse_weight": 0.4
  }'
```

---

## Checking Component Status

```bash
# Is the FAISS vector store loaded?
curl http://localhost:8000/vs/status

# Is the KG retriever ready?
curl http://localhost:8000/kg/retriever/status
```
