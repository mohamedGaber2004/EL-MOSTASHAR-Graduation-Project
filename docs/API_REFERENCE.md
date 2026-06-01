# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## Root

### `GET /`

Returns a welcome message.

**Response:**
```json
{"message": "Welcome to the Graduation Project API!"}
```

---

## Cases — `/cases`

### `POST /cases/invoke_case`

Upload case documents and run the full 10-agent pipeline.

**Content-Type:** `multipart/form-data`

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `case_id` | string | Yes | Unique identifier for the case |
| `files` | file(s) | Yes | One or more `.txt` document files |

Files are saved to `Datasets/User_Cases/{case_id}/` before processing.

**Success response (200):**
```json
{
  "success": true,
  "message": "Case processed successfully",
  "case_id": "string",
  "files_received": 2,
  "result": { ...AgentState fields... }
}
```

**Error responses:**
- `400` — Missing `case_id` or no valid files uploaded
- `500` — Internal processing error

---

### `GET /cases/get_case_result`

Retrieve a cached case result by `case_id`.

**Query parameters:**

| Param | Type | Required |
|---|---|---|
| `case_id` | string | Yes |

**Response:** AgentState dict if found.

**Error responses:**
- `400` — Missing `case_id`
- `404` — No cached result for this `case_id`

---

## Agents Pipeline — `/agents`

### `POST /agents/ingest_data`

Run only the `DataIngestionAgent` on a case.

**Request body (JSON):**
```json
{
  "case_id": "string",
  "source_documents": ["path/to/dir"]
}
```

`source_documents` is optional. If omitted, the agent looks for `Datasets/User_Cases/{case_id}/`.

**Response:** Extracted AgentState fields as a dict.

**Error:** `500` on internal failure.

---

## Chunking — `/Services`

### `GET /Services/get_chunked_articles`

Return all law article chunks loaded by `CorpusChunker`.

**Response:**
```json
{
  "status_code": 200,
  "Length of articles": 1234,
  "content": [...]
}
```

---

### `GET /Services/get_chunked_principles`

Return all Na2d ruling and principle chunks.

**Response:**
```json
{
  "status_code": 200,
  "Length of principles": 567,
  "content": {...}
}
```

---

## Knowledge Graph — `/kg`

### `GET /kg/statistics`

Node and relationship counts. Cached for 60 seconds.

**Response:**
```json
{
  "nodes": {"Law": 9, "Article": 450, ...},
  "relationships": {"CONTAINS": 450, ...}
}
```

---

### `GET /kg/amendments`

List all amendments, optionally filtered by law.

**Query parameters:**

| Param | Type | Required | Description |
|---|---|---|---|
| `law_id` | string | No | Filter by law ID (e.g. `penal_code`) |

**Response:** Array of amendment objects with fields: `amendment_id`, `amendment_law_number`, `amendment_law_title`, `amendment_date`, `amendment_type`, `description`, `effective_date`, `affected_articles`, `law_id`, `law_title`.

---

### `GET /kg/{law_id}/{article_number}`

Fetch the latest text for a specific article.

**Path parameters:**

| Param | Description |
|---|---|
| `law_id` | Law identifier (e.g. `penal_code`) |
| `article_number` | Article number as string |

**Response:**
```json
{
  "law_id": "penal_code",
  "article_number": "5",
  "text": "..."
}
```

**Error:** `404` if article not found.

---

### `POST /kg/build`

Rebuild the Knowledge Graph in the background.

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `drop_existing` | bool | `true` | Drop existing graph data before rebuilding |

**Response:**
```json
{"message": "Knowledge Graph build started in background."}
```

**Error:** `409` if a build is already in progress.

---

### `DELETE /kg/cache`

Invalidate one or all router-level cache buckets.

**Query parameters:**

| Param | Type | Description |
|---|---|---|
| `name` | string | `statistics`, `amendments`, or `history`. Omit to clear all. |

**Response:**
```json
{
  "cleared": ["statistics"],
  "message": "Cleared 1 cache bucket(s) successfully."
}
```

**Error:** `400` if an unknown cache name is given.

---

## KG Retriever — `/kg/retriever`

### `GET /kg/retriever/status`

Check if the `LegalRetriever` singleton is initialised.

**Response:**
```json
{
  "retriever_ready": true,
  "bm25_article_count": 450
}
```

---

### `POST /kg/retriever/retrieve`

Hybrid legal retrieval: vector ANN + BM25 RRF + graph expansion.

**Request body:**
```json
{
  "question": "ما هي عقوبة السرقة؟",
  "k": 15,
  "threshold": 0.5
}
```

| Field | Type | Default | Range |
|---|---|---|---|
| `question` | string | required | — |
| `k` | int | 15 | 1–50 |
| `threshold` | float | 0.5 | 0.0–1.0 |

**Response:**
```json
{
  "query": "...",
  "context_text": "...",
  "sources": [...],
  "article_count": 3,
  "table_count": 1
}
```

**Error:** `404` if no results above threshold.

---

### `POST /kg/retriever/index/setup`

Create Neo4j vector indexes for `Article` and `TableChunk` nodes.

**Query parameters:**

| Param | Type | Description |
|---|---|---|
| `dimensions` | int | Override embedding dimensions (auto-inferred if omitted) |

**Response:**
```json
{"message": "Vector indexes created (or already existed).", "dimensions": 768}
```

**Error:** `409` on dimension mismatch.

---

### `POST /kg/retriever/index/embed`

Embed all un-embedded `Article` and `TableChunk` nodes.

**Query parameters:**

| Param | Type | Default |
|---|---|---|
| `batch_size` | int | 128 |
| `max_workers` | int | 4 |

**Response:**
```json
{"message": "Embedding complete.", "nodes_written": 120}
```

---

### `POST /kg/retriever/index/reindex`

Force re-embed all `Article` nodes (overwrites existing embeddings).

Same parameters as `/index/embed`.

---

### `POST /kg/retriever/index/rebuild`

Drop, recreate, and fully re-embed both Neo4j vector indexes.

**Query parameters:**

| Param | Type | Default |
|---|---|---|
| `batch_size` | int | 128 |
| `max_workers` | int | 4 |
| `dimensions` | int | auto |

**Response:**
```json
{"message": "Full vector index rebuild complete.", "nodes_written": 450}
```

---

## Vector Store — `/vs`

### `GET /vs/status`

Check whether a FAISS vector store is loaded.

**Response:**
```json
{"loaded": true}
```

---

### `POST /vs/build`

Build a FAISS index from the Na2d corpus (rulings + principles) and load it into memory.

**Query parameters:**

| Param | Type | Default |
|---|---|---|
| `index_path` | string | `na2d_faiss_index` |

**Response:**
```json
{
  "message": "FAISS Vector store constructed and synchronized successfully.",
  "rulings_count": 300,
  "principles_count": 267,
  "total_count": 567,
  "index_path": "na2d_faiss_index"
}
```

**Error:** `422` if corpus is empty, `503` on failure.

---

### `POST /vs/load`

Load a previously saved FAISS index into memory.

**Query parameters:**

| Param | Type | Default |
|---|---|---|
| `index_path` | string | `na2d_faiss_index` |

**Response:**
```json
{"message": "Vector store loaded from 'na2d_faiss_index'.", "index_path": "na2d_faiss_index"}
```

**Error:** `503` if the vector store is not loaded.

---

### `POST /vs/search`

Similarity search over rulings and principles.

**Request body:**
```json
{"query": "...", "k": 5}
```

**Response:**
```json
{
  "query": "...",
  "hits": [{"page_content": "...", "metadata": {...}}]
}
```

**Error:** `503` if vector store not loaded.

---

### `POST /vs/search/scored`

Similarity search returning L2 distance scores (lower = more similar).

**Request body:** same as `/vs/search`

**Response:**
```json
{
  "query": "...",
  "hits": [{"page_content": "...", "metadata": {...}, "score": 0.12}]
}
```

---

## VS Retriever — `/vs/retriever`

### `POST /vs/retriever/dense`

Semantic retrieval via FAISS dense vectors.

**Request body:**
```json
{"query": "...", "k": 5}
```

**Response:**
```json
{
  "query": "...",
  "strategy": "dense",
  "hits": [{"page_content": "...", "metadata": {...}}]
}
```

**Error:** `503` if vector store not loaded.

---

### `POST /vs/retriever/sparse`

Lexical retrieval via cached BM25 (Okapi).

**Request body:**
```json
{"query": "...", "k": 5}
```

**Response:**
```json
{"query": "...", "strategy": "sparse", "hits": [...]}
```

---

### `POST /vs/retriever/hybrid`

Hybrid retrieval: RRF fusion of FAISS and BM25.

**Request body:**
```json
{
  "query": "...",
  "k": 5,
  "dense_weight": 0.6,
  "sparse_weight": 0.4
}
```

| Field | Default | Range |
|---|---|---|
| `dense_weight` | 0.6 | 0.0–1.0 |
| `sparse_weight` | 0.4 | 0.0–1.0 |

At least one weight must be > 0.

**Response:**
```json
{
  "query": "...",
  "strategy": "hybrid(dense=0.60, sparse=0.40)",
  "hits": [...]
}
```

**Error:** `422` if both weights are 0.
