# EL-MOSTASHAR — Egyptian Legal Multi-Agent System

A FastAPI-based AI system that processes Egyptian criminal case files through a pipeline of 10 LangGraph agents to produce a simulated legal verdict.

---

## What It Does

1. Accepts uploaded case documents (Arabic `.txt` files) via a REST API.
2. Extracts structured case data (defendants, charges, evidence, witnesses, confessions, etc.).
3. Passes the extracted data through a sequential/parallel agent pipeline.
4. Produces a final suggested verdict with reasoning.

---

## Project Structure

```
.
├── main.py                        # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env                           # Environment variables (not committed)
├── Datasets/                      # Law texts, Na2d corpus, uploaded case files
└── src/
    ├── pipeline.py                # PipelineManager (FAISS init + run_case)
    ├── Config/
    │   ├── config.py              # Settings (pydantic-settings)
    │   └── log_config.py          # Logging setup (file + console)
    ├── Graph/
    │   ├── state.py               # AgentState (shared LangGraph state)
    │   ├── graph_builder.py       # build_legal_graph(), run_case()
    │   └── shared_resources.py    # get_kg(), get_vector_store() singletons
    ├── agents/
    │   ├── agent_base/            # AgentBase class, enums, prompts
    │   ├── data_ingestion_agent/
    │   ├── procedural_auditor_agent/
    │   ├── legal_research_agent/
    │   ├── evidence_analyst_agent/
    │   ├── confession_validity_agent/
    │   ├── witness_credibility_agent/
    │   ├── prosecution_analyst_agent/
    │   ├── defense_analyst_agent/
    │   ├── sentencing_agent/
    │   └── judge_agent/
    ├── Chunking/
    │   ├── chunking.py            # CorpusChunker (laws + Na2d)
    │   └── chunking_enums.py      # ChunkType, LawKeys, Na2dOutputKey, etc.
    ├── Graphstore/
    │   ├── KG_builder.py          # LegalKnowledgeGraph + build_knowledge_graph()
    │   └── kg_enums.py            # NodeLabel, RelType, KgSchemas, etc.
    ├── Vectorstore/
    │   └── vector_store_builder.py # build_vector_store(), load_vector_store(), search()
    ├── retriever/
    │   ├── kg_retriever/
    │   │   └── kg_retriever.py    # LegalRetriever (vector + BM25 + graph expansion)
    │   └── vs_retriever/
    │       └── vs_reriever.py     # get_dense_retriever(), get_sparse_retriever(), get_hybrid_retriever()
    ├── routers/
    │   ├── case_router.py         # POST /cases/invoke_case
    │   ├── data_ingestion_router.py
    │   ├── chunking_router.py
    │   ├── kg_router.py           # Knowledge Graph CRUD + build
    │   ├── kg_retriever_router.py # KG hybrid retrieval + index management
    │   ├── vs_router.py           # FAISS vector store build/load/search
    │   └── vs_retriever_router.py # Dense / sparse / hybrid retrieval
    └── Utils/
        └── file_utils/            # Text loading, normalization, extractors
```

---

## Agent Pipeline

The pipeline is a LangGraph `StateGraph` with 10 nodes:

```
START
  └─► data_ingestion
        └─► procedural_auditor
              ├─► (fatal nullity) ──────────────────────────────────────► judge
              └─► legal_researcher
                    ├─► evidence_analyst ──┐
                    ├─► confession_validity ┼─► prosecution_analyst
                    └─► witness_credibility ┘       └─► defense_analyst
                                                           └─► sentencing
                                                                 └─► judge
                                                                       └─► END
```

**Conditional routing:** After `procedural_auditor`, if a fatal nullity (e.g., statute of limitations, absolute nullity, lack of jurisdiction) is detected, the case skips directly to `judge`.

**Parallel nodes:** `evidence_analyst`, `confession_validity`, and `witness_credibility` all run in parallel (fan-out from `legal_researcher`, fan-in to `prosecution_analyst`).

---

## Supported Document Types

The `DataIngestionAgent` reads `.txt` files with these recognized names:

| File key | Description |
|---|---|
| `amr_ihala` | Referral order |
| `mahdar_dabt` | Arrest report |
| `mahdar_istijwab` | Interrogation report |
| `aqual_elshuhud` | Witness statements |
| `taqrir_tibbi` | Medical/lab report |
| `mozakeret_difa` | Defense memorandum |
| `sgl_genaey` | Criminal record |

---

## Knowledge Graph Schema (Neo4j)

**Node labels:** `Law`, `Article`, `Penalty`, `Definition`, `Topic`, `Amendment`, `Table`, `TableChunk`

**Relationships:** `CONTAINS`, `HAS_PENALTY`, `DEFINES`, `TAGGED_WITH`, `REFERENCES`, `AMENDED_BY`, `SUPERSEDES`, `HAS_TABLE`, `HAS_CHUNK`, `HAS_AMENDMENT`

**Laws indexed:**

| Folder | Law |
|---|---|
| `3okobat` | قانون العقوبات |
| `8asl_amwal` | قانون مكافحة غسل الأموال |
| `Asle7a` | قانون الأسلحة والذخيرة |
| `dostor_gena2y` | الدستور الجنائي |
| `drugs` | قانون مكافحة المخدرات |
| `egra2at_gena2ya` | قانون الإجراءات الجنائية |
| `erhab` | قانون مكافحة الإرهاب |
| `taware2` | قانون الطوارئ |
| `technology` | قانون مكافحة جرائم تقنية المعلومات |

---

## Running Locally

### Prerequisites

- Python 3.12
- A `.env` file with all required variables (see `src/Config/config.py`)
- Neo4j instance (optional — system degrades gracefully if unavailable)

### Install & run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Overview

| Method | Path | Description |
|---|---|---|
| `POST` | `/cases/invoke_case` | Upload case files and run the full pipeline |
| `GET` | `/cases/get_case_result` | Get cached result by case_id |
| `POST` | `/agents/ingest_data` | Run only the DataIngestionAgent |
| `GET` | `/Services/get_chunked_articles` | Return law article chunks |
| `GET` | `/Services/get_chunked_principles` | Return Na2d ruling/principle chunks |
| `POST` | `/kg/build` | Rebuild the Knowledge Graph (background) |
| `GET` | `/kg/statistics` | Node and relationship counts |
| `GET` | `/kg/amendments` | Query amendments |
| `GET` | `/kg/{law_id}/{article_number}` | Fetch a specific article |
| `DELETE` | `/kg/cache` | Invalidate KG router caches |
| `POST` | `/kg/retriever/retrieve` | Hybrid KG retrieval (vector + BM25 + graph) |
| `GET` | `/kg/retriever/status` | Check if LegalRetriever is initialised |
| `POST` | `/kg/retriever/index/setup` | Create Neo4j vector indexes |
| `POST` | `/kg/retriever/index/embed` | Embed unindexed nodes |
| `POST` | `/kg/retriever/index/reindex` | Force re-embed all articles |
| `POST` | `/kg/retriever/index/rebuild` | Drop + recreate + re-embed indexes |
| `GET` | `/vs/status` | Check if FAISS vector store is loaded |
| `POST` | `/vs/build` | Build FAISS index from Na2d corpus |
| `POST` | `/vs/load` | Load a saved FAISS index |
| `POST` | `/vs/search` | Similarity search |
| `POST` | `/vs/search/scored` | Similarity search with L2 scores |
| `POST` | `/vs/retriever/dense` | Dense (FAISS) retrieval |
| `POST` | `/vs/retriever/sparse` | Sparse (BM25) retrieval |
| `POST` | `/vs/retriever/hybrid` | Hybrid (FAISS + BM25 RRF) retrieval |

---

## Environment Variables

See `src/Config/config.py` for the full list. Key variables:

```
APP_NAME, APP_VERSION
DataPath, na2d_data_path, FAISS_INDEX_PATH
ARABIC_NATIVE_EMBEDDING_MODEL
GRPAH_RAG_MODEL
NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_BASE_URL, OPENAI_BASE_ROUTER_API_KEY
MISTRAL_API_KEY, CLOUDFLARE_API_KEY, CLOUDFLARE_BASE_URL, CELEBRAS_API_KEY
LANGSMITH_API_KEY, LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_PROJECT
INVOKATION_MAX_RETRIES, INTER_AGENTS_DELAY, INGESTION_AGENT_MAX_CHARS
DATA_INGESTION_MODEL, DATA_INGESTION_TEMP, DATA_INGESTION_PROVIDER
PROCEDURAL_AUDITOR_MODEL, PROCEDURAL_AUDITOR_TEMP, PROCEDURAL_AUDITOR_PROVIDER
LEGAL_RESEARCHER_MODEL, LEGAL_RESEARCHER_TEMP, LEGAL_RESEARCHER_PROVIDER
EVIDENCE_SCORING_MODEL, EVIDENCE_SCORING_TEMP, EVIDENCE_SCORING_PROVIDER
DEFENSE_AGENT_MODEL, DEFENSE_AGENT_TEMP, DEFENSE_AGENT_PROVIDER
CONFESSION_VALIDITY_MODEL, CONFESSION_VALIDITY_TEMP, CONFESSION_VALIDITY_PROVIDER
WITNESS_CREDIBILITY_MODEL, WITNESS_CREDIBILITY_TEMP, WITNESS_CREDIBILITY_PROVIDER
PROSECUTION_ANALYST_MODEL, PROSECUTION_ANALYST_TEMP, PROSECUTION_ANALYST_PROVIDER
SENTENCING_MODEL, SENTENCING_TEMP, SENTENCING_PROVIDER
JUDGE_MODEL, JUDGE_TEMP, JUDGE_PROVIDER
```

Each agent's provider can be: `open_router`, `groq`, `mistral`, `google`, `vertex`, `cloudflare`, or `celebras`.

---

## Logs

Logs are written to `src/logs/system.log` and also streamed to the console. Format:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```
