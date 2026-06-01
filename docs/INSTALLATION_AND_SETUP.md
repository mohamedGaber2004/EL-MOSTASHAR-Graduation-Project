# Installation & Setup

## Prerequisites

- Python 3.12
- `pip`
- A Neo4j instance (optional — system degrades gracefully without it)
- API keys for at least one LLM provider

---

## 1. Clone and install dependencies

```bash
git clone <repo-url>
cd GraduationProject
pip install -r requirements.txt
```

---

## 2. Create the `.env` file

Create a `.env` file in the project root. All settings are loaded via `pydantic-settings` from `src/Config/config.py`.

```env
# App
APP_NAME=EL-MOSTASHAR
APP_VERSION=1.0.0

# Data paths
DataPath=Datasets/laws
na2d_data_path=Datasets/na2d
FAISS_INDEX_PATH=Datasets/na2d_faiss_index

# Embedding model (HuggingFace)
ARABIC_NATIVE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# GraphRAG model
GRPAH_RAG_MODEL=<model-name>

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# LLM API keys (set the ones you use)
GROQ_API_KEY=
GOOGLE_API_KEY=
OPENAI_BASE_URL=
OPENAI_BASE_ROUTER_API_KEY=
MISTRAL_API_KEY=
CLOUDFLARE_API_KEY=
CLOUDFLARE_BASE_URL=
CELEBRAS_API_KEY=
HUGGINGFACE_API_TOKEN=
COHERE_API_KEY=

# LangSmith (optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=
LANGSMITH_PROJECT=

# Agent retry / delay settings
INVOKATION_MAX_RETRIES=3
INTER_AGENTS_DELAY=1.0
INGESTION_AGENT_MAX_CHARS=8000

# Per-agent model configuration
# Provider options: open_router | groq | mistral | google | vertex | cloudflare | celebras

DATA_INGESTION_MODEL=<model-name>
DATA_INGESTION_TEMP=0.0
DATA_INGESTION_PROVIDER=open_router

PROCEDURAL_AUDITOR_MODEL=<model-name>
PROCEDURAL_AUDITOR_TEMP=0.0
PROCEDURAL_AUDITOR_PROVIDER=open_router

LEGAL_RESEARCHER_MODEL=<model-name>
LEGAL_RESEARCHER_TEMP=0.0
LEGAL_RESEARCHER_PROVIDER=open_router

EVIDENCE_SCORING_MODEL=<model-name>
EVIDENCE_SCORING_TEMP=0.0
EVIDENCE_SCORING_PROVIDER=open_router

DEFENSE_AGENT_MODEL=<model-name>
DEFENSE_AGENT_TEMP=0.0
DEFENSE_AGENT_PROVIDER=open_router

CONFESSION_VALIDITY_MODEL=<model-name>
CONFESSION_VALIDITY_TEMP=0.0
CONFESSION_VALIDITY_PROVIDER=open_router

WITNESS_CREDIBILITY_MODEL=<model-name>
WITNESS_CREDIBILITY_TEMP=0.0
WITNESS_CREDIBILITY_PROVIDER=open_router

PROSECUTION_ANALYST_MODEL=<model-name>
PROSECUTION_ANALYST_TEMP=0.0
PROSECUTION_ANALYST_PROVIDER=open_router

SENTENCING_MODEL=<model-name>
SENTENCING_TEMP=0.0
SENTENCING_PROVIDER=open_router

JUDGE_MODEL=<model-name>
JUDGE_TEMP=0.0
JUDGE_PROVIDER=open_router
```

---

## 3. Prepare the dataset directories

```
Datasets/
├── laws/               # Law folders (see chunking_enums.py LawKeys)
│   ├── 3okobat/
│   ├── 8asl_amwal/
│   ├── Asle7a/
│   ├── dostor_gena2y/
│   ├── drugs/
│   ├── egra2at_gena2ya/
│   ├── erhab/
│   ├── taware2/
│   └── technology/
├── na2d/               # Na2d corpus
│   ├── *.txt           # Principle books
│   └── <crime-folder>/ # Ruling folders containing *.txt files
└── User_Cases/         # Created automatically when cases are uploaded
```

---

## 4. Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5. Run with Docker

```bash
docker compose up --build
```

The `docker-compose.yml` mounts `./Datasets` and persists HuggingFace model downloads in a named volume `hf_cache`.

---

## 6. First-time Knowledge Graph setup (if using Neo4j)

After the server is running, call these endpoints in order:

```bash
# 1. Build the KG from law chunks (runs in background)
curl -X POST "http://localhost:8000/kg/build?drop_existing=true"

# 2. Create vector indexes in Neo4j
curl -X POST "http://localhost:8000/kg/retriever/index/setup"

# 3. Embed all Article and TableChunk nodes
curl -X POST "http://localhost:8000/kg/retriever/index/embed"
```

---

## 7. First-time Vector Store setup (Na2d FAISS)

```bash
# Build and save the FAISS index
curl -X POST "http://localhost:8000/vs/build"
```

After this, the index is saved to `FAISS_INDEX_PATH` and will be loaded automatically on the next server start (via `get_vector_store()` in `shared_resources.py`).

---

## Logs

Application logs are written to `src/logs/system.log` and printed to the console.
