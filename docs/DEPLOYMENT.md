# Deployment

## Local

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker

```bash
docker compose up --build
```

**What `docker-compose.yml` does:**

- Builds the image from `Dockerfile` (Python 3.12-slim)
- Exposes port `8000`
- Loads environment variables from `.env`
- Mounts `./Datasets` into `/app/Datasets` (persists law data and FAISS index)
- Creates a named volume `hf_cache` mounted at `/cache/hf` to persist downloaded HuggingFace models
- Sets `PYTHONUNBUFFERED=1`, `HF_HOME`, `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`
- Runs `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Restart policy: `unless-stopped`

**Dockerfile summary:**

- Base image: `python:3.12-slim`
- System packages installed: `build-essential`, `gcc`, `curl`
- Dependencies installed from `requirements.txt`
- Working directory: `/app`
- Exposes port `8000`
- Default command: `uvicorn main:app --host 0.0.0.0 --port 8000`

---

## Environment variables

All required environment variables must be present in `.env` before starting the container. See `INSTALLATION_AND_SETUP.md` for the full list.

Neo4j is optional. If `NEO4J_URI` is not set, the Knowledge Graph is disabled and agents that use it fall back to FAISS-only retrieval.

---

## First-time setup after deployment

If running for the first time, follow these steps after the server is up:

1. **Build the FAISS index** (Na2d corpus):
   ```bash
   curl -X POST "http://localhost:8000/vs/build"
   ```

2. **Build the Knowledge Graph** (Neo4j, if configured):
   ```bash
   curl -X POST "http://localhost:8000/kg/build?drop_existing=true"
   ```

3. **Create Neo4j vector indexes**:
   ```bash
   curl -X POST "http://localhost:8000/kg/retriever/index/setup"
   ```

4. **Embed all nodes**:
   ```bash
   curl -X POST "http://localhost:8000/kg/retriever/index/embed"
   ```

On subsequent starts, the FAISS index is loaded automatically from `FAISS_INDEX_PATH` if the `index.faiss` file exists. The KG connection and BM25 index are initialized lazily on first request.
