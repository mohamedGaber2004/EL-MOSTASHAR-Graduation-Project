# Quick Reference

## Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# or
docker compose up --build
```

API docs: `http://localhost:8000/docs`

---

## Run a case

```bash
curl -X POST "http://localhost:8000/cases/invoke_case" \
  -F "case_id=my_case" \
  -F "files=@amr_ihala.txt" \
  -F "files=@mahdar_dabt.txt"
```

---

## Common endpoints

| Action | Command |
|---|---|
| Run full pipeline | `POST /cases/invoke_case` |
| Run ingestion only | `POST /agents/ingest_data` |
| KG statistics | `GET /kg/statistics` |
| Build KG | `POST /kg/build?drop_existing=true` |
| Build FAISS index | `POST /vs/build` |
| Load FAISS index | `POST /vs/load` |
| Hybrid KG retrieval | `POST /kg/retriever/retrieve` |
| Hybrid VS retrieval | `POST /vs/retriever/hybrid` |
| VS status | `GET /vs/status` |
| KG retriever status | `GET /kg/retriever/status` |
| Clear KG caches | `DELETE /kg/cache` |

---

## Agent pipeline order

```
data_ingestion → procedural_auditor
  → (fatal nullity) → judge
  → legal_researcher → [evidence_analyst | confession_validity | witness_credibility]
    → prosecution_analyst → defense_analyst → sentencing → judge
```

---

## LLM providers

`open_router` | `groq` | `mistral` | `google` | `vertex` | `cloudflare` | `celebras`

Set per agent in `.env`: `<AGENT>_PROVIDER=open_router`

---

## Document type keys

`amr_ihala` · `mahdar_dabt` · `mahdar_istijwab` · `aqual_elshuhud` · `taqrir_tibbi` · `mozakeret_difa` · `sgl_genaey`

---

## Laws supported

`penal_code` · `criminal_procedure` · `anti_drugs` · `anti_terror` · `money_laundering` · `weapons_ammunition` · `criminal_constitution` · `emergency_law` · `cybercrime`

---

## Key config variables

```
INVOKATION_MAX_RETRIES=3
INTER_AGENTS_DELAY=1.0
INGESTION_AGENT_MAX_CHARS=8000
ARABIC_NATIVE_EMBEDDING_MODEL=<model>
NEO4J_URI=bolt://localhost:7687
FAISS_INDEX_PATH=Datasets/na2d_faiss_index
```
