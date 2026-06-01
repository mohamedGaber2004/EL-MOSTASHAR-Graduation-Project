# Troubleshooting

## Server won't start

**Error: `ValidationError` on startup**

Missing required environment variables. Check that all fields in `src/Config/config.py` that do not have a default are set in `.env`. Required fields include: `APP_NAME`, `APP_VERSION`, `DataPath`, `na2d_data_path`, `FAISS_INDEX_PATH`, `ARABIC_NATIVE_EMBEDDING_MODEL`, `GRPAH_RAG_MODEL`, all per-agent `*_MODEL`, `*_TEMP`, and agent behavior settings (`INVOKATION_MAX_RETRIES`, `INTER_AGENTS_DELAY`, `INGESTION_AGENT_MAX_CHARS`).

---

## Neo4j connection fails

The system logs a warning and continues without the Knowledge Graph:

```
NEO4J_URI not found — Knowledge Graph is disabled.
```
or
```
Knowledge Graph connection failed: ... — running without KG.
```

Agents that use the KG (`ProceduralAuditorAgent`, `LegalResearcherAgent`) will fall back to FAISS-only retrieval.

If you need Neo4j, verify:
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` are correct in `.env`
- The Neo4j instance is reachable from the server

---

## FAISS vector store not loaded

**`POST /vs/search` returns `503 Service Unavailable`**

The FAISS index has not been built or loaded. Run:

```bash
# Build from Na2d corpus
curl -X POST "http://localhost:8000/vs/build"

# Or load an existing saved index
curl -X POST "http://localhost:8000/vs/load?index_path=Datasets/na2d_faiss_index"
```

On the next server start, `shared_resources.get_vector_store()` will load the index automatically if `FAISS_INDEX_PATH/index.faiss` exists.

---

## KG retriever returns 503

The `LegalRetriever` singleton requires a connected Neo4j graph. If Neo4j is not available, the retriever cannot be initialized and the endpoint returns 503.

---

## LLM invocation failures

**Rate limit (429)**

The `_llm_invoke_with_retries` method handles 429 automatically with exponential back-off. If retries are exhausted, the error is recorded in `state.errors` and the agent returns an empty update (the graph continues).

**Empty response after retries**

Logged as `LLM returned empty content after N attempt(s)`. Increase `INVOKATION_MAX_RETRIES` in `.env` or check the model/provider configuration.

**OpenRouter 402 — insufficient credits**

Logged as `OpenRouter 402 — insufficient credits for <agent>`. No retries are attempted. Add credits to the OpenRouter account.

**Unknown agent key / provider**

`ValueError: Unknown agent key: '...'` — the agent's model config key is not in `AgentBase.AGENT_CONFIG`. Add it and the corresponding factory entry in `AgentBase._FACTORY_MAP`.

`ValueError: Unknown provider '...'` — the provider string is not in `AgentBase._PROVIDER_KEY_MAP`. Valid values: `open_router`, `groq`, `mistral`, `google`, `vertex`, `cloudflare`, `celebras`.

---

## JSON parsing failures

**`JSON parse/repair failed`**

The LLM returned output that could not be parsed even after `json-repair`. Check the raw LLM output in the logs. This may indicate the model is not following the JSON prompt format. Adjust the agent prompt or switch to a more capable model.

**`No valid entities found in JSON output`**

Data ingestion parsed the JSON but found no recognizable entity keys. The LLM output structure may not match the expected schema. Check the data ingestion prompt and the model response.

---

## Chunking produces 0 chunks

**`laws total: 0 chunks`**

The law folder names in `DataPath` do not match any entry in `LawKeys.FOLDER_TO_LAW_KEY`. Verify the folder names exactly match the keys: `3okobat`, `8asl_amwal`, `Asle7a`, `dostor_gena2y`, `drugs`, `egra2at_gena2ya`, `erhab`, `taware2`, `technology`.

**`na2d: rulings=0 | principles=0`**

The `na2d_data_path` directory is empty or contains no `.txt` files (for principles) and no sub-directories (for rulings).

---

## KG build returns 409

A KG build is already in progress. Wait for it to finish before sending another `POST /kg/build` request. The build runs in a FastAPI background task.

---

## Case processing hangs

Each agent calls `time.sleep(INTER_AGENTS_DELAY)` between agent invocations. With 10 agents and a 1-second delay this adds ~10 seconds. Rate-limit back-off can add significantly more. Monitor `src/logs/system.log` to track which agent is currently running.

---

## Logs

All logs are at `src/logs/system.log`. To follow in real time:

```bash
tail -f src/logs/system.log
```
