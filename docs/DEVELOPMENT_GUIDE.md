# Development Guide

## Project layout

```
src/
├── Config/             # Settings and logging
├── Graph/              # LangGraph state and graph builder
├── agents/             # 10 agents, each in its own directory
├── Chunking/           # CorpusChunker for laws and Na2d
├── Graphstore/         # LegalKnowledgeGraph (Neo4j)
├── Vectorstore/        # FAISS vector store builder
├── retriever/
│   ├── kg_retriever/   # LegalRetriever (hybrid KG retrieval)
│   └── vs_retriever/   # Dense / sparse / hybrid FAISS retrieval
├── routers/            # FastAPI routers
└── Utils/              # File loading, text normalization, law extractors
```

---

## Adding a new agent

1. Create a new directory under `src/agents/` (e.g. `my_new_agent/`).

2. Add an output model file (Pydantic `BaseModel`).

3. Add a prompt file.

4. Create the agent class extending `AgentBase`:

```python
from src.agents.agent_base.agent_base import AgentBase

class MyNewAgent(AgentBase):
    def __init__(self):
        super().__init__(
            model_config_key="MY_NEW_AGENT_MODEL",
            temp_config_key="MY_NEW_AGENT_TEMP",
            prompt=MY_NEW_AGENT_PROMPT,
        )

    def run(self, state) -> dict:
        # build messages, call self._llm_invoke_with_retries, parse, return dict
        ...
```

5. Add `MY_NEW_AGENT_MODEL`, `MY_NEW_AGENT_TEMP`, and `MY_NEW_AGENT_PROVIDER` to `.env` and to `src/Config/config.py`.

6. Add the agent's model key to `AgentBase.AGENT_CONFIG` and `AgentBase._FACTORY_MAP`.

7. Create the corresponding LLM factory in `src/LLMs/`.

8. Add the node and edges in `src/Graph/graph_builder.py`.

9. Add any new output fields to `AgentState` in `src/Graph/state.py`.

---

## Adding a new law

1. Create a folder under `DataPath` (e.g. `Datasets/laws/my_new_law/`).

2. Place the law text in one or more `.txt` files inside that folder.
   - Articles file: any `.txt` not starting with `new`.
   - Amendment files: named `new*.txt`.
   - Tables file: name must contain `tables`.

3. Add the folder name and law ID to `LawKeys.FOLDER_TO_LAW_KEY` in `src/Chunking/chunking_enums.py`.

4. Add the human-readable title to `LawKeys.LAW_KEY_TO_TITLE`.

5. Rebuild the KG:
   ```bash
   curl -X POST "http://localhost:8000/kg/build?drop_existing=false"
   ```

---

## LLM providers

Each agent has a configurable provider. The mapping between provider name and LLM key is in `AgentBase._PROVIDER_KEY_MAP`:

| Provider string | Key in factory dict |
|---|---|
| `open_router` | `open_router_llm` |
| `groq` | `groq_llm` |
| `mistral` | `mistral_llm` |
| `google` | `google_llm` |
| `vertex` | `vertex_llm` |
| `cloudflare` | `cloudflare_llm` |
| `celebras` | `celebras_llm` |

Each agent's LLM factory is in `src/LLMs/<AGENT>_MODEL.py` and returns a dict of `{provider_key: llm_instance}`.

---

## Retry behavior

`AgentBase._llm_invoke_with_retries` retries on:
- Empty LLM response
- HTTP 429 / rate-limit errors (keywords: `429`, `rate_limit`, `rate_limit_exceeded`, `tokens per minute`, `please try again`)

On rate-limit: uses `Retry-After` seconds if present in the error, otherwise exponential back-off (base 2.0, capped at 60s) plus random jitter.

On OpenRouter 402 (insufficient credits): raises immediately without retrying.

Max retries is controlled by `INVOKATION_MAX_RETRIES` in `.env`.

---

## JSON parsing

Two parsers are available in `AgentBase`:

| Method | Use case |
|---|---|
| `_parse_llm_json` | `DataIngestionAgent` — also runs entity validation via Pydantic models |
| `_parse_agent_json` | All other agents — parses JSON, no entity validation |

Both strip markdown fences (` ```json `) and run `json-repair` before parsing.

---

## State merging in parallel nodes

Fields annotated with `Annotated[List[...], add]` in `AgentState` are merged by LangGraph rather than overwritten. This is used for:

- `evidence_scores`
- `chain_of_custody_issues`
- `confession_assessments`
- `witness_credibility_scores`
- `completed_agents`
- `errors`

Nodes running in parallel must not write to non-additive state fields. This is enforced by `_safe_run` which strips keys in `_PARALLEL_FORBIDDEN_KEYS` from parallel node outputs.

---

## Logging

Import from `src/Config/log_config.py`:

```python
from src.Config.log_config import logging
logger = logging.getLogger(__name__)
```

Logs go to `src/logs/system.log` and to the console.
