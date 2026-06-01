# Architecture

## Overview

The system is a single FastAPI application (`main.py`) that exposes a set of REST endpoints. Case processing is done through a LangGraph `StateGraph` containing 10 agents. Supporting services (Neo4j Knowledge Graph and FAISS Vector Store) are optional — the system degrades gracefully if either is unavailable.

---

## Layers

```
HTTP Client
    │
    ▼
FastAPI (main.py)
    │
    ├── Routers (src/routers/)
    │       case_router, data_ingestion_router, chunking_router,
    │       kg_router, kg_retriever_router, vs_router, vs_retriever_router
    │
    ├── Agent Graph (src/Graph/)
    │       StateGraph with 10 agents (src/agents/)
    │
    ├── Knowledge Graph (src/Graphstore/)
    │       LegalKnowledgeGraph → Neo4j
    │
    ├── Vector Store (src/Vectorstore/ + src/retriever/)
    │       FAISS index + BM25 in-memory index
    │
    └── Corpus (src/Chunking/)
            Law articles + Na2d rulings + principles
```

---

## Agent Graph

Built in `src/Graph/graph_builder.py` using `build_legal_graph()`.

### Nodes

| Node key | Agent class | Role |
|---|---|---|
| `data_ingestion` | `DataIngestionAgent` | Extract structured entities from uploaded documents |
| `procedural_auditor` | `ProceduralAuditorAgent` | Detect procedural violations and nullities |
| `legal_researcher` | `LegalResearcherAgent` | Retrieve applicable laws and judicial principles |
| `evidence_analyst` | `EvidenceAnalystAgent` | Score each piece of evidence |
| `confession_validity` | `ConfessionValidityAgent` | Assess the validity of each confession |
| `witness_credibility` | `WitnessCredibilityAgent` | Assess the credibility of each witness |
| `prosecution_analyst` | `ProsecutionAnalystAgent` | Build the prosecution theory and arguments |
| `defense_analyst` | `DefenseAnalystAgent` | Analyse defense memoranda and arguments |
| `sentencing` | `SentencingAgent` | Determine aggravating/mitigating factors and sentencing |
| `judge` | `JudgeAgent` | Produce the final verdict and reasoning |

### Edges

```
START → data_ingestion → procedural_auditor
procedural_auditor → legal_researcher          (normal path)
procedural_auditor → judge                     (fatal nullity short-circuit)
legal_researcher → evidence_analyst            (parallel fan-out)
legal_researcher → confession_validity         (parallel fan-out)
legal_researcher → witness_credibility         (parallel fan-out)
evidence_analyst    → prosecution_analyst      (parallel fan-in)
confession_validity → prosecution_analyst      (parallel fan-in)
witness_credibility → prosecution_analyst      (parallel fan-in)
prosecution_analyst → defense_analyst → sentencing → judge → END
```

### Conditional routing

`_route_after_procedural_audit` checks `audit.critical_nullities` and each `violation.issue_description` / `violation.nullity_type` for fatal keywords: `سقوط`, `تقادم`, `بطلان مطلق`, `عدم اختصاص`, `انقضاء الدعوى`, `وفاة المتهم`.

### Error isolation

Every node is wrapped by `_safe_run()`. If a node raises an exception, the error is recorded in `state.errors` and execution continues.

---

## Shared State (`AgentState`)

Defined in `src/Graph/state.py` as a Pydantic `BaseModel`. All agents read from and write to this single object.

Main field groups:

| Group | Fields | Set by |
|---|---|---|
| Case meta | `case_id`, `case_number`, `court`, `court_level`, `jurisdiction`, `filing_date`, `referral_date`, `prosecutor_name`, `referral_order_text` | DataIngestionAgent |
| Entities | `defendants`, `charges`, `incidents`, `evidences`, `witness_statements`, `confessions`, `lab_reports`, `criminal_records`, `criminal_proceedings`, `defense_documents` | DataIngestionAgent |
| Procedural | `procedural_audit` | ProceduralAuditorAgent |
| Evidence | `evidence_scores`, `chain_of_custody_issues` | EvidenceAnalystAgent |
| Legal research | `applied_principles`, `case_articles` | LegalResearcherAgent |
| Confessions | `confession_assessments` | ConfessionValidityAgent |
| Witnesses | `witness_credibility_scores` | WitnessCredibilityAgent |
| Prosecution | `prosecution_theory`, `prosecution_arguments` | ProsecutionAnalystAgent |
| Defense | `defense_arguments` | DefenseAnalystAgent |
| Sentencing | `aggravating_factors`, `mitigating_factors`, `applicable_article_17`, `charge_conviction_map`, `civil_claim`, `civil_award_suggested` | SentencingAgent |
| Verdict | `suggested_verdict`, `verdict_preamble`, `verdict_facts`, `verdict_reasoning`, `verdict_operative` | JudgeAgent |
| Status | `completed_agents`, `errors`, `current_agent`, `last_updated` | All agents |
| Meta | `source_documents`, `extraction_notes` | Set at invocation |

Fields annotated with `Annotated[List[...], add]` are merged (not overwritten) during parallel execution.

---

## AgentBase

All 10 agents extend `AgentBase` (`src/agents/agent_base/agent_base.py`).

**Responsibilities:**
- LLM initialization via a factory map (`_init_llm`). Supported providers: `open_router`, `groq`, `mistral`, `google`, `vertex`, `cloudflare`, `celebras`.
- LLM invocation with retry logic (`_llm_invoke_with_retries`): exponential back-off, rate-limit handling (HTTP 429), Retry-After header parsing.
- JSON output parsing: `_parse_llm_json` (for data ingestion, includes entity validation), `_parse_agent_json` (for all other agents), `_strip_md_fences`.
- Hybrid retrieval over the Na2d FAISS corpus (`hybrid_retrieve_logic`, `retrieve_principles`).
- State update helper (`_empty_update`).

---

## Knowledge Graph

Implemented in `src/Graphstore/KG_builder.py` using the `neo4j` Python driver directly.

**Build pipeline (`build_knowledge_graph`):**
1. Extract law summaries from chunks (`ingest_dataset` → `LawExtractor`)
2. Import laws, articles, penalties, definitions, topics, references into Neo4j
3. Extract and import amendments (`ingest_amendments` → `AmendmentExtractor`)
4. Import tables as `Table` + `TableChunk` nodes

**Querying:**
- `get_statistics()` — node and relationship counts
- `query_amendments(law_id?)` — list amendments
- `get_article(law_id, article_number)` — fetch latest article text

---

## KG Retrieval (`LegalRetriever`)

Defined in `src/retriever/kg_retriever/kg_retriever.py`.

**Retrieval steps:**
1. Embed query with HuggingFace embeddings
2. ANN search over `Article` nodes (Neo4j vector index)
3. ANN search over `TableChunk` nodes (Neo4j vector index)
4. BM25 search over all articles (`ArticleBM25Index` using `rank_bm25`)
5. RRF merge of BM25 + vector article hits
6. Graph expansion for each merged article (penalties, amendments, definitions, topics, referenced articles, tables)
7. Filter by score threshold
8. Budget-aware context assembly (max 9,000 characters)

---

## Vector Store (`FAISS`)

Built from the Na2d corpus (rulings + principles) using `langchain-community` FAISS.

Three retrieval modes via `src/retriever/vs_retriever/vs_reriever.py`:
- **Dense** — FAISS `as_retriever`
- **Sparse** — `BM25Retriever` from `langchain-community`
- **Hybrid** — `EnsembleRetriever` (FAISS + BM25, configurable weights, default 0.6 / 0.4)

---

## Corpus Chunking

`CorpusChunker` in `src/Chunking/chunking.py` processes two corpora:

**Laws** (`laws_dir`): 9 law folders → article chunks, preamble chunks, amended-article chunks, table chunks.

**Na2d** (`na2d_dir`): ruling folders → ruling chunks; `.txt` files at root → principle-book chunks.

Chunk sizes are word-based (`RecursiveCharacterTextSplitter` with `length_function=len(text.split())`):
- Principle books: 400 words, 50 overlap
- Rulings: 300 words, 40 overlap

---

## Singletons and Caching

| Singleton | Location | Cache |
|---|---|---|
| `LegalKnowledgeGraph` | `src/routers/kg_router.py` `get_graph()` | Thread-safe double-checked locking |
| `LegalRetriever` | `src/routers/kg_retriever_router.py` `get_kg_retriever()` | Thread-safe double-checked locking |
| FAISS vector store | `src/routers/vs_router.py` module-level `_vs` | Thread-safe lock |
| BM25 sparse retriever | `src/routers/vs_retriever_router.py` | Thread-safe lock |
| Na2d chunks | `src/agents/agent_base/agent_base.py` `_NA2D_CACHE` | Module-level global |
| KG statistics cache | `kg_router.py` `_STATS_CACHE` | TTL 60s (`cachetools.TTLCache`) |
| Amendments cache | `kg_router.py` `_AMENDMENT_CACHE` | TTL 300s |
| Article history cache | `kg_router.py` `_HISTORY_CACHE` | TTL 300s |
| LangGraph graph | `graph_builder.py` `build_legal_graph()` | `lru_cache(maxsize=1)` |

---

## Logging

Configured in `src/Config/log_config.py`. On import, sets up the root logger with:
- `FileHandler` → `src/logs/system.log` (UTF-8)
- `StreamHandler` → console
- Level: `INFO`
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

LangSmith tracing is optionally enabled via `LANGSMITH_*` environment variables set in `main.py`.
