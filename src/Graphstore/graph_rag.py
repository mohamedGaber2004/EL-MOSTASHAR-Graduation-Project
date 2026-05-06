from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Any, Dict, List, Optional
from pydantic import BaseModel , Field
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from src.Graphstore.KG_builder import LegalKnowledgeGraph

logger = logging.getLogger(__name__)

# =============================================================================
# 0. Queries
# =============================================================================

TABLE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX table_chunk_embeddings IF NOT EXISTS
FOR (t:TableChunk)
ON (t.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`:          $dimensions,
        `vector.similarity_function`: 'cosine'
    }
}
"""

VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX article_embeddings IF NOT EXISTS
FOR (a:Article)
ON (a.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`:   $dimensions,
        `vector.similarity_function`: 'cosine'
    }
}
"""

ANN_QUERY = """
CALL db.index.vector.queryNodes('article_embeddings', $k, $query_vector)
YIELD node AS article, score
RETURN
    article.article_id     AS article_id,
    article.article_number AS article_number,
    article.law_id         AS law_id,
    article.text           AS text,
    score
ORDER BY score DESC
"""

TABLE_ANN_QUERY = """
CALL db.index.vector.queryNodes('table_chunk_embeddings', $k, $query_vector)
YIELD node AS tbl, score
RETURN
    tbl.chunk_id      AS chunk_id,
    tbl.table_id      AS table_id,
    tbl.table_number  AS table_number,
    tbl.law_id        AS law_id,
    tbl.text          AS text,
    score
ORDER BY score DESC
"""

EXPAND_QUERY = """
MATCH (a:Article {article_id: $article_id})
OPTIONAL MATCH (l:Law {law_id: a.law_id})
OPTIONAL MATCH (a)-[:HAS_PENALTY]->(p:Penalty)
OPTIONAL MATCH (a)-[:REFERENCES]->(ref:Article)
OPTIONAL MATCH (l)-[:HAS_TABLE]->(tb:Table)
OPTIONAL MATCH (a)-[:AMENDED_BY]->(am:Amendment)
OPTIONAL MATCH (a)-[:DEFINES]->(d:Definition)
OPTIONAL MATCH (a)-[:TAGGED_WITH]->(t:Topic)
RETURN
    coalesce(l.title, 'غير محدد')                               AS law_title,
    l.promulgation_date                                         AS promulgation_date,
    collect(DISTINCT {type: p.penalty_type, id: p.penalty_id}) AS penalties,
    collect(DISTINCT {
        article_number: ref.article_number,
        law_id: ref.law_id,
        text: ref.text
    })                                                          AS referenced_articles,
    collect(DISTINCT {
        table_id: tb.table_id,
        table_number: tb.table_number
    })                                                          AS tables,
    collect(DISTINCT {
        id: am.amendment_id,
        type: am.amendment_type,
        date: am.amendment_date,
        law_number: am.amendment_law_number,
        description: am.description
    })                                                          AS amendments,
    collect(DISTINCT {
        term: d.term,
        definition: d.definition_text
    })                                                          AS definitions,
    collect(DISTINCT t.name)                                    AS topics
"""

# =============================================================================
# 1. VECTOR INDEX SETUP
# =============================================================================

def _infer_embedding_dimension(embeddings: Embeddings) -> int:
    sample = embeddings.embed_query("__dimension_check__")
    if not isinstance(sample, list) or not sample:
        raise ValueError("Unable to infer embedding dimension.")
    return len(sample)

def _get_existing_index_dimension(driver, index_name: str = "article_embeddings") -> Optional[int]:
    """
    Checks the existing vector index in Neo4j 5.x+ to retrieve its dimensions.
    """
    with driver.session() as s:
        rec = s.run(
            "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
            name=index_name,
        ).single()
    
    if not rec:
        return None
        
    options = rec.get("options") or {}
    index_config = options.get("indexConfig", {})
    
    # Vector dimensions are stored under the 'vector.dimensions' key
    dim = index_config.get("vector.dimensions")
    
    return int(dim) if dim is not None else None

def _drop_vector_index(driver, index_name: str = "article_embeddings") -> None:
    with driver.session() as s:
        s.run(f"DROP INDEX {index_name} IF EXISTS")
    logger.info("Dropped vector index '%s'", index_name)

def create_vector_index(driver, dimensions: int) -> None:
    with driver.session() as s:
        s.run(VECTOR_INDEX_QUERY, dimensions=dimensions)
    logger.info("Vector index 'article_embeddings' ready (dim=%d)", dimensions)

def create_table_vector_index(driver, dimensions: int) -> None:
    with driver.session() as s:
        s.run(TABLE_VECTOR_INDEX_QUERY, dimensions=dimensions)
    logger.info("Vector index 'table_chunk_embeddings' ready (dim=%d)", dimensions)

def embed_and_store_tables(driver, embeddings: Embeddings,
                           batch_size: int = 64, max_workers: int = 4) -> int:
    return _run_embedding(
        driver, embeddings,
        fetch_query="""
            MATCH (t:TableChunk)
            WHERE t.embedding IS NULL AND t.text IS NOT NULL
            RETURN t.chunk_id AS id, t.text AS text
        """,
        batch_size=batch_size,
        max_workers=max_workers,
        node_label="TableChunk",
        id_property="chunk_id",
    )

# =============================================================================
# 2. FAST PARALLEL BATCH EMBEDDING  (~1500 articles)
# =============================================================================

# graph_rag.py

def _run_embedding(
    driver,
    embeddings: Embeddings,
    fetch_query: str,
    batch_size: int = 128,
    max_workers: int = 4,
    node_label: str = "Article",      # ← add this
    id_property: str = "article_id",  # ← add this
) -> int:

    with driver.session() as s:
        records = s.run(fetch_query).data()

    if not records:
        logger.info("No %s nodes to embed.", node_label)
        return 0

    logger.info("%s nodes to embed: %d", node_label, len(records))
    batches = [records[i: i + batch_size] for i in range(0, len(records), batch_size)]

    encoded: Dict[int, List] = {}

    def _encode(idx, batch):
        return idx, embeddings.embed_documents([r["text"] for r in batch])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_encode, i, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            idx, vectors = fut.result()
            encoded[idx] = vectors
            logger.info("  Encoded batch %d/%d (%d nodes)",
                        idx + 1, len(batches), len(vectors))

    # ✅ Use parameterized label and id_property
    store_q = f"""
        UNWIND $rows AS row
        MATCH (n:{node_label} {{{id_property}: row.id}})
        SET n.embedding = row.embedding
    """

    total = 0
    for idx, batch in enumerate(batches):
        rows = [{"id": r["id"], "embedding": v}
                for r, v in zip(batch, encoded[idx])]
        with driver.session() as s:
            s.run(store_q, rows=rows)
        total += len(rows)

    logger.info("Embedding complete — %d %s nodes written", total, node_label)
    return total

def embed_and_store_articles(driver, embeddings, batch_size=128, max_workers=4):
    return _run_embedding(
        driver, embeddings,
        fetch_query="""
            MATCH (a:Article)
            WHERE a.embedding IS NULL AND a.text IS NOT NULL
            RETURN a.article_id AS id, a.text AS text
        """,
        batch_size=batch_size, max_workers=max_workers,
        node_label="Article",
        id_property="article_id",
    )

def force_embed_and_store_articles(driver, embeddings, batch_size=128, max_workers=4):
    return _run_embedding(
        driver, embeddings,
        fetch_query="""
            MATCH (a:Article)
            WHERE a.text IS NOT NULL
            RETURN a.article_id AS id, a.text AS text
        """,
        batch_size=batch_size, max_workers=max_workers,
        node_label="Article",
        id_property="article_id",
    )

def embed_and_store_tables(driver, embeddings, batch_size=64, max_workers=4):
    return _run_embedding(
        driver, embeddings,
        fetch_query="""
            MATCH (t:TableChunk)
            WHERE t.embedding IS NULL AND t.text IS NOT NULL
            RETURN t.chunk_id AS id, t.text AS text
        """,
        batch_size=batch_size,
        max_workers=max_workers,
        node_label="TableChunk",
        id_property="chunk_id",
    )

# =============================================================================
# 3. VECTOR SEARCH  — searches across ALL 9 laws, no filter
# =============================================================================

def vector_search(driver, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search across every article in the graph (all laws)."""
    with driver.session() as s:
        records = s.run(ANN_QUERY, k=k, query_vector=query_vector).data()
    logger.debug("Vector search → %d hits", len(records))
    return records

# =============================================================================
# 4. GRAPH EXPAND  — enrich each ANN hit with related graph nodes
# =============================================================================

class ArticleContext(BaseModel):
    article_id:          str
    law_id:              str
    law_title:           str
    article_number:      str
    text:                str
    score:               float
    version:             Optional[str]      = "original"
    promulgation_date:   Optional[str]      = None
    amendments:          List[Dict]         = Field(default_factory=list)
    penalties:           List[Dict]         = Field(default_factory=list)
    definitions:         List[Dict]         = Field(default_factory=list)
    referenced_articles: List[Dict]         = Field(default_factory=list)
    topics:              List[str]          = Field(default_factory=list)
    tables:              List[Dict]         = Field(default_factory=list)

class TableContext(BaseModel):
    chunk_id:     str
    table_id:     str
    table_number: str
    law_id:       str
    text:         str
    score:        float
    law_title:    str = "غير محدد"


def _to_dict(obj: Any) -> Dict:
    """Safely convert Pydantic model, dataclass, or plain dict to dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):      # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):            # Pydantic v1
        return obj.dict()
    if hasattr(obj, "__dataclass_fields__"):  # dataclass
        from dataclasses import asdict
        return asdict(obj)
    return vars(obj)                    # fallback

def _clean(raw: List[Any], required_key: str) -> List[Dict]:
    """Normalize to plain dicts and drop entries missing the required key."""
    return [
        d for d in (_to_dict(item) for item in (raw or []))
        if d.get(required_key) is not None
    ]

def expand_article(driver, hit: Dict[str, Any]) -> ArticleContext:
    with driver.session() as s:
        rec = s.run(EXPAND_QUERY, article_id=hit["article_id"]).single()

    return ArticleContext(
        article_id          = hit["article_id"],
        article_number      = hit["article_number"],
        law_id              = hit["law_id"],
        law_title           = rec["law_title"] if rec else hit["law_id"],
        promulgation_date   = rec["promulgation_date"] if rec else None,
        text                = hit["text"],
        score               = hit["score"],
        version             = hit.get("version"),
        amendments          = _clean(rec["amendments"] if rec else [], "id"),
        penalties           = _clean(rec["penalties"] if rec else [], "type"),
        definitions         = _clean(rec["definitions"] if rec else [], "term"),
        referenced_articles = _clean(rec["referenced_articles"] if rec else [], "article_number"),
        tables              = _clean(rec["tables"] if rec else [], "table_id"),
        topics              = [t for t in (rec["topics"] if rec else []) if t],
    )

def retrieve_context(driver, query_vector, k:int):
    article_hits = []
    with driver.session() as s:
        for record in s.run(ANN_QUERY, query_vector=query_vector, k=k):
            article_hits.append({
                "article_id":     record["article_id"],
                "article_number": record["article_number"],
                "law_id":         record["law_id"],
                "text":           record["text"],
                "score":          record["score"],
            })
    logger.info("Article ANN hits: %d", len(article_hits))

    article_contexts = []
    for hit in article_hits:
        try:
            article_contexts.append(expand_article(driver, hit))
        except Exception as exc:
            logger.warning("Failed to expand article %s: %s", hit.get("article_id"), exc)

    table_contexts = []
    try:
        with driver.session() as s:
            records = s.run(TABLE_ANN_QUERY, query_vector=query_vector, k=k).data()

        logger.info("Table ANN hits: %d", len(records))
        for record in records:
            logger.info(
                "  Table hit → law_id=%s table=%s score=%.3f",
                record["law_id"],
                record["table_number"],
                record["score"],
            )
            table_contexts.append(TableContext(
                chunk_id     = record["chunk_id"],
                table_id     = record["table_id"],
                table_number = record["table_number"],
                law_id       = record["law_id"],
                text         = record["text"],
                score        = record["score"],
            ))
    except Exception as exc:
        logger.error("Table ANN search FAILED: %s", exc)

    return article_contexts, table_contexts

# =============================================================================
# 5. PROMPT ASSEMBLY
# =============================================================================

def _fmt_penalties(p_list: List[Dict]) -> str:
    if not p_list:
        return ""
    lines = []
    for p in p_list:
        parts = [p.get("type", "")]
        if p.get("min_value") is not None: parts.append(f"من {p['min_value']}")
        if p.get("max_value") is not None: parts.append(f"إلى {p['max_value']}")
        if p.get("unit"):                  parts.append(p["unit"])
        lines.append("  • " + " ".join(str(x) for x in parts if x))
    return "العقوبات:\n" + "\n".join(lines)

def _fmt_amendments(a_list: List[Dict]) -> str:
    if not a_list:
        return ""
    lines = []
    for am in a_list[:3]:
        desc = (am.get("description") or "")[:200]
        lines.append(
            f"  • [{am.get('type','?')}] {am.get('date','?')} "
            f"بموجب قانون {am.get('law_number','?')}"
            + (f": {desc}" if desc else "")
        )
    return "التعديلات:\n" + "\n".join(lines)

def _fmt_definitions(d_list: List[Dict]) -> str:
    if not d_list:
        return ""
    lines = [f"  • {d['term']}: {d['definition']}" for d in d_list[:5]]
    return "التعريفات:\n" + "\n".join(lines)

def build_context_block(ctx: ArticleContext, index: int) -> str:
    header = (
        f"[{index}] القانون: {ctx.law_title} (ID: {ctx.law_id})\n"
        f"    المادة {ctx.article_number}"
        + (f" — نسخة {ctx.version}" if ctx.version else "")
        + f" | درجة الصلة: {ctx.score:.3f}\n"
    )
    parts = [header, textwrap.indent(ctx.text, "    ")]
    extras = "\n".join(filter(None, [
        _fmt_amendments(ctx.amendments),
        _fmt_penalties(ctx.penalties),
        _fmt_definitions(ctx.definitions),
    ]))
    if extras:
        parts.append(textwrap.indent(extras, "    "))
    if ctx.topics:
        parts.append("    الموضوعات: " + ", ".join(ctx.topics))
    return "\n".join(parts)

def assemble_prompt_context(contexts: List[ArticleContext]) -> str:
    sep    = "\n\n" + "-" * 60 + "\n\n"
    blocks = [build_context_block(ctx, i + 1) for i, ctx in enumerate(contexts)]
    return "\n\n" + sep.join(blocks) + "\n"

def _dedupe_top_tables(table_contexts: List[TableContext], max_tables: int = 2) -> List[TableContext]:
    """
    Keep only the highest scoring chunk per table_id, then cap the final number of tables.
    """
    best_by_table: Dict[str, TableContext] = {}

    for t in sorted(table_contexts, key=lambda x: x.score, reverse=True):
        if t.table_id not in best_by_table:
            best_by_table[t.table_id] = t
        if len(best_by_table) >= max_tables:
            break

    return list(best_by_table.values())

def assemble_prompt_context(contexts: List[ArticleContext]) -> str:
    sep = "\n\n" + "-" * 60 + "\n\n"
    blocks = [build_context_block(ctx, i + 1) for i, ctx in enumerate(contexts)]
    return "\n\n" + sep.join(blocks) + "\n"


# =============================================================================
# 6. LLM GENERATION
# =============================================================================

SYSTEM_PROMPT = """أنت مستشار قانوني رقمي خبير في التشريعات المصرية.
مهمتك هي تحليل النصوص القانونية المقدمة وتقديم إجابة دقيقة وموثقة.

قواعد العمل الصارمة:
1. المرجعية الحصرية: أجب فقط بناءً على "السياق القانوني" المرفق. لا تستخدم أي معلومات خارجية أو عامة.
2. إدارة التعديلات: إذا وجدت مادة قانونية ولها "تعديلات" (Amendments) في السياق، يجب اعتماد النص المعدّل وذكر تاريخ التعديل ورقمه.
3. التوثيق الإلزامي: يجب أن تبدأ كل فقرة أو حكم تذكره بالإشارة لاسم القانون ورقم المادة (مثال: "طبقاً للمادة (١٠) من قانون العقوبات...").
4. القصور المعلوماتي: إذا كان السؤال يتطلب تفاصيل غير موجودة في المواد المقدمة (مثل عقوبات لم تذكر أو شروط لم ترد)، صرّح بوضوح: "المواد المتاحة لا توضح [التفصيل الناقص]".
5. الدقة والموضوعية: تجنب الاستنتاجات القانونية الواسعة؛ التزم بالمنطوق المباشر للنصوص.
6. إذا طلب المستخدم قائمة بجميع مواد قانون معين، اذكر أرقام المواد المتاحة في السياق فقط،
   ونبّه أن الإجابة تعرض عيّنة من المواد وليس القانون كاملاً.
"""

def generate_answer(llm, question, contexts, system_prompt: str):
    # If for some reason system_prompt is still None here, 
    # Pydantic will throw the error you saw.
    if system_prompt is None:
        system_prompt = "أنت مستشار قانوني متخصص."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\nContext: {contexts}")
    ]
    response = llm.invoke(messages)
    return response.content

# =============================================================================
# 7. GRAPH RAG CHAIN
# =============================================================================

class LegalGraphRAG:

    def __init__(self, graph: LegalKnowledgeGraph, embeddings: Embeddings, llm: BaseChatModel, k: int = 15, default_system_prompt: Optional[str] = None):
        self.graph = graph
        self.embeddings = embeddings
        self.llm = llm
        self.k = k
        self.default_system_prompt = default_system_prompt or SYSTEM_PROMPT

    def close(self):
        self.graph.close()

    # ── index management ──────────────────────────────────────────────────

    def setup_vector_index(self, dimensions=None):
        dimensions = dimensions or _infer_embedding_dimension(self.embeddings)
        existing = _get_existing_index_dimension(self.graph.driver)

        if existing and existing != dimensions:
            raise RuntimeError(f"Dimension mismatch: {existing} vs {dimensions}")

        create_vector_index(self.graph.driver, dimensions=dimensions)
        create_table_vector_index(self.graph.driver, dimensions=dimensions)

    def index_articles(self, batch_size=128, max_workers=4):
        articles = embed_and_store_articles(
            self.graph.driver, self.embeddings, batch_size, max_workers
        )
        tables = embed_and_store_tables(
            self.graph.driver, self.embeddings, batch_size, max_workers
        )
        return articles + tables

    def reindex_articles(self, batch_size=128, max_workers=4):
        return force_embed_and_store_articles(
            self.graph.driver, self.embeddings,
            batch_size=batch_size, max_workers=max_workers,
        )

    def rebuild_vector_index(self, batch_size=128, max_workers=4, dimensions=None):
        dimensions = dimensions or _infer_embedding_dimension(self.embeddings)

        _drop_vector_index(self.graph.driver)
        with self.graph.driver.session() as s:
            s.run("DROP INDEX table_chunk_embeddings IF EXISTS")

        create_vector_index(self.graph.driver, dimensions=dimensions)
        create_table_vector_index(self.graph.driver, dimensions=dimensions)

        article_count = self.reindex_articles(batch_size, max_workers)
        logger.info("Articles embedded: %d", article_count)

        table_count = embed_and_store_tables(
            self.graph.driver, self.embeddings, batch_size, max_workers
        )
        logger.info("Table chunks embedded: %d", table_count)

        return article_count + table_count

    # ── main query ────────────────────────────────────────────────────────

    def query(self, question, k=None, threshold=0.6, system_prompt=None):
        top_k = min(k or self.k, 15)
        query_vector = self.embeddings.embed_query(question)
        effective_prompt = system_prompt or self.default_system_prompt or SYSTEM_PROMPT

        article_contexts, table_contexts = retrieve_context(
            self.graph.driver, query_vector, k=top_k
        )

        logger.info(
            "Before threshold — articles: %d | tables: %d",
            len(article_contexts),
            len(table_contexts),
        )
        logger.info("Table scores: %s", [f"{t.score:.3f}" for t in table_contexts])

        valid_articles = [c for c in article_contexts if c.score >= threshold]
        valid_tables = [t for t in table_contexts if t.score >= threshold]
        valid_tables = _dedupe_top_tables(valid_tables, max_tables=2)

        logger.info(
            "After threshold (%.2f) — articles: %d | tables: %d",
            threshold,
            len(valid_articles),
            len(valid_tables),
        )

        if not valid_articles and not valid_tables:
            return {
                "query": question,
                "answer": "لم أجد نصوصاً أو جداول قانونية ذات صلة مباشرة بسؤالك.",
                "sources": [],
            }

        context_text = assemble_prompt_context(valid_articles)

        if valid_tables:
            context_text += "\n\n" + "=" * 60 + "\nالجداول القانونية ذات الصلة:\n"
            for t in valid_tables:
                table_text = (t.text or "").strip()[:700]
                context_text += (
                    f"\n[جدول {t.table_number}] من {t.law_id} "
                    f"| درجة الصلة: {t.score:.3f}\n"
                    + table_text
                    + "\n"
                )

        MAX_CONTEXT_CHARS = 9000
        if len(context_text) > MAX_CONTEXT_CHARS:
            context_text = context_text[:MAX_CONTEXT_CHARS]

        answer = generate_answer(self.llm, question, context_text, effective_prompt)

        sources = [
            {
                "law_id": c.law_id,
                "article_number": c.article_number,
                "law_title": c.law_title,
                "score": f"{c.score:.3f}",
            }
            for c in valid_articles
        ] + [
            {
                "law_id": t.law_id,
                "article_number": f"جدول-{t.table_number}",
                "law_title": t.law_title,
                "score": f"{t.score:.3f}",
            }
            for t in valid_tables
        ]

        return {"query": question, "answer": answer, "sources": sources}

    def __enter__(self):    return self
    def __exit__(self, *_): self.close()