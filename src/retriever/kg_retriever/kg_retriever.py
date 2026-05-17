from __future__ import annotations

import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.retriever.kg_retriever.kg_retriever_enums import (
    EmbedIdProperty,
    EmbedNodeLabel,
    IndexName,
    SimilarityFunction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

_Record = Dict[str, Any]


# =============================================================================
# Cypher query constants
# =============================================================================

_CREATE_ARTICLE_INDEX = f"""
CREATE VECTOR INDEX {IndexName.ARTICLES.value} IF NOT EXISTS
FOR (a:{EmbedNodeLabel.ARTICLE.value})
ON  (a.embedding)
OPTIONS {{
    indexConfig: {{
        `vector.dimensions`:          $dimensions,
        `vector.similarity_function`: '{SimilarityFunction.COSINE.value}'
    }}
}}
"""

_CREATE_TABLE_INDEX = f"""
CREATE VECTOR INDEX {IndexName.TABLE_CHUNKS.value} IF NOT EXISTS
FOR (t:{EmbedNodeLabel.TABLE_CHUNK.value})
ON  (t.embedding)
OPTIONS {{
    indexConfig: {{
        `vector.dimensions`:          $dimensions,
        `vector.similarity_function`: '{SimilarityFunction.COSINE.value}'
    }}
}}
"""

_ANN_ARTICLES = f"""
CALL db.index.vector.queryNodes('{IndexName.ARTICLES.value}', $k, $query_vector)
YIELD node AS article, score
RETURN
    article.article_id     AS article_id,
    article.article_number AS article_number,
    article.law_id         AS law_id,
    article.text           AS text,
    score
ORDER BY score DESC
"""

_ANN_TABLES = f"""
CALL db.index.vector.queryNodes('{IndexName.TABLE_CHUNKS.value}', $k, $query_vector)
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

_EXPAND_ARTICLE = """
MATCH (a:Article {article_id: $article_id})
OPTIONAL MATCH (l:Law {law_id: a.law_id})
OPTIONAL MATCH (a)-[:HAS_PENALTY]  ->(p:Penalty)
OPTIONAL MATCH (a)-[:REFERENCES]   ->(ref:Article)
OPTIONAL MATCH (l)-[:HAS_TABLE]    ->(tb:Table)
OPTIONAL MATCH (a)-[:AMENDED_BY]   ->(am:Amendment)
OPTIONAL MATCH (a)-[:DEFINES]      ->(d:Definition)
OPTIONAL MATCH (a)-[:TAGGED_WITH]  ->(t:Topic)
RETURN
    coalesce(l.title, 'غير محدد')                                AS law_title,
    l.promulgation_date                                          AS promulgation_date,
    collect(DISTINCT {type: p.penalty_type, id: p.penalty_id})  AS penalties,
    collect(DISTINCT {
        article_number: ref.article_number,
        law_id:         ref.law_id,
        text:           ref.text
    })                                                           AS referenced_articles,
    collect(DISTINCT {
        table_id:     tb.table_id,
        table_number: tb.table_number
    })                                                           AS tables,
    collect(DISTINCT {
        id:          am.amendment_id,
        type:        am.amendment_type,
        date:        am.amendment_date,
        law_number:  am.amendment_law_number,
        description: am.description
    })                                                           AS amendments,
    collect(DISTINCT {
        term:       d.term,
        definition: d.definition_text
    })                                                           AS definitions,
    collect(DISTINCT t.name)                                     AS topics
"""

_FETCH_ALL_ARTICLES = f"""
MATCH (a:{EmbedNodeLabel.ARTICLE.value})
WHERE a.text IS NOT NULL
RETURN
    a.{EmbedIdProperty.ARTICLE_ID.value} AS article_id,
    a.article_number               AS article_number,
    a.law_id                       AS law_id,
    a.text                         AS text
"""

_STORE_EMBEDDING = """
UNWIND $rows AS row
MATCH (n:{label} {{{id_prop}: row.id}})
SET n.embedding = row.embedding
"""

_MAX_CONTEXT_CHARS = 9_000


# =============================================================================
# Data models
# =============================================================================
class ArticleContext(BaseModel):
    """An article node enriched with its graph neighbours."""
    article_id:          str
    law_id:              str
    law_title:           str
    article_number:      str
    text:                str
    score:               float
    version:             Optional[str]  = "original"
    promulgation_date:   Optional[str]  = None
    amendments:          List[Dict]     = Field(default_factory=list)
    penalties:           List[Dict]     = Field(default_factory=list)
    definitions:         List[Dict]     = Field(default_factory=list)
    referenced_articles: List[Dict]     = Field(default_factory=list)
    topics:              List[str]      = Field(default_factory=list)
    tables:              List[Dict]     = Field(default_factory=list)

class TableContext(BaseModel):
    """A matched table chunk with law provenance."""
    chunk_id:     str
    table_id:     str
    table_number: str
    law_id:       str
    text:         str
    score:        float
    law_title:    str = "غير محدد"

@dataclass
class RetrievalResult:
    """
    Structured output of :meth:`LegalRetriever.retrieve`.

    Attributes
    ----------
    query:            The original question string.
    article_contexts: Deduplicated, threshold-filtered article hits.
    table_contexts:   Deduplicated, threshold-filtered table hits.
    context_text:     Budget-aware formatted string ready for display or downstream use.
    sources:          Flat list of provenance dicts (articles + tables).
    """
    query:            str
    article_contexts: List[ArticleContext]         = field(default_factory=list)
    table_contexts:   List[TableContext]           = field(default_factory=list)
    context_text:     str                          = ""
    sources:          List[Dict[str, Any]]         = field(default_factory=list)

# =============================================================================
# VectorIndexManager
# =============================================================================
class VectorIndexManager:
    """
    Manages the lifecycle of Neo4j vector indexes for articles and table chunks.
    Instantiate with a live Neo4j driver.
    """

    def __init__(self, driver) -> None:
        self.driver = driver

    def get_dimension(self, index_name: IndexName = IndexName.ARTICLES.value) -> Optional[int]:
        """Return the configured dimension for an existing index, or ``None``."""
        with self.driver.session() as session:
            rec = session.run(
                "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
                name=index_name.value,
            ).single()
        if not rec:
            return None
        dim = (rec.get("options") or {}).get("indexConfig", {}).get("vector.dimensions")
        return int(dim) if dim is not None else None

    def create(self, dimensions: int) -> None:
        """Create article and table-chunk vector indexes (idempotent)."""
        with self.driver.session() as session:
            session.run(_CREATE_ARTICLE_INDEX, dimensions=dimensions)
            session.run(_CREATE_TABLE_INDEX,   dimensions=dimensions)
        logger.info(
            "Vector indexes ready — '%s' and '%s' (dim=%d)",
            IndexName.ARTICLES.value, IndexName.TABLE_CHUNKS.value, dimensions,
        )

    def drop(self) -> None:
        """Drop both vector indexes."""
        with self.driver.session() as session:
            session.run(f"DROP INDEX {IndexName.ARTICLES.value}     IF EXISTS")
            session.run(f"DROP INDEX {IndexName.TABLE_CHUNKS.value} IF EXISTS")
        logger.info("Vector indexes dropped")

    def validate_dimension(self, expected: int) -> None:
        """Raise if the existing article index has a different dimension."""
        existing = self.get_dimension(IndexName.ARTICLES.value)
        if existing and existing != expected:
            raise RuntimeError(
                f"Dimension mismatch on '{IndexName.ARTICLES.value}': "
                f"index={existing}, model={expected}"
            )


# =============================================================================
# EmbeddingPipeline
# =============================================================================

class EmbeddingPipeline:
    """
    Parallel batch embedding for :attr:`EmbedNodeLabel.ARTICLE` and
    :attr:`EmbedNodeLabel.TABLE_CHUNK` nodes.
    """

    def __init__(self, driver, embeddings: Embeddings) -> None:
        self.driver     = driver
        self.embeddings = embeddings

    def infer_dimension(self) -> int:
        sample = self.embeddings.embed_query("__dimension_check__")
        if not isinstance(sample, list) or not sample:
            raise ValueError("Unable to infer embedding dimension from model.")
        return len(sample)

    # ── public entry points ───────────────────────────────────────────────

    def embed_articles(self, batch_size: int = 128, max_workers: int = 4, force: bool = False) -> int:
        """
        Embed article nodes. Set *force=True* to overwrite existing embeddings.
        """
        where_clause = "WHERE a.text IS NOT NULL" if force else "WHERE a.embedding IS NULL AND a.text IS NOT NULL"
        return self._run(
            fetch_query=f"""
                MATCH (a:{EmbedNodeLabel.ARTICLE.value})
                {where_clause}
                RETURN a.{EmbedIdProperty.ARTICLE_ID.value} AS id, a.text AS text
            """,
            node_label=EmbedNodeLabel.ARTICLE.value,
            id_property=EmbedIdProperty.ARTICLE_ID.value,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    def embed_tables(self, batch_size: int = 64, max_workers: int = 4) -> int:
        return self._run(
            fetch_query=f"""
                MATCH (t:{EmbedNodeLabel.TABLE_CHUNK.value})
                WHERE t.embedding IS NULL AND t.text IS NOT NULL
                RETURN t.{EmbedIdProperty.CHUNK_ID.value} AS id, t.text AS text
            """,
            node_label=EmbedNodeLabel.TABLE_CHUNK.value,
            id_property=EmbedIdProperty.CHUNK_ID.value,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    def embed_all(self, batch_size: int = 128, max_workers: int = 4) -> int:
        articles = self.embed_articles(batch_size=batch_size, max_workers=max_workers)
        tables   = self.embed_tables(batch_size=batch_size, max_workers=max_workers)
        return articles + tables

    # ── internal runner ───────────────────────────────────────────────────
    def _run(
        self,
        fetch_query:  str,
        node_label:   EmbedNodeLabel,
        id_property:  EmbedIdProperty,
        batch_size:   int,
        max_workers:  int,
    ) -> int:
        with self.driver.session() as session:
            records = session.run(fetch_query).data()

        if not records:
            logger.info("No %s nodes to embed.", node_label)
            return 0

        logger.info("%s nodes to embed: %d", node_label, len(records))
        batches = [records[i: i + batch_size] for i in range(0, len(records), batch_size)]

        encoded: Dict[int, List] = {}

        def _encode(idx: int, batch: List[_Record]):
            return idx, self.embeddings.embed_documents([r["text"] for r in batch])

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_encode, i, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futures):
                idx, vectors = fut.result()
                encoded[idx] = vectors
                logger.info("  Encoded batch %d/%d (%d nodes)", idx + 1, len(batches), len(vectors))

        store_query = _STORE_EMBEDDING.format(
            label=node_label, id_prop=id_property
        )
        total = 0
        for idx, batch in enumerate(batches):
            rows = [{"id": r["id"], "embedding": v} for r, v in zip(batch, encoded[idx])]
            with self.driver.session() as session:
                session.run(store_query, rows=rows)
            total += len(rows)

        logger.info("Embedding complete — %d %s nodes written", total, node_label)
        return total


# =============================================================================
# BM25 in-memory index
# =============================================================================
def _tokenize_arabic(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer for Arabic text."""
    return re.sub(r'[^\w\s]', ' ', text or '').split()

class ArticleBM25Index:
    """In-memory BM25Okapi index over all :attr:`EmbedNodeLabel.ARTICLE` nodes."""

    def __init__(self, driver) -> None:
        with driver.session() as session:
            self._records: List[_Record] = session.run(_FETCH_ALL_ARTICLES).data()

        if not self._records:
            raise RuntimeError("BM25: no Article nodes found in Neo4j.")

        self._bm25 = BM25Okapi([_tokenize_arabic(r["text"]) for r in self._records])
        logger.info("BM25 index built — %d articles", len(self._records))

    def search(self, query: str, k: int = 15) -> List[_Record]:
        scores = self._bm25.get_scores(_tokenize_arabic(query))
        ranked = sorted(zip(self._records, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**rec, "score": float(score)} for rec, score in ranked if score > 0]

# =============================================================================
# RRF merge
# =============================================================================
def _rrf_merge(
    bm25_hits:     List[_Record],
    vector_hits:   List[_Record],
    k:             int   = 60,
    bm25_weight:   float = 0.4,
    vector_weight: float = 0.6,
) -> List[_Record]:
    """
    Reciprocal Rank Fusion (Robertson 2009).
    *k=60* is the standard smoothing constant.
    Returns deduplicated hits sorted by normalised fused score.
    """
    scores: Dict[str, float]  = {}
    meta:   Dict[str, _Record] = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        aid = hit["article_id"]
        scores[aid] = scores.get(aid, 0.0) + bm25_weight / (k + rank)
        meta[aid]   = {**hit, "bm25_score": hit["score"], "score": 0.0}

    for rank, hit in enumerate(vector_hits, start=1):
        aid = hit["article_id"]
        scores[aid] = scores.get(aid, 0.0) + vector_weight / (k + rank)
        if aid in meta:
            meta[aid]["vector_score"] = hit["score"]
        else:
            meta[aid] = {**hit, "vector_score": hit["score"], "score": 0.0}

    for aid, fused in scores.items():
        meta[aid]["score"] = fused

    ranked = sorted(meta.values(), key=lambda x: x["score"], reverse=True)

    if ranked:
        lo, hi = ranked[-1]["score"], ranked[0]["score"]
        spread = (hi - lo) or 1e-9
        for hit in ranked:
            hit["score"] = (hit["score"] - lo) / spread

    return ranked


# =============================================================================
# Graph expansion
# =============================================================================
def _to_dict(obj: Any) -> Dict:
    """Safely convert Pydantic model, dataclass, or plain dict to dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    return vars(obj)

def _clean(raw: List[Any], required_key: str) -> List[Dict]:
    """Normalise to plain dicts and drop entries missing *required_key*."""
    return [
        d for d in (_to_dict(item) for item in (raw or []))
        if d.get(required_key) is not None
    ]

def expand_article(driver, hit: _Record) -> ArticleContext:
    """
    Run :data:`_EXPAND_ARTICLE` for *hit* and return a fully populated
    :class:`ArticleContext`.
    """
    with driver.session() as session:
        rec = session.run(_EXPAND_ARTICLE, article_id=hit["article_id"]).single()

    return ArticleContext(
        article_id          = hit["article_id"],
        article_number      = hit["article_number"],
        law_id              = hit["law_id"],
        text                = hit["text"],
        score               = hit["score"],
        version             = hit.get("version"),
        law_title           = rec["law_title"]        if rec else hit["law_id"],
        promulgation_date   = rec["promulgation_date"] if rec else None,
        amendments          = _clean(rec["amendments"]          if rec else [], "id"),
        penalties           = _clean(rec["penalties"]           if rec else [], "type"),
        definitions         = _clean(rec["definitions"]         if rec else [], "term"),
        referenced_articles = _clean(rec["referenced_articles"] if rec else [], "article_number"),
        tables              = _clean(rec["tables"]              if rec else [], "table_id"),
        topics              = [t for t in (rec["topics"] if rec else []) if t],
    )

# =============================================================================
# Context formatting
# =============================================================================
def _fmt_penalties(p_list: List[Dict]) -> str:
    if not p_list:
        return ""
    lines = []
    for p in p_list:
        parts: List[str] = [p.get("type", "")]
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
            f"  • [{am.get('type', '?')}] {am.get('date', '?')} "
            f"بموجب قانون {am.get('law_number', '?')}"
            + (f": {desc}" if desc else "")
        )
    return "التعديلات:\n" + "\n".join(lines)

def _fmt_definitions(d_list: List[Dict]) -> str:
    if not d_list:
        return ""
    lines = [f"  • {d['term']}: {d['definition']}" for d in d_list[:5]]
    return "التعريفات:\n" + "\n".join(lines)

def build_context_block(ctx: ArticleContext, index: int) -> str:
    """Format one :class:`ArticleContext` as a numbered, indented text block."""
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
    """Join all article context blocks with a visual separator."""
    sep    = "\n\n" + "-" * 60 + "\n\n"
    blocks = [build_context_block(ctx, i + 1) for i, ctx in enumerate(contexts)]
    return "\n\n" + sep.join(blocks) + "\n"

def _dedupe_top_tables(
    table_contexts: List[TableContext],
    max_tables: int = 5,
) -> List[TableContext]:
    """Keep the highest-scoring chunk per ``table_id``, then cap at *max_tables*."""
    best: Dict[str, TableContext] = {}
    for t in sorted(table_contexts, key=lambda x: x.score, reverse=True):
        best.setdefault(t.table_id, t)
        if len(best) >= max_tables:
            break
    return list(best.values())

def _budget_aware_context(
    article_contexts: List[ArticleContext],
    table_contexts:   List[TableContext],
    budget:           int = _MAX_CONTEXT_CHARS,
) -> str:
    """
    Assemble articles then tables into a single string without exceeding *budget*
    characters.  Never cuts a block in half.
    """
    sep    = "\n\n" + "-" * 60 + "\n\n"
    parts: List[str] = []
    used   = 0

    for i, ctx in enumerate(article_contexts):
        block = build_context_block(ctx, i + 1)
        chunk = (sep if parts else "\n\n") + block + "\n"
        if used + len(chunk) > budget:
            logger.info("Context budget reached after %d/%d articles", i, len(article_contexts))
            break
        parts.append(chunk)
        used += len(chunk)

    if table_contexts and used < budget:
        header = "\n\n" + "=" * 60 + "\nالجداول القانونية ذات الصلة:\n"
        if used + len(header) <= budget:
            parts.append(header)
            used += len(header)

        for t in table_contexts:
            block = (
                f"\n[جدول {t.table_number}] من {t.law_id} "
                f"| درجة الصلة: {t.score:.3f}\n"
                + (t.text or "").strip()[:700] + "\n"
            )
            if used + len(block) > budget:
                break
            parts.append(block)
            used += len(block)

    return "".join(parts)

# =============================================================================
# LegalRetriever
# =============================================================================
class LegalRetriever:
    """
    Hybrid (vector + BM25 + graph expansion) retriever over the Legal
    Knowledge Graph.

    Parameters
    ----------
    graph:
        A connected :class:`~src.KG.kg_builder.LegalKnowledgeGraph` instance.
    embeddings:
        Any :class:`~langchain_core.embeddings.Embeddings` implementation.
    k:
        Default number of ANN / BM25 candidates per retrieval call.

    Usage
    -----
    ::

        retriever = LegalRetriever(graph, embeddings)
        result    = retriever.retrieve("ما هي عقوبة السرقة؟")
        for src in result.sources:
            print(src["article_number"], src["score"])
    """

    def __init__(
        self,
        graph:      Any,
        embeddings: Embeddings,
        k:          int = 15,
    ) -> None:
        self.graph      = graph
        self.embeddings = embeddings
        self.k          = k

        self._index_mgr   = VectorIndexManager(graph.driver)
        self._embed_pipe  = EmbeddingPipeline(graph.driver, embeddings)
        self._bm25_index  = ArticleBM25Index(graph.driver)

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> LegalRetriever:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.graph.close()

    # ── index management (delegated) ──────────────────────────────────────

    def setup_vector_index(self, dimensions: Optional[int] = None) -> None:
        """Create vector indexes, validating dimension consistency."""
        dim = dimensions or self._embed_pipe.infer_dimension()
        self._index_mgr.validate_dimension(dim)
        self._index_mgr.create(dim)

    def index_nodes(self, batch_size: int = 128, max_workers: int = 4) -> int:
        """Embed unindexed article and table-chunk nodes."""
        return self._embed_pipe.embed_all(batch_size=batch_size, max_workers=max_workers)

    def reindex_articles(self, batch_size: int = 128, max_workers: int = 4) -> int:
        """Force-re-embed all article nodes (overwrites existing embeddings)."""
        return self._embed_pipe.embed_articles(
            batch_size=batch_size, max_workers=max_workers, force=True
        )

    def rebuild_vector_index(
        self,
        batch_size:  int           = 128,
        max_workers: int           = 4,
        dimensions:  Optional[int] = None,
    ) -> int:
        """Drop, recreate, and fully re-embed both vector indexes."""
        dim = dimensions or self._embed_pipe.infer_dimension()
        self._index_mgr.drop()
        self._index_mgr.create(dim)
        return self._embed_pipe.embed_all(batch_size=batch_size, max_workers=max_workers)

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        question:  str,
        k:         Optional[int] = None,
        threshold: float         = 0.5,
    ) -> RetrievalResult:
        """
        Run hybrid retrieval for *question* and return a :class:`RetrievalResult`.

        Steps
        -----
        1. Embed the question.
        2. ANN search over articles and table chunks.
        3. BM25 search + RRF merge for articles.
        4. Expand each article hit via graph traversal.
        5. Filter by *threshold*, deduplicate.
        6. Assemble budget-aware context string.
        """
        top_k        = min(k or self.k, 15)
        query_vector = self.embeddings.embed_query(question)

        article_hits, table_hits = self._run_searches(query_vector, question, top_k)

        valid_articles = [c for c in article_hits if c.score >= threshold]
        valid_tables   = _dedupe_top_tables(
            [t for t in table_hits if t.score >= threshold], max_tables=5
        )

        logger.info(
            "After threshold (%.2f) — articles: %d | tables: %d",
            threshold, len(valid_articles), len(valid_tables),
        )

        if not valid_articles and not valid_tables:
            return RetrievalResult(query=question)

        context_text = _budget_aware_context(valid_articles, valid_tables)
        sources      = self._build_sources(valid_articles, valid_tables)

        return RetrievalResult(
            query            = question,
            article_contexts = valid_articles,
            table_contexts   = valid_tables,
            context_text     = context_text,
            sources          = sources,
        )

    # ── private retrieval helpers ─────────────────────────────────────────

    def _run_searches(
        self,
        query_vector: List[float],
        query_text:   str,
        k:            int,
    ) -> tuple[List[ArticleContext], List[TableContext]]:
        """Execute ANN + BM25 searches and expand article hits."""
        # Vector search — articles
        with self.graph.driver.session() as session:
            raw_articles = [
                {
                    "article_id":     r["article_id"],
                    "article_number": r["article_number"],
                    "law_id":         r["law_id"],
                    "text":           r["text"],
                    "score":          r["score"],
                }
                for r in session.run(_ANN_ARTICLES, query_vector=query_vector, k=k)
            ]
        logger.info("Vector hits (articles): %d", len(raw_articles))

        # BM25 + RRF merge
        bm25_hits    = self._bm25_index.search(query_text, k=k)
        merged       = _rrf_merge(bm25_hits, raw_articles)
        logger.info("After RRF merge: %d unique articles", len(merged))

        # Graph expansion
        article_contexts: List[ArticleContext] = []
        seen_ids: set = set()
        for hit in merged:
            aid = hit["article_id"]
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            try:
                article_contexts.append(expand_article(self.graph.driver, hit))
            except Exception as exc:
                logger.warning("Failed to expand article %s: %s", aid, exc)

        # Vector search — tables
        table_contexts: List[TableContext] = []
        try:
            with self.graph.driver.session() as session:
                table_records = session.run(
                    _ANN_TABLES, query_vector=query_vector, k=k
                ).data()
            logger.info("Vector hits (tables): %d", len(table_records))
            for r in table_records:
                logger.debug(
                    "  Table hit → law_id=%s table=%s score=%.3f",
                    r["law_id"], r["table_number"], r["score"],
                )
                table_contexts.append(TableContext(
                    chunk_id     = r["chunk_id"],
                    table_id     = r["table_id"],
                    table_number = r["table_number"],
                    law_id       = r["law_id"],
                    text         = r["text"],
                    score        = r["score"],
                ))
        except Exception as exc:
            logger.error("Table ANN search failed: %s", exc)

        return article_contexts, table_contexts

    @staticmethod
    def _build_sources(
        articles: List[ArticleContext],
        tables:   List[TableContext],
    ) -> List[Dict[str, Any]]:
        article_sources = [
            {
                "type":           "article",
                "law_id":         c.law_id,
                "article_number": c.article_number,
                "law_title":      c.law_title,
                "score":          f"{c.score:.3f}",
                "text":           c.text,
            }
            for c in articles
        ]
        table_sources = [
            {
                "type":           "table",
                "law_id":         t.law_id,
                "article_number": f"جدول-{t.table_number}",
                "law_title":      t.law_title,
                "score":          f"{t.score:.3f}",
                "text":           t.text,
            }
            for t in tables
        ]
        return article_sources + table_sources