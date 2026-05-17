from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document
from neo4j import GraphDatabase

from src.Config import get_settings
from src.Utils import (
    Amendment,
    AmendmentExtractor,
    LawExtractor,
    _stable_id,
)
from src.Chunking.chunking_enums  import AmendmentType, ChunkType
from src.Graphstore.kg_enums import (
    ArticleTag,
    Language,
    NodeLabel,
    ReferenceType,
    RelType,
    KgSchemas,

)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_SummaryDict    = Dict[str, Any]
_AmendmentMap   = Dict[str, List[Amendment]]
_TableMap       = Dict[str, List[Dict[str, Any]]]
_StatsDict      = Dict[str, Any]

_LIST_KEYS = frozenset(
    {"articles", "penalties", "definitions", "references",
     "topics", "schedules", "law_meta"}
)

# =============================================================================
# Module-level ingestion helpers  (pure — no Neo4j dependency)
# ============================================================================
def _split_table_text(text: str, max_chars: int = 1200) -> List[str]:
    """
    Split a large table string into retrieval-friendly chunks without
    breaking lines mid-way.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks:      List[str]  = []
    current:     List[str]  = []
    current_len: int        = 0

    for line in (ln.rstrip() for ln in text.splitlines()):
        if not line.strip():
            continue
        line_len = len(line)
        if current and current_len + line_len + 1 > max_chars:
            chunks.append("\n".join(current).strip())
            current     = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len + (1 if current_len else 0)

    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c]

def ingest_dataset(chunks: List[Document]) -> List[_SummaryDict]:
    """
    Extract structured law metadata from article chunks.

    Groups chunks by ``law_id``, reconstructs the raw text per law, then
    runs :class:`LawExtractor` to pull penalties, definitions, topics, and
    cross-references.
    """
    law_data: Dict[str, Dict] = {}

    for chunk in chunks:
        if chunk.metadata.get("chunk_type") != ChunkType.ARTICLE:
            continue
        law_key = chunk.metadata["law_id"]
        law_data.setdefault(law_key, {
            "law_title": chunk.metadata.get("law_title", law_key),
            "articles":  [],
        })["articles"].append({
            "article_number": chunk.metadata.get("article_number"),
            "text":           chunk.page_content,
            "language":       Language.ARABIC,
        })

    summaries: List[_SummaryDict] = []
    for law_key, data in law_data.items():
        articles = data["articles"]
        if not articles:
            summaries.append({"law_key": law_key, "articles": [], "error": "empty"})
            continue
        try:
            reconstructed = "\n\n".join(
                f"المادة {a['article_number']}\n{a['text']}"
                if a.get("article_number") else a["text"]
                for a in articles
            )
            ext = LawExtractor(law_key, reconstructed)
            summaries.append({
                "law_key":     law_key,
                "law_meta":    ext.extract().law_meta,
                "articles":    articles,
                "penalties":   ext._penalties(articles),
                "definitions": ext._definitions(articles),
                "references":  ext._references(articles),
                "topics":      ext._topics(articles),
                "schedules":   [],   # tables are imported separately in Phase 4
                "error":       None,
            })
        except Exception as exc:
            logger.error("  [ERROR] %s: %s", law_key, exc)
            summaries.append({"law_key": law_key, "articles": [], "error": str(exc)})

    return summaries

def ingest_amendments(chunks: List[Document]) -> _AmendmentMap:
    """
    Group amendment chunks by ``(law_id, amendment_law_number, amendment_date)``
    and run :class:`AmendmentExtractor` on each group.
    """
    groups: Dict[tuple, Dict] = {}

    for chunk in chunks:
        if chunk.metadata.get("chunk_type") != ChunkType.AMENDED_ARTICLE:
            continue
        m       = chunk.metadata
        law_key = m["law_id"]
        key     = (law_key, m.get("amendment_law_number", "unknown"), m.get("amendment_date", "unknown"))
        groups.setdefault(key, {"law_key": law_key, "texts": [], "articles": []})

        if (no := m.get("article_number")) and no not in groups[key]["articles"]:
            groups[key]["articles"].append(no)
        groups[key]["texts"].append(chunk.page_content)

    result: _AmendmentMap = {}
    for (law_key, _, _), data in groups.items():
        amendment = AmendmentExtractor("\n".join(data["texts"])).extract(
            target_law_id=law_key
        )
        if amendment:
            amendment.amended_article_numbers = (
                data["articles"] or amendment.amended_article_numbers
            )
            result.setdefault(law_key, []).append(amendment)

    return result

def ingest_tables(chunks: List[Document]) -> _TableMap:
    """
    Group table chunks by ``law_id``, sort by table number and chunk index,
    and return a structured mapping ready for :meth:`LegalKnowledgeGraph.create_table_node`.
    """
    result: _TableMap = {}

    for chunk in chunks:
        if chunk.metadata.get("chunk_type") != ChunkType.TABLE:
            continue
        law_id = chunk.metadata.get("law_id")
        if not law_id:
            logger.warning("table chunk missing law_id — skipped")
            continue
        result.setdefault(law_id, []).append({
            "table_number":     str(chunk.metadata.get("table_number", "0")),
            "source_file":      chunk.metadata.get("source_file", "unknown"),
            "text":             chunk.page_content,
            "chunk_index_hint": int(chunk.metadata.get("chunk_index", 0) or 0),
        })

    for law_id, tables in result.items():
        tables.sort(key=lambda t: (t["table_number"], t["chunk_index_hint"]))
        logger.info(
            "  %s: %d table(s) -> %s",
            law_id, len(tables),
            ", ".join(f"table {t['table_number']}" for t in tables),
        )

    return result

# =============================================================================
# LegalKnowledgeGraph
# =============================================================================
class LegalKnowledgeGraph:
    """
    Thin wrapper around a Neo4j driver that exposes typed CRUD methods for
    every node label and relationship in the Legal Knowledge Graph schema.

    Call :meth:`connect` before any write/read operation, and :meth:`close`
    when done (or use it as a context manager).
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri      = uri
        self._user     = user
        self._password = password
        self.driver    = None

    # ── context manager support ───────────────────────────────────────────

    def __enter__(self) -> LegalKnowledgeGraph:
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        """Establish the Neo4j driver. Safe to call multiple times (idempotent)."""
        if self.driver is not None:
            return
        self.driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self.driver.verify_connectivity()
        logger.info("Neo4j connected")

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    # ── internal query runner ─────────────────────────────────────────────

    def _run(self, query: str, **params) -> None:
        with self.driver.session() as session:
            session.run(query, **params)

    # ── schema ────────────────────────────────────────────────────────────

    def setup_schema(self, drop_existing: bool = False) -> None:
        with self.driver.session() as session:
            if drop_existing:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Existing data cleared")
            for stmt in KgSchemas.CREATION_OF_SCHEMA.value:
                try:
                    session.run(stmt)
                except Exception:
                    pass
        logger.info("Schema ready")

    # ── node / relationship creators ──────────────────────────────────────

    def create_law(self, meta: Dict[str, Any]) -> None:
        self._run(f"""
            MERGE (l:{NodeLabel.LAW.value} {{law_id: $law_id}})
            SET l.title             = $title,
                l.promulgation_date = $promulgation_date,
                l.source            = $source,
                l.language          = $language
        """,
            law_id=meta["law_id"],
            title=meta.get("title"),
            promulgation_date=meta.get("promulgation_date"),
            source=meta.get("source"),
            language=meta.get("language", Language.ARABIC.value),
        )

    def create_article(
        self,
        law_id:         str,
        article_number: Optional[str],
        text:           str,
        position:       int,
    ) -> None:
        self._run(f"""
            MATCH (l:{NodeLabel.LAW.value} {{law_id: $law_id}})
            MERGE (a:{NodeLabel.ARTICLE.value} {{article_id: $article_id}})
            SET a.article_number = $article_number,
                a.text           = $text,
                a.law_id         = $law_id
            MERGE (l)-[r:{RelType.CONTAINS.value}]->(a)
            SET r.position = $position
        """,
            law_id=law_id,
            article_id=_stable_id(law_id, article_number or ArticleTag.PREAMBLE.value, position),
            article_number=article_number,
            text=text,
            position=position,
        )

    def create_penalty(self, penalty: Dict, law_id: str) -> None:
        self._run(f"""
            MATCH (a:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
            MERGE (p:{NodeLabel.PENALTY.value} {{penalty_id: $penalty_id}})
            SET p.penalty_type = $penalty_type,
                p.min_value    = $min_value,
                p.max_value    = $max_value,
                p.unit         = $unit
            MERGE (a)-[:{RelType.HAS_PENALTY.value}]->(p)
        """,
            law_id=law_id,
            penalty_id=_stable_id(law_id, penalty.get("article_number"), penalty.get("penalty_type")),
            article_number=penalty.get("article_number"),
            penalty_type=penalty.get("penalty_type"),
            min_value=penalty.get("min_value"),
            max_value=penalty.get("max_value"),
            unit=penalty.get("unit"),
        )

    def create_definition(self, definition: Dict, law_id: str) -> None:
        self._run(f"""
            MATCH (a:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
            MERGE (d:{NodeLabel.DEFINITION.value} {{definition_id: $def_id}})
            SET d.term            = $term,
                d.definition_text = $definition_text
            MERGE (a)-[:{RelType.DEFINES.value}]->(d)
        """,
            law_id=law_id,
            def_id=_stable_id(law_id, definition.get("term")),
            article_number=definition.get("defined_in_article"),
            term=definition.get("term"),
            definition_text=definition.get("definition_text"),
        )

    def create_topic(self, topic_name: str) -> None:
        self._run(
            f"MERGE (t:{NodeLabel.TOPIC.value} {{topic_id: $id, name: $name}})",
            id=_stable_id(topic_name),
            name=topic_name,
        )

    def link_article_topic(
        self,
        law_id:         str,
        article_number: str,
        topic_name:     str,
        confidence:     float,
    ) -> None:
        self._run(f"""
            MATCH (a:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
            MATCH (t:{NodeLabel.TOPIC.value} {{name: $topic_name}})
            MERGE (a)-[r:{RelType.TAGGED_WITH.value}]->(t)
            SET r.confidence = $confidence
        """,
            law_id=law_id,
            article_number=article_number,
            topic_name=topic_name,
            confidence=confidence,
        )

    def create_reference(
        self, law_id: str, from_article: str, to_article: str
    ) -> None:
        self._run(f"""
            MATCH (from:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $from_article}})
            MATCH (to:{NodeLabel.ARTICLE.value}   {{law_id: $law_id, article_number: $to_article}})
            MERGE (from)-[r:{RelType.REFERENCES.value}]->(to)
            SET r.reference_type = $ref_type
        """,
            law_id=law_id,
            from_article=from_article,
            to_article=to_article,
            ref_type=ReferenceType.DIRECT.value,
        )

    def create_table_node(
        self,
        law_id:          str,
        table_number:    str,
        source_file:     str,
        text:            str,
        position:        int,
        max_chunk_chars: int = 1200,
    ) -> None:
        """
        Create one logical :attr:`NodeLabel.TABLE` node (metadata only) then
        split the raw text into :attr:`NodeLabel.TABLE_CHUNK` child nodes.
        """
        table_id = _stable_id(law_id, table_number)
        chunks   = _split_table_text(text, max_chars=max_chunk_chars)

        self._run(f"""
            MATCH (l:{NodeLabel.LAW.value} {{law_id: $law_id}})
            MERGE (t:{NodeLabel.TABLE.value} {{table_id: $table_id}})
            SET t.table_number = $table_number,
                t.source_file  = $source_file,
                t.law_id       = $law_id,
                t.text_preview = $text_preview,
                t.chunk_count  = $chunk_count,
                t.total_chars  = $total_chars
            MERGE (l)-[r:{RelType.HAS_TABLE.value}]->(t)
            SET r.position = $position
        """,
            law_id=law_id,
            table_id=table_id,
            table_number=table_number,
            source_file=source_file,
            text_preview=(text or "")[:1000].strip(),
            chunk_count=len(chunks),
            total_chars=len(text or ""),
            position=position,
        )

        for idx, chunk_text in enumerate(chunks, start=1):
            self._create_table_chunk(
                law_id=law_id,
                table_id=table_id,
                table_number=table_number,
                chunk_number=idx,
                text=chunk_text,
                source_file=source_file,
            )

    def _create_table_chunk(
        self,
        law_id:       str,
        table_id:     str,
        table_number: str,
        chunk_number: int,
        text:         str,
        source_file:  str = "unknown",
    ) -> str:
        chunk_id = _stable_id(law_id, table_number, "chunk", chunk_number)
        self._run(f"""
            MATCH (t:{NodeLabel.TABLE.value} {{table_id: $table_id}})
            MERGE (c:{NodeLabel.TABLE_CHUNK.value} {{chunk_id: $chunk_id}})
            SET c.table_id     = $table_id,
                c.law_id       = $law_id,
                c.table_number = $table_number,
                c.chunk_number = $chunk_number,
                c.source_file  = $source_file,
                c.text         = $text,
                c.char_count   = $char_count
            MERGE (t)-[:{RelType.HAS_CHUNK.value}]->(c)
        """,
            law_id=law_id,
            table_id=table_id,
            chunk_id=chunk_id,
            table_number=table_number,
            chunk_number=chunk_number,
            source_file=source_file,
            text=text,
            char_count=len(text or ""),
        )
        return chunk_id

    # ── amendment import ──────────────────────────────────────────────────

    def import_amendment(self, law_id: str, amendment: Amendment) -> None:
        self._create_amendment_node(law_id, amendment)

        is_addition = amendment.amendment_type == AmendmentType.ADDITION.value
        for article_number in filter(None, amendment.amended_article_numbers):
            if is_addition:
                self._import_article_addition(law_id, amendment, article_number)
            else:
                self._import_article_modification(law_id, amendment, article_number)

    def _create_amendment_node(self, law_id: str, amendment: Amendment) -> None:
        self._run(f"""
            MATCH (l:{NodeLabel.LAW.value} {{law_id: $law_id}})
            MERGE (am:{NodeLabel.AMENDMENT.value} {{amendment_id: $amendment_id}})
            SET am.amendment_law_number = $law_num,
                am.amendment_law_title  = $law_title,
                am.amendment_date       = $date,
                am.amendment_type       = $atype,
                am.description          = $desc,
                am.effective_date       = $effective_date
            MERGE (l)-[:{RelType.HAS_AMENDMENT.value}]->(am)
        """,
            law_id=law_id,
            amendment_id=amendment.amendment_id,
            law_num=amendment.amendment_law_number,
            law_title=amendment.amendment_law_title,
            date=amendment.amendment_date,
            atype=amendment.amendment_type,
            desc=(amendment.description or "")[:500],
            effective_date=amendment.effective_date,
        )

    def _import_article_addition(
        self, law_id: str, amendment: Amendment, article_number: str
    ) -> None:
        new_id = _stable_id(law_id, article_number, ArticleTag.ADDED.value, amendment.amendment_date)
        desc   = (amendment.description or "")[:500]

        self._run(f"""
            MATCH (l:{NodeLabel.LAW.value} {{law_id: $law_id}})
            MATCH (am:{NodeLabel.AMENDMENT.value} {{amendment_id: $amendment_id}})
            MERGE (a_new:{NodeLabel.ARTICLE.value} {{article_id: $new_id}})
            ON CREATE SET
                a_new.article_number = $article_number,
                a_new.law_id         = $law_id,
                a_new.text           = $desc,
                a_new.version        = $date,
                a_new.is_addition    = true
            MERGE (l)-[:{RelType.CONTAINS.value}]->(a_new)
            MERGE (a_new)-[:{RelType.AMENDED_BY.value} {{
                amendment_type: $atype, amendment_date: $date
            }}]->(am)
        """,
            law_id=law_id,
            amendment_id=amendment.amendment_id,
            new_id=new_id,
            article_number=article_number,
            desc=desc,
            date=amendment.amendment_date,
            atype=amendment.amendment_type,
        )

        # Wire SUPERSEDES if an original article already exists
        self._run(f"""
            MATCH (a_new:{NodeLabel.ARTICLE.value} {{article_id: $new_id}})
            MATCH (a_old:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
            WHERE a_old.article_id <> $new_id
              AND a_old.is_addition IS NULL
            MERGE (a_new)-[:{RelType.SUPERSEDES.value} {{
                amendment_id: $amendment_id, amendment_date: $date
            }}]->(a_old)
        """,
            new_id=new_id,
            law_id=law_id,
            article_number=article_number,
            amendment_id=amendment.amendment_id,
            date=amendment.amendment_date,
        )

    def _import_article_modification(
        self, law_id: str, amendment: Amendment, article_number: str
    ) -> None:
        new_id = _stable_id(law_id, article_number, ArticleTag.AMENDED.value, amendment.amendment_date)
        desc   = (amendment.description or "")[:500]

        # Tag existing article as AMENDED_BY
        self._run(f"""
            MATCH (am:{NodeLabel.AMENDMENT.value} {{amendment_id: $amendment_id}})
            MATCH (a:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
            MERGE (a)-[r:{RelType.AMENDED_BY.value}]->(am)
            SET r.amendment_type = $atype,
                r.amendment_date = $date
        """,
            amendment_id=amendment.amendment_id,
            law_id=law_id,
            article_number=article_number,
            atype=amendment.amendment_type,
            date=amendment.amendment_date,
        )

        # Create new versioned article and wire SUPERSEDES
        self._run(f"""
            MATCH (l:{NodeLabel.LAW.value} {{law_id: $law_id}})
            MATCH (am:{NodeLabel.AMENDMENT.value} {{amendment_id: $amendment_id}})
            MATCH (a_old:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
            WHERE a_old.article_id <> $new_id
              AND a_old.is_amended IS NULL
            MERGE (a_new:{NodeLabel.ARTICLE.value} {{article_id: $new_id}})
            ON CREATE SET
                a_new.article_number = $article_number,
                a_new.law_id         = $law_id,
                a_new.text           = $desc,
                a_new.version        = $date,
                a_new.is_amended     = true
            MERGE (l)-[:{RelType.CONTAINS.value}]->(a_new)
            MERGE (a_new)-[:{RelType.SUPERSEDES.value} {{
                amendment_id: $amendment_id, amendment_date: $date
            }}]->(a_old)
        """,
            law_id=law_id,
            amendment_id=amendment.amendment_id,
            article_number=article_number,
            new_id=new_id,
            desc=desc,
            date=amendment.amendment_date,
        )

    # ── law import (orchestrates all sub-creators) ────────────────────────

    def import_law(self, summary: _SummaryDict, verbose: bool = True) -> None:
        """
        Import one law from an :func:`ingest_dataset` summary dict.
        Tables are intentionally excluded here — they are imported in Phase 4.
        """
        law_id = summary["law_meta"]["law_id"]
        if verbose:
            logger.info("IMPORTING: %s", summary["law_meta"].get("title"))

        self.create_law(summary["law_meta"])

        for i, article in enumerate(summary["articles"]):
            self.create_article(law_id, article.get("article_number"), article.get("text", ""), i)
        for penalty in summary["penalties"]:
            self.create_penalty(penalty, law_id)
        for definition in summary["definitions"]:
            self.create_definition(definition, law_id)

        unique_topics = {t["topic_name"] for t in summary["topics"]}
        for topic_name in unique_topics:
            self.create_topic(topic_name)
        for topic in summary["topics"]:
            self.link_article_topic(
                law_id, topic["article_number"],
                topic["topic_name"], topic.get("confidence", 1.0)
            )
        for ref in summary["references"]:
            self.create_reference(law_id, ref["from_article"], ref["to_article"])

        if verbose:
            logger.info(
                "  %d articles | %d penalties | %d schedules",
                len(summary["articles"]),
                len(summary["penalties"]),
                len(summary["schedules"]),
            )

    # ── statistics / queries ──────────────────────────────────────────────

    def get_statistics(self) -> _StatsDict:
        stats: _StatsDict = {"nodes": {}, "relationships": {}}
        with self.driver.session() as session:
            for label in NodeLabel.NODE_LABELS.value:
                stats["nodes"][label] = session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS c"
                ).single()["c"]
            for rel in NodeLabel.RELATIONSHIPS.value:
                stats["relationships"][rel] = session.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                ).single()["c"]
        return stats

    def query_amendments(self, law_id: Optional[str] = None) -> List[Dict]:
        with self.driver.session() as session:
            if law_id:
                result = session.run(f"""
                    MATCH (l:{NodeLabel.LAW.value} {{law_id: $law_id}})-[:{RelType.HAS_AMENDMENT.value}]->(am:{NodeLabel.AMENDMENT.value})
                    OPTIONAL MATCH (a:{NodeLabel.ARTICLE.value})-[:{RelType.AMENDED_BY.value}]->(am)
                    RETURN am, collect(DISTINCT a.article_number) AS affected_articles
                    ORDER BY am.amendment_date
                """, law_id=law_id)
            else:
                result = session.run(f"""
                    MATCH (l:{NodeLabel.LAW.value})-[:{RelType.HAS_AMENDMENT.value}]->(am:{NodeLabel.AMENDMENT.value})
                    OPTIONAL MATCH (a:{NodeLabel.ARTICLE.value})-[:{RelType.AMENDED_BY.value}]->(am)
                    RETURN l.law_id AS law_id, l.title AS law_title,
                           am, collect(DISTINCT a.article_number) AS affected_articles
                    ORDER BY am.amendment_date
                """)

            out = []
            for record in result:
                row = {**dict(record["am"]), "affected_articles": record["affected_articles"]}
                if not law_id:
                    row["law_id"]    = record["law_id"]
                    row["law_title"] = record["law_title"]
                out.append(row)
            return out

    def get_article(self, law_id: str, article_number: str) -> Optional[str]:
        with self.driver.session() as session:
            record = session.run(f"""
                MATCH (a:{NodeLabel.ARTICLE.value} {{law_id: $law_id, article_number: $article_number}})
                RETURN a.text AS text
                ORDER BY a.version DESC
                LIMIT 1
            """, law_id=law_id, article_number=article_number).single()
        return record["text"] if record else None


# =============================================================================
# Pipeline orchestration
# =============================================================================

def build_knowledge_graph(
    neo4j_uri:             str,
    neo4j_user:            str,
    neo4j_password:        str,
    chunks:                Optional[List[Document]] = None,
    drop_existing:         bool = True,
    verbose:               bool = True,
    table_chunk_max_chars: int  = 1200,
) -> LegalKnowledgeGraph:

    if chunks is None:
        raise ValueError("chunks must be provided.")

    _log_chunk_type_counts(chunks)

    # ── Phase 1: extraction ───────────────────────────────────────────────
    logger.info("PHASE 1: extracting law summaries")
    summaries    = ingest_dataset(chunks)
    successful   = [s for s in summaries if not s.get("error")]
    _log_extraction_summary(summaries)
    logger.info("%d/%d laws extracted successfully", len(successful), len(summaries))

    # ── Phase 2: import laws ──────────────────────────────────────────────
    logger.info("PHASE 2: importing laws into Knowledge Graph")
    graph = LegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.connect()
    graph.setup_schema(drop_existing=drop_existing)
    for summary in successful:
        graph.import_law(summary, verbose=verbose)

    # ── Phase 3: amendments ───────────────────────────────────────────────
    logger.info("PHASE 3: importing amendments")
    total_amendments = 0
    for law_id, amendments in ingest_amendments(chunks).items():
        logger.info("  %s: %d amendment(s)", law_id, len(amendments))
        for amendment in amendments:
            try:
                graph.import_amendment(law_id, amendment)
                total_amendments += 1
            except Exception as exc:
                logger.error(
                    "Failed to import amendment %s: %s",
                    getattr(amendment, "amendment_id", "<unknown>"), exc,
                )
                if verbose:
                    traceback.print_exc()
    logger.info("Total amendments imported: %d", total_amendments)

    # ── Phase 4: tables ───────────────────────────────────────────────────
    logger.info("PHASE 4: importing tables as chunked nodes")
    total_tables = total_chunks = 0
    for law_id, table_list in ingest_tables(chunks).items():
        logger.info("  %s: %d table(s)", law_id, len(table_list))
        for i, table in enumerate(table_list):
            try:
                graph.create_table_node(
                    law_id=law_id,
                    table_number=table["table_number"],
                    source_file=table["source_file"],
                    text=table["text"],
                    position=i,
                    max_chunk_chars=table_chunk_max_chars,
                )
                total_tables  += 1
                total_chunks  += max(
                    1, len(_split_table_text(table["text"], max_chars=table_chunk_max_chars))
                )
            except Exception as exc:
                logger.error(
                    "Failed to import table %s for %s: %s",
                    table.get("table_number"), law_id, exc,
                )
                if verbose:
                    traceback.print_exc()
    logger.info("Tables imported: %d | Table chunks created: %d", total_tables, total_chunks)

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("BUILD COMPLETE")
    try:
        logger.info("KG statistics: %s", graph.get_statistics())
    except Exception:
        logger.exception("Failed to fetch KG statistics")

    return graph


# =============================================================================
# Convenience entry point
# =============================================================================

def run_KG() -> None:
    """Build the Knowledge Graph using application settings."""
    graph: Optional[LegalKnowledgeGraph] = None
    try:
        graph = build_knowledge_graph(
            neo4j_uri=get_settings().NEO4J_URI,
            neo4j_user=get_settings().NEO4J_USERNAME,
            neo4j_password=get_settings().NEO4J_PASSWORD,
            drop_existing=True,
            verbose=True,
            table_chunk_max_chars=1200,
        )
    except Exception:
        logger.exception("Failed to build knowledge graph")
    finally:
        if graph is not None:
            graph.close()


# =============================================================================
# Private pipeline logging helpers
# =============================================================================

def _log_chunk_type_counts(chunks: List[Document]) -> None:
    counts: Dict[str, int] = {}
    for c in chunks:
        ct = c.metadata.get("chunk_type", "unknown")
        counts[ct] = counts.get(ct, 0) + 1
    for ct, count in sorted(counts.items()):
        logger.info("  %s: %d", ct, count)

def _log_extraction_summary(summaries: List[_SummaryDict]) -> None:
    df_rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if k not in _LIST_KEYS}
        for k in ("articles", "penalties", "definitions", "references", "topics"):
            row[k] = len(s.get(k) or [])
        df_rows.append(row)
    try:
        logger.info("Extraction summary:\n%s", pd.DataFrame(df_rows).to_string())
    except Exception:
        logger.info("Extraction summary: %d laws processed", len(df_rows))