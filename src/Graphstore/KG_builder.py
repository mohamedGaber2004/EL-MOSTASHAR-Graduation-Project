from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================
 
import logging
import traceback
from typing import Any, Dict, List, Optional
 
import pandas as pd
from langchain_core.documents import Document
from neo4j import GraphDatabase
 
from src.Config import get_settings
from src.Utils import (
    _stable_id,
    Amendment,
    norm_regu,
    LawExtractor,
    AmendmentExtractor,
)
 
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER
# =============================================================================

def _split_table_text(text: str, max_chars: int = 1200) -> List[str]:
    """
    Split a large table into retrieval-friendly chunks without breaking lines badly.
    """
    text = (text or "").strip()
    if not text:
        return []

    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in lines:
        if not line.strip():
            continue

        line_len = len(line)

        if current and current_len + line_len + 1 > max_chars:
            chunks.append("\n".join(current).strip())
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len + (1 if current_len else 0)

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c]

# =============================================================================
# CHUNK-BASED INGESTION
# =============================================================================
 
def ingest_dataset(chunks: List[Document]) -> List[Dict]:
    law_data: Dict[str, Dict] = {}
    for chunk in chunks:
        if chunk.metadata.get("chunk_type") != "article":
            continue
        law_key = chunk.metadata["law_id"]
        if law_key not in law_data:
            law_data[law_key] = {
                "law_title": chunk.metadata.get("law_title", law_key),
                "articles":  [],
            }
        law_data[law_key]["articles"].append({
            "article_number": chunk.metadata.get("article_number"),
            "text":           chunk.page_content,
            "language":       "ar",
        })
 
    summaries = []
    for law_key, data in law_data.items():
        articles = data["articles"]
        try:
            if not articles:
                summaries.append({"law_key": law_key, "articles": [], "error": "empty"})
                continue
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
                "schedules":   [],   # always empty — tables live in separate chunks
                "error":       None,
            })
        except Exception as e:
            logger.error(f"  [ERROR] {law_key}: {e}")
            summaries.append({"law_key": law_key, "articles": [], "error": str(e)})
    return summaries
 
def ingest_amendments(chunks: List[Document]) -> Dict[str, List[Amendment]]:
    groups: Dict[tuple, Dict] = {}
    for chunk in chunks:
        if chunk.metadata.get("chunk_type") != "amended_article":
            continue
        m       = chunk.metadata
        law_key = m["law_id"]
        law_num = m.get("amendment_law_number", "unknown")
        adate   = m.get("amendment_date",       "unknown")
        key     = (law_key, law_num, adate)
        if key not in groups:
            groups[key] = {"law_key": law_key, "texts": [], "articles": []}
        if (no := m.get("article_number")) and no not in groups[key]["articles"]:
            groups[key]["articles"].append(no)
        groups[key]["texts"].append(chunk.page_content)
 
    result: Dict[str, List[Amendment]] = {}
    for (law_key, _, _), data in groups.items():
        full_text = "\n".join(data["texts"])
        ext       = AmendmentExtractor(full_text)
        amendment = ext.extract(target_law_id=law_key)
        if amendment:
            amendment.amended_article_numbers = data["articles"] or amendment.amended_article_numbers
            result.setdefault(law_key, []).append(amendment)
    return result
 
def ingest_tables(chunks: List[Document]) -> Dict[str, List[Dict]]:
    """
    Group table documents by law_id.
    Each input Document is still treated as one logical table unit,
    and we will split it later into smaller TableChunk nodes.
    """
    result: Dict[str, List[Dict]] = {}

    for chunk in chunks:
        if chunk.metadata.get("chunk_type") != "table":
            continue

        law_id = chunk.metadata.get("law_id")
        if not law_id:
            logger.warning("table chunk missing law_id — skipped")
            continue

        entry = {
            "table_number": str(chunk.metadata.get("table_number", "0")),
            "source_file": chunk.metadata.get("source_file", "unknown"),
            "text": chunk.page_content,
            "chunk_index_hint": int(chunk.metadata.get("chunk_index", 0) or 0),
        }
        result.setdefault(law_id, []).append(entry)

    for law_id, tables in result.items():
        tables.sort(key=lambda t: (t.get("table_number", ""), t.get("chunk_index_hint", 0)))
        logger.info(
            f"  {law_id}: {len(tables)} table(s) -> "
            + ", ".join(f"table {t['table_number']}" for t in tables)
        )

    return result

# =============================================================================
# SECTION 5: KNOWLEDGE GRAPH
# =============================================================================
class LegalKnowledgeGraph:

    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password
        self.driver = None

    def connect(self):
        """Establish the Neo4j driver connection. Call explicitly to verify connectivity."""
        if self.driver is not None:
            return
        self.driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self.driver.verify_connectivity()
        logger.info("Neo4j connected")
 
    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None
 
    def _run(self, query: str, **params):
        with self.driver.session() as s:
            s.run(query, **params)
 
    # ── schema ────────────────────────────────────────────────────────────
 
    def setup_schema(self, drop_existing: bool = False):
        with self.driver.session() as s:
            if drop_existing:
                s.run("MATCH (n) DETACH DELETE n")
                logger.info("Existing data cleared")
            for stmt in norm_regu.CREATION_OF_SCHEMA.value:
                try:
                    s.run(stmt)
                except Exception:
                    pass
        logger.info("Schema ready")
 
    # ── node / relationship creators ──────────────────────────────────────
 
    def create_law(self, meta: Dict[str, Any]):
        self._run("""
            MERGE (l:Law {law_id: $law_id})
            SET l.title             = $title,
                l.promulgation_date = $promulgation_date,
                l.source            = $source,
                l.language          = $language
        """, law_id=meta["law_id"], title=meta.get("title"),
            promulgation_date=meta.get("promulgation_date"),
            source=meta.get("source"), language=meta.get("language", "ar"))
 
    def create_article(self, law_id: str, article_number: Optional[str],
                       text: str, position: int):
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (a:Article {article_id: $article_id})
            SET a.article_number = $article_number,
                a.text           = $text,
                a.law_id         = $law_id
            MERGE (l)-[r:CONTAINS]->(a)
            SET r.position = $position
        """, law_id=law_id,
            article_id=_stable_id(law_id, article_number or "preamble", position),
            article_number=article_number, text=text, position=position)
 
    def create_penalty(self, penalty: Dict, law_id: str):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MERGE (p:Penalty {penalty_id: $penalty_id})
            SET p.penalty_type = $penalty_type,
                p.min_value    = $min_value,
                p.max_value    = $max_value,
                p.unit         = $unit
            MERGE (a)-[:HAS_PENALTY]->(p)
        """, law_id=law_id,
            penalty_id=_stable_id(law_id, penalty.get("article_number"), penalty.get("penalty_type")),
            article_number=penalty.get("article_number"),
            penalty_type=penalty.get("penalty_type"),
            min_value=penalty.get("min_value"),
            max_value=penalty.get("max_value"),
            unit=penalty.get("unit"))
 
    def create_definition(self, definition: Dict, law_id: str):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MERGE (d:Definition {definition_id: $def_id})
            SET d.term            = $term,
                d.definition_text = $definition_text
            MERGE (a)-[:DEFINES]->(d)
        """, law_id=law_id,
            def_id=_stable_id(law_id, definition.get("term")),
            article_number=definition.get("defined_in_article"),
            term=definition.get("term"),
            definition_text=definition.get("definition_text"))
 
    def create_topic(self, topic_name: str):
        self._run("MERGE (t:Topic {topic_id: $id, name: $name})",
                  id=_stable_id(topic_name), name=topic_name)
 
    def link_article_topic(self, law_id: str, article_number: str,
                           topic_name: str, confidence: float):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MATCH (t:Topic {name: $topic_name})
            MERGE (a)-[r:TAGGED_WITH]->(t)
            SET r.confidence = $confidence
        """, law_id=law_id, article_number=article_number,
            topic_name=topic_name, confidence=confidence)
 
    def create_reference(self, law_id: str, from_article: str, to_article: str):
        self._run("""
            MATCH (from:Article {law_id: $law_id, article_number: $from_article})
            MATCH (to:Article   {law_id: $law_id, article_number: $to_article})
            MERGE (from)-[r:REFERENCES]->(to)
            SET r.reference_type = 'direct'
        """, law_id=law_id, from_article=from_article, to_article=to_article)
 
    def create_table_chunk_node(
        self,
        law_id: str,
        table_id: str,
        table_number: str,
        chunk_number: int,
        text: str,
        source_file: str = "unknown",
    ):
        """
        Create a small retrieval-friendly chunk for a logical table.
        """
        chunk_id = _stable_id(law_id, table_number, "chunk", chunk_number)

        self._run(
            """
            MATCH (t:Table {table_id: $table_id})
            MERGE (c:TableChunk {chunk_id: $chunk_id})
            SET c.table_id     = $table_id,
                c.law_id       = $law_id,
                c.table_number = $table_number,
                c.chunk_number  = $chunk_number,
                c.source_file   = $source_file,
                c.text          = $text,
                c.char_count    = $char_count
            MERGE (t)-[:HAS_CHUNK]->(c)
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

    def create_table_node(
        self,
        law_id: str,
        table_number: str,
        source_file: str,
        text: str,
        position: int,
        max_chunk_chars: int = 1200,
    ):
        """
        Create one logical Table node as metadata only,
        then split the raw table text into TableChunk nodes.
        """
        table_id = _stable_id(law_id, table_number)
        table_preview = (text or "")[:1000].strip()
        chunks = _split_table_text(text, max_chars=max_chunk_chars)

        self._run(
            """
            MATCH (l:Law {law_id: $law_id})
            MERGE (t:Table {table_id: $table_id})
            SET t.table_number = $table_number,
                t.source_file  = $source_file,
                t.law_id       = $law_id,
                t.text_preview = $text_preview,
                t.chunk_count   = $chunk_count,
                t.total_chars   = $total_chars
            MERGE (l)-[r:HAS_TABLE]->(t)
            SET r.position = $position
            """,
            law_id=law_id,
            table_id=table_id,
            table_number=table_number,
            source_file=source_file,
            text_preview=table_preview,
            chunk_count=len(chunks),
            total_chars=len(text or ""),
            position=position,
        )

        for idx, chunk_text in enumerate(chunks, start=1):
            self.create_table_chunk_node(
                law_id=law_id,
                table_id=table_id,
                table_number=table_number,
                chunk_number=idx,
                text=chunk_text,
                source_file=source_file,
            )
 
    def import_amendment(self, law_id: str, amendment: Amendment):
        """
        Create Amendment node and all article-version relationships.
 
        modification / deletion:
            existing Article -[:AMENDED_BY]-> Amendment
            new versioned Article -[:SUPERSEDES]-> old Article
 
        addition:
            new Article -[:AMENDED_BY]-> Amendment
            new Article -[:SUPERSEDES]-> old Article (if one exists)
        """
        desc = (amendment.description or "")[:500]
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (am:Amendment {amendment_id: $amendment_id})
            SET am.amendment_law_number = $law_num,
                am.amendment_law_title  = $law_title,
                am.amendment_date       = $date,
                am.amendment_type       = $atype,
                am.description          = $desc,
                am.effective_date       = $effective_date
            MERGE (l)-[:HAS_AMENDMENT]->(am)
        """, law_id=law_id, amendment_id=amendment.amendment_id,
            law_num=amendment.amendment_law_number,
            law_title=amendment.amendment_law_title,
            date=amendment.amendment_date, atype=amendment.amendment_type,
            desc=desc, effective_date=amendment.effective_date)
 
        for article_number in amendment.amended_article_numbers:
            if not article_number:
                continue
            is_addition = amendment.amendment_type == "addition"
            tag         = "added" if is_addition else "amended"
            new_id      = _stable_id(law_id, article_number, tag, amendment.amendment_date)
 
            if is_addition:
                self._run("""
                    MATCH (l:Law {law_id: $law_id})
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MERGE (a_new:Article {article_id: $new_id})
                    ON CREATE SET
                        a_new.article_number = $article_number,
                        a_new.law_id         = $law_id,
                        a_new.text           = $desc,
                        a_new.version        = $date,
                        a_new.is_addition    = true
                    MERGE (l)-[:CONTAINS]->(a_new)
                    MERGE (a_new)-[:AMENDED_BY {
                        amendment_type: $atype, amendment_date: $date
                    }]->(am)
                """, law_id=law_id, amendment_id=amendment.amendment_id,
                    new_id=new_id, article_number=article_number, desc=desc,
                    date=amendment.amendment_date, atype=amendment.amendment_type)
                self._run("""
                    MATCH (a_new:Article {article_id: $new_id})
                    MATCH (a_old:Article {law_id: $law_id, article_number: $article_number})
                    WHERE a_old.article_id <> $new_id
                      AND a_old.is_addition IS NULL
                    MERGE (a_new)-[:SUPERSEDES {
                        amendment_id: $amendment_id, amendment_date: $date
                    }]->(a_old)
                """, new_id=new_id, law_id=law_id, article_number=article_number,
                    amendment_id=amendment.amendment_id, date=amendment.amendment_date)
            else:
                self._run("""
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                    MERGE (a)-[r:AMENDED_BY]->(am)
                    SET r.amendment_type = $atype,
                        r.amendment_date = $date
                """, amendment_id=amendment.amendment_id, law_id=law_id,
                    article_number=article_number,
                    atype=amendment.amendment_type, date=amendment.amendment_date)
                self._run("""
                    MATCH (l:Law {law_id: $law_id})
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MATCH (a_old:Article {law_id: $law_id, article_number: $article_number})
                    WHERE a_old.article_id <> $new_id
                      AND a_old.is_amended IS NULL
                    MERGE (a_new:Article {article_id: $new_id})
                    ON CREATE SET
                        a_new.article_number = $article_number,
                        a_new.law_id         = $law_id,
                        a_new.text           = $desc,
                        a_new.version        = $date,
                        a_new.is_amended     = true
                    MERGE (l)-[:CONTAINS]->(a_new)
                    MERGE (a_new)-[:SUPERSEDES {
                        amendment_id: $amendment_id, amendment_date: $date
                    }]->(a_old)
                """, law_id=law_id, amendment_id=amendment.amendment_id,
                    article_number=article_number, new_id=new_id,
                    desc=desc, date=amendment.amendment_date)
 
    def import_law(self, summary: Dict, verbose: bool = True):
        """
        Import one law from an ``ingest_dataset()`` summary dict.
        All structured data is pre-extracted — no re-parsing.
        NOTE: schedules are intentionally empty here; they are imported
        separately in Phase 4 from dedicated table chunks.
        """
        law_id = summary["law_meta"]["law_id"]
        if verbose:
            logger.info(f"IMPORTING: {summary['law_meta'].get('title')}")
        self.create_law(summary["law_meta"])
        for i, a in enumerate(summary["articles"]):
            self.create_article(law_id, a.get("article_number"), a.get("text", ""), i)
        for p in summary["penalties"]:
            self.create_penalty(p, law_id)
        for d in summary["definitions"]:
            self.create_definition(d, law_id)
        for topic_name in set(t["topic_name"] for t in summary["topics"]):
            self.create_topic(topic_name)
        for t in summary["topics"]:
            self.link_article_topic(law_id, t["article_number"],
                                    t["topic_name"], t.get("confidence", 1.0))
        for r in summary["references"]:
            self.create_reference(law_id, r["from_article"], r["to_article"])
        if verbose:
            logger.info(f"  {len(summary['articles'])} articles "
                        f"| {len(summary['penalties'])} penalties "
                        f"| {len(summary['schedules'])} schedules")
 
    # ── statistics / queries ──────────────────────────────────────────────
 
    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"nodes": {}, "relationships": {}}

        with self.driver.session() as s:
            for label in norm_regu.NODE_LABELS.value:
                stats["nodes"][label] = s.run(
                    f"MATCH (n:{label}) RETURN count(n) AS c"
                ).single()["c"]

            for rel in norm_regu.RELATIONSHIPS.value:
                stats["relationships"][rel] = s.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                ).single()["c"]

        return stats
 
    def query_amendments(self, law_id: Optional[str] = None) -> List[Dict]:
        with self.driver.session() as s:
            if law_id:
                result = s.run("""
                    MATCH (l:Law {law_id: $law_id})-[:HAS_AMENDMENT]->(am:Amendment)
                    OPTIONAL MATCH (a:Article)-[:AMENDED_BY]->(am)
                    RETURN am, collect(DISTINCT a.article_number) AS affected_articles
                    ORDER BY am.amendment_date
                """, law_id=law_id)
            else:
                result = s.run("""
                    MATCH (l:Law)-[:HAS_AMENDMENT]->(am:Amendment)
                    OPTIONAL MATCH (a:Article)-[:AMENDED_BY]->(am)
                    RETURN l.law_id AS law_id, l.title AS law_title,
                           am, collect(DISTINCT a.article_number) AS affected_articles
                    ORDER BY am.amendment_date
                """)
            out = []
            for record in result:
                row = dict(record["am"])
                row["affected_articles"] = record["affected_articles"]
                if not law_id:
                    row["law_id"]    = record["law_id"]
                    row["law_title"] = record["law_title"]
                out.append(row)
            return out
 
    def get_article(self, law_id: str, article_number: str) -> Optional[str]:
        with self.driver.session() as s:
            record = s.run("""
                MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                RETURN a.text AS text
                ORDER BY a.version DESC
                LIMIT 1
            """, law_id=law_id, article_number=article_number).single()
            
            return record["text"] if record else None
        
# =============================================================================
# PIPELINE
# =============================================================================

def build_knowledge_graph(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    chunks: Optional[List[Document]] = None,
    drop_existing: bool = True,
    verbose: bool = True,
    table_chunk_max_chars: int = 1200,
) -> LegalKnowledgeGraph:
    # ── Phase 0 ───────────────────────────────────────────────────────────
    if chunks is None:
        raise ValueError("chunks must be provided; get_chunks() integration is incomplete.")

    chunk_type_counts = {}
    for c in chunks:
        ct = c.metadata.get("chunk_type", "unknown")
        chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1
    for ct, count in sorted(chunk_type_counts.items()):
        logger.info(f"  {ct}: {count}")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    logger.info("PHASE 1: extraction summary — ingesting dataset")
    summaries = ingest_dataset(chunks)

    _LIST_KEYS = {"articles", "penalties", "definitions", "references",
                  "topics", "schedules", "law_meta"}

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

    successful = [s for s in summaries if not s.get("error")]
    logger.info("%d/%d laws extracted successfully", len(successful), len(summaries))

    # ── Phase 2 ───────────────────────────────────────────────────────────
    logger.info("PHASE 2: importing laws — initializing Knowledge Graph")
    graph = LegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.connect()
    graph.setup_schema(drop_existing=drop_existing)

    for summary in successful:
        graph.import_law(summary, verbose=verbose)

    # ── Phase 3 ───────────────────────────────────────────────────────────
    logger.info("PHASE 3: importing amendments")
    amendments_by_law = ingest_amendments(chunks)
    total_amendments = 0

    for law_id, amendments in amendments_by_law.items():
        logger.info("Processing amendments for %s: %d", law_id, len(amendments))
        for amendment in amendments:
            try:
                graph.import_amendment(law_id, amendment)
                total_amendments += 1
                logger.debug("Imported amendment %s for law %s", amendment.amendment_id, law_id)
            except Exception as e:
                logger.error(
                    "Failed to import amendment %s: %s",
                    getattr(amendment, "amendment_id", "<unknown>"),
                    e,
                )
                if verbose:
                    traceback.print_exc()

    logger.info("Total amendments imported: %d", total_amendments)

    # ── Phase 4 ───────────────────────────────────────────────────────────
    logger.info("PHASE 4: importing tables as chunked nodes")
    tables_by_law = ingest_tables(chunks)
    total_tables = 0
    total_table_chunks = 0

    for law_id, table_list in tables_by_law.items():
        logger.info("Processing %d table(s) for %s", len(table_list), law_id)
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
                total_tables += 1
                total_table_chunks += max(1, len(_split_table_text(table["text"], max_chars=table_chunk_max_chars)))
                logger.debug("Imported table %s for law %s", table.get("table_number"), law_id)
            except Exception as e:
                logger.error(
                    "Failed to import table %s for law %s: %s",
                    table.get("table_number"),
                    law_id,
                    e,
                )
                if verbose:
                    traceback.print_exc()

    logger.info("Tables imported: %d", total_tables)
    logger.info("Table chunks created: %d", total_table_chunks)

    logger.info("BUILD COMPLETE")
    try:
        stats = graph.get_statistics()
        logger.info("KG statistics: %s", stats)
    except Exception:
        logger.exception("Failed to fetch KG statistics")

    return graph

def run_KG():
    graph = None
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