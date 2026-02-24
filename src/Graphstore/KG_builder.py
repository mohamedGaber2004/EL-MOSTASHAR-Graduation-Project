from __future__ import annotations

import logging , fnmatch
from pathlib import Path
from typing import Any , Dict , List, Optional
import pandas as pd
from langchain_core.documents import Document
from neo4j import GraphDatabase
from src.Config.config import get_settings
from src.Utils.regex_utils import _stable_id
from src.Utils.norm_and_regu import norm_regu
from src.Utils.text_loader import _read_file
from src.Utils.files_extractors import (
    LawExtractor , 
    AmendmentExtractor , 
    Amendment , 
    Schedule , 
    ScheduleEntry , 
    ExtractedLaw
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LegalKnowledgeGraph:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        logger.info("✓ Neo4j connected")

    def close(self):
        self.driver.close()


    def _read_law_folder(self , folder: Path) -> str:
        parts = []
        for fp in sorted(f for f in folder.glob("*.txt") if not fnmatch.fnmatch(f.name.lower(), "new*")):
            text = _read_file(fp)
            if text:
                parts.append(text)
        return "\n\n".join(parts).strip()


    def ingest_dataset(self,root_dir: str, folder_to_law_key: Dict[str, str]) -> List[Dict]:
        summaries = []
        for folder in sorted(p for p in Path(root_dir).iterdir() if p.is_dir()):
            if folder.name not in folder_to_law_key:
                continue
            law_key = folder_to_law_key[folder.name]
            try:
                raw = self._read_law_folder(folder)
                if not raw:
                    summaries.append({"folder": folder.name, "law_key": law_key, "error": "empty"})
                    continue
                bundle = LawExtractor(law_key, raw).extract()
                summaries.append({"folder": folder.name, "law_key": law_key,
                                "articles": len(bundle.articles), "penalties": len(bundle.penalties),
                                "definitions": len(bundle.definitions), "references": len(bundle.references),
                                "schedules": len(bundle.schedules), "error": None})
            except Exception as e:
                summaries.append({"folder": folder.name, "law_key": law_key, "error": str(e)})
        return summaries


    def ingest_amendments(self,root_dir: str, folder_to_law_key: Dict[str, str]) -> Dict[str, List[Amendment]]:
        result: Dict[str, List[Amendment]] = {}
        for folder in sorted(p for p in Path(root_dir).iterdir() if p.is_dir()):
            if folder.name not in folder_to_law_key:
                continue
            law_key, law_amendments = folder_to_law_key[folder.name], []
            for af in sorted(folder.glob("new*.txt")):
                text = _read_file(af)
                if not text or not text.strip():
                    continue
                amendment = AmendmentExtractor(text, af.name).extract(law_key)
                if amendment:
                    law_amendments.append(amendment)
            if law_amendments:
                result[law_key] = law_amendments
                logger.info(f"  '{folder.name}' → {len(law_amendments)} amendment(s)")
        return result



    def setup_schema(self, drop_existing: bool = False):
        with self.driver.session() as s:
            if drop_existing:
                s.run("MATCH (n) DETACH DELETE n")
            for stmt in norm_regu.CREATION_OF_SCHEMA.value:
                try:
                    s.run(stmt)
                except Exception:
                    pass
        logger.info("✓ Schema ready")

    def _run(self, query: str, **params):
        with self.driver.session() as s:
            s.run(query, **params)

    def create_law(self, meta: Dict[str, Any]):
        self._run("""
            MERGE (l:Law {law_id: $law_id})
            SET l.title=$title, l.promulgation_date=$promulgation_date,
                l.source=$source, l.language=$language
        """, law_id=meta.get("law_id"), title=meta.get("title"),
            promulgation_date=meta.get("promulgation_date"),
            source=meta.get("source"), language=meta.get("language", "ar"))

    def create_article(self, law_id: str, article_number: Optional[str], text: str, position: int):
        article_id = _stable_id(law_id, article_number or "preamble", position)
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (a:Article {article_id: $article_id})
            SET a.article_number=$article_number, a.text=$text, a.law_id=$law_id
            MERGE (l)-[r:CONTAINS]->(a) SET r.position=$position
        """, law_id=law_id, article_id=article_id,
            article_number=article_number, text=text, position=position)

    def create_penalty(self, penalty: Dict, law_id: str):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MERGE (p:Penalty {penalty_id: $penalty_id})
            SET p.penalty_type=$penalty_type, p.min_value=$min_value,
                p.max_value=$max_value, p.unit=$unit
            MERGE (a)-[:HAS_PENALTY]->(p)
        """, law_id=law_id,
            penalty_id=_stable_id(law_id, penalty.get("article_number"), penalty.get("penalty_type")),
            article_number=penalty.get("article_number"), penalty_type=penalty.get("penalty_type"),
            min_value=penalty.get("min_value"), max_value=penalty.get("max_value"), unit=penalty.get("unit"))

    def create_definition(self, definition: Dict, law_id: str):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MERGE (d:Definition {definition_id: $def_id})
            SET d.term=$term, d.definition_text=$definition_text
            MERGE (a)-[:DEFINES]->(d)
        """, law_id=law_id, def_id=_stable_id(law_id, definition.get("term")),
            article_number=definition.get("defined_in_article"),
            term=definition.get("term"), definition_text=definition.get("definition_text"))

    def create_topic(self, topic_name: str):
        self._run("MERGE (t:Topic {topic_id: $id, name: $name})",
                  id=_stable_id(topic_name), name=topic_name)

    def link_article_topic(self, law_id: str, article_number: str, topic_name: str, confidence: float):
        self._run("""
            MATCH (a:Article {law_id: $law_id, article_number: $article_number})
            MATCH (t:Topic {name: $topic_name})
            MERGE (a)-[r:TAGGED_WITH]->(t) SET r.confidence=$confidence
        """, law_id=law_id, article_number=article_number, topic_name=topic_name, confidence=confidence)

    def create_reference(self, law_id: str, from_article: str, to_article: str):
        self._run("""
            MATCH (from:Article {law_id: $law_id, article_number: $from_article})
            MATCH (to:Article   {law_id: $law_id, article_number: $to_article})
            MERGE (from)-[r:REFERENCES]->(to) SET r.reference_type='direct'
        """, law_id=law_id, from_article=from_article, to_article=to_article)

    def create_schedule(self, law_id: str, schedule: Schedule):
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (s:Schedule {schedule_id: $schedule_id})
            SET s.schedule_number=$schedule_number, s.title=$title, s.entry_count=$count
            MERGE (l)-[:HAS_SCHEDULE]->(s)
        """, law_id=law_id, schedule_id=schedule.schedule_id,
            schedule_number=schedule.schedule_number, title=schedule.title, count=len(schedule.entries))

    def create_schedule_entry(self, schedule_id: str, entry: ScheduleEntry, position: int):
        entry_id = _stable_id(schedule_id, entry.entry_number, position)
        self._run("""
            MATCH (s:Schedule {schedule_id: $schedule_id})
            MERGE (e:ScheduleEntry {entry_id: $entry_id})
            SET e.entry_number=$entry_number, e.arabic_name=$arabic_name,
                e.english_name=$english_name, e.description=$description
            MERGE (s)-[r:CONTAINS_ENTRY]->(e) SET r.position=$position
        """, schedule_id=schedule_id, entry_id=entry_id, entry_number=entry.entry_number,
            arabic_name=entry.arabic_name, english_name=entry.english_name,
            description=entry.description, position=position)
        if entry.chemical_name:
            sub_id = _stable_id("substance", entry.arabic_name or entry.english_name)
            self._run("""
                MATCH (e:ScheduleEntry {entry_id: $entry_id})
                MERGE (sub:Substance {substance_id: $sub_id})
                SET sub.arabic_name=$arabic_name, sub.english_name=$english_name,
                    sub.chemical_name=$chemical_name, sub.trade_names=$trade_names
                MERGE (e)-[:DEFINES_SUBSTANCE]->(sub)
            """, entry_id=entry_id, sub_id=sub_id, arabic_name=entry.arabic_name,
                english_name=entry.english_name, chemical_name=entry.chemical_name,
                trade_names=entry.trade_names or [])

    def create_amendment(self, law_id: str, amendment: Amendment):
        desc = (amendment.description or "")[:500]
        self._run("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (am:Amendment {amendment_id: $amendment_id})
            SET am.amendment_law_number=$law_num, am.amendment_law_title=$law_title,
                am.amendment_date=$date, am.amendment_type=$atype,
                am.description=$desc, am.effective_date=$effective_date
            MERGE (l)-[:HAS_AMENDMENT]->(am)
        """, law_id=law_id, amendment_id=amendment.amendment_id,
            law_num=amendment.amendment_law_number, law_title=amendment.amendment_law_title,
            date=amendment.amendment_date, atype=amendment.amendment_type,
            desc=desc, effective_date=amendment.effective_date)

        for article_number in amendment.amended_article_numbers:
            if not article_number:
                continue
            new_id = _stable_id(law_id, article_number,
                                "added" if amendment.amendment_type == "addition" else "amended",
                                amendment.amendment_date)
            if amendment.amendment_type == "addition":
                self._run("""
                    MATCH (l:Law {law_id: $law_id})
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MERGE (a_new:Article {article_id: $new_id})
                    ON CREATE SET a_new.article_number=$article_number, a_new.law_id=$law_id,
                                  a_new.text=$desc, a_new.version=$date, a_new.is_addition=true
                    MERGE (l)-[:CONTAINS]->(a_new)
                    MERGE (a_new)-[:AMENDED_BY {amendment_type: $atype, amendment_date: $date}]->(am)
                """, law_id=law_id, amendment_id=amendment.amendment_id, new_id=new_id,
                    article_number=article_number, desc=desc,
                    date=amendment.amendment_date, atype=amendment.amendment_type)
                self._run("""
                    MATCH (a_new:Article {article_id: $new_id})
                    MATCH (a_old:Article {law_id: $law_id, article_number: $article_number})
                    WHERE a_old.article_id <> $new_id AND a_old.is_addition IS NULL
                    MERGE (a_new)-[:SUPERSEDES {amendment_id: $amendment_id, amendment_date: $date}]->(a_old)
                """, new_id=new_id, law_id=law_id, article_number=article_number,
                    amendment_id=amendment.amendment_id, date=amendment.amendment_date)
            else:
                self._run("""
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MATCH (a:Article {law_id: $law_id, article_number: $article_number})
                    MERGE (a)-[r:AMENDED_BY]->(am)
                    SET r.amendment_type=$atype, r.amendment_date=$date
                """, amendment_id=amendment.amendment_id, law_id=law_id,
                    article_number=article_number, atype=amendment.amendment_type, date=amendment.amendment_date)
                self._run("""
                    MATCH (l:Law {law_id: $law_id})
                    MATCH (am:Amendment {amendment_id: $amendment_id})
                    MATCH (a_old:Article {law_id: $law_id, article_number: $article_number})
                    WHERE a_old.article_id <> $new_id AND a_old.is_amended IS NULL
                    MERGE (a_new:Article {article_id: $new_id})
                    ON CREATE SET a_new.article_number=$article_number, a_new.law_id=$law_id,
                                  a_new.text=$desc, a_new.version=$date, a_new.is_amended=true
                    MERGE (l)-[:CONTAINS]->(a_new)
                    MERGE (a_new)-[:SUPERSEDES {amendment_id: $amendment_id, amendment_date: $date}]->(a_old)
                """, law_id=law_id, amendment_id=amendment.amendment_id,
                    article_number=article_number, new_id=new_id,
                    desc=desc, date=amendment.amendment_date)

    def import_law(self, bundle: ExtractedLaw):
        law_id = bundle.law_meta["law_id"]
        self.create_law(bundle.law_meta)
        for i, a in enumerate(bundle.articles):
            self.create_article(law_id, a.get("article_number"), a.get("text", ""), i)
        for p in bundle.penalties:
            self.create_penalty(p, law_id)
        for d in bundle.definitions:
            self.create_definition(d, law_id)
        for topic_name in set(t["topic_name"] for t in bundle.topics):
            self.create_topic(topic_name)
        for t in bundle.topics:
            self.link_article_topic(law_id, t["article_number"], t["topic_name"], t.get("confidence", 1.0))
        for r in bundle.references:
            self.create_reference(law_id, r["from_article"], r["to_article"])
        for sched in bundle.schedules:
            self.create_schedule(law_id, sched)
            for i, entry in enumerate(sched.entries):
                self.create_schedule_entry(sched.schedule_id, entry, i)
        logger.info(f"  ✓ {law_id}: {len(bundle.articles)} articles | {len(bundle.penalties)} penalties")

    def import_amendments(self, law_id: str, amendments: List[Amendment]):
        for am in amendments:
            try:
                self.create_amendment(law_id, am)
            except Exception as e:
                logger.error(f"  ✗ {am.amendment_id}: {e}")

    def build_from_documents(self, docs: List[Document]) -> Dict[str, int]:
        stats = {"laws": 0, "articles": 0, "amendments": 0, "skipped": 0}
        seen_laws: set = set()
        for doc in docs:
            law_id = doc.metadata.get("law_id")
            if law_id and law_id not in seen_laws:
                self.create_law({"law_id": law_id, "title": doc.metadata.get("law_title", law_id),
                                 "promulgation_date": None, "source": "chunking_pipeline", "language": "ar"})
                seen_laws.add(law_id)
                stats["laws"] += 1
        for pos, doc in enumerate(docs):
            m, law_id, text = doc.metadata, doc.metadata.get("law_id"), doc.page_content
            if not law_id or not text:
                stats["skipped"] += 1
                continue
            try:
                chunk_type = m.get("chunk_type")
                if chunk_type in ("article", "preamble"):
                    self.create_article(law_id, m.get("article_number"), text, pos)
                    stats["articles"] += 1
                elif chunk_type == "amended_article":
                    law_num = m.get("amendment_law_number")
                    adate   = m.get("amendment_date")
                    atype   = m.get("amendment_type", "modification")
                    if not law_num or not adate:
                        stats["skipped"] += 1
                        continue
                    year = adate.split("-")[0]
                    self.create_amendment(law_id, Amendment(
                        amendment_id            = _stable_id(law_id, law_num, year),
                        amendment_law_number    = law_num,
                        amendment_date          = adate,
                        amendment_law_title     = f"قانون رقم {law_num} لسنة {year}",
                        amended_article_numbers = [m["article_number"]] if m.get("article_number") else [],
                        amendment_type          = atype,
                        description             = text[:500],
                        effective_date          = adate,
                    ))
                    stats["amendments"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"  ✗ pos={pos} law={law_id}: {e}")
                stats["skipped"] += 1
        logger.info(f"✓ KG import: {stats}")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"nodes": {}, "relationships": {}}
        with self.driver.session() as s:
            for label in norm_regu.NODE_LABELS.value:
                stats["nodes"][label] = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for rel in norm_regu.RELATIONSHIPS.value:
                stats["relationships"][rel] = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
        return stats

    def print_statistics(self):
        stats = self.get_statistics()
        print("\n" + "="*60 + "\nKNOWLEDGE GRAPH STATISTICS")
        print("\nNodes:")
        for k, v in stats["nodes"].items():
            print(f"  {k}: {v}")
        print("\nRelationships:")
        for k, v in stats["relationships"].items():
            print(f"  {k}: {v}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    root_dir:          str,
    neo4j_uri:         str,
    neo4j_user:        str,
    neo4j_password:    str,
    folder_to_law_key: Optional[Dict[str, str]] = None,
    drop_existing:     bool = True,
) -> LegalKnowledgeGraph:
    folder_to_law_key = folder_to_law_key or norm_regu.FOLDER_TO_LAW_KEY.value


    graph = LegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.setup_schema(drop_existing=drop_existing)

    print("\nPHASE 1: EXTRACTION SUMMARY")
    summaries = graph.ingest_dataset(root_dir, folder_to_law_key)
    print(pd.DataFrame(summaries).to_string())

    print("\nPHASE 2: IMPORTING LAWS")
    for s in (s for s in summaries if not s.get("error")):
        raw    = graph._read_law_folder(Path(root_dir) / s["folder"])
        bundle = LawExtractor(s["law_key"], raw).extract()
        graph.import_law(bundle)

    print("\nPHASE 3: IMPORTING AMENDMENTS")
    for law_id, amendments in graph.ingest_amendments(root_dir, folder_to_law_key).items():
        graph.import_amendments(law_id, amendments)

    graph.print_statistics()
    return graph


def build_knowledge_graph(
    docs:           List[Document],
    neo4j_uri:      str,
    neo4j_user:     str,
    neo4j_password: str,
    drop_existing:  bool = False,
) -> LegalKnowledgeGraph:
    graph = LegalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.setup_schema(drop_existing=drop_existing)
    graph.build_from_documents(docs)
    graph.print_statistics()
    return graph


def run_KG():
    run_pipeline(get_settings().DataPath, get_settings().NEO4J_URI, get_settings().NEO4J_USERNAME, get_settings().NEO4J_PASSWORD, drop_existing=True).close()