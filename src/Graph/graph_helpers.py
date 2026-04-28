import json
from src.Config.log_config import logging
import re
from typing import Any, Dict
from datetime import datetime, timezone

from src.Graph.state import AgentState

logger = logging.getLogger(__name__)


# =============================================================================
# _now
# =============================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# _parse_llm_json
# =============================================================================

def _parse_llm_json(raw: str) -> dict:
    if not raw:
        return {}

    text = raw.strip()

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()

    brace_start = text.find("{")
    if brace_start > 0:
        text = text[brace_start:]
    elif brace_start == -1:
        logger.error("No JSON found in response: %s", raw[:200])
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                nested = json.loads(parsed)
                if isinstance(nested, dict):
                    return nested
            except Exception:
                pass
        if isinstance(parsed, list):
            logger.warning("LLM returned array, wrapping in dict.")
            return {"items": parsed}
        logger.warning("LLM returned unexpected type: %s", type(parsed).__name__)
        return {}

    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            logger.warning("Multiple JSON objects found, using the first only.")
            depth, in_string, escape_next = 0, False, False
            for i, char in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[: i + 1])
                        except json.JSONDecodeError:
                            break

        logger.warning("JSON parsing failed, attempting to repair: %s", str(e))
        open_braces   = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        if open_braces > 0:
            text += "}" * open_braces
        if open_brackets > 0:
            text += "]" * open_brackets

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("JSON repair failed, returning empty dict.")
            return {}


# =============================================================================
# _merge_extracted
# =============================================================================

def _merge_extracted(base: dict, new: dict) -> dict:
    merged = {k: v for k, v in base.items()}

    if "case_meta" in new:
        if "case_meta" not in merged:
            merged["case_meta"] = {}
        for k, v in new["case_meta"].items():
            if v is not None and merged["case_meta"].get(k) is None:
                merged["case_meta"][k] = v

    id_keys = {
        "defendants":         "name",
        "charges":            "charge_id",
        "incidents":          "incident_id",
        "evidences":          "evidence_id",
        "lab_reports":        "report_id",
        "witness_statements": "witness_id",
        "confessions":        "confession_id",
        "procedural_issues":  "issue_id",
        "prior_judgments":    "judgment_id",
        "defense_documents":  "doc_id",
    }

    for field, id_key in id_keys.items():
        if field not in new:
            continue
        existing = merged.get(field, [])
        seen_ids = {item.get(id_key) for item in existing if item.get(id_key)}
        for item in new.get(field, []):
            if not isinstance(item, dict):
                continue
            item_id = item.get(id_key)
            if item_id and item_id in seen_ids:
                continue
            existing.append(item)
            if item_id:
                seen_ids.add(item_id)
        merged[field] = existing

    return merged


# =============================================================================
# _apply_extracted_to_state
# =============================================================================

def _apply_extracted_to_state(state: AgentState, extracted: dict, doc_id: str) -> dict:
    from src.Utils import (
        Defendant, Charge, CaseIncident, Evidence,
        LabReport, WitnessStatement, Confession,
        ProceduralIssue, PriorJudgment, DefenseDocument,
    )

    def _safe_parse(model_cls, data: dict):
        try:
            data = {k: v for k, v in data.items() if v is not None}
            if "source_document_id" not in data or not data.get("source_document_id"):
                data["source_document_id"] = doc_id
            return model_cls.model_validate(data)
        except Exception as e:
            logger.warning("Failed to parse %s from %s: %s", model_cls.__name__, doc_id, e)
            return None

    updates: dict = {}

    # case_meta
    meta = extracted.get("case_meta", {})
    if meta:
        for field in ("case_number", "court", "jurisdiction", "prosecutor_name"):
            if getattr(state, field, None) is None and meta.get(field):
                updates[field] = meta[field]

    # lists
    field_map = {
        "defendants":         (Defendant,        "defendants"),
        "charges":            (Charge,           "charges"),
        "incidents":          (CaseIncident,     "incidents"),
        "evidences":          (Evidence,         "evidences"),
        "lab_reports":        (LabReport,        "lab_reports"),
        "witness_statements": (WitnessStatement, "witness_statements"),
        "confessions":        (Confession,       "confessions"),
        "procedural_issues":  (ProceduralIssue,  "procedural_issues"),
        "prior_judgments":    (PriorJudgment,    "prior_judgments"),
        "defense_documents":  (DefenseDocument,  "defense_documents"),
    }

    for json_key, (model_cls, state_field) in field_map.items():
        raw_list = extracted.get(json_key, [])
        if not raw_list:
            continue
        existing = list(getattr(state, state_field, []))
        new_parsed = [
            obj for item in raw_list
            if isinstance(item, dict)
            for obj in [_safe_parse(model_cls, item)]
            if obj is not None
        ]
        if new_parsed:
            updates[state_field] = existing + new_parsed

    return updates


# =============================================================================
# _retrieve_for_charge  (للـ LegalResearcherAgent)
# =============================================================================

def _build_charge_query(charge) -> str:
    parts = []
    if getattr(charge, "article_number", None):
        parts.append(f"المادة {charge.article_number}")
    if getattr(charge, "statute", None):
        parts.append(charge.statute)
    if getattr(charge, "law_code", None):
        parts.append(str(charge.law_code))
    if getattr(charge, "description", None):
        parts.append(charge.description)
    if getattr(charge, "elements_required", None):
        parts.append(" ".join(charge.elements_required[:4]))
    return " | ".join(parts)


def _lawcode_to_law_id(charge) -> str | None:
    if not getattr(charge, "law_code", None):
        return None
    mapping = {
        "قانون العقوبات":              "penal_code",
        "قانون مكافحة المخدرات":        "anti_drugs",
        "قانون مكافحة الإرهاب":         "anti_terror",
        "قانون الأسلحة والذخائر":       "weapons_ammunition",
        "قانون مكافحة جرائم المعلومات": "cybercrime",
        "قانون مكافحة غسيل الأموال":    "money_laundering",
    }
    return mapping.get(str(charge.law_code))


def _docs_to_text_list(docs: list, max_chars: int = 500) -> list[dict]:
    result = []
    for doc in docs:
        meta = doc.metadata or {}
        result.append({
            "content":        doc.page_content[:max_chars],
            "law_id":         meta.get("law_id"),
            "article_number": meta.get("article_number"),
            "chunk_type":     meta.get("chunk_type"),
            "law_title":      meta.get("law_title"),
        })
    return result


def _retrieve_for_charge(charge, lv, kg, k_statutes: int = 6) -> dict:
    query = _build_charge_query(charge)
    law_id = _lawcode_to_law_id(charge)

    # FAISS search
    statute_docs = []
    try:
        from src.Vectorstore.vector_store_builder import search
        statute_docs = search(vs=lv, query=query, k=k_statutes, law_id=law_id)
        logger.info("FAISS statutes found: %d", len(statute_docs))
    except Exception as e:
        logger.warning("FAISS statutes failed: %s", e)

    # KG expansion
    kg_context = {}
    article_numbers = [
        d.metadata.get("article_number")
        for d in statute_docs
        if d.metadata.get("article_number")
    ]

    if kg and article_numbers and law_id:
        try:
            with kg.driver.session() as session:
                penalties = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:HAS_PENALTY]->(p:Penalty)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number AS article,
                           p.penalty_type AS type, p.min_value AS min,
                           p.max_value AS max, p.unit AS unit
                """, law_id=law_id, article_numbers=article_numbers).data()

                definitions = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:DEFINES]->(d:Definition)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number AS article,
                           d.term AS term, d.definition_text AS text
                """, law_id=law_id, article_numbers=article_numbers).data()

                amendments = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:AMENDED_BY]->(am:Amendment)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number AS article,
                           am.amendment_date AS date, am.amendment_type AS type,
                           am.description AS description
                    ORDER BY am.amendment_date DESC LIMIT 5
                """, law_id=law_id, article_numbers=article_numbers).data()

                latest_versions = session.run("""
                    MATCH (new:Article)-[:SUPERSEDES]->(old:Article {law_id: $law_id})
                    WHERE old.article_number IN $article_numbers
                    RETURN old.article_number AS article,
                           new.text AS latest_text, new.version AS version
                    ORDER BY new.version DESC
                """, law_id=law_id, article_numbers=article_numbers).data()

            kg_context = {
                "penalties":       penalties,
                "definitions":     definitions,
                "amendments":      amendments,
                "latest_versions": latest_versions,
            }
            logger.info("KG expanded: %d penalties | %d definitions | %d amendments",
                        len(penalties), len(definitions), len(amendments))
        except Exception as e:
            logger.warning("KG expansion failed: %s", e)

    return {
        "charge_statute": getattr(charge, "statute", None),
        "article_number": getattr(charge, "article_number", None),
        "law_code":       str(charge.law_code) if getattr(charge, "law_code", None) else None,
        "query_used":     query,
        "statute_docs":   _docs_to_text_list(statute_docs),
        "cassation_docs": [],
        "kg_context":     kg_context,
    }


# =============================================================================
# _build_fallback_package  (للـ LegalResearcherAgent)
# =============================================================================

def _build_fallback_package(contexts: list[dict]) -> dict:
    return {
        "research_packages": [
            {
                "charge_statute":       ctx.get("charge_statute"),
                "law_code":             ctx.get("law_code"),
                "article_number":       ctx.get("article_number"),
                "elements_of_crime":    [],
                "statutes":             [d["content"] for d in ctx.get("statute_docs", [])[:3]],
                "aggravating_articles": [],
                "mitigating_articles":  [],
                "principles":           [],
                "precedents":           [],
                "penalty_range":        "يحدده القاضي — فشل استخلاص العقوبة",
            }
            for ctx in contexts
        ],
        "procedural_principles":      [],
        "relevant_cassation_rulings": [],
        "_fallback": True,
    }


# =============================================================================
# _agent_context  (context builder لكل agent)
# =============================================================================

def _json_safe(obj) -> Any:
    from enum import Enum as _Enum
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, _Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    return obj


def _safe_dump(pydantic_obj) -> dict:
    return _json_safe(pydantic_obj.model_dump())


def _state_summary(state: AgentState) -> str:
    return json.dumps({
        "case_id":           state.case_id,
        "case_number":       state.case_number,
        "court":             state.court,
        "defendants":        [d.name for d in state.defendants],
        "charges":           [c.statute for c in state.charges],
        "evidence_count":    len(state.evidences),
        "witness_count":     len(state.witness_statements),
        "confession_count":  len(state.confessions),
        "procedural_issues": len(state.procedural_issues),
        "completed_agents":  state.completed_agents,
    }, ensure_ascii=False, indent=2)


def _agent_context(state: AgentState, agent_name: str) -> str:
    base  = _state_summary(state)
    extra: Dict[str, Any] = {}

    if agent_name == "procedural_auditor":
        extra = {
            "procedural_issues": [_safe_dump(p) for p in state.procedural_issues],
            "confessions":       [_safe_dump(c) for c in state.confessions],
            "evidences":         [_safe_dump(e) for e in state.evidences],
        }
    elif agent_name == "legal_researcher":
        extra = {
            "charges":         [_safe_dump(c) for c in state.charges],
            "incidents":       [_safe_dump(i) for i in state.incidents],
            "prior_judgments": [_safe_dump(j) for j in state.prior_judgments],
        }
    elif agent_name == "evidence_analyst":
        extra = {
            "evidences":          [_safe_dump(e) for e in state.evidences],
            "lab_reports":        [_safe_dump(r) for r in state.lab_reports],
            "witness_statements": [_safe_dump(w) for w in state.witness_statements],
            "confessions":        [_safe_dump(c) for c in state.confessions],
            "incidents":          [_safe_dump(i) for i in state.incidents],
            "procedural_audit":   state.agent_outputs.get("procedural_auditor", {}),
        }
    elif agent_name == "defense_analyst":
        extra = {
            "defense_documents": [_safe_dump(d) for d in state.defense_documents],
            "procedural_audit":  state.agent_outputs.get("procedural_auditor", {}),
            "evidence_matrix":   state.agent_outputs.get("evidence_analyst", {}),
            "defendants":        [_safe_dump(d) for d in state.defendants],
        }
    elif agent_name == "judge":
        extra = {
            "procedural_audit": state.agent_outputs.get("procedural_auditor", {}),
            "legal_research":   state.agent_outputs.get("legal_researcher", {}),
            "evidence_matrix":  state.agent_outputs.get("evidence_analyst", {}),
            "defense_analysis": state.agent_outputs.get("defense_analyst", {}),
            "charges":          [_safe_dump(c) for c in state.charges],
            "defendants":       [_safe_dump(d) for d in state.defendants],
            "incidents":        [_safe_dump(i) for i in state.incidents],
        }

    return json.dumps({"summary": json.loads(base), **extra}, ensure_ascii=False, indent=2)