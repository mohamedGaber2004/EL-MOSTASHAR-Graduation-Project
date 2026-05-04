import json
from src.Config.log_config import logging
from typing import Any, Dict
from datetime import datetime, timezone

# Try to import json_repair, but make it optional with fallback
try:
    from json_repair import repair_json
except ImportError:
    def repair_json(s: str) -> str:
        """Fallback JSON repair when json_repair is not available"""
        return s

from src.Graph.state import AgentState
import re

logger = logging.getLogger(__name__)


# =============================================================================
# _now
# =============================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
#  Dedup helpers
# ──────────────────────────────────────────────────────────────────────────────
 
def _dedup_by_keys(existing: list[dict], new_items: list[dict], *keys: str) -> list[dict]:
    """
    Append items from new_items that don't already exist in existing,
    using `keys` as the identity fields for deduplication.
 
    If none of the keys are present in an item, the item is always appended
    (we can't tell if it's a duplicate).
    """
    def fingerprint(item: dict) -> tuple:
        return tuple(item.get(k) for k in keys)
 
    seen: set[tuple] = set()
    for item in existing:
        fp = fingerprint(item)
        if any(v is not None for v in fp):
            seen.add(fp)
 
    result = list(existing)
    for item in new_items:
        if not isinstance(item, dict):
            continue
        fp = fingerprint(item)
        has_key = any(v is not None for v in fp)
        if has_key and fp in seen:
            continue
        result.append(item)
        seen.add(fp)
 
    return result
 
 
def _merge_list(existing: list[dict], incoming: list[dict], *dedup_keys: str) -> list[dict]:
    if not incoming:
        return existing
    if dedup_keys:
        return _dedup_by_keys(existing, incoming, *dedup_keys)
    return existing + incoming


# ──────────────────────────────────────────────────────────────────────────────
#  _merge_extracted
# ──────────────────────────────────────────────────────────────────────────────
 
def _merge_extracted(base: dict, incoming: dict) -> dict:
    """
    Merge `incoming` extraction result into `base`.
    List fields are appended with deduplication where meaningful keys exist.
    Scalar fields (case_meta) are merged with 'first-write-wins'.
    """
    merged = dict(base)
 
    # ── case_meta: first-write-wins per sub-key ──────────────────────────────
    if "case_meta" in incoming and isinstance(incoming["case_meta"], dict):
        base_meta = merged.get("case_meta") or {}
        for k, v in incoming["case_meta"].items():
            if v is not None and base_meta.get(k) is None:
                base_meta[k] = v
        merged["case_meta"] = base_meta
 
    # ── lists: merge with dedup ───────────────────────────────────────────────
    LIST_DEDUP: dict[str, tuple[str, ...]] = {
        "defendants":         ("national_id", "name"),  # Enhanced dedup for defendants
        "charges":            ("law_code", "article_number", "description"),
        "incidents":          ("incident_type", "incident_date", "incident_location"),
        "evidences":          ("evidence_type", "description", "seizure_date"),
        "lab_reports":        ("report_type", "examiner_name", "examination_date"),
        "witness_statements": ("witness_name", "statement_date"),
        "confessions":        ("defendant_name", "confession_date", "text"),
        "procedural_issues":  ("procedure_type", "issue_description"),
        "prior_judgments":    ("judgment_number", "court_name"),
        "defense_documents":  ("submitted_by", "alibi_claimed"),
    }
 
    for key, dedup_keys in LIST_DEDUP.items():
        if key in incoming and isinstance(incoming[key], list):
            merged[key] = _merge_list(
                merged.get(key, []),
                incoming[key],
                *dedup_keys,
            )
 
    return merged


# ──────────────────────────────────────────────────────────────────────────────
#  _apply_extracted_to_state
# ──────────────────────────────────────────────────────────────────────────────
 
def _apply_extracted_to_state(state: Any, extracted: dict) -> dict:
    """
    Convert the raw `extracted` dict (from _merge_extracted) to the
    keyword arguments needed for state.model_copy(update={...}).
 
    Pydantic models inside lists are constructed here so the AgentState
    receives typed objects rather than raw dicts.
    """
    from src.Utils import (
        Defendant, Charge, CaseIncident, Evidence, LabReport,
        WitnessStatement, Confession, ProceduralIssue, PriorJudgment,
        DefenseDocument, CourtLevel,
    )
 
    def safe_make(cls, data: dict) -> Any:
        try:
            return cls(**{k: v for k, v in data.items() if v is not None})
        except Exception as exc:
            logger.warning("Could not construct %s from %s: %s", cls.__name__, data, exc)
            return None
 
    def make_list(cls, raw: list[dict]) -> list:
        return [obj for d in raw if isinstance(d, dict) if (obj := safe_make(cls, d)) is not None]
 
    updates: dict = {}
 
    # ── case_meta scalars ─────────────────────────────────────────────────────
    meta = extracted.get("case_meta") or {}
    for field in ("case_number", "court", "jurisdiction", "prosecutor_name",
                  "referral_order_text", "filing_date", "referral_date"):
        val = meta.get(field)
        if val is not None:
            updates[field] = val
 
    raw_level = meta.get("court_level")
    if raw_level and state.court_level is None:
        for member in CourtLevel:
            if raw_level in (member.value, member.name):
                updates["court_level"] = member
                break
 
    # ── entity lists ──────────────────────────────────────────────────────────
    mapping = {
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
 
    for raw_key, (cls, state_field) in mapping.items():
        raw_list = extracted.get(raw_key, [])
        if raw_list:
            existing = getattr(state, state_field, [])
            new_objs = make_list(cls, raw_list)
            updates[state_field] = existing + new_objs
 
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


# =============================================================================
# _parse_llm_json  (Parse and validate LLM JSON responses using Pydantic)
# =============================================================================

def _parse_llm_json(content: Any) -> dict:
    """
    Parse LLM JSON response using Pydantic classes for validation.
    
    Handles extraction schemas from different document types (AmrIhala, MahdarDabt, etc.)
    and returns a validated dictionary with proper structure.
    
    Args:
        content: Raw JSON string from LLM response or list of blocks
        
    Returns:
        dict: Validated extraction data with proper structure, ready for merging
    """
    from src.Utils import (
        AmrIhalaExtraction,
        MahdarDabtExtraction,
        MahdarIstijwabExtraction,
        AqwalShuhudExtraction,
        TaqrirTibbiExtraction,
        MozakaretDifaExtraction,
    )
    
    if isinstance(content, list):
        # Extract text if content is a list of blocks
        extracted = ""
        for block in content:
            if isinstance(block, dict) and "text" in block:
                extracted += block["text"]
            elif isinstance(block, str):
                extracted += block
        content = extracted

    if not content or not isinstance(content, str):
        logger.warning(f"Empty or invalid content provided to _parse_llm_json: type={type(content)}, repr={repr(content)[:100]}")
        return {}
    
    # Try to repair malformed JSON
    try:
        # Strip markdown codeblocks manually in case json_repair is fallback
        cleaned_content = content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        elif cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        cleaned_content = cleaned_content.strip()

        repaired = repair_json(cleaned_content)
        data = json.loads(repaired)
        
        # Handle case where LLM returned a JSON-encoded string instead of raw JSON object
        if isinstance(data, str):
            try:
                inner_repaired = repair_json(data)
                data = json.loads(inner_repaired)
            except Exception as inner_e:
                logger.debug("Failed to parse double-encoded JSON: %s", inner_e)
                
    except Exception as e:
        logger.error("Failed to parse and repair JSON: %s", e)
        logger.debug("Content: %s", content[:200])
        return {}
    
    if not isinstance(data, dict):
        logger.warning("Parsed JSON is not a dict: %s", type(data))
        return {}
    
    result = _extract_with_entity_validation(data)
    
    if result:
        logger.info("Successfully validated %d entity categories", len(result))
    else:
        logger.warning("No valid entities found in the JSON output.")
        
    return result


def _extract_with_entity_validation(data: dict) -> dict:
    """
    Extract entities from raw dict and validate each using appropriate Pydantic class.
    
    This is a fallback method when the overall extraction schema doesn't fit.
    It validates individual entity lists using their specific Pydantic models.
    
    Args:
        data: Raw extraction dict from LLM
        
    Returns:
        dict: Dict with validated entity lists
    """
    from src.Utils import (
        Defendant, Charge, CaseIncident, Evidence, LabReport,
        WitnessStatement, Confession, ProceduralIssue, PriorJudgment,
        DefenseDocument, CaseMeta,
    )
    
    result = {}
    
    # Entity class mapping
    entity_mappings = {
        "defendants":         Defendant,
        "charges":            Charge,
        "incidents":          CaseIncident,
        "evidences":          Evidence,
        "lab_reports":        LabReport,
        "witness_statements": WitnessStatement,
        "confessions":        Confession,
        "procedural_issues":  ProceduralIssue,
        "prior_judgments":    PriorJudgment,
        "defense_documents":  DefenseDocument,
    }
    
    # Validate and extract each entity list
    for key, pydantic_class in entity_mappings.items():
        if key in data and isinstance(data[key], list):
            validated_items = []
            for item in data[key]:
                if not isinstance(item, dict):
                    continue
                try:
                    validated = pydantic_class(**{k: v for k, v in item.items() if v is not None})
                    validated_items.append(validated.model_dump(exclude_none=True))
                except Exception as e:
                    logger.debug(
                        "Could not validate %s item: %s",
                        pydantic_class.__name__, str(e)[:100]
                    )
                    # Still include the raw item if validation fails
                    validated_items.append(item)
            
            if validated_items:
                result[key] = validated_items
    
    # Handle case_meta separately
    if "case_meta" in data and isinstance(data["case_meta"], dict):
        try:
            validated_meta = CaseMeta(**{k: v for k, v in data["case_meta"].items() if v is not None})
            result["case_meta"] = validated_meta.model_dump(exclude_none=True)
        except Exception as e:
            logger.debug("Could not validate case_meta: %s", str(e)[:100])
            result["case_meta"] = data["case_meta"]
    
    return result