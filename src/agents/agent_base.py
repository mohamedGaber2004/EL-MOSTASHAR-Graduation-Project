import logging, time, random, json 
from datetime import datetime, timezone

from typing import Any, Callable, List, Dict

from src.Config import get_settings
from src.Utils.Enums.agents_enums import AgentsEnums
from src.Graph.states_and_schemas.state import AgentState
from src.Config.log_config import logging

try:
    from json_repair import repair_json
except ImportError:
    def repair_json(s: str) -> str:
        return s


logger = logging.getLogger(__name__)


class AgentBase:
    """Base class for all court-simulation agents."""

    def __init__(self, model_config_key: str, temp_config_key: str, prompt: str) -> None:
        self.cfg = get_settings()
        self.agent: str = model_config_key
        self.model_name: str   = getattr(self.cfg, model_config_key)
        self.temperature: float = getattr(self.cfg, temp_config_key)
        self.prompt: str | List[str] = prompt
        self._llm = self._init_llm(self.agent)

        logger.debug(
            "Initialized %s | model=%s | temp=%s | provider=%s",
            self.__class__.__name__,
            self.model_name,
            self.temperature,
        )

    # ── Agents helpers ───────────────────────────────────────────
    
    # now function
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # Dedup helpers
    def _dedup_by_keys(self,existing: list[dict], new_items: list[dict], *keys: str) -> list[dict]:
        """
        Append items from new_items that don't already exist in existing,
        using `keys` as the identity fields for deduplication.

        If none of the keys are present in an item, the item is always appended.
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

    def _merge_list(self,existing: list[dict], incoming: list[dict], *dedup_keys: str) -> list[dict]:
        if not incoming:
            return existing
        if dedup_keys:
            return self._dedup_by_keys(existing, incoming, *dedup_keys)
        return existing + incoming

    # _merge_extracted
    def _merge_extracted(self,base: dict, incoming: dict) -> dict:
        """
        Merge `incoming` extraction result into `base`.
        List fields are appended with deduplication where meaningful keys exist.
        Scalar fields (case_meta) are merged with first-write-wins.
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
            "defendants":                ("national_id", "name"),
            "charges":                   ("law_code", "article_number", "description"),
            "incidents":                 ("incident_type", "incident_date", "incident_location"),
            "evidences":                 ("evidence_type", "description", "seizure_date"),
            "lab_reports":               ("report_type", "examiner_name", "examination_date"),
            "witness_statements":        ("witness_name", "statement_date"),
            "confessions":               ("defendant_name", "confession_date", "text"),
            "procedural_issues":         ("procedure_type", "issue_description"),
            "defense_procedural_issues": ("procedure_type", "issue_description"),
            "criminal_records":          ("defendant_name",),
            "defense_documents":         ("submitted_by", "defendant_name"),
        }

        for key, dedup_keys in LIST_DEDUP.items():
            if key in incoming and isinstance(incoming[key], list):
                merged[key] = self._merge_list(
                    merged.get(key, []),
                    incoming[key],
                    *dedup_keys,
                )

        return merged

    # _apply_extracted_to_state
    def _apply_extracted_to_state(self,state: Any, extracted: dict) -> dict:
        """
        Convert the raw `extracted` dict (from _merge_extracted) to the
        keyword arguments needed for state.model_copy(update={...}).

        Pydantic models inside lists are constructed here so AgentState
        receives typed objects rather than raw dicts.
        """
        from src.Utils import (
            Defendant, Charge, CaseIncident, Evidence, LabReport,
            WitnessStatement, Confession, ProceduralIssue,
            DefenseDocument, CriminalRecord,
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
        for field in (
            "case_number", "court", "court_level", "jurisdiction",
            "prosecutor_name", "referral_order_text", "filing_date", "referral_date",
        ):
            val = meta.get(field)
            if val is not None:
                updates[field] = val

        # ── entity lists ──────────────────────────────────────────────────────────
        mapping = {
            "defendants":                (Defendant,        "defendants"),
            "charges":                   (Charge,           "charges"),
            "incidents":                 (CaseIncident,     "incidents"),
            "evidences":                 (Evidence,         "evidences"),
            "lab_reports":               (LabReport,        "lab_reports"),
            "witness_statements":        (WitnessStatement, "witness_statements"),
            "confessions":               (Confession,       "confessions"),
            "procedural_issues":         (ProceduralIssue,  "procedural_issues"),
            "defense_procedural_issues": (ProceduralIssue,  "defense_procedural_issues"),
            "defense_documents":         (DefenseDocument,  "defense_documents"),
            "criminal_records":          (CriminalRecord,   "criminal_records"),
        }

        for raw_key, (cls, state_field) in mapping.items():
            raw_list = extracted.get(raw_key, [])
            if raw_list:
                existing = getattr(state, state_field, [])
                new_objs = make_list(cls, raw_list)
                updates[state_field] = existing + new_objs

        return updates

    # _retrieve_for_charge  (للـ LegalResearcherAgent)
    def _build_charge_query(self,charge) -> str:
        parts = []
        if getattr(charge, "article_number", None):
            parts.append(f"المادة {charge.article_number}")
        if getattr(charge, "law_code", None):
            parts.append(str(charge.law_code))
        if getattr(charge, "description", None):
            parts.append(charge.description)
        if getattr(charge, "incident_type", None):
            parts.append(charge.incident_type)
        return " | ".join(parts)
    
    def _lawcode_to_law_id(self,charge) -> str | None:
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

    def _docs_to_text_list(self,docs: list, max_chars: int = 500) -> list[dict]:
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

    def _retrieve_for_charge(self,charge, lv, kg, k_statutes: int = 6) -> dict:
        query = self._build_charge_query(charge)
        law_id = self._lawcode_to_law_id(charge)

        statute_docs = []
        try:
            from src.Vectorstore.vector_store_builder import search
            statute_docs = search(vs=lv, query=query, k=k_statutes, law_id=law_id)
            logger.info("FAISS statutes found: %d", len(statute_docs))
        except Exception as e:
            logger.warning("FAISS statutes failed: %s", e)

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
                logger.info(
                    "KG expanded: %d penalties | %d definitions | %d amendments",
                    len(penalties), len(definitions), len(amendments),
                )
            except Exception as e:
                logger.warning("KG expansion failed: %s", e)

        return {
            "article_number": getattr(charge, "article_number", None),
            "law_code":       str(charge.law_code) if getattr(charge, "law_code", None) else None,
            "query_used":     query,
            "statute_docs":   self._docs_to_text_list(statute_docs),
            "cassation_docs": [],
            "kg_context":     kg_context,
        }

    # _build_fallback_package  (للـ LegalResearcherAgent)
    def _build_fallback_package(self,contexts: list[dict]) -> dict:
        return {
            "research_packages": [
                {
                    "law_code":              ctx.get("law_code"),
                    "article_number":        ctx.get("article_number"),
                    "elements_of_crime":     [],
                    "statutes":              [d["content"] for d in ctx.get("statute_docs", [])[:3]],
                    "aggravating_articles":  [],
                    "mitigating_articles":   [],
                    "principles":            [],
                    "precedents":            [],
                    "penalty_range":         "يحدده القاضي — فشل استخلاص العقوبة",
                }
                for ctx in contexts
            ],
            "procedural_principles":      [],
            "relevant_cassation_rulings": [],
            "_fallback": True,
        }

    # _agent_context  (context builder لكل agent)
    def _json_safe(self,obj) -> Any:
        from enum import Enum as _Enum
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, _Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(i) for i in obj]
        return obj

    def _safe_dump(self,pydantic_obj) -> dict:
        return self._json_safe(pydantic_obj.model_dump())

    def _state_summary(self,state: AgentState) -> str:
        return json.dumps({
            "case_id":           state.case_id,
            "case_number":       state.case_number,
            "court":             state.court,
            "defendants":        [d.name for d in state.defendants],
            "charges":           [c.description for c in state.charges],   # was c.statute (no such field)
            "evidence_count":    len(state.evidences),
            "witness_count":     len(state.witness_statements),
            "confession_count":  len(state.confessions),
            "procedural_issues": len(state.procedural_issues),
            "completed_agents":  state.completed_agents,
        }, ensure_ascii=False, indent=2)

    def _agent_context(self,state: AgentState, agent_name: str) -> str:
        base  = self._state_summary(state)
        extra: Dict[str, Any] = {}

        if agent_name == "procedural_auditor":
            extra = {
                "procedural_issues":         [self._safe_dump(p) for p in state.procedural_issues],
                "defense_procedural_issues": [self._safe_dump(p) for p in state.defense_procedural_issues],
                "confessions":               [self._safe_dump(c) for c in state.confessions],
                "evidences":                 [self._safe_dump(e) for e in state.evidences],
            }
        elif agent_name == "legal_researcher":
            extra = {
                "charges":          [self._safe_dump(c) for c in state.charges],
                "incidents":        [self._safe_dump(i) for i in state.incidents],
                "criminal_records": [self._safe_dump(r) for r in state.criminal_records],  # was prior_judgments
            }
        elif agent_name == "evidence_analyst":
            extra = {
                "evidences":          [self._safe_dump(e) for e in state.evidences],
                "lab_reports":        [self._safe_dump(r) for r in state.lab_reports],
                "witness_statements": [self._safe_dump(w) for w in state.witness_statements],
                "confessions":        [self._safe_dump(c) for c in state.confessions],
                "incidents":          [self._safe_dump(i) for i in state.incidents],
                "procedural_audit":   state.agent_outputs.get("procedural_auditor", {}),
            }
        elif agent_name == "defense_analyst":
            extra = {
                "defense_documents":         [self._safe_dump(d) for d in state.defense_documents],
                "defense_procedural_issues": [self._safe_dump(p) for p in state.defense_procedural_issues],
                "procedural_audit":          state.agent_outputs.get("procedural_auditor", {}),
                "evidence_matrix":           state.agent_outputs.get("evidence_analyst", {}),
                "defendants":                [self._safe_dump(d) for d in state.defendants],
            }
        elif agent_name == "judge":
            extra = {
                "procedural_audit":  state.agent_outputs.get("procedural_auditor", {}),
                "legal_research":    state.agent_outputs.get("legal_researcher", {}),
                "evidence_matrix":   state.agent_outputs.get("evidence_analyst", {}),
                "defense_analysis":  state.agent_outputs.get("defense_analyst", {}),
                "charges":           [self._safe_dump(c) for c in state.charges],
                "defendants":        [self._safe_dump(d) for d in state.defendants],
                "incidents":         [self._safe_dump(i) for i in state.incidents],
                "criminal_records":  [self._safe_dump(r) for r in state.criminal_records],
            }

        return json.dumps({"summary": json.loads(base), **extra}, ensure_ascii=False, indent=2)

    # _parse_llm_json
    def _parse_llm_json(self,content: Any) -> dict:
        """
        Parse and lightly validate LLM JSON output.
        Returns a plain dict ready for _merge_extracted.
        """
        if isinstance(content, list):
            extracted = ""
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    extracted += block["text"]
                elif isinstance(block, str):
                    extracted += block
            content = extracted

        if not content or not isinstance(content, str):
            logger.warning("Empty or invalid content: type=%s repr=%s", type(content), repr(content)[:100])
            return {}

        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            repaired = repair_json(cleaned)
            data = json.loads(repaired)

            if isinstance(data, str):
                try:
                    data = json.loads(repair_json(data))
                except Exception:
                    pass

        except Exception as e:
            logger.error("JSON parse/repair failed: %s | content[:200]=%s", e, content[:200])
            return {}

        if not isinstance(data, dict):
            logger.warning("Parsed JSON is not a dict: %s", type(data))
            return {}

        result = self._extract_with_entity_validation(data)

        if result:
            logger.info("Validated %d entity categories", len(result))
        else:
            logger.warning("No valid entities found in JSON output")

        return result

    def _extract_with_entity_validation(self,data: dict) -> dict:
        """
        Validate each entity list in `data` using the appropriate Pydantic model.
        Falls back to including the raw item dict if validation fails.
        """
        from src.Utils import (
            Defendant, Charge, CaseIncident, Evidence, LabReport,
            WitnessStatement, Confession, ProceduralIssue,
            DefenseDocument, CriminalRecord, CaseMeta,
        )

        result = {}

        entity_mappings = {
            "defendants":                Defendant,
            "charges":                   Charge,
            "incidents":                 CaseIncident,
            "evidences":                 Evidence,
            "lab_reports":               LabReport,
            "witness_statements":        WitnessStatement,
            "confessions":               Confession,
            "procedural_issues":         ProceduralIssue,
            "defense_procedural_issues": ProceduralIssue,
            "defense_documents":         DefenseDocument,
            "criminal_records":          CriminalRecord,
        }

        for key, pydantic_class in entity_mappings.items():
            if key not in data or not isinstance(data[key], list):
                continue

            raw_list = data[key]
            validated_items = []
            for item in raw_list:
                if not isinstance(item, dict):
                    logger.debug("[%s] skipping non-dict item: %s", key, repr(item)[:80])
                    continue
                try:
                    validated = pydantic_class(**{k: v for k, v in item.items() if v is not None})
                    dumped = validated.model_dump(exclude_none=True)
                    validated_items.append(dumped if dumped else item)
                except Exception as e:
                    logger.debug(
                        "[%s] Pydantic validation failed for %s — keeping raw. Error: %s | item: %s",
                        key, pydantic_class.__name__, str(e)[:120], repr(item)[:120],
                    )
                    validated_items.append(item)

            # Always write the key even if list is empty — key presence != parse failure
            result[key] = validated_items
            if not validated_items and raw_list:
                logger.warning(
                    "[%s] All %d item(s) dropped during validation — raw sample: %s",
                    key, len(raw_list), repr(raw_list[0])[:200],
                )

        # case_meta
        if "case_meta" in data and isinstance(data["case_meta"], dict):
            try:
                validated_meta = CaseMeta(**{k: v for k, v in data["case_meta"].items() if v is not None})
                result["case_meta"] = validated_meta.model_dump(exclude_none=True)
            except Exception as e:
                logger.debug("case_meta validation failed: %s", str(e)[:100])
                result["case_meta"] = data["case_meta"]

        return result

    # ── LLM factory ───────────────────────────────────────────────────

    def _init_llm(self, agent_name:str):
        """Instantiate only the requested LLM provider."""
        if agent_name == "DATA_INGESTION_MODEL":
            from src.LLMs.DATA_INGESTION_MODEL import get_ingesion_model
            return get_ingesion_model()['open_router_llm']
        if agent_name == "PROCEDURAL_AUDITOR_MODEL":
            from src.LLMs.PROCEDURAL_AUDITOR_MODEL import get_procedural_model
            return get_procedural_model()['open_router_llm']
        if agent_name == "LEGAL_RESEARCHER_MODEL":
            from src.LLMs.LEGAL_RESEARCHER_MODEL import get_legal_reasearcher_model
            return get_legal_reasearcher_model()['open_router_llm']
        if agent_name == "EVIDENCE_SCORING_MODEL":
            from src.LLMs.EVIDENCE_SCORING_MODEL import get_evidence_scoring_model
            return get_evidence_scoring_model()['open_router_llm']
        if agent_name == "DEFENSE_AGENT_MODEL":
            from src.LLMs.DEFENSE_AGENT_MODEL import get_defense_model
            return get_defense_model()['open_router_llm']
        if agent_name == "JUDJICAL_PRINCIPLE_AGENT":
            from src.LLMs.JUDJICAL_PRINCIPLE_AGENT import get_judjical_principle_model
            return get_judjical_principle_model()['open_router_llm']
        if agent_name == "JUDGE_MODEL":
            from src.LLMs.JUDGE_MODEL import get_judge_model
            return get_judge_model()['open_router_llm']
        raise ValueError(f"Unknown LLM provider: {agent_name!r}")

    # ── retry wrapper ─────────────────────────────────────────────────

    def _llm_invoke_with_retries(self,llm_callable: Callable,messages: List[Any],max_retries: int = None,backoff_base: float = 2.0) -> Any:
        if max_retries is None:
            max_retries = self.cfg.INVOKATION_MAX_RETRIES

        logger.debug(
            "Invoking LLM for %s (max_retries=%d)",
            self.__class__.__name__,
            max_retries,
        )

        for attempt in range(1, max_retries + 1):
            try:
                result = llm_callable.invoke(messages)
                if attempt > 1:
                    logger.info("LLM succeeded on attempt %d", attempt)
                return result
            except Exception as e:
                err = str(e).lower()
                is_rate = any(
                    k in err for k in AgentsEnums.AGENT_INVOKATION_ERRORS
                )
                if is_rate and attempt < max_retries:
                    wait = (
                        min(60.0, backoff_base * (2 ** (attempt - 1)))
                        + random.random()
                    )
                    logger.warning(
                        "Rate-limit / timeout on attempt %d — waiting %.2fs",
                        attempt,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "LLM invocation failed after %d attempt(s): %s", attempt, e
                )
                raise

    # ── state update helper ───────────────────────────────────────────

    def _empty_update(self,state: Any,agent_key: str,output: Any,inter_agent_delay: float = None) -> Any:
        if inter_agent_delay is None:
            inter_agent_delay = self.cfg.INTER_AGENTS_DELAY

        logger.info("Agent [%s] complete — updating state.", agent_key)
        if inter_agent_delay > 0:
            logger.debug("Inter-agent delay: %.2fs", inter_agent_delay)
            time.sleep(inter_agent_delay)

        return state.model_copy(
            update={
                "completed_agents": state.completed_agents + [agent_key],
                "current_agent":    agent_key,
                "last_updated":     self._now(),          # was missing in original
            }
        )
    
    # ── Agents utils ───────────────────────────────────────────
    
    def remove_nulls(self,obj: Any) -> Any:
        """
        Recursively remove keys with None/null values from dicts and lists.
        """
        if isinstance(obj, dict):
            return {k: self.remove_nulls(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self.remove_nulls(v) for v in obj if v is not None]
        else:
            return obj
        
    def clean_agent_output(self,output: Dict) -> Dict:
        """
        Clean agent output dict by removing nulls and empty fields.
        """
        cleaned = self.remove_nulls(output)
        if isinstance(cleaned, dict):
            return {k: v for k, v in cleaned.items() if v not in (None, [], {}, "")}
        return cleaned
    
