import logging, time, random, json 
from datetime import datetime, timezone
from typing import Any, Callable, List, Dict

from src.Config import get_settings
from src.Graph.shared_resources import get_vector_store 
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

    # retrieve principles based on query text
    def retrieve_principles(self, query_text: str, vs) -> list:
        if not vs:
            logger.warning("No vector store available — skipping principles retrieval.")
            return []
        if not query_text or not query_text.strip():
            logger.warning("Empty query text — skipping principles retrieval.")
            return []

        # ── Clamp k to index size before any call ─────────────────────────────
        ntotal = getattr(getattr(vs, "index", None), "ntotal", None)
        k = max(1, min(4, ntotal)) if ntotal else 4

        try:
            docs = vs.similarity_search(query_text, k=k)
            return docs or []

        except AssertionError as e:
            logger.warning(
                "retrieve_principles — FAISS assertion failed "
                "(ntotal=%s, k=%d) — likely embedding dimension mismatch: %s",
                ntotal, k, str(e) or "(no message)",
            )
            return []

        except Exception as e:
            logger.warning(
                "retrieve_principles failed for [%s...]: %s — %s",
                query_text[:60], type(e).__name__, str(e) or "(no message)",
            )
            return []

    # _agent_helpers
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

    def _lawcode_to_law_id(self,charge) -> str | None:
        if not getattr(charge, "law_code", None):
            return None
        mapping = AgentsEnums.law_code_to_kg_id(str(charge.law_code))
        return mapping
    
    def _safe_dump(self,pydantic_obj) -> dict:
        return self._json_safe(pydantic_obj.model_dump())

    def _extract_with_entity_validation(self, data: dict) -> dict:
        """
        Validate each entity list in `data` using the appropriate Pydantic model.
        Falls back to including the raw item dict if validation fails.
        """
        from src.Graph.states_and_schemas.main_entity_classes import (
            Charge,
            CaseIncident,
            Confession,
            CriminalRecord,
            DefenseDocument,
            Defendant,
            Evidence,
            LabReport,
            WitnessStatement,
            CriminalProceedings
        )

        result = {}

        entity_mappings = {
            "defendants":           Defendant,
            "charges":              Charge,
            "incidents":            CaseIncident,
            "evidences":            Evidence,
            "lab_reports":          LabReport,
            "witness_statements":   WitnessStatement,
            "confessions":          Confession,
            "defense_documents":    DefenseDocument,
            "criminal_records":     CriminalRecord,
            "criminal_proceedings": CriminalProceedings,
        }

        for key, pydantic_class in entity_mappings.items():
            if key not in data:
                continue

            raw_value = data[key]  # ✅ renamed to avoid conflict with outer scope

            # coerce singular dict to list
            if isinstance(raw_value, dict):
                raw_value = [raw_value]

            if not isinstance(raw_value, list):
                continue

            raw_list = raw_value  # ✅ now safe
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

            result[key] = validated_items
            if not validated_items and raw_list:
                logger.warning(
                    "[%s] All %d item(s) dropped during validation — raw sample: %s",
                    key, len(raw_list), repr(raw_list[0])[:200],
                )

        # case_meta — no CaseMeta Pydantic model exists; pass through as raw dict
        if "case_meta" in data and isinstance(data["case_meta"], dict):
            result["case_meta"] = data["case_meta"]

        return result

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
    
    def _parse_agent_json(self, content) -> dict:
        """
        Parse LLM JSON output for non-ingestion agents.
        Does NOT run entity validation — just cleans and parses.
        """
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        if not content or not isinstance(content, str):
            logger.warning("Empty or invalid content: type=%s", type(content))
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
            data     = json.loads(repaired)

            if isinstance(data, str):
                data = json.loads(repair_json(data))

        except Exception as e:
            logger.error("JSON parse failed: %s | content[:200]=%s", e, content[:200])
            return {}

        if not isinstance(data, dict):
            logger.warning("Parsed JSON is not a dict: %s", type(data))
            return data 

        logger.info("Agent JSON parsed — %d top-level key(s)", len(data))
        return data

    # ── LLM factory ───────────────────────────────────────────────────
    AGENT_CONFIG: dict[str, tuple[str, str]] = {
        "DATA_INGESTION_MODEL":      ("DATA_INGESTION_MODEL",      "DATA_INGESTION_PROVIDER"),
        "PROCEDURAL_AUDITOR_MODEL":  ("PROCEDURAL_AUDITOR_MODEL",  "PROCEDURAL_AUDITOR_PROVIDER"),
        "LEGAL_RESEARCHER_MODEL":    ("LEGAL_RESEARCHER_MODEL",    "LEGAL_RESEARCHER_PROVIDER"),
        "EVIDENCE_SCORING_MODEL":    ("EVIDENCE_SCORING_MODEL",    "EVIDENCE_SCORING_PROVIDER"),
        "DEFENSE_AGENT_MODEL":       ("DEFENSE_AGENT_MODEL",       "DEFENSE_AGENT_PROVIDER"),
        "JUDJICAL_PRINCIPLE_AGENT":  ("JUDJICAL_PRINCIPLE_AGENT",  "JUDJICAL_PRINCIPLE_PROVIDER"),
        "JUDGE_MODEL":               ("JUDGE_MODEL",               "JUDGE_PROVIDER"),
        "CONFESSION_VALIDITY_MODEL": ("CONFESSION_VALIDITY_MODEL", "CONFESSION_VALIDITY_PROVIDER"),
        "WITNESS_CREDIBILITY_MODEL": ("WITNESS_CREDIBILITY_MODEL", "WITNESS_CREDIBILITY_PROVIDER"),
        "PROSECUTION_ANALYST_MODEL": ("PROSECUTION_ANALYST_MODEL", "PROSECUTION_ANALYST_PROVIDER"),
        "SENTENCING_MODEL":          ("SENTENCING_MODEL",          "SENTENCING_PROVIDER"),
    }

    def _init_llm(self, agent_name:str):
        """Instantiate only the requested LLM provider."""

        if agent_name not in self.AGENT_CONFIG:
            raise ValueError(
                f"Unknown agent key: {agent_name!r}. "
                f"Add it to AgentBase.AGENT_CONFIG."
            )
        
        model_attr, provider_attr = self.AGENT_CONFIG[agent_name]
        model_name   = getattr(self.cfg, model_attr)
        provider     = getattr(self.cfg, provider_attr, "open_router").strip().lower()
        temperature  = self.temperature

        logger.debug(
            "Initialising LLM — agent=%s | provider=%s | model=%s | temp=%s",
            agent_name, provider, model_name, temperature,
        )

        factory_map = {
            "DATA_INGESTION_MODEL":      ("src.LLMs.DATA_INGESTION_MODEL",      "get_ingesion_model"),
            "PROCEDURAL_AUDITOR_MODEL":  ("src.LLMs.PROCEDURAL_AUDITOR_MODEL",  "get_procedural_model"),
            "LEGAL_RESEARCHER_MODEL":    ("src.LLMs.LEGAL_RESEARCHER_MODEL",    "get_legal_reasearcher_model"),
            "EVIDENCE_SCORING_MODEL":    ("src.LLMs.EVIDENCE_SCORING_MODEL",    "get_evidence_scoring_model"),
            "DEFENSE_AGENT_MODEL":       ("src.LLMs.DEFENSE_AGENT_MODEL",       "get_defense_model"),
            "JUDJICAL_PRINCIPLE_AGENT":  ("src.LLMs.JUDJICAL_PRINCIPLE_AGENT",  "get_judjical_principle_model"),
            "JUDGE_MODEL":               ("src.LLMs.JUDGE_MODEL",               "get_judge_model"),
            "CONFESSION_VALIDITY_MODEL": ("src.LLMs.CONFESSION_VALIDITY_MODEL", "get_confession_validity_model"),
            "WITNESS_CREDIBILITY_MODEL": ("src.LLMs.WITNESS_CREDIBILITY_MODEL", "get_witness_credibility_model"),
            "PROSECUTION_ANALYST_MODEL": ("src.LLMs.PROSECUTION_ANALYST_MODEL", "get_prosecution_analyst_model"),
            "SENTENCING_MODEL":          ("src.LLMs.SENTENCING_MODEL",           "get_sentencing_model"),
        }

        module_path, func_name = factory_map[agent_name]
        import importlib
        module  = importlib.import_module(module_path)
        factory = getattr(module, func_name)
        llm_map = factory() 


        provider_key_map = {
            "open_router": "open_router_llm",
            "groq":        "groq_llm",
            "mistral":      "mistral_llm",
        }

        if provider not in provider_key_map:
            raise ValueError(
                f"Unknown provider {provider!r} for agent {agent_name!r}. "
                f"Choose from: {list(provider_key_map.keys())}"
            )
 
        key = provider_key_map[provider]
 
        if key not in llm_map:
            raise KeyError(
                f"Factory for {agent_name!r} did not return key {key!r}. "
                f"Available keys: {list(llm_map.keys())}"
            )
 
        logger.info(
            "✅ LLM ready — agent=%s | provider=%s | model=%s",
            agent_name, provider, model_name,
        )
        return llm_map[key]

    # ── retry wrapper ─────────────────────────────────────────────────
    def _llm_invoke_with_retries(self,llm_callable: Callable,messages: List[Any],max_retries: int = None,backoff_base: float = 2.0) -> Any:
        if max_retries is None:
            max_retries = self.cfg.INVOKATION_MAX_RETRIES

        for attempt in range(1, max_retries + 1):
            try:
                result = llm_callable.invoke(messages)
                if attempt > 1:
                    logger.info("LLM succeeded on attempt %d", attempt)
                return result

            except Exception as e:
                err_raw = str(e)
                err     = err_raw.lower()

                # ── Insufficient credits: no point retrying ───────────────
                if "402" in err_raw and "credits" in err_raw:
                    logger.error(
                        "OpenRouter 402 — insufficient credits for %s.",
                        self.__class__.__name__,
                    )
                    raise RuntimeError(
                        f"Insufficient OpenRouter credits — cannot invoke {self.__class__.__name__}."
                    ) from e

                # ── Rate-limit / transient: respect Retry-After if present ─
                is_rate = any(k in err for k in AgentsEnums.AGENT_INVOKATION_ERRORS)
                if is_rate and attempt < max_retries:

                    # try to extract Retry-After from the error payload
                    retry_after = self._extract_retry_after(err_raw)

                    if retry_after:
                        # honour the header but cap at 120s and add jitter
                        wait = min(120.0, retry_after) + random.uniform(1.0, 3.0)
                        logger.warning(
                            "Rate-limit on attempt %d — honouring Retry-After=%.0fs → waiting %.1fs",
                            attempt, retry_after, wait,
                        )
                    else:
                        # no header → exponential backoff, cap at 60s
                        wait = min(60.0, backoff_base * (2 ** (attempt - 1))) + random.random()
                        logger.warning(
                            "Rate-limit / timeout on attempt %d — waiting %.2fs", attempt, wait,
                        )

                    time.sleep(wait)
                    continue

                logger.error("LLM invocation failed after %d attempt(s): %s", attempt, e)
                raise

        # should never reach here, but satisfies type checkers
        raise RuntimeError(f"LLM failed after {max_retries} attempt(s)")

    @staticmethod
    def _extract_retry_after(err_raw: str) -> float | None:
        """
        Parse retry_after_seconds out of the OpenRouter error payload string.
        Handles both the nested dict repr and raw JSON.
        Examples seen in logs:
        'retry_after_seconds': 30
        'retry_after_seconds_raw': 29.313
        Returns the value in seconds as a float, or None if not found.
        """
        import re
        # prefer the exact integer field first
        m = re.search(r"['\"]retry_after_seconds['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)", err_raw)
        if m:
            return float(m.group(1))
        # fall back to the _raw variant
        m = re.search(r"['\"]retry_after_seconds_raw['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)", err_raw)
        if m:
            return float(m.group(1))
        return None

    # ── state update helper ───────────────────────────────────────────
    def _empty_update(self,state: Any,agent_key: str,output: Any,inter_agent_delay: float = None) -> dict:
        """
        Return a partial-state dict containing only the keys this agent writes.
        LangGraph merges partial dicts from parallel nodes safely.

        Fields emitted here:
        - completed_agents : Annotated[List, add]  → appended by reducer, safe in parallel
        - errors           : Annotated[List, add]  → appended by reducer, safe in parallel
        Fields NOT emitted here (scalar, would collide in parallel):
        - current_agent, last_updated → dropped entirely; not worth the collision risk
        """
        if inter_agent_delay is None:
            inter_agent_delay = self.cfg.INTER_AGENTS_DELAY

        logger.info("Agent [%s] complete — updating state.", agent_key)
        if inter_agent_delay > 0:
            logger.debug("Inter-agent delay: %.2fs", inter_agent_delay)
            time.sleep(inter_agent_delay)

        # Only completed_agents uses the add reducer — safe for parallel writes.
        update: dict = {
            "completed_agents": [agent_key],
        }

        if isinstance(output, dict):
            update.update(output)

        return update
    