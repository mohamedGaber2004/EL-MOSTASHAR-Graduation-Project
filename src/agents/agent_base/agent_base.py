import logging
import time
import random
import json
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from src.Config import get_settings
from src.Config.log_config import logging
from src.Chunking.chunking import get_na2d_chunks
from src.Chunking.chunking_enums import Na2dOutputKey
from src.retriever.vs_retriever.vs_reriever import get_hybrid_retriever
from src.agents.agent_base.agents_enums import AgentsEnums
from src.agents.agent_base.agent_base_prompts import LEGAL_QUERY_TRANSFORMATION_PROMPT
from src.routers.vs_retriever_router import HybridRetrieveRequest, RetrieveResponse, RetrievedDoc

try:
    from json_repair import repair_json
except ImportError:
    def repair_json(s: str) -> str:
        return s

logger = logging.getLogger(__name__)

# ── module-level na2d singleton ───────────────────────────────────────────────

_NA2D_CACHE: Optional[dict] = None

def _get_cached_na2d() -> dict:
    global _NA2D_CACHE
    if _NA2D_CACHE is None:
        logger.info("Building na2d chunk cache (first call)…")
        _NA2D_CACHE = get_na2d_chunks()
        rulings    = len(_NA2D_CACHE.get(Na2dOutputKey.RULINGS,    []))
        principles = len(_NA2D_CACHE.get(Na2dOutputKey.PRINCIPLES, []))
        logger.info("na2d cache ready — rulings=%d | principles=%d", rulings, principles)
    return _NA2D_CACHE


class AgentBase:
    """Base class for all court-simulation agents."""

    def __init__(self, model_config_key: str, temp_config_key: str, prompt: str) -> None:
        self.cfg          = get_settings()
        self.agent        = model_config_key
        self.model_name   = getattr(self.cfg, model_config_key)
        self.temperature  = getattr(self.cfg, temp_config_key)
        self.prompt       = prompt
        self._llm         = self._init_llm(self.agent)

        # ── instance-level retriever cache ────────────────────────────────────
        self._retriever_cache: dict = {}

        logger.debug(
            "Initialized %s | model=%s | temp=%s",
            self.__class__.__name__, self.model_name, self.temperature,
        )

    # ── time helper ───────────────────────────────────────────────────────────
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # ── hybrid retrieval ──────────────────────────────────────────────────────
    def hybrid_retrieve_logic(
        self,
        req: HybridRetrieveRequest,
        vs,
    ) -> RetrieveResponse:
        """
        Run a hybrid dense+sparse retrieval over the na2d corpus.
        The hybrid retriever is cached per (vs identity, k, weights) so the
        BM25 index is only built once per unique parameter combination.
        """
        if req.dense_weight <= 0 and req.sparse_weight <= 0:
            raise ValueError(
                "At least one of dense_weight / sparse_weight must be > 0."
            )

        na2d = _get_cached_na2d()
        docs = na2d[Na2dOutputKey.RULINGS] + na2d[Na2dOutputKey.PRINCIPLES]

        cache_key = (id(vs), req.k, req.dense_weight, req.sparse_weight)
        if cache_key not in self._retriever_cache:
            logger.debug(
                "Building hybrid retriever — k=%d dense=%.2f sparse=%.2f (cache miss)",
                req.k, req.dense_weight, req.sparse_weight,
            )
            self._retriever_cache[cache_key] = get_hybrid_retriever(
                vs,
                docs,
                k=req.k,
                dense_weight=req.dense_weight,
                sparse_weight=req.sparse_weight,
            )

        retriever = self._retriever_cache[cache_key]
        results   = retriever.invoke(req.query)

        return RetrieveResponse(
            query    = req.query,
            strategy = f"hybrid(dense={req.dense_weight:.2f}, sparse={req.sparse_weight:.2f})",
            hits     = [
                RetrievedDoc(page_content=d.page_content, metadata=d.metadata)
                for d in results
            ],
        )

    def retrieve_principles(self, query_text: str, vs) -> List[RetrievedDoc]:
        """
        Retrieve judicial principles for `query_text`.
        Returns a list of RetrievedDoc — empty list on any failure.
        """
        if not vs or not query_text or not query_text.strip():
            return []

        ntotal = getattr(getattr(vs, "index", None), "ntotal", None)
        k      = max(1, min(4, ntotal)) if ntotal else 4

        try:
            response = self.hybrid_retrieve_logic(
                req=HybridRetrieveRequest(query=query_text, k=k),
                vs=vs,
            )
            return response.hits  # List[RetrievedDoc]
        except Exception as e:
            logger.warning("retrieve_principles failed: %s", e)
            return []

    # ── agent helpers ─────────────────────────────────────────────────────────
    def _json_safe(self, obj) -> Any:
        from enum import Enum as _Enum
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):          # ← add this
            return self._json_safe(obj.model_dump())
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, _Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(i) for i in obj]
        return obj

    def _lawcode_to_law_id(self, charge) -> Optional[str]:
        if not getattr(charge, "law_code", None):
            return None
        return AgentsEnums.law_code_to_kg_id(str(charge.law_code))

    def _safe_dump(self, pydantic_obj) -> dict:
        return self._json_safe(pydantic_obj.model_dump())

    def _extract_with_entity_validation(self, data: dict) -> dict:
        from src.agents.data_ingestion_agent.data_ingestion_output_model import (
            Charge, CaseIncident, Confession, CriminalRecord,
            DefenseDocument, Defendant, Evidence, LabReport,
            WitnessStatement, CriminalProceedings,
        )

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

        result = {}
        for key, pydantic_class in entity_mappings.items():
            if key not in data:
                continue

            raw_value = data[key]
            if isinstance(raw_value, dict):
                raw_value = [raw_value]
            if not isinstance(raw_value, list):
                continue

            validated_items = []
            for item in raw_value:
                if not isinstance(item, dict):
                    logger.debug("[%s] skipping non-dict item: %s", key, repr(item)[:80])
                    continue
                try:
                    validated    = pydantic_class(**{k: v for k, v in item.items() if v is not None})
                    dumped       = validated.model_dump(exclude_none=True)
                    validated_items.append(dumped if dumped else item)
                except Exception as e:
                    logger.debug(
                        "[%s] validation failed — keeping raw. error=%s item=%s",
                        key, str(e)[:120], repr(item)[:120],
                    )
                    validated_items.append(item)

            result[key] = validated_items
            if not validated_items and raw_value:
                logger.warning(
                    "[%s] all %d item(s) dropped during validation — sample: %s",
                    key, len(raw_value), repr(raw_value[0])[:200],
                )

        if "case_meta" in data and isinstance(data["case_meta"], dict):
            result["case_meta"] = data["case_meta"]

        return result

    # ── JSON parsers ──────────────────────────────────────────────────────────
    def _parse_llm_json(self, content: Any) -> dict:
        """Parse + entity-validate LLM output (data ingestion agent)."""
        if isinstance(content, list):
            content = "".join(
                block["text"] if isinstance(block, dict) and "text" in block else str(block)
                for block in content
            )

        if not content or not isinstance(content, str):
            logger.warning("Empty or invalid content: type=%s repr=%s", type(content), repr(content)[:100])
            return {}

        try:
            cleaned  = self._strip_md_fences(content)
            repaired = repair_json(cleaned)
            data     = json.loads(repaired)
            if isinstance(data, str):
                data = json.loads(repair_json(data))
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

    def _parse_agent_json(self, content: Any) -> dict:
        """Parse LLM output for non-ingestion agents (no entity validation)."""
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        logger.debug("_parse_agent_json received: type=%s | value=%s", type(content), repr(content[:300]) if isinstance(content, str) else repr(content))
        if not content or not isinstance(content, str):
            logger.warning("Empty or invalid content: type=%s", type(content))
            return {}

        try:
            cleaned  = self._strip_md_fences(content)
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

    @staticmethod
    def _strip_md_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    # ── query transformation ──────────────────────────────────────────────────
    def query_transformation(self, query: str) -> dict:
        fallback = {"article_query": query, "principle_query": query}

        if not query:
            logger.warning("No query for transforming.")
            return fallback

        KEY_ALIASES = {
            "article_text":     "article_query",
            "article":          "article_query",
            "query_text":       "article_query",
            "principle":        "principle_query",
            "principles_query": "principle_query",
        }

        try:
            chain  = self._llm | JsonOutputParser()
            result = chain.invoke([
                SystemMessage(content=LEGAL_QUERY_TRANSFORMATION_PROMPT),
                HumanMessage(content=f"النصوص القانونيه: {query}"),
            ])

            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}
            if not isinstance(result, dict):
                logger.warning("query_transformation: unexpected type=%s — using fallback", type(result))
                return fallback

            result = {KEY_ALIASES.get(k, k): v for k, v in result.items()}
            result.setdefault("article_query",   query)
            result.setdefault("principle_query", query)
            return result

        except Exception as e:
            logger.warning("query_transformation failed: %s", e)
            return fallback

    # ── LLM factory ───────────────────────────────────────────────────────────
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

    _FACTORY_MAP: dict[str, tuple[str, str]] = {
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

    _PROVIDER_KEY_MAP: dict[str, str] = {
        "open_router": "open_router_llm",
        "groq":        "groq_llm",
        "mistral":     "mistral_llm",
        "google":      "google_llm",
        "vertex":      "vertex_llm",
        "cloudflare":  "cloudflare_llm",
        "celebras":    "celebras_llm",
    }

    def _init_llm(self, agent_name: str):
        if agent_name not in self.AGENT_CONFIG:
            raise ValueError(f"Unknown agent key: {agent_name!r}. Add it to AgentBase.AGENT_CONFIG.")

        model_attr, provider_attr = self.AGENT_CONFIG[agent_name]
        provider = getattr(self.cfg, provider_attr, "open_router").strip().lower()

        module_path, func_name = self._FACTORY_MAP[agent_name]
        import importlib
        factory = getattr(importlib.import_module(module_path), func_name)
        llm_map = factory()

        if provider not in self._PROVIDER_KEY_MAP:
            raise ValueError(
                f"Unknown provider {provider!r} for agent {agent_name!r}. "
                f"Choose from: {list(self._PROVIDER_KEY_MAP)}"
            )

        key = self._PROVIDER_KEY_MAP[provider]
        if key not in llm_map:
            raise KeyError(
                f"Factory for {agent_name!r} did not return key {key!r}. "
                f"Available: {list(llm_map)}"
            )

        logger.info(
            "✅ LLM ready — agent=%s | provider=%s | model=%s",
            agent_name, provider, getattr(self.cfg, model_attr),
        )
        return llm_map[key]

    # ── retry wrapper ─────────────────────────────────────────────────────────
    def _llm_invoke_with_retries(self,llm_callable: Callable,messages: List[Any],max_retries: int = None,backoff_base: float = 2.0) -> Any:
        if max_retries is None:
            max_retries = self.cfg.INVOKATION_MAX_RETRIES

        for attempt in range(1, max_retries + 1):
            try:
                result = llm_callable.invoke(messages)

                if not getattr(result, "content", None) or not str(result.content).strip():
                    if attempt < max_retries:
                        wait = backoff_base * (2 ** (attempt - 1)) + random.random()
                        logger.warning("Empty response on attempt %d — retrying in %.2fs", attempt, wait)
                        time.sleep(wait)
                        continue
                    logger.error("Empty response after %d attempt(s) — giving up", attempt)
                    raise RuntimeError(f"LLM returned empty content after {attempt} attempt(s)")

                if attempt > 1:
                    logger.info("LLM succeeded on attempt %d", attempt)
                return result

            except Exception as e:
                err_raw = str(e)
                err     = err_raw.lower()

                if "402" in err_raw and "credits" in err_raw:
                    logger.error("OpenRouter 402 — insufficient credits for %s.", self.__class__.__name__)
                    raise RuntimeError(
                        f"Insufficient OpenRouter credits — cannot invoke {self.__class__.__name__}."
                    ) from e

                is_rate = any(k in err for k in AgentsEnums.AGENT_INVOKATION_ERRORS)
                if is_rate and attempt < max_retries:
                    retry_after = self._extract_retry_after(err_raw)
                    if retry_after:
                        wait = min(120.0, retry_after) + random.uniform(1.0, 3.0)
                        logger.warning(
                            "Rate-limit on attempt %d — honouring Retry-After=%.0fs → waiting %.1fs",
                            attempt, retry_after, wait,
                        )
                    else:
                        wait = min(60.0, backoff_base * (2 ** (attempt - 1))) + random.random()
                        logger.warning("Rate-limit / timeout on attempt %d — waiting %.2fs", attempt, wait)
                    time.sleep(wait)
                    continue

                logger.error("LLM invocation failed after %d attempt(s): %s", attempt, e)
                raise

        raise RuntimeError(f"LLM failed after {max_retries} attempt(s)")

    @staticmethod
    def _extract_retry_after(err_raw: str) -> Optional[float]:
        import re
        m = re.search(r"['\"]retry_after_seconds['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)", err_raw)
        if m:
            return float(m.group(1))
        m = re.search(r"['\"]retry_after_seconds_raw['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)", err_raw)
        if m:
            return float(m.group(1))
        return None

    # ── state update helper ───────────────────────────────────────────────────
    def _empty_update(
        self,
        state: Any,
        agent_key: str,
        output: Any,
        inter_agent_delay: float = None,
    ) -> dict:
        if inter_agent_delay is None:
            inter_agent_delay = self.cfg.INTER_AGENTS_DELAY

        logger.info("Agent [%s] complete — updating state.", agent_key)
        if inter_agent_delay > 0:
            logger.debug("Inter-agent delay: %.2fs", inter_agent_delay)
            time.sleep(inter_agent_delay)

        update: dict = {"completed_agents": [agent_key]}
        if isinstance(output, dict):
            update.update(output)
        return update