import json
import logging
from typing import Optional

from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage

from ..agent_base.agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Graph.state import AgentState
from src.routers.kg_retriever_router import get_kg_retriever
from src.Utils.file_utils.article_number_normalizer import generate_article_number_variants
from src.agents.legal_research_agent.legal_researcher_prompt import (
    LEGAL_RESEARCHER_AGENT_PROMPT,
)

logger = logging.getLogger(__name__)


class LegalResearcherAgent(AgentBase):

    def __init__(self, kg: LegalKnowledgeGraph, embeddings, vector_store):
        super().__init__(
            "LEGAL_RESEARCHER_MODEL",
            "LEGAL_RESEARCHER_TEMP",
            LEGAL_RESEARCHER_AGENT_PROMPT,
        )
        self.kg         = kg
        self.vs         = vector_store
        self.embeddings = embeddings

    # =========================================================================
    # Helpers
    # =========================================================================
    def _build_fallback_package(self) -> dict:
        return {"case_articles": [], "applied_principles": [], "_fallback": True}

    @staticmethod
    def _resolve_law_id(law_code: str) -> Optional[str]:
        """Arabic law name → Neo4j law_id (or pass-through if already English)."""
        clean_code = (law_code or "").strip()
        
        _MAP = {
            "قانون العقوبات":                      "penal_code",
            "قانون الإجراءات الجنائية":            "criminal_procedure",
            "قانون مكافحة المخدرات":               "anti_drugs",
            "قانون مكافحة الإرهاب":                "anti_terror",
            "قانون الأسلحة والذخائر":              "weapons_ammunition",
            "قانون مكافحة جرائم تقنية المعلومات": "cybercrime",
            "قانون مكافحة غسل الأموال":            "money_laundering",
            "قانون الطوارئ":                       "emergency_law",
            "الدستور الجنائي":                     "criminal_constitution",
        }
        
        # If it's already an English target value, return it directly
        if clean_code in _MAP.values():
            return clean_code
            
        return _MAP.get(clean_code)

    # =========================================================================
    # Article retrieval
    # =========================================================================
    @traceable(name="fetch_article_exact", run_type="retriever")
    def _fetch_article_exact(self, law_code: str, article_number: str) -> Optional[dict]:
        if self.kg is None:
            logger.warning("KG not available for exact lookup")
            return None

        resolved_law_id = self._resolve_law_id(law_code)
        if not resolved_law_id:
            logger.warning(
                "Cannot resolve law_id for law_code=%s — will fall back to semantic",
                law_code,
            )
            return None

        variants = generate_article_number_variants(article_number)
        if not variants:
            variants = [str(article_number).strip()]

        for variant in variants:
            try:
                text = self.kg.get_article(
                    law_id=resolved_law_id,
                    article_number=variant,
                )
            except Exception as exc:
                logger.warning(
                    "KG.get_article raised for %s/%s: %s",
                    resolved_law_id, variant, exc,
                )
                continue

            if text is not None:
                logger.info(
                    "Exact KG hit ✓ — law_id=%s article=%s (variant=%s)",
                    resolved_law_id, article_number, variant,
                )
                return {
                    "article_number": variant,
                    "article_number_input": article_number,
                    "law_id": resolved_law_id,
                    "text": text,
                }

        logger.info(
            "Exact KG miss — law_id=%s article=%s variants=%s",
            resolved_law_id, article_number, variants,
        )
        return None

    @traceable(name="kg_retrieve_semantic", run_type="retriever")
    def _retrieve_semantic(self, law_code: str, article_number: str, k: int, threshold: float) -> list[dict]:
        """
        Semantic fallback — builds a natural-language query from law_code +
        article_number, transforms it, then calls the hybrid retriever.
        """
        if self.kg is None or self.embeddings is None:
            logger.warning("KG or embeddings not available — skipping semantic retrieval")
            return []

        # Build a natural-language query for the transformer
        raw_query = f"المادة {article_number} من {law_code}".strip(" -من")
        if not raw_query:
            return []

        # Query transformation
        try:
            transformed = (self.query_transformation(raw_query) or {})
            clean_query = (transformed.get("article_query") or "").strip()
        except Exception as exc:
            logger.warning("query_transformation failed for [%s]: %s", raw_query[:60], exc)
            clean_query = ""

        if not clean_query:
            logger.warning(
                "query_transformation returned empty article_query for [%s] — "
                "skipping semantic retrieval",
                raw_query[:80],
            )
            return []

        # Hybrid retrieval
        try:
            records = get_kg_retriever().retrieve(
                question  = clean_query,
                k         = k,
                threshold = threshold,
            )
        except Exception as exc:
            logger.warning("kg retriever failed for [%s]: %s", clean_query[:60], exc)
            return []

        if records is None:
            return []

        return [
            {
                "article_number": str(r["article_number"]),
                "law_id":         str(r["law_id"]),
                "text":           str(r["text"]),
            }
            for r in records.sources
            if r.get("article_number") and float(r.get("score", 0)) >= threshold
        ]

    @traceable(name="retrieve_articles_for_charge", run_type="chain")
    def _retrieve_articles_for_charge(self, law_code:str, article_number: str, k:int= 15, threshold:float = 0.45) -> list[dict]:
        # Path 1
        if law_code and article_number:
            article = self._fetch_article_exact(law_code, article_number)
            if article:
                return [article]
            logger.warning(
                "Exact miss for %s / %s — falling back to semantic",
                law_code, article_number,
            )

        # Path 2
        return self._retrieve_semantic(law_code, article_number, k, threshold)

    @traceable(name="resolve_articles_for_charges", run_type="chain")
    def _resolve_articles_for_charges(self, charges: list) -> list[dict]:
        """Deduplicated articles across all charges."""
        seen:   set[str]   = set()
        result: list[dict] = []

        for charge in charges:
            law_code       = (getattr(charge, "law_code",       None) or "").strip()
            article_number = (getattr(charge, "article_number", None) or "").strip()

            if not law_code and not article_number:
                logger.warning("Skipping charge with no law_code or article_number: %s", charge)
                continue

            for article in self._retrieve_articles_for_charge(
                law_code       = law_code,
                article_number = article_number,
            ):
                key = f"{article['law_id']}:{article['article_number']}"
                if key not in seen:
                    seen.add(key)
                    result.append(article)

        if result:
            logger.info("Article resolution complete — %d unique article(s)", len(result))
        else:
            logger.warning("No articles resolved for any charge")

        return result

    # =========================================================================
    # Principles retrieval
    # =========================================================================
    @traceable(name="retrieve_principles_for_query", run_type="chain")
    def _retrieve_principles_for_query(self, query_text: str, source: str) -> list[dict]:
        """Transform query then retrieve from vector store."""
        if not query_text or not query_text.strip():
            return []

        try:
            transformed = (self.query_transformation(query_text) or {})
            clean_query = (transformed.get("principle_query") or "").strip()
        except Exception as exc:
            logger.warning(
                "query_transformation failed for source=%s: %s", source, exc
            )
            return []

        if not clean_query:
            logger.warning(
                "Empty principle_query for source=%s input=[%s...]",
                source, query_text[:80],
            )
            return []

        try:
            docs = self.retrieve_principles(clean_query, self.vs)
        except Exception as exc:
            logger.warning("retrieve_principles failed for source=%s: %s", source, exc)
            return []

        return [
            {
                "principle_text": doc.page_content,
                "court_source":   (doc.metadata.get("source_file") if doc.metadata else None),
                "applicable_to":  source,
            }
            for doc in docs
        ]

    @traceable(name="resolve_principles", run_type="chain")
    def _resolve_principles(self, charges:list, procedural_violations: list, incidents:list) -> list[dict]:
        """Deduplicated principles across charges, violations, and incidents."""
        if self.vs is None:
            logger.warning("No vector store — skipping principles retrieval")
            return []

        seen:   set[str] = set()
        result: list     = []

        def _accumulate(query_text: str, source: str) -> None:
            for item in self._retrieve_principles_for_query(query_text, source):
                key = item["principle_text"][:80]
                if key not in seen:
                    seen.add(key)
                    result.append(item)

        for charge in charges:
            parts = [
                getattr(charge, attr, None)
                for attr in (
                    "law_code", "article_number", "description",
                    "incident_type", "charge_classification", "attempt_flag",
                )
            ]
            _accumulate("---".join(str(p) for p in parts if p), "charge")

        for violation in procedural_violations:
            desc = (getattr(violation, "issue_description", None) or str(violation)).strip()
            _accumulate(desc, "procedural_violation")

        for incident in incidents:
            desc = (getattr(incident, "incident_description", None) or str(incident)).strip()
            _accumulate(desc[:120], "incident_narrative")

        if result:
            logger.info("Principles resolution complete — %d unique principle(s)", len(result))
        else:
            logger.warning("No principles resolved from any source")

        return result

    # =========================================================================
    # Prompt builder
    # =========================================================================

    @traceable(name="build_legal_research_prompt", run_type="prompt")
    def _build_prompt(self, state: AgentState, case_articles: list[dict], applied_principles: list) -> str:
        case_summary = json.dumps(
            {
                "charges": [
                    {
                        "incident_type": getattr(c, "incident_type", None),
                        "description":   getattr(c, "description",   None),
                    }
                    for c in (state.charges or [])
                ],
                "incidents": [
                    getattr(i, "incident_description", None)
                    for i in (state.incidents or [])
                    if getattr(i, "incident_description", None)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            "## معطيات القضية:\n"
            f"{case_summary}\n\n"
            "## المواد القانونية المسترجعة من قاعدة المعرفة (KG):\n"
            f"{json.dumps(case_articles, ensure_ascii=False, indent=2)}\n\n"
            "## مبادئ محكمة النقض المسترجعة من الـ Vector Store:\n"
            f"{json.dumps(applied_principles, ensure_ascii=False, indent=2)}\n\n"
        )

    # =========================================================================
    # Main entry
    # =========================================================================

    @traceable(name="legal_researcher_agent_run", run_type="chain")
    def run(self, state: AgentState):
        logger.info(
            "LegalResearcherAgent starting — %d charge(s)",
            len(state.charges or []),
        )

        if not state.charges:
            return self._empty_update(state, "legal_researcher", {
                "case_articles":      [],
                "applied_principles": [],
            })

        procedural_violations = (
            state.procedural_audit.violations if state.procedural_audit else []
        )

        case_articles      = self._resolve_articles_for_charges(state.charges)
        applied_principles = self._resolve_principles(
            state.charges, procedural_violations, state.incidents or []
        )

        prompt = self._build_prompt(state, case_articles, applied_principles)

        try:
            response      = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            legal_package = self._parse_agent_json(response.content)
        except Exception as exc:
            logger.error("LLM invocation failed: %s — using fallback", exc)
            legal_package = self._build_fallback_package()

        if not isinstance(legal_package, dict):
            logger.error("LLM response is not a dict — using fallback")
            legal_package = self._build_fallback_package()
        else:
            logger.info(
                "Legal research complete — %d case_article(s), %d principle(s)",
                len(legal_package.get("case_articles",      [])),
                len(legal_package.get("applied_principles", [])),
            )

        return self._empty_update(state, "legal_researcher", {
            "case_articles": legal_package.get("case_articles", []),
            "applied_principles": (
                legal_package.get("applied_principles") or applied_principles
            ),
        })