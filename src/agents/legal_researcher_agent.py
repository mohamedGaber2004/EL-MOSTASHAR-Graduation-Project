import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Utils.agents_enums import AgentsEnums
from src.Prompts.legal_researcher_agent import (
    LEGAL_RESEARCHER_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)
from src.Graph.graph_helpers import (
    _build_fallback_package,
    _parse_llm_json,
)

logger = logging.getLogger(__name__)


class LegalResearcherAgent(AgentBase):
    """Builds a legal research package for every charge."""

    def __init__(self, kg: LegalKnowledgeGraph, vector_store):
        super().__init__("LEGAL_RESEARCHER_MODEL","LEGAL_RESEARCHER_TEMP",LEGAL_RESEARCHER_AGENT_PROMPT,llm_provider="llama")
        self.kg = kg
        self.vs = vector_store

    # ── context retrieval ─────────────────────────────────────────────

    def _extract_search_keywords(self, charge) -> list[str]:
        """
        Break description into short searchable terms instead of
        searching the full sentence which likely won't match verbatim.
        """
        candidates = []

        # incident_type is usually short and precise — try first
        if getattr(charge, "incident_type", None):
            candidates.append(charge.incident_type)          # e.g. "اتجار في مخدرات"

        # from description, take first 3 words only
        desc = getattr(charge, "description", None)
        if desc:
            words = desc.strip().split()
            candidates.append(" ".join(words[:3]))            # e.g. "الاتجار بالمواد المخدرة" → "الاتجار بالمواد"

        # single important words as fallback
        if getattr(charge, "law_code", None):
            law_map = {
                "قانون مكافحة المخدرات": "مخدر",
                "قانون العقوبات":        "عقوبة",
                "قانون الأسلحة والذخائر": "سلاح",
            }
            keyword = law_map.get(charge.law_code)
            if keyword:
                candidates.append(keyword)

        return candidates

    def _retrieve_statute_from_kg(self, charge) -> list[dict]:
        if self.kg is None:
            return []

        results  = []
        law_code = getattr(charge, "law_code", None)
        law_id   = AgentsEnums.law_code_to_kg_id(law_code) if law_code else None

        # ── strategy 1: direct lookup by article_number ──────────────────
        article_number = getattr(charge, "article_number", None)
        if article_number and law_id:
            try:
                history = self.kg.query_article_history(law_id, str(article_number))
                if history:
                    latest = sorted(history, key=lambda x: x.get("version") or "0000-00-00")[-1]
                    text   = (latest.get("text") or "").strip()
                    if text:
                        results.append({
                            "article_number": article_number,
                            "law_code":       law_code,
                            "text":           text[:400],
                            "is_amended":     latest.get("is_amended", False),
                        })
                        logger.info("KG direct lookup — found م%s من %s", article_number, law_code)
                        return results
            except Exception as e:
                logger.warning("KG direct lookup failed for م%s: %s", article_number, e)

        # ── strategy 2: short keyword search ─────────────────────────────
        for keyword in self._extract_search_keywords(charge):
            if not keyword:
                continue
            try:
                hits = self.kg.search_articles_by_keyword(
                    keyword=keyword,
                    law_id=law_id,
                    k=2,
                )
                for hit in hits:
                    text = (hit.get("text") or "").strip()
                    if text and hit.get("article_number") not in [r["article_number"] for r in results]:
                        results.append({
                            "article_number": hit.get("article_number"),
                            "law_code":       hit.get("law_id") or law_code,
                            "text":           text[:400],
                            "is_amended":     hit.get("is_amended", False),
                        })
                if results:
                    logger.info("KG keyword [%s] → %d result(s)", keyword, len(results))
                    break   # found something — stop trying more keywords
            except Exception as e:
                logger.warning("KG keyword search failed for [%s]: %s", keyword, e)

        return results

    def _retrieve_all_contexts(self, charges: list) -> tuple[list, list]:
        contexts: list = []
        failed:   list = []

        for charge in charges:
            try:
                # ── KG: statute text ─────────────────────────────────────
                kg_statutes = self._retrieve_statute_from_kg(charge)

                # ── VS: cassation rulings only ───────────────────────────
                cassation   = self._retrieve_cassation_rulings(charge)

                contexts.append({
                    "charge":         getattr(charge, "statute", str(charge)),
                    "law_code":       getattr(charge, "law_code", None),
                    "article_number": getattr(charge, "article_number", None),
                    "description":    getattr(charge, "description", None),
                    "kg_statutes":    kg_statutes,      # ← from KG only
                    "cassation_docs": cassation,        # ← from VS only
                })

            except Exception as e:
                failed.append(getattr(charge, "statute", str(charge)))
                contexts.append({
                    "charge":         getattr(charge, "statute", str(charge)),
                    "error":          str(e),
                    "kg_statutes":    [],
                    "cassation_docs": [],
                })

        return contexts, failed

    # ── prompt builder ────────────────────────────────────────────────

    def _build_prompt(self, state, all_contexts: list) -> str:
        case_summary = json.dumps(
            {
                "defendants": [d.name for d in state.defendants],
                "charges":    [c.statute for c in state.charges],
                "incidents":  [
                    i.incident_description
                    for i in state.incidents
                    if i.incident_description
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            "## معطيات القضية:\n"
            f"{case_summary}\n\n"

            "## السياق المسترجع من قاعدة البيانات والـ Vector Store\n"
            "(يشمل: نصوص المواد + العقوبات + مبادئ النقض + التعديلات):\n"
            f"{json.dumps(all_contexts, ensure_ascii=False, indent=2)}\n\n"

            "## التعليمات:\n"
            "ابنِ حزمة بحث قانوني لكل تهمة من المعطيات أعلاه فقط.\n"
            "لكل تهمة:\n"
            "  - حدد أركانها ونطاق عقوبتها والمواد ذات الصلة.\n"
            "  - أدرج مبادئ محكمة النقض الواردة في cassation_docs إن وجدت.\n"
            "  - استند فقط إلى ما في السياق المسترجع — لا تخترع مواد أو أحكاماً.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        charges_count = len(state.charges or [])
        logger.info("LegalResearcherAgent starting — %d charge(s)", charges_count)

        if not state.charges:
            logger.info("No charges — returning empty research package")
            return self._empty_update(state, "legal_researcher", {
                "research_packages":          [],
                "procedural_principles":      [],
                "relevant_cassation_rulings": [],
            })

        all_contexts, failed_charges = self._retrieve_all_contexts(state.charges)

        if len(failed_charges) == charges_count:
            logger.warning("All charges failed context retrieval — using fallback")
            legal_package = _build_fallback_package(all_contexts)
            legal_package["_meta"] = {
                "source":         "fallback",
                "failed_charges": failed_charges,
            }
            return self._empty_update(state, "legal_researcher", legal_package)

        prompt = self._build_prompt(state, all_contexts)
        logger.debug("Prompt built — invoking LLM")

        try:
            response = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            legal_package = _parse_llm_json(response.content)
        except Exception as e:
            logger.error("LLM invocation failed: %s — using fallback", e)
            legal_package = _build_fallback_package(all_contexts)

        if not isinstance(legal_package, dict):
            logger.error("LLM response not a dict — using fallback")
            legal_package = _build_fallback_package(all_contexts)
        else:
            logger.info(
                "Legal research complete — %d package(s)",
                len(legal_package.get("research_packages", [])),
            )

        legal_package["_meta"] = {
            "charges_count":  charges_count,
            "failed_charges": failed_charges,
            "contexts_count": len(all_contexts),
            "kg_connected":   self.kg is not None,
            "vs_connected":   self.vs is not None,
        }

        return self._empty_update(state, "legal_researcher", legal_package)