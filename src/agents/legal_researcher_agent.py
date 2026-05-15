import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Utils.Enums.agents_enums import AgentsEnums
from src.routers.kg_router import fetch_article_text
from src.Prompts.legal_researcher_agent import (
    LEGAL_RESEARCHER_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)


class LegalResearcherAgent(AgentBase):
    """
    Maps charges → case_articles (KG) and applied_principles (VS).
    INPUT : charges[], procedural_issues[], incidents[]
    OUTPUT: case_articles[], applied_principles[]
    """

    def __init__(self, kg: LegalKnowledgeGraph, vector_store):
        super().__init__("LEGAL_RESEARCHER_MODEL", "LEGAL_RESEARCHER_TEMP", LEGAL_RESEARCHER_AGENT_PROMPT)
        self.kg = kg
        self.vs = vector_store

    # ── keyword extraction ────────────────────────────────────────────

    def _extract_search_keywords(self, charge) -> list[str]:
        candidates = []

        if getattr(charge, "incident_type", None):
            candidates.append(charge.incident_type)

        desc = getattr(charge, "description", None)
        if desc:
            words = desc.strip().split()
            candidates.append(" ".join(words[:3]))

        if getattr(charge, "law_code", None):
            law_map = {
                "قانون مكافحة المخدرات":  "مخدر",
                "قانون العقوبات":          "عقوبة",
                "قانون الأسلحة والذخائر": "سلاح",
            }
            keyword = law_map.get(charge.law_code)
            if keyword:
                candidates.append(keyword)

        return candidates

    # ── KG retrieval → case_articles ─────────────────────────────────

    def _retrieve_statute_from_kg(self, charge) -> list[dict]:
        """
        Three-strategy KG lookup:
          1. Direct article number lookup
          2. Embedding similarity search (semantic)
        Returns a list of case_article dicts.
        """
        if self.kg is None:
            return []

        results  = []
        law_code = getattr(charge, "law_code", None)
        law_id   = AgentsEnums.law_code_to_kg_id(law_code) if law_code else None

        # ── strategy 1: direct article number lookup ──────────────────
        article_number = getattr(charge, "article_number", None)
        if article_number and law_id:
            try:
                history = fetch_article_text(law_id, str(article_number))
                if history:
                    latest = sorted(history, key=lambda x: x.get("version") or "0000-00-00")[-1]
                    text   = (latest.get("text") or "").strip()
                    if text:
                        results.append(self._build_article_dict(
                            article_number=article_number,
                            law_code=law_code,
                            text=text,
                            is_amended=latest.get("is_amended", False),
                            retrieval_strategy="direct_lookup",
                        ))
                        logger.info("KG direct lookup — found م%s من %s", article_number, law_code)
                        return results
            except Exception as e:
                logger.warning("KG direct lookup failed for م%s: %s", article_number, e)


        # ── strategy 2: embedding similarity search ────────────────────
        if not results:
            query_text = self._build_embedding_query(charge)
            if query_text:
                try:
                    hits = self.kg.search_articles_by_embedding(
                        query=query_text,
                        law_id=law_id,
                        k=3,
                    )
                    for hit in hits:
                        text   = (hit.get("text") or "").strip()
                        art_no = hit.get("article_number")
                        score  = hit.get("score", 0.0)
                        if text and art_no not in seen_articles and score >= 0.70:
                            seen_articles.add(art_no)
                            results.append(self._build_article_dict(
                                article_number=art_no,
                                law_code=hit.get("law_id") or law_code,
                                text=text,
                                is_amended=hit.get("is_amended", False),
                                retrieval_strategy="embedding_search",
                                similarity_score=round(score, 3),
                            ))
                    if results:
                        logger.info("KG embedding [%s] → %d result(s)", query_text[:40], len(results))
                    else:
                        logger.info("KG embedding search — no results above threshold for: %s", query_text[:40])
                except Exception as e:
                    logger.warning("KG embedding search failed: %s", e)

        return results

    def _build_embedding_query(self, charge) -> str:
        """Compose a semantic query from the charge's most descriptive fields."""
        parts = []
        if getattr(charge, "description", None):
            parts.append(charge.description)
        if getattr(charge, "incident_type", None):
            parts.append(charge.incident_type)
        if getattr(charge, "law_code", None):
            parts.append(charge.law_code)
        return " — ".join(parts)

    def _build_article_dict(
        self,
        article_number,
        law_code,
        text,
        is_amended,
        retrieval_strategy,
        similarity_score=None,
    ) -> dict:
        """Uniform case_article dict shape."""
        d = {
            "article_number":     article_number,
            "law_code":           law_code,
            "text":               text[:400],
            "is_amended":         is_amended,
            "retrieval_strategy": retrieval_strategy,
        }
        if similarity_score is not None:
            d["similarity_score"] = similarity_score
        return d

    # ── VS retrieval → applied_principles ────────────────────────────

    def _retrieve_applied_principles(self, charge, procedural_issues: list, incidents: list) -> list[dict]:
        """
        Retrieve cassation rulings and judicial principles from the
        vector store, enriched with context from procedural_issues
        and incidents.
        """
        principles = []

        # primary: charge-based cassation rulings
        principles.extend(self._retrieve_cassation_rulings(charge))

        # secondary: procedural issue based search
        for issue in procedural_issues:
            issue_desc = getattr(issue, "issue_description", None) or str(issue)
            if not issue_desc:
                continue
            try:
                hits = self.vs.similarity_search(issue_desc, k=2)
                for hit in hits:
                    content = getattr(hit, "page_content", None) or hit.get("text", "")
                    if content:
                        principles.append({
                            "text":   content[:400],
                            "source": "procedural_issue",
                            "query":  issue_desc[:80],
                        })
            except Exception as e:
                logger.warning("VS procedural issue search failed: %s", e)

        # tertiary: incident narrative based search
        for incident in incidents:
            incident_desc = getattr(incident, "incident_description", None) or str(incident)
            if not incident_desc:
                continue
            try:
                hits = self.vs.similarity_search(incident_desc[:120], k=2)
                for hit in hits:
                    content = getattr(hit, "page_content", None) or hit.get("text", "")
                    if content:
                        principles.append({
                            "text":   content[:400],
                            "source": "incident_narrative",
                            "query":  incident_desc[:80],
                        })
            except Exception as e:
                logger.warning("VS incident search failed: %s", e)

        # deduplicate by text prefix
        seen, deduped = set(), []
        for p in principles:
            key = (p.get("text") or "")[:80]
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        return deduped

    # ── context assembly ──────────────────────────────────────────────

    def _retrieve_all_contexts(
        self,
        charges: list,
        procedural_issues: list,
        incidents: list,
    ) -> tuple[list, list]:
        contexts: list = []
        failed:   list = []

        for charge in charges:
            try:
                # KG  → case_articles
                case_articles = self._retrieve_statute_from_kg(charge)

                # VS  → applied_principles
                applied_principles = self._retrieve_applied_principles(
                    charge, procedural_issues, incidents
                )

                contexts.append({
                    "charge":            getattr(charge, "statute", str(charge)),
                    "law_code":          getattr(charge, "law_code", None),
                    "article_number":    getattr(charge, "article_number", None),
                    "description":       getattr(charge, "description", None),
                    "case_articles":     case_articles,       # ← from KG
                    "applied_principles": applied_principles, # ← from VS
                })

            except Exception as e:
                failed.append(getattr(charge, "statute", str(charge)))
                contexts.append({
                    "charge":             getattr(charge, "statute", str(charge)),
                    "error":              str(e),
                    "case_articles":      [],
                    "applied_principles": [],
                })

        return contexts, failed

    # ── prompt builder ────────────────────────────────────────────────

    def _build_prompt(self, state, all_contexts: list) -> str:
        case_summary = json.dumps(
            {
                "defendants":        [d.name for d in state.defendants],
                "charges":           [c.statute for c in state.charges],
                "procedural_issues": [
                    getattr(pi, "issue_description", str(pi))
                    for pi in (state.procedural_issues or [])
                ],
                "incidents": [
                    i.incident_description
                    for i in (state.incidents or [])
                    if i.incident_description
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            "## معطيات القضية:\n"
            f"{case_summary}\n\n"

            "## السياق المسترجع\n"
            "• case_articles    : نصوص المواد القانونية من قاعدة المعرفة (KG)\n"
            "• applied_principles: مبادئ محكمة النقض من الـ Vector Store\n"
            f"{json.dumps(all_contexts, ensure_ascii=False, indent=2)}\n\n"

            "## التعليمات:\n"
            "ابنِ حزمة بحث قانوني لكل تهمة من المعطيات أعلاه فقط.\n"
            "لكل تهمة:\n"
            "  - حدد أركانها ونطاق عقوبتها من case_articles.\n"
            "  - أدرج مبادئ النقض ذات الصلة من applied_principles.\n"
            "  - راعِ الدفوع الإجرائية في procedural_issues عند تحديد المواد.\n"
            "  - استند فقط إلى ما في السياق المسترجع — لا تخترع مواد أو أحكاماً.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        charges_count = len(state.charges or [])
        logger.info("LegalResearcherAgent starting — %d charge(s)", charges_count)

        if not state.charges:
            return self._empty_update(state, "legal_researcher", {
                "case_articles":      [],
                "applied_principles": [],
            })

        procedural_issues = state.procedural_issues or []
        incidents         = state.incidents or []

        all_contexts, failed_charges = self._retrieve_all_contexts(
            state.charges, procedural_issues, incidents
        )

        if len(failed_charges) == charges_count:
            logger.warning("All charges failed context retrieval — using fallback")
            legal_package = self._build_fallback_package(all_contexts)
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
            legal_package = self._parse_llm_json(response.content)
        except Exception as e:
            logger.error("LLM invocation failed: %s — using fallback", e)
            legal_package = self._build_fallback_package(all_contexts)

        if not isinstance(legal_package, dict):
            logger.error("LLM response not a dict — using fallback")
            legal_package = self._build_fallback_package(all_contexts)
        else:
            logger.info(
                "Legal research complete — %d case_article(s), %d principle(s)",
                len(legal_package.get("case_articles", [])),
                len(legal_package.get("applied_principles", [])),
            )

        legal_package["_meta"] = {
            "charges_count":        charges_count,
            "procedural_issues_count": len(procedural_issues),
            "incidents_count":      len(incidents),
            "failed_charges":       failed_charges,
            "contexts_count":       len(all_contexts),
            "kg_connected":         self.kg is not None,
            "vs_connected":         self.vs is not None,
        }

        base_state = self._empty_update(state, "legal_researcher", legal_package)
        return base_state.model_copy(update={
            "case_articles":      legal_package.get("case_articles", []),
            "applied_principles": legal_package.get("applied_principles", []),
        })