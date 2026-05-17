import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.agents.agents_enums import AgentsEnums
from src.Graph.state import AgentState
from src.agents.legal_research_agent.legal_researcher_prompt import (
    LEGAL_RESEARCHER_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
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

    # ── fallback ──────────────────────────────────────────────────────
    def _build_fallback_package(self, all_contexts: list[dict]) -> dict:
        return {
            "case_articles":      [],
            "applied_principles": [],
            "_fallback":          True,
        }

    # ── ANN query ────────────
    _CHARGE_ANN_QUERY = """
        CALL db.index.vector.queryNodes('article_embeddings', $k, $query_vector)
        YIELD node AS article, score
        RETURN
            article.article_id     AS article_id,
            article.article_number AS article_number,
            article.law_id         AS law_id,
            article.text           AS text,
            score
        ORDER BY score DESC
    """

    def _retrieve_articles_for_charge(self,text: str, k: int= 15, threshold: float = 0.45) -> list[dict]:
        if self.kg is None or self.embeddings is None:
            logger.warning("KG or embeddings not available — skipping article retrieval")
            return []

        try:
            query_vector = self.embeddings.embed_query(text)
        except Exception as e:
            logger.warning("embed_query failed for [%s...]: %s", text[:60], e)
            return []

        try:
            with self.kg.driver.session() as s:
                records = s.run(
                    self._CHARGE_ANN_QUERY,
                    k=k,
                    query_vector=query_vector,
                ).data()
        except Exception as e:
            logger.warning("ANN query failed for [%s...]: %s", text[:60], e)
            return []

        articles = [
            {
                "article_number":     str(r["article_number"]),
                "law_id":             str(r["law_id"]),
                "article_id":         str(r["article_id"]),
                "text":               str(r["text"])[:400],
            }
            for r in records
            if r.get("article_number") and float(r.get("score", 0)) >= threshold
        ]

        logger.info(
            "ANN retrieval — text=[%s...] | law=%s | hits=%d (threshold=%.2f)",
            text[:60], len(articles), threshold,
        )
        return articles

    def _build_charge_query_text(self, charge) -> str:
        """Compose a single query string from a charge's descriptive fields."""
        parts = []
        if getattr(charge, "description", None):
            parts.append(charge.description)
        if getattr(charge, "incident_type", None):
            parts.append(charge.incident_type)
        if getattr(charge, "law_code", None):
            parts.append(charge.law_code)
        return " — ".join(parts)

    # ── KG retrieval: accumulate unique articles across all charges ────
    def _resolve_articles_for_charges(self, charges: list) -> list[dict]:
        """
        For every charge, build a query text, resolve its law_id,
        run the ANN query, and accumulate unique articles.
        Mirrors ProceduralAuditorAgent._resolve_article_numbers.
        """
        seen:   set[str]   = set()
        result: list[dict] = []

        for charge in charges:
            query_text = self._build_charge_query_text(charge)
            if not query_text.strip():
                logger.warning("Skipping charge with no queryable text: %s", charge)
                continue

            law_code = getattr(charge, "law_code", None)
            law_id   = AgentsEnums.law_code_to_kg_id(law_code) if law_code else None

            for article in self._retrieve_articles_for_charge(text=query_text):
                key = article["article_number"]
                if key not in seen:
                    seen.add(key)
                    result.append(article)

        if not result:
            logger.warning(
                "No articles found via ANN for any charge — LLM will rely on internal knowledge"
            )
        else:
            logger.info(
                "Article resolution complete — %d unique article(s) from ANN search",
                len(result),
            )

        return result

    # ── VS retrieval: accumulate unique principles ────────────────────
    def _resolve_principles_for_charges(self,charges: list,procedural_violations: list,incidents: list) -> list:
        if self.vs is None:
            logger.warning("No vector store — skipping principles retrieval.")
            return []

        seen:   set[str] = set()
        result: list     = []

        def _accumulate(query_text: str, source: str) -> None:
            if not query_text or not query_text.strip():
                return
            docs = self.retrieve_principles(query_text, self.vs)
            for doc in docs:
                text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                key  = text[:80]
                if key not in seen:
                    seen.add(key)
                    result.append({
                        "principle_text": text,                                          # ✅ matches model
                        "court_source":   doc.metadata.get("source_file", None) if hasattr(doc, "metadata") else None,
                        "applicable_to":  source,                                        # charge / violation / incident
                    })

        for charge in charges:
            _accumulate(self._build_charge_query_text(charge), "charge")

        for violation in procedural_violations:
            issue_desc = (
                getattr(violation, "issue_description", None) or str(violation)
            ).strip()
            _accumulate(issue_desc, "procedural_violation")

        for incident in incidents:
            incident_desc = (
                getattr(incident, "incident_description", None) or str(incident)
            ).strip()
            _accumulate(incident_desc[:120], "incident_narrative")

        if not result:
            logger.warning("No principles found via VS for any query source.")
        else:
            logger.info(
                "Principles resolution complete — %d unique principle(s)", len(result)
            )

        return result

    # ── prompt builder ────────────────────────────────────────────────
    def _build_prompt(self, state: AgentState, case_articles: list[dict], applied_principles: list) -> str:
        # ── charges + incidents only — no defendants, no procedural data ──────
        case_summary = json.dumps(
            {
                "charges": [
                    {
                        "statute":     getattr(c, "statute", None),
                        "description": getattr(c, "description", None),
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

            "## التعليمات:\n"
            "ابنِ حزمة بحث قانوني لكل تهمة من المعطيات أعلاه فقط.\n"
            "لكل تهمة:\n"
            "  - حدد أركانها ونطاق عقوبتها من المواد المسترجعة.\n"
            "  - أدرج مبادئ النقض ذات الصلة من المبادئ المسترجعة.\n"
            "  - استند فقط إلى ما في السياق المسترجع — لا تخترع مواد أو أحكاماً.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )



    # ── main entry ────────────────────────────────────────────────────
    def run(self, state):
        charges_count = len(state.charges or [])
        logger.info("LegalResearcherAgent starting — %d charge(s)", charges_count)

        if not state.charges:
            base_state = self._empty_update(state, "legal_researcher", {})
            return self._empty_update(state, "legal_researcher", {
                "case_articles":      [],
                "applied_principles": [],
            })

        procedural_violations = (
            state.procedural_audit.violations
            if state.procedural_audit
            else []
        )
        incidents = state.incidents or []

        # ── retrieval (no KG method call — pure ANN like ProceduralAuditorAgent) ──
        case_articles      = self._resolve_articles_for_charges(state.charges)
        applied_principles = self._resolve_principles_for_charges(
            state.charges, procedural_violations, incidents
        )

        prompt = self._build_prompt(state, case_articles, applied_principles)
        logger.debug("LegalResearcherAgent prompt built — invoking LLM")

        try:
            response      = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            legal_package = self._parse_agent_json(response.content)
        except Exception as e:
            logger.error("LLM invocation failed: %s — using fallback", e)
            legal_package = self._build_fallback_package([])

        if not isinstance(legal_package, dict):
            logger.error("LLM response is not a dict — using fallback")
            legal_package = self._build_fallback_package([])
        else:
            logger.info(
                "Legal research complete — %d case_article(s), %d principle(s)",
                len(legal_package.get("case_articles", [])),
                len(legal_package.get("applied_principles", [])),
            )

        return self._empty_update(state, "legal_researcher", {
            "case_articles":      legal_package.get("case_articles", []),
            "applied_principles": (
                legal_package.get("applied_principles", [])   # from LLM
                or applied_principles 
            ),
        })