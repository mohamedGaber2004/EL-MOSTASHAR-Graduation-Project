from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..agent_base.agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Graph.state import AgentState
from src.agents.procedural_auditor_agent.procedural_auditor_output_model import (
    ProceduralAuditResult,
    ProceduralIssue,
    ExcludedDefenseClaim,
)
from src.routers.kg_retriever_router import get_kg_retriever
from src.agents.procedural_auditor_agent.procedural_auditor_prompt import (
    PROCEDURAL_AUDITOR_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)

class ProceduralAuditorAgent(AgentBase):
    """Audits procedural issues against articles retrieved from the KG."""

    def __init__(self, kg: LegalKnowledgeGraph, embeddings, vector_store):
        super().__init__(
            "PROCEDURAL_AUDITOR_MODEL",
            "PROCEDURAL_AUDITOR_TEMP",
            PROCEDURAL_AUDITOR_AGENT_PROMPT,
        )
        self.kg         = kg
        self.vs         = vector_store
        self.embeddings = embeddings

    # ── KG article retrieval ──────────────────────────────────────────────────
    def _retrieve_articles_for_proceeding(
        self, text: str, k: int = 15, threshold: float = 0.45
    ) -> list[dict]:
        if self.kg is None or self.embeddings is None:
            logger.warning("KG or embeddings not available — skipping retrieval")
            return []

        try:
            clean_query = self.query_transformation(text)["article_query"]
        except Exception as e:
            logger.warning("query_transformation failed for [%s...]: %s", text[:60], e)
            return []

        try:
            records = get_kg_retriever().retrieve(
                question=clean_query, k=k, threshold=threshold
            )
        except Exception as e:
            logger.warning("kg retriever failed for [%s...]: %s", text[:60], e)
            return []

        articles = [
            {
                "Article Number":        str(r["article_number"]),
                "The Law of the Article": str(r["law_id"]),
                # truncate article text to avoid context overflow
                "The Article Text":       str(r["text"]),
            }
            for r in records.sources
            if r.get("article_number") and float(r.get("score", 0)) >= threshold
        ]

        logger.info(
            "query=%s | articles=%d | tables=%d",
            records.query,
            len(records.article_contexts),
            len(records.table_contexts),
        )
        return articles

    # ── text extractor ────────────────────────────────────────────────────────
    @staticmethod
    def _extract_text(entry) -> str:
        if isinstance(entry, dict):
            return (
                entry.get("issue_description")
                or entry.get("procedure_type")
                or entry.get("warrant_present")
                or entry.get("description")
                or entry.get("text")
                or ""
            ).strip()
        return str(entry).strip()

    # ── deduplicated article accumulator ──────────────────────────────────────
    def _accumulate_unique_articles(
        self, entries: list, label: str
    ) -> list[dict]:
        seen:   set[str]   = set()
        result: list[dict] = []

        for entry in entries:
            text = self._extract_text(entry)
            if not text or text.lower() == "none":
                logger.warning("Skipping empty entry in [%s]", label)
                continue

            for article in self._retrieve_articles_for_proceeding(text):
                key = article["Article Number"]
                if key not in seen:
                    seen.add(key)
                    result.append(article)

        if result:
            logger.info("Article resolution [%s] — %d unique article(s)", label, len(result))
        else:
            logger.warning(
                "No articles found for [%s] — LLM will rely on internal knowledge", label
            )
        return result

    # ── deduplicated principle accumulator ────────────────────────────────────
    def _accumulate_unique_principles(
        self, entries: list, label: str
    ) -> list[dict]:
        """
        Retrieve unique judicial principles for a list of entries.
        Returns plain dicts (not RetrievedDoc objects) for easy JSON serialisation.
        """
        seen:   set[str] = set()
        result: list     = []

        for entry in entries:
            text = self._extract_text(entry)
            if not text or text.lower() == "none":
                logger.warning("Skipping empty entry in [%s] while retrieving principles", label)
                continue

            try:
                transformed = self.query_transformation(text)["principle_query"]
            except Exception as e:
                logger.warning("query_transformation failed in [%s]: %s", label, e)
                continue

            # retrieve_principles returns List[RetrievedDoc]
            for doc in self.retrieve_principles(transformed, self.vs):
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    result.append({
                        "principle_text": doc.page_content,
                        "court_source":   (
                            doc.metadata.get("source_file") if doc.metadata else None
                        ),
                    })

        if result:
            logger.info("Principles resolution [%s] — %d unique principle(s)", label, len(result))
        else:
            logger.warning("No principles found for [%s]", label)

        return result

    # ── resolve articles + principles for proceedings and defense ─────────────
    def _resolve_all(
        self,
        criminal_proceedings: list,
        defense_procedural_issues: list,
    ) -> tuple[list, list, list, list]:
        """
        Returns (proceedings_articles, defense_articles,
                 proceedings_principles, defense_principles).
        Articles are capped at _MAX_ARTICLES_IN_PROMPT to prevent context overflow.
        """
        proceedings_articles = self._accumulate_unique_articles(
            criminal_proceedings, "criminal_proceedings"
        )

        defense_articles = self._accumulate_unique_articles(
            defense_procedural_issues, "defense_procedural_issues"
        )

        proceedings_principles = self._accumulate_unique_principles(
            criminal_proceedings, "criminal_proceedings"
        )
        defense_principles = self._accumulate_unique_principles(
            defense_procedural_issues, "defense_procedural_issues"
        )

        return proceedings_articles, defense_articles, proceedings_principles, defense_principles

    # ── prompt builder ────────────────────────────────────────────────────────
    def _build_prompt(self, state: AgentState) -> str:
        incidents = getattr(state, "incidents", []) or []
        criminal_proceedings = getattr(state, "criminal_proceedings", []) or []
        defense_procedural_issues = [
            {"formal_defenses": getattr(doc, "formal_defenses", [])}
            for doc in (state.defense_documents or [])
        ]

        (
            proceedings_articles,
            defense_articles,
            proceedings_principles,
            defense_principles,
        ) = self._resolve_all(criminal_proceedings, defense_procedural_issues)

        return (
            "## المسائل الإجرائية المستخرجة من وقائع القضية:\n"
            f"{criminal_proceedings}\n\n"

            "## الإشكاليات الإجرائية المُثارة من محامي الدفاع (تحتاج تحققاً):\n"
            f"{defense_procedural_issues}\n\n"

            "## تفاصيل الحوادث والقبض:\n"
            f"{incidents}\n\n"

            "---\n"
            "## المواد القانونية ذات الصلة بوقائع القضية:\n"
            f"{proceedings_articles}\n\n"

            "## المبادئ القضائية ذات الصلة بوقائع القضية:\n"
            f"{proceedings_principles}\n\n"

            "## المواد القانونية ذات الصلة بدفوع المحامي:\n"
            f"{defense_articles}\n\n"

            "## المبادئ القضائية ذات الصلة بدفوع المحامي:\n"
            f"{defense_principles}\n\n"

            "---\n"
            "## التعليمات النهائية:\n"
            "1. استخرج جميع المسائل الإجرائية التي تراها أنت في الوقائع — بصرف النظر عما أثاره المحامي.\n"
            "2. لكل مسألة أثارها محامي الدفاع: إن كانت صحيحة وثابتة → أدرجها. إن كانت خاطئة → أدرجها في excluded_defense_claims فقط مع بيان سبب الاستبعاد.\n"
            "3. حدد لكل مخالفة: نوع البطلان (مطلق/نسبي) وأثره على الأدلة المترتبة.\n"
            "4. الاعتراف المنتزع بإكراه = بطلان مطلق حتماً.\n"
            "5. لا تذكر مادة قانونية لم ترد في السياق أعلاه.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا أي نص خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ────────────────────────────────────────────────────────────
    def run(self, state):
        criminal_proceedings = getattr(state, "criminal_proceedings", []) or []
        defense_procedural_issues = [
            item
            for doc in (state.defense_documents or [])
            for item in (getattr(doc, "formal_defenses", []) or [])
        ]

        logger.info(
            "ProceduralAuditorAgent starting — %d proceeding(s), %d defense issue(s)",
            len(criminal_proceedings),
            len(defense_procedural_issues),
        )

        if not criminal_proceedings and not defense_procedural_issues:
            return self._empty_update(state, "procedural_auditor", {
                "procedural_audit": ProceduralAuditResult(
                    overall_assessment="لا توجد مسائل إجرائية للمراجعة",
                )
            })

        _fallback = ProceduralAuditResult(
            overall_assessment="فشل استدعاء النموذج — لا يمكن إتمام المراجعة الإجرائية",
        )

        try:
            prompt   = self._build_prompt(state)
            response = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            raw_package = self._parse_agent_json(response.content)
        except Exception as e:
            logger.error("ProceduralAuditorAgent LLM invocation failed: %s", e)
            raw_package = None

        if not isinstance(raw_package, dict):
            logger.error("ProceduralAuditorAgent: LLM response is not a dict — using fallback")
            audit_result = _fallback
        else:
            try:
                audit_result = ProceduralAuditResult(
                    violations=[
                        ProceduralIssue(**v)
                        for v in raw_package.get("violations", [])
                        if isinstance(v, dict)
                    ],
                    excluded_defense_claims=[
                        ExcludedDefenseClaim(**c)
                        for c in raw_package.get("excluded_defense_claims", [])
                        if isinstance(c, dict)
                    ],
                    overall_assessment = raw_package.get("overall_assessment"),
                    critical_nullities = raw_package.get("critical_nullities", []),
                    kg_articles_used   = raw_package.get("kg_articles_used", []),
                )
            except Exception as e:
                logger.error("ProceduralAuditorAgent: Pydantic validation failed: %s", e)
                audit_result = _fallback

            logger.info(
                "Procedural audit complete — %d violation(s), %d excluded claim(s), %d critical nullit(ies)",
                len(audit_result.violations),
                len(audit_result.excluded_defense_claims),
                len(audit_result.critical_nullities),
            )

        audit_result._meta = {
            "proceedings_count":    len(criminal_proceedings),
            "defense_issues_count": len(defense_procedural_issues),
            "kg_connected":         self.kg is not None,
            "vs_connected":         self.vs is not None,
        }

        return self._empty_update(state, "procedural_auditor", {
            "procedural_audit": audit_result,
        })