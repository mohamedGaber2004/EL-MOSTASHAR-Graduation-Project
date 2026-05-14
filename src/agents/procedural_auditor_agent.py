from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Graph.graph_helpers import _parse_llm_json
from src.Prompts.procedural_auditor_agent import (
    PROCEDURAL_AUDITOR_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)
from src.Utils.Enums.agents_enums import AgentsEnums

logger = logging.getLogger(__name__)


class ProceduralAuditorAgent(AgentBase):
    """Audits procedural issues against articles retrieved from the KG."""

    def __init__(self, kg: LegalKnowledgeGraph, vector_store=None):
        super().__init__("PROCEDURAL_AUDITOR_MODEL","PROCEDURAL_AUDITOR_TEMP",PROCEDURAL_AUDITOR_AGENT_PROMPT)
        self.kg = kg
        self.vs = vector_store

    # ── dynamic article discovery ─────────────────────────────────────
    def _retrieve_articles_for_issue(self, issue_text: str, k: int = 5) -> list[str]:
        if self.kg is None:
            return []
        # use first 2 words only for better match
        short_keyword = " ".join(issue_text.strip().split()[:2])
        try:
            hits = self.kg.search_articles_by_keyword(
                keyword=short_keyword,
                law_id=AgentsEnums.CRIMINAL_PROCEDURE_LAW_ID.value,
                k=k,
            )
            article_nums = [str(h["article_number"]) for h in hits if h.get("article_number")]
            logger.info("KG retrieval for [%s]: %d article(s)", short_keyword, len(article_nums))
            return article_nums
        except Exception as e:
            logger.warning("KG retrieval failed for [%s]: %s", short_keyword, e)
            return []

    def _resolve_article_numbers(self, procedural_issues: list) -> list[str]:
        """
        Purely dynamic — NO hardcoded baseline whatsoever.
        """
        dynamic: list[str] = []

        for issue in procedural_issues:
            # ── extract meaningful text from issue ──────────────────────
            if isinstance(issue, dict):
                issue_text = (
                    issue.get("issue_description")
                    or issue.get("procedure_type")
                    or issue.get("violation_type")
                    or ""
                ).strip()
            else:
                issue_text = str(issue).strip()

            if not issue_text or issue_text.lower() == "none":
                logger.warning("Skipping empty procedural issue")
                continue

            for num in self._retrieve_articles_for_issue(issue_text):
                if num not in dynamic:
                    dynamic.append(num)

        if not dynamic:
            logger.warning(
                "No articles found from KG for any issue — "
                "LLM will rely on internal knowledge"
            )

        logger.info("Article resolution — dynamic from KG: %d", len(dynamic))
        return dynamic

    # ── KG fetcher ────────────────────────────────────────────────────

    def _fetch_procedural_articles(
        self, article_numbers: list[str]
    ) -> tuple[str, list[str]]:
        """Fetch article texts from the KG for the given article numbers."""
        if self.kg is None:
            logger.warning("KG not available — LLM will rely on internal knowledge")
            return "لا توجد مواد متاحة من قاعدة البيانات.", []

        articles_text: list[str] = []
        fetched: list[str] = []

        for article_num in article_numbers:
            try:
                history = self.kg.query_article_history(
                    AgentsEnums.CRIMINAL_PROCEDURE_LAW_ID.value, article_num
                )
                if not history:
                    continue

                latest = sorted(
                    history,
                    key=lambda x: x.get("version") or "0000-00-00",
                )[-1]

                text = (latest.get("text") or "").strip()
                if not text:
                    continue

                amendment_note = (
                    f" [معدَّلة — {latest.get('version', '')}]"
                    if latest.get("is_amended")
                    else ""
                )

                articles_text.append(f"م{article_num}{amendment_note}:\n{text}")
                fetched.append(article_num)

            except Exception as e:
                logger.error("Error fetching article %s: %s", article_num, e)

        if not articles_text:
            return "لا توجد مواد متاحة من قاعدة البيانات.", []

        return "\n\n".join(articles_text), fetched

    # ── prompt builder ────────────────────────────────────────────────

    def _build_prompt(self, state, legal_references: str, fetched_articles: list[str]) -> str:
        
        procedural_issues = state.procedural_issues or []
        confessions       = getattr(state, "confessions", [])
        incidents         = getattr(state, "incidents", [])

        return (
            "## المسائل الإجرائية المرصودة:\n"
            f"{procedural_issues}\n\n"

            "## تفاصيل الحوادث والقبض:\n"
            f"{incidents}\n\n"

            "## الاعترافات (إن وجدت):\n"
            f"{confessions}\n\n"

            "## بيانات استخلاص الوثائق:\n"
            f"{ingestion_data}\n\n"

            f"## النصوص القانونية المرجعية (مواد: {', '.join(fetched_articles)}):\n"
            f"{legal_references}\n\n"

            "## التعليمات:\n"
            "راجع كل مسألة إجرائية في ضوء النصوص القانونية أعلاه وحدد:\n"
            "1. هل توجد مخالفة إجرائية؟\n"
            "2. هل تُفضي إلى بطلان؟ (مطلق أم نسبي؟)\n"
            "3. ما أثر البطلان على الأدلة المترتبة عليه؟\n"
            "4. الاعتراف المنتزع بالإكراه باطل ولا يُعتد به حتى لو صادقه المتهم لاحقاً.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        issues_count = len(state.procedural_issues or [])
        logger.info(
            "ProceduralAuditorAgent starting — %d issue(s) to review", issues_count
        )

        # ← dynamic resolution replaces the hardcoded enum loop
        article_numbers = self._resolve_article_numbers(state.procedural_issues or [])

        legal_references, fetched_articles = self._fetch_procedural_articles(
            article_numbers
        )
        logger.debug("Fetched %d procedural articles from KG", len(fetched_articles))

        prompt = self._build_prompt(state, legal_references, fetched_articles)
        response = self._llm_invoke_with_retries(
            self._llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        audit_report = _parse_llm_json(response.content)

        if isinstance(audit_report, dict):
            logger.info(
                "Procedural audit complete — %d violation(s) found",
                len(audit_report.get("violations", [])),
            )
            audit_report["_meta"] = {
                "kg_law_id":           AgentsEnums.CRIMINAL_PROCEDURE_LAW_ID.value,
                "kg_articles_fetched": fetched_articles,
                "articles_dynamic":    article_numbers,
                "issues_reviewed":     issues_count,
                "vs_connected":        self.vs is not None
            }
        else:
            logger.error("Procedural audit — LLM response could not be parsed as dict")
            audit_report = {
                "raw_response": audit_report,
                "violations":   [],
                "_meta":        {"kg_articles_fetched": fetched_articles},
            }

        return self._empty_update(state, "procedural_auditor", audit_report)