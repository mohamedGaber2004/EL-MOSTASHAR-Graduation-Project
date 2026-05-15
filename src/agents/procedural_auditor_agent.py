from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.routers.kg_router import fetch_article_text, get_statistics
from src.Prompts.procedural_auditor_agent import (
    PROCEDURAL_AUDITOR_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)
from src.Utils.Enums.agents_enums import AgentsEnums

logger = logging.getLogger(__name__)





class ProceduralAuditorAgent(AgentBase):
    """Audits procedural issues against articles retrieved from the KG."""

    def __init__(self, kg: LegalKnowledgeGraph, embeddings):
        super().__init__(
            "PROCEDURAL_AUDITOR_MODEL",
            "PROCEDURAL_AUDITOR_TEMP",
            PROCEDURAL_AUDITOR_AGENT_PROMPT,
        )
        self.kg         = kg
        self.embeddings = embeddings


    # ── dynamic article discovery ─────────────────────────────────────

    def _resolve_article_numbers(self, procedural_issues: list) -> list[str]:
        """
        Pass the full issue text to the embedder — no truncation needed
        (unlike keyword search, embeddings benefit from complete context).
        """
        dynamic: list[str] = []

        for issue in procedural_issues:
            if isinstance(issue, dict):
                # prefer the most descriptive field available
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
                "No articles found via embedding search for any issue — "
                "LLM will rely on internal knowledge"
            )
        else:
            logger.info(
                "Article resolution complete — %d unique article(s) from embedding search",
                len(dynamic),
            )

        return dynamic

    # ── KG fetcher ────────────────────────────────────────────────────

    def _fetch_procedural_articles(self, article_numbers: list[str]) -> tuple[str, list[str]]:
        """Fetch article texts from the KG for the given article numbers."""
        if get_statistics() is None:
            logger.warning("KG not available — LLM will rely on internal knowledge")
            return "لا توجد مواد متاحة من قاعدة البيانات.", []

        articles_text: list[str] = []
        fetched: list[str] = []

        for article_num in article_numbers:
            try:
                history = fetch_article_text(
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
        incidents = getattr(state, "incidents", [])
        defendents = getattr(state, "defendents", [])
        evidences = getattr(state, "evidences", [])
        confessions = getattr(state, "confessions", [])
        criminal_proceedings = getattr(state, "criminal_proceedings", [])
        defense_procedural_issues = getattr(state.defense_documents,"formal_defenses",[])

        return (
            "## المسائل الإجرائية المرصودة من وقائع القضيه:\n"
            f"{criminal_proceedings}\n\n"

            "## المسائل الاجرائيه المرصوده من دفاع المتهم:"
            f"{defense_procedural_issues}\n\n"

            "## تفاصيل الحوادث والقبض:\n"
            f"{incidents}\n\n"

            "## المتهمين:\n"
            f"{defendents}\n\n"

            "## الاعترافات (إن وجدت):\n"
            f"{confessions}\n\n"

            "## كل الادله:\n"
            f"{evidences}\n\n"

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

        audit_report = self._parse_llm_json(response.content)

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