from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from .agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Graph.graph_helpers import _parse_llm_json
from src.Prompts.procedural_auditor_agent import PROCEDURAL_AUDITOR_AGENT_PROMPT , EXPECTED_OUTPUT_SCHEMA
from src.Utils.agents_enums import AgentsEnums

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Procedural Auditor Agent
# ─────────────────────────────────────────────────────────────────────────────

class ProceduralAuditorAgent(AgentBase):

    def __init__(self, kg: LegalKnowledgeGraph):
        super().__init__("PROCEDURAL_AUDITOR_MODEL","PROCEDURAL_AUDITOR_TEMP",PROCEDURAL_AUDITOR_AGENT_PROMPT)
        self.kg = kg

    # ── KG helpers ────────────────────────────────────────────────────────

    def _fetch_procedural_articles(self) -> tuple[str, list[str]]:
        if self.kg is None:
            logger.warning("KG not available - LLM will rely on its internal knowledge")
            return "لا توجد مواد متاحة من قاعدة البيانات.", []
        articles_text: list[str] = []
        fetched: list[str] = []

        for article_num in AgentsEnums.PROCEDURAL_ARTICLE_NUMBERS.value:
            try:
                history = self.kg.query_article_history(
                    AgentsEnums.CRIMINAL_PROCEDURE_LAW_ID.value, article_num
                )
                if not history:
                    continue

                latest = sorted(history,key=lambda x: x.get("version") or "0000-00-00")[-1]

                text = (latest.get("text") or "").strip()
                if not text:
                    continue

                amendment_note = ""
                if latest.get("is_amended"):
                    amendment_note = f" [معدَّلة — {latest.get('version', '')}]"

                articles_text.append(f"م{article_num}{amendment_note}:\n{text}")
                fetched.append(article_num)

            except Exception as e:
                logger.error("Error fetching article: %s", e)

        if not articles_text:
            return "لا توجد مواد متاحة من قاعدة البيانات.", []

        return "\n\n".join(articles_text), fetched

    # ── prompt builder ────────────────────────────────────────────────────

    def _build_prompt(self, state, legal_references: str, fetched_articles: list[str]) -> str:
        ingestion_data   = state.agent_outputs.get("data_ingestion", {})
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

    # ── main entry ────────────────────────────────────────────────────────

    def run(self, state):
        issues_count = len(state.procedural_issues or [])
        logger.info("Starting procedural auditor... Total issues to review: %d", issues_count)
        # 1. جلب النصوص من الـ KG
        legal_references, fetched_articles = self._fetch_procedural_articles()
        logger.debug("Fetched %d procedural articles from KG.", len(fetched_articles))
        # 2. بناء الـ prompt
        prompt = self._build_prompt(state, legal_references, fetched_articles)
        # 3. استدعاء الـ LLM مع retry
        logger.debug("Invoking LLM for procedural auditing...")
        response = self._llm_invoke_with_retries(
            self.lama_llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        # 4. تحليل الرد
        audit_report = _parse_llm_json(response.content)

        # 5. إضافة metadata للتوثيق
        if isinstance(audit_report, dict):
            logger.info("Procedural auditing complete. Found %d violations.", len(audit_report.get("violations", [])))
            audit_report["_meta"] = {
                "kg_law_id"          : AgentsEnums.CRIMINAL_PROCEDURE_LAW_ID.value,
                "kg_articles_fetched": fetched_articles,
                "articles_requested" : AgentsEnums.PROCEDURAL_ARTICLE_NUMBERS.value,
                "issues_reviewed"    : issues_count,
            }
        else:
            logger.error("LLM procedural audit failed to parse as valid dict.")
            audit_report = {
                "raw_response": audit_report,
                "violations": [],
                "_meta": {"kg_articles_fetched": fetched_articles},
            }

        return self._empty_update(state, "procedural_auditor", audit_report)