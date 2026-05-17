from __future__ import annotations
import logging

from ..agent_base import AgentBase
from langchain_core.messages import HumanMessage, SystemMessage
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Utils.Enums.agents_enums import AgentsEnums
from src.agents.Agents_output_models import ProceduralAuditResult, ProceduralIssue, ExcludedDefenseClaim
from src.Graph.state import AgentState
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
        self.kg = kg
        self.vs = vector_store
        self.embeddings = embeddings

    # ── ANN query scoped to Criminal Procedure Law ────────────────────
    _PROCEDURAL_LAW_ANN_QUERY = """
        CALL db.index.vector.queryNodes('article_embeddings', $k, $query_vector)
        YIELD node AS article, score
        WHERE article.law_id = $law_id
        RETURN
            article.article_id     AS article_id,
            article.article_number AS article_number,
            article.law_id         AS law_id,
            article.text           AS text,
            score
        ORDER BY score DESC
    """

    def _retrieve_articles_for_proceeding(self,text: str,k: int = 15,threshold: float = 0.45) -> list[dict]:
        if self.kg is None or self.embeddings is None:
            logger.warning("KG or embeddings not available — skipping retrieval")
            return []

        try:
            query_vector = self.embeddings.embed_query(text)
        except Exception as e:
            logger.warning("embed_query failed for [%s...]: %s", text[:60], e)
            return []

        law_id = AgentsEnums.CRIMINAL_PROCEDURE_LAW_ID.value

        try:
            with self.kg.driver.session() as s:
                records = s.run(
                    self._PROCEDURAL_LAW_ANN_QUERY,
                    k=k,
                    query_vector=query_vector,
                    law_id=law_id,
                ).data()
        except Exception as e:
            logger.warning("ANN query failed for [%s...]: %s", text[:60], e)
            return []

        articles = [
            {
                "Article Number": str(r["article_number"]),
                "The Law of the Article": str(r["article_id"]),
                "The Article Text": str(r["text"]),
            }
            for r in records
            if r.get("article_number") and float(r.get("score", 0)) >= threshold
        ]

        logger.info(
            "Embedding retrieval — text=[%s...] | law=%s | hits=%d (threshold=%.2f)",
            text[:60], law_id, len(articles), threshold,
        )
        return articles

    # ── extract plain text from a proceeding or defense issue entry ───
    @staticmethod
    def _extract_text(entry) -> str:
        """
        Safely extract a searchable text string from either a dict entry
        (criminal proceeding or formal defense issue) or a plain string.
        """
        if isinstance(entry, dict):
            return (
                entry.get("issue_description")
                or entry.get("warrant_present")
                or entry.get("procedure_type")
                or entry.get("description")
                or entry.get("text")
                or ""
            ).strip()
        return str(entry).strip()

    # ── deduplicated article accumulator ──────────────────────────────
    @staticmethod
    def _accumulate_unique(entries: list,fetch_fn,label: str) -> list[dict]:
        """
        Iterate `entries`, call `fetch_fn(text)` for each, and accumulate
        unique article dicts (deduplicated by Article Number).
        """
        seen: set[str] = set()
        result: list[dict] = []

        for entry in entries:
            text = ProceduralAuditorAgent._extract_text(entry)
            if not text or text.lower() == "none":
                logger.warning("Skipping empty entry in [%s]", label)
                continue

            for article in fetch_fn(text):
                key = article["Article Number"]
                if key not in seen:
                    seen.add(key)
                    result.append(article)

        if not result:
            logger.warning("No articles found for [%s] — LLM will rely on internal knowledge", label)
        else:
            logger.info("Article resolution [%s] — %d unique article(s)", label, len(result))

        return result

    # ── dynamic article discovery ─────────────────────────────────────
    def _resolve_article_numbers(self,criminal_proceedings: list,defense_procedural_issues: list) -> tuple[list[dict], list[dict]]:
        """
        Returns two lists of unique article dicts:
          1. Articles relevant to the criminal proceedings.
          2. Articles relevant to the defense procedural issues.
        """
        proceedings_articles = self._accumulate_unique(
            criminal_proceedings,
            self._retrieve_articles_for_proceeding,
            "criminal_proceedings",
        )
        defense_articles = self._accumulate_unique(
            defense_procedural_issues,
            self._retrieve_articles_for_proceeding,
            "defense_procedural_issues",
        )
        return proceedings_articles, defense_articles

    # ── judicial principles retrieval ─────────────────────────────────
    def _retrieve_principles_for_entries(self,entries: list,label: str) -> list:
        """
        Retrieve unique judicial principles for a list of proceeding/issue entries.
        """
        seen: set = set()
        result = []

        for entry in entries:
            text = self._extract_text(entry)
            if not text or text.lower() == "none":
                logger.warning("Skipping empty entry in [%s] while retrieving principles", label)
                continue

            for principle in self.retrieve_principles(text, self.vs):
                # Deduplicate by string representation (adjust key if principle is a dict)
                key = str(principle)
                if key not in seen:
                    seen.add(key)
                    result.append(principle)

        if not result:
            logger.warning("No principles found for [%s]", label)
        else:
            logger.info("Principles resolution [%s] — %d unique principle(s)", label, len(result))

        return result

    def retrieve_proceedings_and_defense_principles(self,criminal_proceedings: list,defense_procedural_issues: list) -> tuple[list, list]:
        proceedings_principles = self._retrieve_principles_for_entries(
            criminal_proceedings, "criminal_proceedings"
        )
        issues_principles = self._retrieve_principles_for_entries(
            defense_procedural_issues, "defense_procedural_issues"
        )
        return proceedings_principles, issues_principles

    # ── prompt builder ────────────────────────────────────────────────
    def _build_prompt(self, state: AgentState) -> str:
        incidents = getattr(state, "incidents", [])
        confessions = getattr(state, "confessions", [])
        criminal_proceedings = getattr(state, "criminal_proceedings", [])
        defendants = [
            {"name": getattr(d, "name", None), "id": getattr(d, "id", None)}
            for d in (state.defendants or [])
        ]
        defense_procedural_issues = [
            {"formal_defenses": getattr(document, "formal_defenses")}
            for document in state.defense_documents
        ]
        proceedings_articles, defense_articles = self._resolve_article_numbers(
            criminal_proceedings, defense_procedural_issues
        )
        proceedings_principles, defense_principles = (
            self.retrieve_proceedings_and_defense_principles(
                criminal_proceedings, defense_procedural_issues
            )
        )

        return (
            "## المسائل الإجرائية المستخرجة من وقائع القضية:\n"
            f"{criminal_proceedings}\n\n"

            "## الإشكاليات الإجرائية المُثارة من محامي الدفاع (تحتاج تحققاً):\n"
            f"{defense_procedural_issues}\n\n"

            "## تفاصيل الحوادث والقبض:\n"
            f"{incidents}\n\n"

            "## المتهمون:\n"
            f"{defendants}\n\n"

            "## الاعترافات (إن وجدت):\n"
            f"{confessions}\n\n"

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
    
    # ── main entry ────────────────────────────────────────────────────
    def run(self, state):
        criminal_proceedings      = getattr(state, "criminal_proceedings", []) or []
        defense_procedural_issues = getattr(state.defense_documents, "formal_defenses", []) or []

        proceedings_count = len(criminal_proceedings)
        defense_count     = len(defense_procedural_issues)

        logger.info(
            "ProceduralAuditorAgent starting — %d proceeding(s), %d defense issue(s)",
            proceedings_count, defense_count,
        )

        if not criminal_proceedings and not defense_procedural_issues:
            return self._empty_update(state, "procedural_auditor", {
                "procedural_audit": ProceduralAuditResult(
                    overall_assessment="لا توجد مسائل إجرائية للمراجعة",
                )
            })

        prompt = self._build_prompt(state)
        logger.debug("ProceduralAuditorAgent prompt built — invoking LLM")

        _fallback = ProceduralAuditResult(
            overall_assessment="فشل استدعاء النموذج — لا يمكن إتمام المراجعة الإجرائية",
        )

        try:
            response     = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            raw_package  = self._parse_agent_json(response.content)
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
                    overall_assessment=raw_package.get("overall_assessment"),
                    critical_nullities=raw_package.get("critical_nullities", []),
                    kg_articles_used=raw_package.get("kg_articles_used", []),
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
            "proceedings_count":    proceedings_count,
            "defense_issues_count": defense_count,
            "kg_connected":         self.kg is not None,
            "vs_connected":         self.vs is not None,
        }

        return self._empty_update(state, "procedural_auditor", {
            "procedural_audit": audit_result,
        })