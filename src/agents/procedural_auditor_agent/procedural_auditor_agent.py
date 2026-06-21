from __future__ import annotations
import logging
import json
from typing import Optional

from langsmith import traceable
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

    # =========================================================================
    # Text extraction
    # =========================================================================
    @staticmethod
    def _extract_text(entry) -> str:
        """
        Extracts a descriptive query string from a CriminalProceedings
        object or dict for use in KG / VS retrieval.
        """
        if isinstance(entry, dict):
            parts = []
            if entry.get("procedure_type"):
                parts.append(f"نوع الإجراء: {entry['procedure_type']}")
            if entry.get("description"):
                parts.append(f"الوصف: {entry['description']}")
            if entry.get("formal_defenses"):
                defenses = entry["formal_defenses"]
                if isinstance(defenses, list):
                    parts.append("دفوع: " + " / ".join(str(d) for d in defenses))
            if entry.get("legal_basis_present") is not None:
                parts.append(
                    f"السند القانوني: {'موجود' if entry['legal_basis_present'] else 'غير موجود'}"
                )
            if entry.get("legal_basis_type"):
                parts.append(f"نوع السند: {entry['legal_basis_type']}")
            return " | ".join(parts)

        parts = []
        if getattr(entry, "procedure_type", None):
            parts.append(f"نوع الإجراء: {entry.procedure_type}")
        if getattr(entry, "description", None):
            parts.append(f"الوصف: {entry.description}")
        if getattr(entry, "conducting_officer", None):
            parts.append(f"المنفذ: {entry.conducting_officer}")

        proc_dt = getattr(entry, "procedure_datetime", None)
        parts.append(
            f"توقيت الإجراء: {proc_dt}" if proc_dt
            else "توقيت الإجراء: غير مذكور بالمحضر"
        )

        basis_present = getattr(entry, "legal_basis_present", None)
        if basis_present is not None:
            parts.append(f"السند القانوني: {'موجود' if basis_present else 'غير موجود'}")

        basis_type = getattr(entry, "legal_basis_type", None)
        if basis_type:
            parts.append(f"نوع السند: {basis_type}")

        return " | ".join(parts)

    # =========================================================================
    # Retrieval (Single-pass optimized)
    # =========================================================================
    @traceable(name="retrieve_for_entry", run_type="retriever")
    def _retrieve_for_entry(self, text: str, k: int = 25, threshold: float = 0.45) -> tuple[list[dict], list[dict]]:
        """
        Run query_transformation ONCE for the aggregated text block, 
        then hit KG and VS. We use a higher k (25) to account for grouped text.
        """
        # ── 1. Transform once ─────────────────────────────────────────────
        try:
            transformed     = self.query_transformation(text) or {}
            article_query   = (transformed.get("article_query")  or "").strip()
            principle_query = (transformed.get("principle_query") or "").strip()
        except Exception as exc:
            logger.warning("query_transformation failed for [%s...]: %s", text[:60], exc)
            return [], []

        # ── 2. KG retrieval ───────────────────────────────────────────────
        articles: list[dict] = []
        if article_query and self.kg is not None and self.embeddings is not None:
            try:
                records = get_kg_retriever().retrieve(
                    question  = article_query,
                    k         = k,
                    threshold = threshold,
                )
                articles = [
                    {
                        "Article Number":         str(r["article_number"]),
                        "The Law of the Article": str(r["law_id"]),
                        "The Article Text":       str(r["text"]),
                    }
                    for r in records.sources
                    if r.get("article_number") and float(r.get("score", 0)) >= threshold
                ]
            except Exception as exc:
                logger.warning("KG retrieval failed for [%s...]: %s", article_query[:60], exc)

        # ── 3. VS retrieval ───────────────────────────────────────────────
        principles: list[dict] = []
        if principle_query and self.vs is not None:
            try:
                for doc in self.retrieve_principles(principle_query, self.vs):
                    principles.append({
                        "principle_text": doc.page_content,
                        "court_source": (
                            doc.metadata.get("source_file") if doc.metadata else None
                        ),
                    })
            except Exception as exc:
                logger.warning("VS retrieval failed for [%s...]: %s", principle_query[:60], exc)

        return articles, principles

    # =========================================================================
    # Accumulate across a list of entries (Batched Optimization)
    # =========================================================================
    @traceable(name="accumulate_for_entries", run_type="chain")
    def _accumulate_for_entries(self, entries: list, label: str) -> tuple[list[dict], list[dict]]:
        """
        Combines all entry texts into a single context block. 
        This prevents hitting the LLM/DB N times in a loop.
        """
        if not entries:
            return [], []

        # 1. Gather all non-empty texts
        texts = []
        for entry in entries:
            text = self._extract_text(entry)
            if text and text.lower() != "none":
                texts.append(text)

        if not texts:
            return [], []

        # 2. Combine into a single unified context block
        combined_text = "\n".join(f"- {t}" for t in texts)

        # Protect against massive prompts if cases are excessively large
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."

        logger.info("[%s] Sending batched text to query transformer (%d items aggregated)", label, len(texts))

        # 3. Retrieve ONCE using the combined text
        articles, principles = self._retrieve_for_entry(combined_text)

        # Deduplicate results
        seen_articles, all_articles = set(), []
        for article in articles:
            key = article["Article Number"]
            if key not in seen_articles:
                seen_articles.add(key)
                all_articles.append(article)

        seen_principles, all_principles = set(), []
        for principle in principles:
            key = principle["principle_text"][:80]
            if key not in seen_principles:
                seen_principles.add(key)
                all_principles.append(principle)

        logger.info(
            "[%s] batched resolution complete — %d article(s), %d principle(s)",
            label, len(all_articles), len(all_principles),
        )
        return all_articles, all_principles

    # =========================================================================
    # Build entries lists
    # =========================================================================
    @staticmethod
    def _build_proceedings_entries(criminal_proceedings: list, confessions: list) -> list:
        """
        Merge criminal proceedings and confessions into a unified list
        of retrieval-ready entries.
        """
        entries = list(criminal_proceedings)

        for c in confessions:
            entries.append({
                "procedure_type":        "استجواب",
                "description":           getattr(c, "text", ""),
                "procedure_datetime":    getattr(c, "confession_date", None),
                "legal_basis_present":   None,
                "legal_basis_type":      "unknown",
                "legal_basis_note":      None,
                "conducting_officer":    None,
            })

        return entries

    @staticmethod
    def _build_defense_entries(defense_documents: list) -> list:
        """
        Flatten defense documents into retrieval-ready dicts.
        """
        return [
            {"formal_defenses": getattr(doc, "formal_defenses", []) or []}
            for doc in defense_documents
            if getattr(doc, "formal_defenses", None)
        ]

    # =========================================================================
    # Prompt builder
    # =========================================================================
    @traceable(name="build_procedural_audit_prompt", run_type="prompt")
    def _build_prompt(
        self,
        state: AgentState,
        proceedings_articles: list[dict],
        proceedings_principles: list[dict],
        defense_articles: list[dict],
        defense_principles: list[dict],
    ) -> str:

        def serialize_proceedings(procs: list) -> str:
            return json.dumps([
                {
                    "procedure_type":        getattr(p, "procedure_type", None),
                    "description":           getattr(p, "description", None),
                    "procedure_datetime":    getattr(p, "procedure_datetime", None),
                    "legal_basis_present":   getattr(p, "legal_basis_present", None),
                    "legal_basis_type":      getattr(p, "legal_basis_type", None),
                    "legal_basis_note":      getattr(p, "legal_basis_note", None),
                    "conducting_officer":    getattr(p, "conducting_officer", None),
                    "source_document_id":    getattr(p, "source_document_id", None),
                }
                for p in procs
                if any(getattr(p, attr, None) for attr in [
                    "procedure_type",
                    "description",
                    "legal_basis_present",
                    "legal_basis_type",
                    "legal_basis_note",
                ])
            ], ensure_ascii=False, indent=2)

        def serialize_defense(docs: list) -> str:
            return json.dumps([
                {
                    "formal_defense": defense,
                    "defendant_name": getattr(doc, "defendant_name", None),
                    "submitted_by":   getattr(doc, "submitted_by", None),
                }
                for doc in docs
                for defense in (getattr(doc, "formal_defenses", []) or [])
            ], ensure_ascii=False, indent=2)

        def serialize_confessions(confessions: list) -> str:
            return json.dumps([
                {
                    "procedure_type":        "استجواب",
                    "defendant_name":        getattr(c, "defendant_name", None),
                    "confession_stage":      getattr(c, "confession_stage", None),
                    "legal_counsel_present": getattr(c, "legal_counsel_present", None),
                    "coercion_claimed":      getattr(c, "coercion_claimed", None),
                    "summary":               getattr(c, "text", None),
                }
                for c in confessions
            ], ensure_ascii=False, indent=2)

        criminal_proceedings = getattr(state, "criminal_proceedings", []) or []
        confessions          = getattr(state, "confessions", []) or []
        defense_documents    = state.defense_documents or []

        return (
            "## الإجراءات الجنائية في القضية:\n"
            f"{serialize_proceedings(criminal_proceedings)}\n\n"

            "## الدفوع الإجرائية للمحامي:\n"
            f"{serialize_defense(defense_documents)}\n\n"

            "## إجراءات الاستجواب (من محضر الاستجواب):\n"
            f"{serialize_confessions(confessions)}\n\n"
            "⚠️ انتبه: coercion_claimed=true = بطلان مطلق للاعتراف فوراً.\n"
            "⚠️ انتبه: legal_counsel_present=false لا يعني بطلاناً تلقائياً — "
            "لكن يستوجب الفحص إذا كان الاستجواب في مرحلة التحقيق.\n\n"

            "---\n"
            "## المواد القانونية ذات الصلة بوقائع القضية:\n"
            f"{json.dumps(proceedings_articles, ensure_ascii=False, indent=2)}\n\n"

            "## المبادئ القضائية ذات الصلة بوقائع القضية:\n"
            f"{json.dumps(proceedings_principles, ensure_ascii=False, indent=2)}\n\n"

            "## المواد القانونية ذات الصلة بدفوع المحامي:\n"
            f"{json.dumps(defense_articles, ensure_ascii=False, indent=2)}\n\n"

            "## المبادئ القضائية ذات الصلة بدفوع المحامي:\n"
            f"{json.dumps(defense_principles, ensure_ascii=False, indent=2)}\n\n"

            "---\n"
            "## التعليمات النهائية:\n"
            "1. طبّق الاختبار المزدوج على كل إجراء: (أ) القاعدة جوهرية؟ (ب) ضرر فعلي على حق الدفاع؟\n"
            "2. حدد sanction_type الصحيح: بطلان مطلق / بطلان نسبي / انعدام / سقوط / عدم القبول.\n"
            "3. وضّح الأثر على ثلاث طبقات: الإجراء ذاته / الإجراءات السابقة / الإجراءات اللاحقة.\n"
            "4. لكل مخالفة: هل هي قابلة للتصحيح (م.335)؟ هل تحققت الغاية رغم العيب (م.334)؟\n"
            "5. كل دفع أثاره المحامي يظهر في violations أو excluded_defense_claims — لا يُحذف صمتاً.\n"
            "6. الاعتراف المنتزع بإكراه = بطلان مطلق دائماً.\n"
            "7. لا تذكر مادة قانونية لم ترد في السياق أعلاه.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا أي نص خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # =========================================================================
    # Main entry
    # =========================================================================
    @traceable(name="procedural_auditor_agent_run", run_type="chain")
    def run(self, state: AgentState, config=None):
        criminal_proceedings = getattr(state, "criminal_proceedings", []) or []
        confessions          = getattr(state, "confessions",          []) or []
        defense_documents    = state.defense_documents or []

        defense_procedural_issues = [
            item
            for doc in defense_documents
            for item in (getattr(doc, "formal_defenses", []) or [])
        ]

        logger.info(
            "ProceduralAuditorAgent starting — "
            "%d proceeding(s) | %d confession(s) | %d defense issue(s)",
            len(criminal_proceedings),
            len(confessions),
            len(defense_procedural_issues),
        )

        if not criminal_proceedings and not confessions and not defense_procedural_issues:
            return self._empty_update(state, "procedural_auditor", {
                "procedural_audit": ProceduralAuditResult(
                    overall_assessment="لا توجد مسائل إجرائية للمراجعة",
                )
            })

        # ── Build Entries ──────────────────────────────────────────────────
        proceedings_entries = self._build_proceedings_entries(
            criminal_proceedings, confessions
        )
        defense_entries = self._build_defense_entries(defense_documents)

        # ── Batched Retrieval ──────────────────────────────────────────────
        # This reduces N loops down to exactly 2 calls total
        proceedings_articles, proceedings_principles = self._accumulate_for_entries(
            proceedings_entries, "proceedings"
        )
        defense_articles, defense_principles = self._accumulate_for_entries(
            defense_entries, "defense"
        )

        # ── LLM call ─────────────────────────────────────────────────────
        _fallback = ProceduralAuditResult(
            overall_assessment="فشل استدعاء النموذج — لا يمكن إتمام المراجعة الإجرائية",
        )

        raw_package = None
        try:
            prompt   = self._build_prompt(
                state,
                proceedings_articles,
                proceedings_principles,
                defense_articles,
                defense_principles,
            )
            response = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            raw_package = self._parse_agent_json(response.content)
        except Exception as exc:
            logger.error("ProceduralAuditorAgent LLM invocation failed: %s", exc)

        if not isinstance(raw_package, dict):
            logger.error("LLM response is not a dict — using fallback")
            return self._empty_update(state, "procedural_auditor", {
                "procedural_audit": _fallback,
            })

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
                critical_nullities = raw_package.get("critical_nullities",  []),
                correctable_issues = raw_package.get("correctable_issues",  []),
                kg_articles_used   = raw_package.get("kg_articles_used",    []),
                meta={
                    "proceedings_count":    len(criminal_proceedings),
                    "confessions_count":    len(confessions),
                    "defense_issues_count": len(defense_procedural_issues),
                    "kg_connected":         self.kg is not None,
                    "vs_connected":         self.vs is not None,
                },
            )
        except Exception as exc:
            logger.error("Pydantic validation failed: %s", exc)
            audit_result = _fallback

        logger.info(
            "Procedural audit complete — "
            "%d violation(s) | %d excluded claim(s) | "
            "%d critical nullit(ies) | %d correctable issue(s)",
            len(audit_result.violations),
            len(audit_result.excluded_defense_claims),
            len(audit_result.critical_nullities),
            len(audit_result.correctable_issues),
        )

        return self._empty_update(state, "procedural_auditor", {
            "procedural_audit": audit_result,
        })