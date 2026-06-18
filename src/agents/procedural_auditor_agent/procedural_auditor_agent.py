from __future__ import annotations
import logging, json

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
    def _retrieve_articles_for_proceeding(self, text: str, k: int = 15, threshold: float = 0.45) -> list[dict]:
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
        """
        يستخرج نص وصفي كامل من CriminalProceedings object أو dict
        لاستخدامه في KG retrieval query.
        """
        # لو Pydantic model
        if hasattr(entry, "__dict__") or hasattr(entry, "model_fields"):
            parts = []
            if getattr(entry, "procedure_type", None):
                parts.append(f"نوع الإجراء: {entry.procedure_type}")
            if getattr(entry, "description", None):
                parts.append(f"الوصف: {entry.description}")
            if getattr(entry, "conducting_officer", None):
                parts.append(f"المنفذ: {entry.conducting_officer}")

            # ← الإضافة الجديدة هنا
            proc_dt = getattr(entry, "procedure_datetime", None)
            parts.append(f"توقيت الإجراء: {proc_dt}" if proc_dt else "توقيت الإجراء: غير مذكور بالمحضر")

            warrant = getattr(entry, "warrant_present", None)
            if warrant is not None:
                parts.append(f"إذن النيابة: {'موجود' if warrant else 'غير موجود'}")
            return " | ".join(parts)

        # لو dict
        if isinstance(entry, dict):
            parts = []
            if entry.get("procedure_type"):
                parts.append(f"نوع الإجراء: {entry['procedure_type']}")
            if entry.get("description"):
                parts.append(f"الوصف: {entry['description']}")
            if entry.get("formal_defenses"):
                defenses = entry["formal_defenses"]
                if isinstance(defenses, list):
                    parts.append("دفوع: " + " / ".join(defenses))
            return " | ".join(parts)

        return str(entry).strip()

    # ── deduplicated article accumulator ──────────────────────────────────────
    def _accumulate_unique_articles(self, entries: list, label: str) -> list[dict]:
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
    def _accumulate_unique_principles(self, entries: list, label: str) -> list[dict]:
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
    def _resolve_all(self,all_proceedings: list,defense_procedural_issues: list) -> tuple[list, list, list, list]:

        proceedings_articles = self._accumulate_unique_articles(
            all_proceedings, "all_proceedings"
        )
        defense_articles = self._accumulate_unique_articles(
            defense_procedural_issues, "defense_procedural_issues"
        )
        proceedings_principles = self._accumulate_unique_principles(
            all_proceedings, "all_proceedings"
        )
        defense_principles = self._accumulate_unique_principles(
            defense_procedural_issues, "defense_procedural_issues"
        )
        return proceedings_articles, defense_articles, proceedings_principles, defense_principles

    # ── prompt builder ────────────────────────────────────────────────────────
    def _build_prompt(self, state: AgentState) -> str:
            
            def serialize_proceedings(procs):
                result = []
                for p in procs:
                    result.append({
                        "procedure_type":     getattr(p, "procedure_type", None),
                        "description":        getattr(p, "description", None),
                        "warrant_present":    getattr(p, "warrant_present", None),
                        "conducting_officer": getattr(p, "conducting_officer", None),
                        "source_document_id": getattr(p, "source_document_id", None),
                    })
                return json.dumps(result, ensure_ascii=False, indent=2)

            def serialize_defense(docs):
                result = []
                for doc in docs:
                    for defense in (getattr(doc, "formal_defenses", []) or []):
                        result.append({
                            "formal_defense": defense,
                            "defendant_name": getattr(doc, "defendant_name", None),
                            "submitted_by":   getattr(doc, "submitted_by", None),
                        })
                return json.dumps(result, ensure_ascii=False, indent=2)

            def serialize_confessions_as_proceedings(confessions):
                """
                يحول الاعترافات لإجراءات إجرائية للمراجعة:
                - غياب المحامي → إجراء محتمل الخلل
                - ادعاء الإكراه → بطلان مطلق
                """
                result = []
                for c in confessions:
                    result.append({
                        "procedure_type":      "استجواب",
                        "defendant_name":      getattr(c, "defendant_name", None),
                        "confession_stage":    getattr(c, "confession_stage", None),
                        "legal_counsel_present": getattr(c, "legal_counsel_present", None),
                        "coercion_claimed":    getattr(c, "coercion_claimed", None),
                        "summary":             getattr(c, "text", None),
                    })
                return json.dumps(result, ensure_ascii=False, indent=2)

            confessions          = getattr(state, "confessions", []) or []
            criminal_proceedings      = getattr(state, "criminal_proceedings", []) or []
            defense_procedural_issues = [
                {"formal_defenses": getattr(doc, "formal_defenses", [])}
                for doc in (state.defense_documents or [])
            ]

            all_proceedings_for_retrieval = list(criminal_proceedings) + [
                type("obj", (), {
                    "procedure_type":   "استجواب",
                    "description":      getattr(c, "text", ""),
                    "procedure_datetime": getattr(c, "confession_date", None),
                    "warrant_present":  None,
                    "conducting_officer": None,
                })()
                for c in confessions
            ]

            (
                proceedings_articles, defense_articles,
                proceedings_principles, defense_principles,
            ) = self._resolve_all(all_proceedings_for_retrieval, defense_procedural_issues)

            return (
                "## الإجراءات الجنائية في القضية:\n"
                f"{serialize_proceedings(criminal_proceedings)}\n\n"

                "## الدفوع الإجرائية للمحامي:\n"
                f"{serialize_defense(state.defense_documents or [])}\n\n"

                "## إجراءات الاستجواب (من محضر الاستجواب):\n"
                f"{serialize_confessions_as_proceedings(confessions)}\n\n"
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

    def run(self, state):
            criminal_proceedings = getattr(state, "criminal_proceedings", []) or []
            confessions          = getattr(state, "confessions", []) or []
            defense_procedural_issues = [
                item
                for doc in (state.defense_documents or [])
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

            _fallback = ProceduralAuditResult(
                overall_assessment="فشل استدعاء النموذج — لا يمكن إتمام المراجعة الإجرائية",
            )

            raw_package = None
            try:
                prompt   = self._build_prompt(state)
                response = self._llm_invoke_with_retries(
                    self._llm,
                    [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
                )
                raw_package = self._parse_agent_json(response.content)
            except Exception as e:
                logger.error("ProceduralAuditorAgent LLM invocation failed: %s", e)

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
                        critical_nullities  = raw_package.get("critical_nullities", []),
                        correctable_issues  = raw_package.get("correctable_issues", []),
                        kg_articles_used    = raw_package.get("kg_articles_used", []),
                            meta = {
                                "proceedings_count":    len(criminal_proceedings),
                                "confessions_count":    len(confessions),
                                "defense_issues_count": len(defense_procedural_issues),
                                "kg_connected":         self.kg is not None,
                                "vs_connected":         self.vs is not None,
                            },
                    )
                except Exception as e:
                    logger.error("ProceduralAuditorAgent: Pydantic validation failed: %s", e)
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