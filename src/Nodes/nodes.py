import logging, json
from pathlib import Path
from src.Utils import VerdictType
from src.Graph.state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from src.LLMs import get_llm_llama, get_llm_oss, get_llm_qwn
from src.Prompts.data_ingestion_agent import DATA_INGESTION_AGENT_PROMPT
from src.Prompts.procedural_auditor_agent import PROCEDURAL_AUDITOR_AGENT_PROMPT
from src.Prompts.legal_researcher_agent import LEGAL_RESEARCHER_AGENT_PROMPT
from src.Prompts.evidence_analyst_agent import EVIDENCE_ANALYST_AGENT_PROMPT
from src.Prompts.defense_analysis_agent import DEFENSE_ANALYSIS_AGENT_PROMPT
from src.Prompts.judge_agent import JUDGE_AGENT_PROMPT
from src.Graph.graph_helpers import (
    _retrieve_for_charge,
    _build_fallback_package,
    _merge_extracted,
    _parse_llm_json,
    _apply_extracted_to_state,
    _now,
    _agent_context,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
#  NODE 1 — DATA INGESTION & GRAPH BUILDER
# =============================================================================

_INGESTION_SYSTEM = DATA_INGESTION_AGENT_PROMPT


def _extract_file(txt_doc_id: str, raw_text: str, all_extracted: dict, file_errors: list, processed_docs: list):
    """Single LLM call for one .txt file — mutates all_extracted in place."""
    prompt = f"اسم الملف: {txt_doc_id}\n\nالنص:\n{raw_text}"
    try:
        response = get_llm_qwn().invoke([
            SystemMessage(content=_INGESTION_SYSTEM),
            HumanMessage(content=prompt),
        ])
        result = _parse_llm_json(response.content)

        # Ensure the LLM returned a dict we can merge.
        if not isinstance(result, dict):
            # If it's a string that can be parsed to JSON, try that
            try:
                if isinstance(result, str):
                    import json as _json
                    parsed = _json.loads(result)
                    if isinstance(parsed, dict):
                        result = parsed
                    else:
                        raise ValueError("parsed non-dict")
                else:
                    raise ValueError("non-dict result")
            except Exception as e:
                err = f"[{txt_doc_id}] LLM returned non-dict JSON — {e}"
                logger.warning("  ⚠️  %s", err)
                file_errors.append(err)
                return

        all_extracted.update(_merge_extracted(all_extracted, result))
        processed_docs.append(txt_doc_id)
        logger.info("  ✅ Done: %s", txt_doc_id)

    except json.JSONDecodeError as e:
        err = f"[{txt_doc_id}] JSON parsing فاشل — {e}"
        logger.warning("  ⚠️  %s", err)
        file_errors.append(err)

    except Exception as e:
        err = f"[{txt_doc_id}] LLM call فاشل — {e}"
        logger.error("  ❌ %s", err)
        file_errors.append(err)


def data_ingestion_node(state: AgentState) -> AgentState:

    all_extracted: dict = {
        "case_meta": {}, "defendants": [], "charges": [], "incidents": [],
        "evidences": [], "lab_reports": [], "witness_statements": [],
        "confessions": [], "procedural_issues": [], "prior_judgments": [],
        "defense_documents": [],
    }
    file_errors:    list[str] = []
    processed_docs: list[str] = []

    for file_path in state.source_documents:
        doc_id = file_path.split("/")[-1]
        path   = Path(file_path)
        logger.info("  📄 Reading: %s", doc_id)

        try:
            if path.is_dir():
                txt_files = sorted(path.glob("*.txt"))
                if not txt_files:
                    err = f"[{doc_id}] المجلد لا يحتوي على ملفات .txt"
                    logger.warning("  ⚠️  %s", err)
                    file_errors.append(err)
                    continue

                for txt_file in txt_files:
                    try:
                        raw_text = txt_file.read_text(encoding="utf-8")
                    except OSError as e:
                        err = f"[{txt_file.name}] قراءة فاشلة: {e}"
                        logger.error("  ❌ %s", err)
                        file_errors.append(err)
                        continue

                    if not raw_text.strip():
                        err = f"[{txt_file.name}] الملف فارغ"
                        logger.warning("  ⚠️  %s", err)
                        file_errors.append(err)
                        continue

                    logger.info("  ✅ Read %d chars from %s", len(raw_text), txt_file.name)
                    _extract_file(txt_file.name, raw_text, all_extracted, file_errors, processed_docs)
                continue

            else:
                raw_text = path.read_text(encoding="utf-8")

        except OSError as e:
            err = f"[{doc_id}] قراءة فاشلة: {e}"
            logger.error("  ❌ %s", err)
            file_errors.append(err)
            continue

        if not raw_text.strip():
            err = f"[{doc_id}] الملف فارغ"
            logger.warning("  ⚠️  %s", err)
            file_errors.append(err)
            continue

        logger.info("  ✅ Read %d chars from %s", len(raw_text), doc_id)
        _extract_file(doc_id, raw_text, all_extracted, file_errors, processed_docs)

    # ── Apply & finalize ──────────────────────────────────────────────────────
    logger.info("🔗 Applying extracted entities to state ...")
    state_updates = _apply_extracted_to_state(state, all_extracted, doc_id="__merged__")

    provenance = {
        "processed_at":   _now().isoformat(),
        "processed_docs": processed_docs,
        "failed_docs":    file_errors,
        "extracted_counts": {
            field: len(all_extracted.get(field, []))
            for field in [
                "defendants", "charges", "incidents", "evidences", "lab_reports",
                "witness_statements", "confessions", "procedural_issues",
                "prior_judgments", "defense_documents",
            ]
        },
        "graph_status": "entities_extracted" if processed_docs else "no_documents_processed",
    }

    return state.model_copy(update={
        **state_updates,
        "agent_outputs":    {**state.agent_outputs, "data_ingestion": provenance},
        "completed_agents": state.completed_agents + ["data_ingestion"],
        "current_agent":    "data_ingestion",
        "errors":           state.errors + file_errors,
        "last_updated":     _now(),
    })


# =============================================================================
#  NODE 2 — PROCEDURAL AUDITOR
# =============================================================================

_PROCEDURAL_SYSTEM = PROCEDURAL_AUDITOR_AGENT_PROMPT


def procedural_auditor_node(state: AgentState) -> AgentState:
    logger.info("⚖️  ProceduralAuditor — Reviewing %d procedural issues", len(state.procedural_issues))

    prompt = f"""
راجع الإجراءات التالية وأصدر تقرير البطلان:

{_agent_context(state, "procedural_auditor")}

تذكر:
- م 41 دستور: القبض يستلزم أمرًا قضائيًا أو تلبسًا.
- م 45 إجراءات: التفتيش يستلزم إذنًا مسببًا.
- م 135 إجراءات: الاستجواب يستلزم إخطار النيابة خلال 24 ساعة.
- الاعتراف الباطل لا يُعتد به حتى لو صادقه المتهم لاحقًا.
"""

    response    = get_llm_llama().invoke([SystemMessage(content=_PROCEDURAL_SYSTEM), HumanMessage(content=prompt)])
    audit_report = _parse_llm_json(response.content)

    return state.model_copy(update={
        "agent_outputs":    {**state.agent_outputs, "procedural_auditor": audit_report},
        "completed_agents": state.completed_agents + ["procedural_auditor"],
        "current_agent":    "procedural_auditor",
        "last_updated":     _now(),
    })


# =============================================================================
#  NODE 3 — LEGAL RESEARCHER
# =============================================================================

_LEGAL_RESEARCH_SYSTEM = LEGAL_RESEARCHER_AGENT_PROMPT


def legal_researcher_node(state: AgentState) -> AgentState:
    from src.Vectorstore.vector_store_builder import vs, search
    from src.Graphstore.KG_builder import LegalKnowledgeGraph
    from src.Config.config import get_settings

    logger.info("📚 LegalResearcher — %d charge(s)", len(state.charges))

    if not state.charges:
        return state.model_copy(update={
            "agent_outputs":    {**state.agent_outputs, "legal_researcher": {
                "research_packages": [], "procedural_principles": [], "relevant_cassation_rulings": [],
            }},
            "completed_agents": state.completed_agents + ["legal_researcher"],
            "current_agent":    "legal_researcher",
            "last_updated":     _now(),
        })

    cfg = get_settings()
    kg  = LegalKnowledgeGraph(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD)

    try:
        all_contexts = []
        for charge in state.charges:
            logger.info("  🔎 Charge: %s", charge.statute)
            # _retrieve_for_charge now receives the raw FAISS `vs` instead of LegalVectorStore
            ctx = _retrieve_for_charge(charge, vs, kg)
            all_contexts.append(ctx)

        case_summary  = json.dumps({
            "defendants": [d.name for d in state.defendants],
            "charges":    [c.statute for c in state.charges],
            "incidents":  [i.incident_description for i in state.incidents if i.incident_description],
        }, ensure_ascii=False, indent=2)

        prompt = f"""
معطيات القضية:
{case_summary}

المعطيات المسترجعة من قاعدة المعرفة (FAISS + Knowledge Graph):
{json.dumps(all_contexts, ensure_ascii=False, indent=2)}

كل context يحتوي على:
- statute_docs: نصوص المواد القانونية المتشابهة دلاليًا
- cassation_docs: مبادئ وأحكام النقض ذات الصلة
- kg_context.penalties: العقوبات المنصوص عليها في الـ Graph مباشرة
- kg_context.definitions: تعريفات المصطلحات من نص القانون
- kg_context.references: المواد المُحال إليها
- kg_context.amendments: التعديلات التي طرأت على هذه المواد
- kg_context.latest_versions: أحدث نسخة للمادة بعد التعديل

المطلوب:
لكل تهمة ابنِ حزمة بحث قانوني من هذه المعطيات فقط — لا تخترع معلومة.
استخدم kg_context.penalties كمصدر أول للعقوبة.
استخدم kg_context.latest_versions إن وُجدت بدلًا من النص الأصلي.
"""

        try:
            response     = get_llm_qwn().invoke([SystemMessage(content=_LEGAL_RESEARCH_SYSTEM), HumanMessage(content=prompt)])
            legal_package = _parse_llm_json(response.content)
            logger.info("✅ LegalResearcher done")
        except Exception as e:
            logger.error("❌ LLM failed: %s", e)
            legal_package = _build_fallback_package(all_contexts)

    finally:
        kg.close()

    return state.model_copy(update={
        "agent_outputs":    {**state.agent_outputs, "legal_researcher": legal_package},
        "completed_agents": state.completed_agents + ["legal_researcher"],
        "current_agent":    "legal_researcher",
        "last_updated":     _now(),
    })


# =============================================================================
#  NODE 4 — EVIDENCE ANALYST
# =============================================================================

_EVIDENCE_SYSTEM = EVIDENCE_ANALYST_AGENT_PROMPT


def evidence_analyst_node(state: AgentState) -> AgentState:
    logger.info("🔍 EvidenceAnalyst — %d evidences, %d witnesses, %d confessions",
                len(state.evidences), len(state.witness_statements), len(state.confessions))

    audit           = state.agent_outputs.get("procedural_auditor", {})
    invalidated_ids = [
        eid
        for result in audit.get("audit_results", [])
        if result.get("status") == "باطل"
        for eid in result.get("affected_evidence_ids", [])
    ]

    prompt = f"""
حلل الأدلة التالية وأصدر مصفوفة الإثبات:

{_agent_context(state, "evidence_analyst")}

الأدلة الباطلة إجرائيًا (يجب استبعادها): {invalidated_ids}

لكل واقعة:
1. حدد الأدلة الداعمة وقوتها.
2. حدد التناقضات والتعارضات.
3. استبعد الأدلة الباطلة إجرائيًا.
4. احسب درجة الإثبات الإجمالية.
"""

    response       = get_llm_oss().invoke([SystemMessage(content=_EVIDENCE_SYSTEM), HumanMessage(content=prompt)])
    evidence_matrix = _parse_llm_json(response.content)

    return state.model_copy(update={
        "agent_outputs":    {**state.agent_outputs, "evidence_analyst": evidence_matrix},
        "completed_agents": state.completed_agents + ["evidence_analyst"],
        "current_agent":    "evidence_analyst",
        "last_updated":     _now(),
    })


# =============================================================================
#  NODE 5 — DEFENSE ANALYST
# =============================================================================

_DEFENSE_SYSTEM = DEFENSE_ANALYSIS_AGENT_PROMPT


def defense_analyst_node(state: AgentState) -> AgentState:
    logger.info("🛡️  DefenseAnalyst — %d defense documents", len(state.defense_documents))

    prompt = f"""
حلل دفوع المتهم التالية:

{_agent_context(state, "defense_analyst")}

لكل دفع:
1. صنّفه (شكلي / موضوعي).
2. حدد سنده القانوني.
3. قيّم قوته في مواجهة الأدلة المتاحة.
4. ارتباطه بأي بطلان إجرائي.
"""

    get_llm_qwn().invoke([SystemMessage(content=_DEFENSE_SYSTEM), HumanMessage(content=prompt)])

    # ── Build structured defense analysis ────────────────────────────────────
    audit_nullities  = state.agent_outputs.get("procedural_auditor", {}).get("critical_nullities", [])
    all_formal       = [d for doc in state.defense_documents for d in doc.formal_defenses]
    all_substantive  = [d for doc in state.defense_documents for d in doc.substantive_defenses]
    all_principles   = [p for doc in state.defense_documents for p in doc.supporting_principles]
    any_alibi        = any(doc.alibi_claimed for doc in state.defense_documents)

    defense_analysis = {
        "formal_defenses": [
            {"defense": d, "legal_basis": "قانون الإجراءات الجنائية", "strength": "متوسط", "notes": ""}
            for d in all_formal
        ] + [
            {"defense": f"بطلان إجرائي — {n}", "legal_basis": "م 45 إجراءات", "strength": "قوي", "notes": ""}
            for n in audit_nullities
        ],
        "substantive_defenses": [
            {"defense": d, "legal_basis": "قانون العقوبات", "strength": "متوسط", "notes": ""}
            for d in all_substantive
        ],
        "alibi_analysis": {
            "claimed":               any_alibi,
            "supported_by_evidence": False,
            "notes":                 "يستلزم تحقيقًا في أدلة دفاع الحضور" if any_alibi else "",
        },
        "supporting_principles":    all_principles,
        "overall_defense_strength": "قوي" if audit_nullities else "متوسط" if (all_formal or all_substantive) else "ضعيف",
        "critical_defense_points":  audit_nullities + all_formal[:2],
    }

    return state.model_copy(update={
        "agent_outputs":    {**state.agent_outputs, "defense_analyst": defense_analysis},
        "completed_agents": state.completed_agents + ["defense_analyst"],
        "current_agent":    "defense_analyst",
        "last_updated":     _now(),
    })


# =============================================================================
#  NODE 6 — JUDGE
# =============================================================================

_JUDGE_SYSTEM = JUDGE_AGENT_PROMPT


def judge_node(state: AgentState) -> AgentState:
    logger.info("👨‍⚖️  JudgeAgent — Deliberating on case %s", state.case_id)

    prompt = f"""
أصدر مسودة حكم مسببة في القضية التالية:

{_agent_context(state, "judge")}

تأكد من:
1. مناقشة كل دليل على حدة.
2. الرد الصريح على كل دفع من دفوع الدفاع.
3. الإشارة الصريحة لأي نقص أو شك أو قصور.
4. عدم الإدانة إلا باليقين — الشك يُفسَّر لصالح المتهم.
"""

    response = get_llm_llama().invoke([SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)])
    judgment = _parse_llm_json(response.content)

    verdict_text = judgment.get("verdict", "")
    verdict      = VerdictType(verdict_text) if verdict_text else VerdictType.ACQUITTAL

    return state.model_copy(update={
        "agent_outputs":     {**state.agent_outputs, "judge": judgment},
        "suggested_verdict": verdict,
        "confidence_score":  judgment.get("confidence_score", 0.0),
        "reasoning_trace":   [{"agent": "judge", "reasoning": judgment.get("verdict_reasoning", "")}],
        "completed_agents":  state.completed_agents + ["judge"],
        "current_agent":     "judge",
        "last_updated":      _now(),
    })