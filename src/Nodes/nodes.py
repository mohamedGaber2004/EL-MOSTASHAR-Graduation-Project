import logging , json
from src.Graph import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from src.LLMs import get_llm_llama , get_llm_oss , get_llm_qwn
from src.Utils import VerdictType
from src.Graph import (
    _retrieve_for_charge,
    _build_fallback_package,
    _read_file,
    _chunk_text,
    _merge_extracted,
    _parse_llm_json,
    _apply_extracted_to_state,
    _now,
    _agent_context,
)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



# =============================================================================
#  NODE 1 — ORCHESTRATOR
# =============================================================================

def orchestrator_node(state: AgentState) -> AgentState:
    """
    يستقبل الحالة، يحدد الخطوة التالية، يصرّح بالـ Context لكل Agent،
    ويحتفظ بالـ Global Context.
    """
    logger.info("🎯 Orchestrator — Case: %s | Completed: %s",
                state.case_id, state.completed_agents)

    # ── Validate minimum inputs ───────────────────────────────────────────────
    errors = []
    if not state.defendants:
        errors.append("لا يوجد متهمون مستخلصون من الملفات")
    if not state.charges:
        errors.append("لا توجد تهم مستخلصة من أمر الإحالة")

    if errors:
        logger.warning("⚠️  Orchestrator found input errors: %s", errors)

    return state.model_copy(update={
        "errors":       state.errors + errors,
        "current_agent": "orchestrator",
        "last_updated":  _now(),
    })



# =============================================================================
#  NODE 2 — DATA INGESTION & GRAPH BUILDER
# =============================================================================

_INGESTION_SYSTEM = """
أنت Data Ingestion Agent متخصص في القانون الجنائي المصري.
مهمتك الوحيدة: استخلاص المعلومات من النص كما هي — لا تفسير، لا ترجيح، لا استنتاج.
إذا لم يُذكر شيء صراحةً في النص، لا تضعه.

استخلص الكيانات التالية من النص المُعطى وأجب بـ JSON صارم فقط بدون أي نص إضافي:

{
  "case_meta": {
    "case_number":    "رقم القضية إن وُجد",
    "court":          "اسم المحكمة إن وُجد",
    "court_level":    "محكمة الجنح|محكمة الجنايات|محكمة الاستئناف|محكمة النقض|...",
    "jurisdiction":   "الدائرة إن وُجدت",
    "filing_date":    "YYYY-MM-DD أو null",
    "referral_date":  "YYYY-MM-DD أو null",
    "prosecutor_name":"اسم المحقق/المدعي إن وُجد"
  },
  "defendants": [
    {
      "name": "", "alias": null, "national_id": null, "gender": null,
      "age": null, "date_of_birth": null, "nationality": null,
      "occupation": null, "address": null,
      "mental_state": null, "mental_report_id": null,
      "prior_record": false, "prior_crimes": [], "prior_sentences": [],
      "complicity_role": null,
      "in_custody": null, "arrest_date": null, "detention_order_id": null,
      "source_document_id": "DOC_ID", "notes": null
    }
  ],
  "charges": [
    {
      "charge_id": "CHG-001",
      "statute": "نص التهمة",
      "law_code": "قانون العقوبات|قانون مكافحة المخدرات|...",
      "article_number": null, "article_paragraph": null,
      "incident_type": null, "description": null,
      "charge_date": null, "charge_location": null,
      "elements_required": [], "elements_proven": {},
      "attempt_flag": false, "complicity_role": null,
      "penalty_range": null, "penalty_min": null, "penalty_max": null,
      "aggravating_factors": [], "mitigating_factors": [],
      "linked_defendant_names": [],
      "source_document_id": "DOC_ID", "notes": null
    }
  ],
  "incidents": [
    {
      "incident_id": "INC-001",
      "incident_type": null,
      "incident_date": null, "incident_location": null,
      "incident_description": "",
      "perpetrator_names": [], "victim_names": [], "witness_names": [],
      "outcome": null, "outcome_severity": null,
      "linked_evidence_ids": [], "linked_charge_ids": [],
      "source_document_id": "DOC_ID", "notes": null
    }
  ],
  "evidences": [
    {
      "evidence_id": "EV-001",
      "evidence_type": null, "description": null,
      "seizure_date": null, "seizure_location": null,
      "seized_by": null, "seizure_warrant_present": null,
      "chain_of_custody_ok": null, "chain_of_custody_notes": null,
      "storage_conditions_ok": null,
      "validity_status": null, "invalidity_reason": null,
      "linked_charge_ids": [], "linked_charge_elements": [],
      "linked_defendant_name": null,
      "source_document_id": "DOC_ID", "page_reference": null, "notes": null
    }
  ],
  "lab_reports": [
    {
      "report_id": "LAB-001",
      "report_type": null, "lab_name": null,
      "examiner_name": null, "examination_date": null,
      "sample_id": null, "sample_type": null,
      "sample_source": null, "sample_chain_intact": null,
      "result": null, "result_is_positive": null,
      "confirms_charge_element": null, "linked_charge_element": null,
      "linked_evidence_id": null, "linked_defendant_name": null,
      "source_document_id": "DOC_ID", "notes": null
    }
  ],
  "witness_statements": [
    {
      "witness_name": "",
      "witness_national_id": null, "witness_type": null,
      "witness_occupation": null, "relation_to_defendant": null,
      "statement_date": null, "statement_stage": null,
      "was_sworn_in": null, "presence_at_scene": null,
      "statement_summary": null, "key_facts_mentioned": [],
      "consistency_flag": null, "contradicts_other_evidence": null,
      "contradiction_details": null,
      "prior_statement_exists": null, "retracted_statement": null,
      "credibility_flags": [],
      "source_document_id": "DOC_ID", "page_reference": null, "notes": null
    }
  ],
  "confessions": [
    {
      "confession_id": "CONF-001", "defendant_name": "",
      "confession_date": null, "confession_stage": null,
      "obtained_before_arrest": null, "obtained_after_arrest": null,
      "legal_counsel_present": null, "informed_of_rights": null,
      "coercion_claimed": null, "coercion_evidence": null,
      "text": null, "key_admissions": [],
      "voluntary": null, "consistent_with_facts": null,
      "consistent_with_evidence": null,
      "retracted": null, "retraction_date": null, "retraction_reason": null,
      "source_document_id": "DOC_ID", "page_reference": null, "notes": null
    }
  ],
  "procedural_issues": [
    {
      "issue_id": "PROC-001",
      "procedure_type": null,
      "procedure_date": null, "conducting_officer": null,
      "conducting_authority": null,
      "warrant_present": null, "warrant_id": null,
      "notification_to_da": null, "notification_time": null,
      "legal_timeframes_respected": null,
      "issue_description": null,
      "nullity_type": null, "article_basis": null,
      "affects_evidence_ids": [], "may_invalidate_case": null,
      "tainted_fruit_risk": null,
      "source_document_id": "DOC_ID", "page_reference": null, "notes": null
    }
  ],
  "prior_judgments": [
    {
      "judgment_id": "JUD-001", "case_reference": null,
      "court_name": null, "court_level": null,
      "verdict_date": null, "verdict_type": null,
      "charge_statutes": [], "key_legal_principles": [],
      "penalty_imposed": null, "relevance_to_current_case": null,
      "source_document_id": "DOC_ID", "notes": null
    }
  ],
  "defense_documents": [
    {
      "doc_id": "DEF-001", "doc_type": null,
      "submission_date": null, "submitted_by": null,
      "formal_defenses": [], "substantive_defenses": [],
      "alibi_claimed": null, "alibi_details": null,
      "defense_evidence_ids": [], "supporting_principles": [],
      "source_document_id": "DOC_ID", "notes": null
    }
  ]
}

قواعد صارمة:
- لا تختلق معلومة غير موجودة في النص.
- إذا لم تجد قيمة لحقل، ضع null أو [] حسب النوع.
- الـ source_document_id يساوي اسم الملف المُعالج.
- أجب بـ JSON فقط — بدون أي مقدمة أو تعليق.
"""


# ── Document type classifier prompt ──────────────────────────────────────────
_DOC_CLASSIFIER_PROMPT = """
صنّف هذا المستند القانوني إلى واحد من الأنواع التالية:
amr_ihalah | mahdar_dabt | taqrir_tibbi | aqwal_shuhud | mozakart_difa |
hukm_sabiq | taqrir_lab | other

أجب بكلمة واحدة فقط.

النص:
{text}
"""

def data_ingestion_node(state: AgentState) -> AgentState:
    """
    يقرأ ملفات .txt و .pdf النصية، يقطّعها، يُرسلها للـ LLM لاستخلاص
    الكيانات القانونية المنظمة، ويدمج النتائج في الـ AgentState.

    إذا كانت الكيانات موجودة مسبقًا في الـ state (مثل create_sample_case)،
    يتخطى قراءة الملفات ويُسجل فقط الـ Provenance Metadata.

    Pipeline:
        source_documents (paths)
            → _read_file()          [txt / pdf text extraction]
            → _chunk_text()         [overlap chunking]
            → LLM(_INGESTION_SYSTEM)[per-chunk structured extraction]
            → _parse_llm_json()     [safe JSON parsing]
            → _merge_extracted()    [dedup merge across chunks]
            → _apply_extracted_to_state() [Pydantic validation → state]
    """
    # ── Short-circuit: entities already loaded (e.g. create_sample_case) ──────
    already_loaded = any([
        state.defendants,
        state.charges,
        state.incidents,
        state.evidences,
        state.witness_statements,
        state.confessions,
        state.procedural_issues,
    ])

    if already_loaded:
        logger.info("📂 DataIngestion — entities already in state, skipping file read.")
        provenance = {
            "processed_at":   _now().isoformat(),
            "processed_docs": list(state.source_documents),
            "failed_docs":    [],
            "extracted_counts": {
                "defendants":        len(state.defendants),
                "charges":           len(state.charges),
                "incidents":         len(state.incidents),
                "evidences":         len(state.evidences),
                "lab_reports":       len(state.lab_reports),
                "witness_statements":len(state.witness_statements),
                "confessions":       len(state.confessions),
                "procedural_issues": len(state.procedural_issues),
                "prior_judgments":   len(state.prior_judgments),
                "defense_documents": len(state.defense_documents),
            },
            "graph_status": "entities_pre_loaded",
        }
        updated_outputs = {**state.agent_outputs, "data_ingestion": provenance}
        return state.model_copy(update={
            "agent_outputs":    updated_outputs,
            "completed_agents": state.completed_agents + ["data_ingestion"],
            "current_agent":    "data_ingestion",
            "last_updated":     _now(),
        })

    logger.info("📂 DataIngestion — %d file(s) to process", len(state.source_documents))

    all_extracted: dict = {
        "case_meta":         {},
        "defendants":        [],
        "charges":           [],
        "incidents":         [],
        "evidences":         [],
        "lab_reports":       [],
        "witness_statements":[],
        "confessions":       [],
        "procedural_issues": [],
        "prior_judgments":   [],
        "defense_documents": [],
    }

    file_errors:   list[str] = []
    processed_docs: list[str] = []

    for file_path in state.source_documents:

        doc_id = file_path.split("/")[-1]   # اسم الملف كـ ID
        logger.info("  📄 Reading: %s", doc_id)

        # ── 1. Read file ──────────────────────────────────────────────────────
        try:
            raw_text, file_type = _read_file(file_path)
        except (ValueError, ImportError, OSError) as e:
            err = f"[{doc_id}] قراءة فاشلة: {e}"
            logger.error("  ❌ %s", err)
            file_errors.append(err)
            continue

        if not raw_text.strip():
            err = f"[{doc_id}] الملف فارغ أو لا يحتوي على نص"
            logger.warning("  ⚠️  %s", err)
            file_errors.append(err)
            continue

        logger.info("  ✅ Read %d chars from %s (%s)", len(raw_text), doc_id, file_type)

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        chunks = _chunk_text(raw_text, chunk_size=3000, overlap=200)
        logger.info("  🔪 Split into %d chunk(s)", len(chunks))

        doc_extracted: dict = {k: ([] if isinstance(v, list) else {})
                               for k, v in all_extracted.items()}

        # ── 3. Extract per chunk ──────────────────────────────────────────────
        for chunk_idx, chunk in enumerate(chunks):
            logger.info("    🤖 Extracting chunk %d/%d ...", chunk_idx + 1, len(chunks))

            prompt = (
                f"اسم الملف: {doc_id}\n"
                f"رقم الـ chunk: {chunk_idx + 1} من {len(chunks)}\n\n"
                f"النص:\n{chunk}"
            )

            try:
                # ── Real LLM call ──────────────────────────────────────────────
                response = get_llm_qwn.invoke([
                SystemMessage(content=_INGESTION_SYSTEM),
                HumanMessage(content=prompt)
                ])
                chunk_result = _parse_llm_json(response.content)

                # ── Mock (remove when plugging real LLM) ──────────────────────
                chunk_result = {
                    "case_meta":         {},
                    "defendants":        [],
                    "charges":           [],
                    "incidents":         [],
                    "evidences":         [],
                    "lab_reports":       [],
                    "witness_statements":[],
                    "confessions":       [],
                    "procedural_issues": [],
                    "prior_judgments":   [],
                    "defense_documents": [],
                }
                # ─────────────────────────────────────────────────────────────

                doc_extracted = _merge_extracted(doc_extracted, chunk_result)

            except json.JSONDecodeError as e:
                err = f"[{doc_id}] chunk {chunk_idx+1}: JSON parsing فاشل — {e}"
                logger.warning("  ⚠️  %s", err)
                file_errors.append(err)
                continue

            except Exception as e:
                err = f"[{doc_id}] chunk {chunk_idx+1}: LLM call فاشل — {e}"
                logger.error("  ❌ %s", err)
                file_errors.append(err)
                continue

        # ── 4. Merge this doc's results into global accumulator ────────────────
        all_extracted = _merge_extracted(all_extracted, doc_extracted)
        processed_docs.append(doc_id)
        logger.info("  ✅ Done: %s", doc_id)

    # ── 5. Apply all extracted data to state ──────────────────────────────────
    logger.info("🔗 Applying extracted entities to state ...")
    state_updates = _apply_extracted_to_state(state, all_extracted, doc_id="__merged__")

    # ── 6. Build provenance metadata ──────────────────────────────────────────
    provenance = {
        "processed_at":      _now().isoformat(),
        "processed_docs":    processed_docs,
        "failed_docs":       file_errors,
        "extracted_counts": {
            field: len(all_extracted.get(field, []))
            for field in [
                "defendants", "charges", "incidents", "evidences",
                "lab_reports", "witness_statements", "confessions",
                "procedural_issues", "prior_judgments", "defense_documents",
            ]
        },
        "graph_status": "entities_extracted" if processed_docs else "no_documents_processed",
    }

    updated_outputs = {**state.agent_outputs, "data_ingestion": provenance}

    return state.model_copy(update={
        **state_updates,
        "agent_outputs":    updated_outputs,
        "completed_agents": state.completed_agents + ["data_ingestion"],
        "current_agent":    "data_ingestion",
        "errors":           state.errors + file_errors,
        "last_updated":     _now(),
    })



# =============================================================================
#  NODE 3 — PROCEDURAL AUDITOR
# =============================================================================

_PROCEDURAL_SYSTEM = """
أنت Procedural Auditor Agent.
مهمتك: مراجعة سلامة الإجراءات وفق قانون الإجراءات الجنائية المصري فقط.
تطبق مبادئ النقض الخاصة بالبطلان (نسبي / مطلق).
لا تحلل وقائع، لا ترجح أدلة.

لكل إجراء:
1. طابقه بشرطه القانوني (م 34-46 إجراءات جنائية).
2. حدد: صحة أم بطلان.
3. حدد نوع البطلان: مطلق أم نسبي.
4. حدد الأدلة المتأثرة (ثمرة الشجرة المسمومة).

أجب بـ JSON:
{
  "audit_results": [
    {
      "procedure": "...",
      "issue_id": "...",
      "status": "صحيح | باطل",
      "nullity_type": "بطلان مطلق | بطلان نسبي | لا بطلان",
      "affected_evidence_ids": [],
      "legal_basis": [],
      "tainted_fruit_applies": false,
      "notes": "..."
    }
  ],
  "overall_procedural_validity": "سليم | مشكوك | باطل",
  "critical_nullities": [],
  "recommendation": "..."
}
"""

def procedural_auditor_node(state: AgentState) -> AgentState:
    """
    يراجع سلامة إجراءات القبض والتفتيش والاستجواب.
    """
    logger.info("⚖️  ProceduralAuditor — Reviewing %d procedural issues",
                len(state.procedural_issues))

    context = _agent_context(state, "procedural_auditor")

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = f"""
راجع الإجراءات التالية وأصدر تقرير البطلان:

{context}

تذكر:
- م 41 دستور: القبض يستلزم أمرًا قضائيًا أو تلبسًا.
- م 45 إجراءات: التفتيش يستلزم إذنًا مسببًا.
- م 135 إجراءات: الاستجواب يستلزم إخطار النيابة خلال 24 ساعة.
- الاعتراف الباطل لا يُعتد به حتى لو صادقه المتهم لاحقًا.
"""

    # ── Call LLM ──────────────────────────────────────────────────────────────
    response = get_llm_llama().invoke([
    SystemMessage(content=_PROCEDURAL_SYSTEM),
    HumanMessage(content=prompt)
    ])
    audit_report = _parse_llm_json(response.content)

    # ── Mock output (replace with real LLM call) ──────────────────────────────
    audit_report = {
        "audit_results": [
            {
                "procedure":               pi.procedure_type,
                "issue_id":                pi.issue_id or "unknown",
                "status":                  "باطل" if pi.nullity_type and pi.nullity_type != "لا بطلان" else "صحيح",
                "nullity_type":            pi.nullity_type or "لا بطلان",
                "affected_evidence_ids":   pi.affects_evidence_ids,
                "legal_basis":             [pi.article_basis] if pi.article_basis else [],
                "tainted_fruit_applies":   pi.tainted_fruit_risk or False,
                "notes":                   pi.notes or "",
            }
            for pi in state.procedural_issues
        ],
        "overall_procedural_validity": (
            "باطل" if any(
                p.nullity_type and "مطلق" in str(p.nullity_type)
                for p in state.procedural_issues
            ) else "سليم"
        ),
        "critical_nullities": [
            p.issue_id for p in state.procedural_issues
            if p.nullity_type and "مطلق" in str(p.nullity_type)
        ],
        "recommendation": "مراجعة صحة الأدلة المتأثرة بالبطلان المطلق",
    }

    updated_outputs = {**state.agent_outputs, "procedural_auditor": audit_report}

    return state.model_copy(update={
        "agent_outputs":    updated_outputs,
        "completed_agents": state.completed_agents + ["procedural_auditor"],
        "current_agent":    "procedural_auditor",
        "last_updated":     _now(),
    })




# =============================================================================
#  NODE 4 — LEGAL RESEARCHER  (Graph RAG — FAISS + Neo4j)
#  Drop this into legal_graph.py replacing the old legal_researcher_node
# =============================================================================

_LEGAL_RESEARCH_SYSTEM = """
أنت Legal Researcher Agent متخصص في القانون الجنائي المصري.
مهمتك: البحث القانوني المنضبط فقط — لا تحلل وقائع، لا ترجح أدلة.
مصادرك الوحيدة: ما يُعطى لك من نصوص قانونية ومبادئ نقض مسترجعة.
استبعد أي مصدر غير مصري تلقائيًا.
أجب بـ JSON صارم فقط بدون أي نص إضافي أو markdown.

{
  "research_packages": [
    {
      "charge_statute":       "نص التهمة",
      "law_code":             "القانون المنطبق",
      "article_number":       "رقم المادة",
      "elements_of_crime":    ["ركن 1", "ركن 2"],
      "statutes":             ["نص المادة كاملًا من المصادر المُعطاة"],
      "aggravating_articles": ["مواد الظروف المشددة"],
      "mitigating_articles":  ["مواد الظروف المخففة"],
      "principles":           ["مبدأ قانوني من أحكام النقض المُعطاة"],
      "precedents":           ["نقض جلسة ... من المصادر المُعطاة"],
      "penalty_range":        "العقوبة المقررة بالنص"
    }
  ],
  "procedural_principles": ["مبدأ إجرائي من المصادر المُعطاة"],
  "relevant_cassation_rulings": ["حكم نقض ذي صلة من المصادر المُعطاة"]
}
"""


# ── Node ──────────────────────────────────────────────────────────────────────

def legal_researcher_node(state: "AgentState") -> "AgentState":
    from src.LLMs.Qwen_Model import get_llm_qwn
    from src.Vectorstore.vector_store_builder import LegalVectorStore
    from src.Graphstore.KG_builder import LegalKnowledgeGraph
    from src.Config.config import get_settings

    logger.info("📚 LegalResearcher — %d charge(s)", len(state.charges))

    if not state.charges:
        empty = {
            "research_packages":          [],
            "procedural_principles":      [],
            "relevant_cassation_rulings": [],
        }
        return state.model_copy(update={
            "agent_outputs":    {**state.agent_outputs, "legal_researcher": empty},
            "completed_agents": state.completed_agents + ["legal_researcher"],
            "current_agent":    "legal_researcher",
            "last_updated":     _now(),
        })

    # ── Init both retrieval systems ───────────────────────────────────────────
    lv = LegalVectorStore().build_laws(docs_filter="latest").build_na2d()
    kg = LegalKnowledgeGraph(get_settings().NEO4J_URI, get_settings().NEO4J_USERNAME, get_settings().NEO4J_PASSWORD)

    try:
        # ── Retrieve for each charge ──────────────────────────────────────────
        all_contexts = []
        for charge in state.charges:
            logger.info("  🔎 Charge: %s", charge.statute)
            ctx = _retrieve_for_charge(charge, lv, kg)
            all_contexts.append(ctx)

        # ── Build prompt ──────────────────────────────────────────────────────
        case_summary = json.dumps({
            "defendants": [d.name for d in state.defendants],
            "charges":    [c.statute for c in state.charges],
            "incidents":  [i.incident_description for i in state.incidents if i.incident_description],
        }, ensure_ascii=False, indent=2)

        retrieved_json = json.dumps(all_contexts, ensure_ascii=False, indent=2)

        prompt = f"""
معطيات القضية:
{case_summary}

المعطيات المسترجعة من قاعدة المعرفة (FAISS + Knowledge Graph):
{retrieved_json}

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

        # ── LLM synthesis ─────────────────────────────────────────────────────
        try:
            response = get_llm_qwn().invoke([
                SystemMessage(content=_LEGAL_RESEARCH_SYSTEM),
                HumanMessage(content=prompt),
            ])
            legal_package = _parse_llm_json(response.content)
            logger.info("✅ LegalResearcher done")

        except (json.JSONDecodeError, Exception) as e:
            logger.error("❌ LLM failed: %s", e)
            legal_package = _build_fallback_package(all_contexts)

    finally:
        # ── Always close KG connection ────────────────────────────────────────
        kg.close()

    return state.model_copy(update={
        "agent_outputs":    {**state.agent_outputs, "legal_researcher": legal_package},
        "completed_agents": state.completed_agents + ["legal_researcher"],
        "current_agent":    "legal_researcher",
        "last_updated":     _now(),
    })



# =============================================================================
#  NODE 5 — EVIDENCE ANALYST
# =============================================================================

_EVIDENCE_SYSTEM = """
أنت Evidence Analyst Agent.
مهمتك: تحليل الثبوت دون ترجيح نهائي — لا تُصدر حكمًا.
قيّم قوة كل دليل وارتباطه بكل واقعة وتهمة.
انتبه للتناقضات بين الأدلة.
أخذ في الحسبان أي أدلة باطلة إجرائيًا من تقرير المراقب الإجرائي.

أجب بـ JSON:
{
  "evidence_matrix": [
    {
      "fact": "...",
      "charge_statute": "...",
      "supporting_evidence": [
        {"evidence_id": "...", "type": "...", "strength": 0.0-1.0, "notes": "..."}
      ],
      "contradicting_evidence": [
        {"evidence_id": "...", "issue": "...", "impact": "..."}
      ],
      "invalidated_evidence": [],
      "fact_proven_score": 0.0-1.0
    }
  ],
  "overall_evidence_strength": 0.0-1.0,
  "key_weaknesses": [],
  "key_strengths": []
}
"""

def evidence_analyst_node(state: AgentState) -> AgentState:
    """
    يحلل الأدلة ويبني مصفوفة الإثبات.
    """
    logger.info("🔍 EvidenceAnalyst — Analyzing %d evidences, %d witnesses, %d confessions",
                len(state.evidences), len(state.witness_statements), len(state.confessions))

    context = _agent_context(state, "evidence_analyst")
    audit   = state.agent_outputs.get("procedural_auditor", {})

    # ── Collect invalidated evidence IDs from procedural audit ────────────────
    invalidated_ids: list[str] = []
    for result in audit.get("audit_results", []):
        if result.get("status") == "باطل":
            invalidated_ids.extend(result.get("affected_evidence_ids", []))

    prompt = f"""
حلل الأدلة التالية وأصدر مصفوفة الإثبات:

{context}

الأدلة الباطلة إجرائيًا (يجب استبعادها): {invalidated_ids}

لكل واقعة:
1. حدد الأدلة الداعمة وقوتها.
2. حدد التناقضات والتعارضات.
3. استبعد الأدلة الباطلة إجرائيًا.
4. احسب درجة الإثبات الإجمالية.
"""

    response = get_llm_oss().invoke([
    SystemMessage(content=_EVIDENCE_SYSTEM),
    HumanMessage(content=prompt)
    ])
    evidence_matrix = _parse_llm_json(response.content)

    # ── Mock ──────────────────────────────────────────────────────────────────
    valid_evidences = [e for e in state.evidences if e.evidence_id not in invalidated_ids]
    overall_strength = (
        sum(1 for e in valid_evidences if e.validity_status and "سليم" in str(e.validity_status))
        / max(len(valid_evidences), 1)
    )

    evidence_matrix = {
        "evidence_matrix": [
            {
                "fact":                  incident.incident_description or str(incident.incident_type),
                "charge_statute":        ", ".join(
                    c.statute for c in state.charges
                    if incident.incident_id in c.linked_defendant_names or True
                )[:1 or 1],
                "supporting_evidence": [
                    {
                        "evidence_id": e.evidence_id,
                        "type":        str(e.evidence_type),
                        "strength":    0.8 if e.validity_status and "سليم" in str(e.validity_status) else 0.3,
                        "notes":       e.notes or "",
                    }
                    for e in valid_evidences
                    if e.evidence_id in (incident.linked_evidence_ids or [])
                ],
                "contradicting_evidence": [
                    {
                        "evidence_id": w.witness_name,
                        "issue":       w.contradiction_details or "تناقض",
                        "impact":      "متوسط",
                    }
                    for w in state.witness_statements
                    if w.contradicts_other_evidence
                ],
                "invalidated_evidence":  invalidated_ids,
                "fact_proven_score":     overall_strength,
            }
            for incident in state.incidents
        ],
        "overall_evidence_strength": round(overall_strength, 2),
        "key_weaknesses": [
            f"دليل {e.evidence_id} مشكوك في صحته" for e in state.evidences
            if e.validity_status and "مشكوك" in str(e.validity_status)
        ],
        "key_strengths": [
            f"تقرير {r.report_id} يُثبت {r.linked_charge_element}" for r in state.lab_reports
            if r.confirms_charge_element
        ],
    }

    updated_outputs = {**state.agent_outputs, "evidence_analyst": evidence_matrix}

    return state.model_copy(update={
        "agent_outputs":    updated_outputs,
        "completed_agents": state.completed_agents + ["evidence_analyst"],
        "current_agent":    "evidence_analyst",
        "last_updated":     _now(),
    })



# =============================================================================
#  NODE 6 — DEFENSE ANALYST
# =============================================================================

_DEFENSE_SYSTEM = """
أنت Defense Analysis Agent.
مهمتك: تحليل دفوع المتهم شكلًا وموضوعًا.
لا تخترع دفوعًا غير موجودة في مستندات الدفاع.
صنف الدفوع: بطلان / قصور أدلة / تناقض / انتفاء ركن.
اربط كل دفع بسنده القانوني.

أجب بـ JSON:
{
  "formal_defenses": [
    {"defense": "...", "legal_basis": "...", "strength": "قوي|متوسط|ضعيف", "notes": "..."}
  ],
  "substantive_defenses": [
    {"defense": "...", "legal_basis": "...", "strength": "قوي|متوسط|ضعيف", "notes": "..."}
  ],
  "alibi_analysis": {"claimed": false, "supported_by_evidence": false, "notes": ""},
  "supporting_principles": [],
  "overall_defense_strength": "قوي|متوسط|ضعيف",
  "critical_defense_points": []
}
"""

def defense_analyst_node(state: AgentState) -> AgentState:
    """
    يحلل دفوع المتهم ويربطها بسندها القانوني.
    """
    logger.info("🛡️  DefenseAnalyst — Analyzing %d defense documents",
                len(state.defense_documents))

    context = _agent_context(state, "defense_analyst")

    prompt = f"""
حلل دفوع المتهم التالية:

{context}

لكل دفع:
1. صنّفه (شكلي / موضوعي).
2. حدد سنده القانوني.
3. قيّم قوته في مواجهة الأدلة المتاحة.
4. ارتباطه بأي بطلان إجرائي.
"""

    response = get_llm_qwn().invoke([
    SystemMessage(content=_DEFENSE_SYSTEM),
    HumanMessage(content=prompt)
    ])
    defense_analysis = _parse_llm_json(response.content)

    # ── Mock ──────────────────────────────────────────────────────────────────
    all_formal      = [d for doc in state.defense_documents for d in doc.formal_defenses]
    all_substantive = [d for doc in state.defense_documents for d in doc.substantive_defenses]
    all_principles  = [p for doc in state.defense_documents for p in doc.supporting_principles]
    any_alibi       = any(doc.alibi_claimed for doc in state.defense_documents)

    # Pull critical nullities from procedural audit
    audit_nullities = state.agent_outputs.get("procedural_auditor", {}).get("critical_nullities", [])

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
            "claimed":                    any_alibi,
            "supported_by_evidence":      False,   # يحتاج تحقق
            "notes":                      "يستلزم تحقيقًا في أدلة دفاع الحضور" if any_alibi else "",
        },
        "supporting_principles": all_principles,
        "overall_defense_strength": (
            "قوي"    if audit_nullities else
            "متوسط"  if all_formal or all_substantive else
            "ضعيف"
        ),
        "critical_defense_points": audit_nullities + all_formal[:2],
    }

    updated_outputs = {**state.agent_outputs, "defense_analyst": defense_analysis}

    return state.model_copy(update={
        "agent_outputs":    updated_outputs,
        "completed_agents": state.completed_agents + ["defense_analyst"],
        "current_agent":    "defense_analyst",
        "last_updated":     _now(),
    })


# =============================================================================
#  NODE 7 — JUDGE AGENT
# =============================================================================

_JUDGE_SYSTEM = """
أنت Judge Agent — قاضٍ جنائي مصري متخصص.
مهمتك: الموازنة بين كل المدخلات وإصدار مسودة حكم مسببة.
لا حكم بلا تسبيب — لا ترجيح بلا سند.

منهجيتك:
1. عرض الوقائع الثابتة.
2. تحديد التهمة وأركانها.
3. مناقشة الأدلة عنصرًا عنصرًا.
4. الرد على دفوع الدفاع.
5. بحث صحة الإجراءات.
6. الوصول للنتيجة.

أجب بـ JSON:
{
  "established_facts": [],
  "charge_analysis": [
    {
      "charge_statute": "...",
      "elements_proven": {},
      "all_elements_satisfied": false
    }
  ],
  "evidence_discussion": [],
  "defense_responses": [],
  "procedural_findings": "",
  "verdict": "إدانة | براءة | عدم كفاية الأدلة | عدم الاختصاص | بطلان المحاكمة",
  "verdict_reasoning": "...",
  "applicable_penalty": "...",
  "confidence_score": 0.0-1.0,
  "deficiencies": [],
  "doubts": []
}
"""

def judge_node(state: AgentState) -> AgentState:
    """
    يوازن بين كل المدخلات ويُصدر مسودة الحكم المسببة.
    """
    logger.info("👨‍⚖️  JudgeAgent — Deliberating on case %s", state.case_id)

    context = _agent_context(state, "judge")

    prompt = f"""
أصدر مسودة حكم مسببة في القضية التالية:

{context}

تأكد من:
1. مناقشة كل دليل على حدة.
2. الرد الصريح على كل دفع من دفوع الدفاع.
3. الإشارة الصريحة لأي نقص أو شك أو قصور.
4. عدم الإدانة إلا باليقين — الشك يُفسَّر لصالح المتهم.
"""

    response = get_llm_llama().invoke([
    SystemMessage(content=_JUDGE_SYSTEM),
    HumanMessage(content=prompt)
    ])
    judgment = _parse_llm_json(response.content)

    # ── Mock deliberation logic ───────────────────────────────────────────────
    evidence_strength  = state.agent_outputs.get("evidence_analyst", {}).get("overall_evidence_strength", 0.0)
    has_critical_nullity = bool(
        state.agent_outputs.get("procedural_auditor", {}).get("critical_nullities", [])
    )

    if has_critical_nullity and evidence_strength < 0.5:
        verdict      = VerdictType.MISTRIAL
        verdict_text = "بطلان المحاكمة"
        confidence   = 0.85
        reasoning    = "وجود بطلان مطلق في الإجراءات مع قصور في الأدلة المشروعة."
    elif evidence_strength >= 0.75:
        verdict      = VerdictType.CONVICTION
        verdict_text = "إدانة"
        confidence   = evidence_strength
        reasoning    = "الأدلة كافية ومتعاضدة لإثبات أركان التهمة."
    elif evidence_strength >= 0.5:
        verdict      = VerdictType.INSUFFICIENT_EVD
        verdict_text = "عدم كفاية الأدلة"
        confidence   = 0.6
        reasoning    = "الأدلة غير كافية لإثبات التهمة بما لا يدع مجالًا للشك."
    else:
        verdict      = VerdictType.ACQUITTAL
        verdict_text = "براءة"
        confidence   = 0.7
        reasoning    = "الشك يُفسَّر لصالح المتهم — لا يوجد دليل كافٍ."

    judgment = {
        "established_facts": [
            i.incident_description for i in state.incidents if i.incident_description
        ],
        "charge_analysis": [
            {
                "charge_statute":         c.statute,
                "elements_proven":        c.elements_proven,
                "all_elements_satisfied": all(c.elements_proven.values()) if c.elements_proven else False,
            }
            for c in state.charges
        ],
        "evidence_discussion":   state.agent_outputs.get("evidence_analyst", {}).get("evidence_matrix", []),
        "defense_responses":     state.agent_outputs.get("defense_analyst", {}).get("critical_defense_points", []),
        "procedural_findings":   state.agent_outputs.get("procedural_auditor", {}).get("overall_procedural_validity", ""),
        "verdict":               verdict_text,
        "verdict_reasoning":     reasoning,
        "applicable_penalty":    next((c.penalty_range for c in state.charges if c.penalty_range), "يحدده القاضي"),
        "confidence_score":      confidence,
        "deficiencies":          state.agent_outputs.get("evidence_analyst", {}).get("key_weaknesses", []),
        "doubts":                [] if evidence_strength >= 0.75 else ["قصور في الإثبات"],
    }

    updated_outputs = {**state.agent_outputs, "judge": judgment}

    return state.model_copy(update={
        "agent_outputs":    updated_outputs,
        "suggested_verdict":verdict,
        "confidence_score": confidence,
        "reasoning_trace":  [{"agent": "judge", "reasoning": reasoning}],
        "completed_agents": state.completed_agents + ["judge"],
        "current_agent":    "judge",
        "last_updated":     _now(),
    })