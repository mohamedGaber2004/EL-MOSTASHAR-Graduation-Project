import logging , json
from typing import  Dict , Any
from datetime import datetime
from datetime import datetime
from src.Graph import AgentState
from src.Vectorstore import LegalVectorStore
from src.Graphstore import LegalKnowledgeGraph
from src.Utils import (
    Defendant,
    Charge,
    Evidence,
    LabReport,
    WitnessStatement,
    Confession,
    ProceduralIssue,
    DefenseDocument,
    PriorJudgment,
    CaseIncident
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_charge_query(charge: "Charge") -> str:
    """
    يبني query نصي من حقول التهمة لتحقيق أفضل recall في البحث.
    يرتب الحقول من الأكثر تخصصًا للأقل.
    """
    parts = []

    if charge.article_number:
        parts.append(f"المادة {charge.article_number}")
    if charge.statute:
        parts.append(charge.statute)
    if charge.law_code:
        parts.append(str(charge.law_code))
    if charge.description:
        parts.append(charge.description)
    if charge.elements_required:
        parts.append(" ".join(charge.elements_required[:4]))
    if charge.aggravating_factors:
        parts.append(" ".join(charge.aggravating_factors[:2]))

    return " | ".join(parts)


def _docs_to_text_list(docs: list, max_chars: int = 500) -> list[dict]:
    result = []
    for doc in docs:
        meta = doc.metadata or {}
        result.append({
            "content":        doc.page_content[:max_chars],
            "law_id":         meta.get("law_id"),
            "article_number": meta.get("article_number"),
            "chunk_type":     meta.get("chunk_type"),
            "law_title":      meta.get("law_title"),
            "crime_category": meta.get("crime_category"),
            "chunk_id":       meta.get("chunk_id"),
        })
    return result

def _lawcode_to_law_id(charge: "Charge") -> str | None:
    if not charge.law_code:
        return None
    mapping = {
        "قانون العقوبات":               "penal_code",
        "قانون مكافحة المخدرات":         "anti_drugs",
        "قانون مكافحة الإرهاب":          "anti_terror",
        "قانون الأسلحة والذخائر":        "weapons_ammunition",
        "قانون مكافحة جرائم المعلومات":  "cybercrime",
        "قانون مكافحة غسيل الأموال":     "money_laundering",
    }
    return mapping.get(str(charge.law_code))  # None = no filter

def _retrieve_for_charge(
    charge:     "Charge",
    lv:         "LegalVectorStore",
    kg:         "LegalKnowledgeGraph",
    k_statutes: int = 6,
    k_cassation: int = 6,
) -> dict:
    """
    Hybrid retrieval:
      1. FAISS vector search  → top-k semantically similar articles
      2. KG Cypher expansion  → penalties, definitions, amendments,
                                cross-references for those articles
      3. Na2d FAISS search    → cassation rulings + principles
    """
    query = _build_charge_query(charge)

    # ── 1. FAISS: semantic search for relevant articles ───────────────────────
    statute_docs = []
    try:
        statute_docs = lv.search_laws(
            query      = query,
            k          = k_statutes,
            law_id     = _lawcode_to_law_id(charge),
        )
        logger.info("    📖 FAISS statutes: %d", len(statute_docs))
    except Exception as e:
        logger.warning("    ⚠️  FAISS statutes failed: %s", e)

    # ── 2. KG expansion: for each retrieved article, pull its full subgraph ───
    kg_context = {}
    article_numbers = [
        d.metadata.get("article_number")
        for d in statute_docs
        if d.metadata.get("article_number")
    ]
    law_id = _lawcode_to_law_id(charge)

    if article_numbers and law_id:
        try:
            with kg.driver.session() as session:

                # ── Penalties for these articles ──────────────────────────────
                penalties = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:HAS_PENALTY]->(p:Penalty)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number AS article, 
                           p.penalty_type  AS type,
                           p.min_value     AS min,
                           p.max_value     AS max,
                           p.unit          AS unit
                """, law_id=law_id, article_numbers=article_numbers).data()

                # ── Definitions inside these articles ─────────────────────────
                definitions = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:DEFINES]->(d:Definition)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number AS article,
                           d.term           AS term,
                           d.definition_text AS text
                """, law_id=law_id, article_numbers=article_numbers).data()

                # ── Articles referenced BY these articles ─────────────────────
                references = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:REFERENCES]->(ref:Article)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number   AS from_article,
                           ref.article_number AS to_article,
                           ref.text           AS ref_text
                    LIMIT 10
                """, law_id=law_id, article_numbers=article_numbers).data()

                # ── Amendments that touched these articles ────────────────────
                amendments = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:AMENDED_BY]->(am:Amendment)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number      AS article,
                           am.amendment_date     AS date,
                           am.amendment_type     AS type,
                           am.description        AS description
                    ORDER BY am.amendment_date DESC
                    LIMIT 5
                """, law_id=law_id, article_numbers=article_numbers).data()

                # ── Latest (superseding) version of these articles ────────────
                latest_versions = session.run("""
                    MATCH (new:Article)-[:SUPERSEDES]->(old:Article {law_id: $law_id})
                    WHERE old.article_number IN $article_numbers
                    RETURN old.article_number AS article,
                           new.text           AS latest_text,
                           new.version        AS version
                    ORDER BY new.version DESC
                """, law_id=law_id, article_numbers=article_numbers).data()

                # ── Aggravating/mitigating topics linked to these articles ─────
                topics = session.run("""
                    MATCH (a:Article {law_id: $law_id})-[:TAGGED_WITH]->(t:Topic)
                    WHERE a.article_number IN $article_numbers
                    RETURN a.article_number AS article,
                           t.name           AS topic
                """, law_id=law_id, article_numbers=article_numbers).data()

            kg_context = {
                "penalties":       penalties,
                "definitions":     definitions,
                "references":      references,
                "amendments":      amendments,
                "latest_versions": latest_versions,
                "topics":          topics,
            }
            logger.info(
                "    🕸️  KG expanded: %d penalties | %d defs | %d refs | %d amendments",
                len(penalties), len(definitions), len(references), len(amendments)
            )
        except Exception as e:
            logger.warning("    ⚠️  KG expansion failed: %s", e)

    # ── 3. FAISS: cassation rulings + principles ──────────────────────────────
    cassation_docs = []
    try:
        cassation_docs = lv.search_na2d(
            query          = query,
            k              = k_cassation,
            source         = "both",
            crime_category = _map_incident_to_crime_category(charge),
        )
        logger.info("    ⚖️  Cassation: %d docs", len(cassation_docs))
    except Exception as e:
        logger.warning("    ⚠️  Cassation retrieval failed: %s", e)

    return {
        "charge_statute":   charge.statute,
        "article_number":   charge.article_number,
        "law_code":         str(charge.law_code) if charge.law_code else None,
        "incident_type":    str(charge.incident_type) if charge.incident_type else None,
        "query_used":       query,
        # FAISS results
        "statute_docs":     _docs_to_text_list(statute_docs),
        "cassation_docs":   _docs_to_text_list(cassation_docs),
        # KG structured expansion ← هذا هو الفرق الحقيقي
        "kg_context":       kg_context,
    }

def _map_incident_to_crime_category(charge: "Charge") -> str | None:
    """
    يحاول تحويل IncidentType إلى crime_category يفهمه Na2dRetriever.
    يعيد None إذا لم يكن هناك تطابق — فيبحث بدون تصفية.
    """
    if not charge.incident_type:
        return None

    # هذا الـ mapping مبني على القيم الموجودة في metadata.crime_category
    # في قاعدة بياناتك — عدّله حسب قيمك الفعلية
    mapping = {
        "قتل عمد":             "قتل",
        "قتل خطأ":             "قتل",
        "ضرب وجرح":            "جرائم الإيذاء",
        "ضرب أفضى إلى موت":    "جرائم الإيذاء",
        "سطو مسلح":            "سرقة",
        "سرقة":                "سرقة",
        "حيازة مخدرات":        "مخدرات",
        "اتجار في مخدرات":     "مخدرات",
        "اغتصاب":              "جرائم الشرف",
        "تحرش":                "جرائم الشرف",
        "نصب واحتيال":         "غش وتدليس",
        "تزوير":               "تزوير",
        "إرهاب":               "إرهاب",
    }
    return mapping.get(str(charge.incident_type))


def _build_fallback_package(contexts: list[dict]) -> dict:
    """
    يبني حزمة بحث مبسطة من النصوص المسترجعة مباشرة
    في حالة فشل الـ LLM أو خطأ JSON parsing.
    الـ _fallback flag يُخبر Judge Agent إن البحث ناقص.
    """
    return {
        "research_packages": [
            {
                "charge_statute":       ctx["charge_statute"],
                "law_code":             ctx.get("law_code"),
                "article_number":       ctx.get("article_number"),
                "elements_of_crime":    [],
                "statutes": [
                    d["content"]
                    for d in ctx.get("statute_docs", [])[:3]
                ],
                "aggravating_articles": [],
                "mitigating_articles":  [],
                "principles": [
                    d["content"]
                    for d in ctx.get("cassation_docs", [])[:3]
                ],
                "precedents":           [],
                "penalty_range":        "يحدده القاضي — فشل استخلاص العقوبة",
            }
            for ctx in contexts
        ],
        "procedural_principles":      [],
        "relevant_cassation_rulings": [],
        "_fallback": True,
    }





def _read_file(file_path: str) -> tuple[str, str]:
    """
    يقرأ ملف .txt أو .pdf ويُعيد (النص الخام، نوع الملف).
    لا OCR — فقط نصوص مباشرة.

    Returns:
        (raw_text, file_extension)

    Raises:
        ValueError: إذا كان الملف غير مدعوم أو غير موجود.
    """
    import os
    from pathlib import Path

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"الملف غير موجود: {file_path}")

    ext = path.suffix.lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "txt"

    elif ext == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(f"[صفحة {page_num}]\n{page_text}")
            if not text_parts:
                raise ValueError(f"الملف {file_path} لا يحتوي على نص قابل للاستخراج (مسح ضوئي غير مدعوم)")
            return "\n\n".join(text_parts), "pdf"

        except ImportError:
            raise ImportError(
                "مكتبة pdfplumber غير مثبتة. "
                "ثبّتها بـ: pip install pdfplumber"
            )

    else:
        raise ValueError(
            f"نوع الملف غير مدعوم: {ext} — المدعوم فقط: .txt و .pdf"
        )


def _chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """
    يقسم النص إلى chunks مع تداخل للحفاظ على السياق عند الحدود.
    chunk_size بالأحرف لا بالكلمات — مناسب للعربية.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks   = []
    start    = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # محاولة القطع عند فاصل طبيعي (سطر جديد أو نقطة)
        if end < text_len:
            for sep in ("\n\n", "\n", ".", "،"):
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = max(start + 1, end - overlap)   # overlap للسياق

    return [c for c in chunks if c]


def _merge_extracted(base: dict, new: dict) -> dict:
    """
    يدمج نتائج استخلاص chunk جديد مع ما تم استخلاصه مسبقًا.
    القوائم تُضاف، القاموس المفرد (case_meta) يُحدَّث بالقيم غير الـ null فقط.
    """
    merged = {k: v for k, v in base.items()}

    # case_meta: خذ القيمة الأولى غير null لكل حقل
    if "case_meta" in new and "case_meta" in merged:
        for k, v in new["case_meta"].items():
            if v is not None and merged["case_meta"].get(k) is None:
                merged["case_meta"][k] = v
    elif "case_meta" in new:
        merged["case_meta"] = new["case_meta"]

    # القوائم: أضف عناصر جديدة فقط (تجنب التكرار بناءً على الاسم/الـ ID)
    list_fields = [
        "defendants", "charges", "incidents", "evidences",
        "lab_reports", "witness_statements", "confessions",
        "procedural_issues", "prior_judgments", "defense_documents",
    ]

    id_keys = {
        "defendants":         "name",
        "charges":            "charge_id",
        "incidents":          "incident_id",
        "evidences":          "evidence_id",
        "lab_reports":        "report_id",
        "witness_statements": "witness_name",
        "confessions":        "confession_id",
        "procedural_issues":  "issue_id",
        "prior_judgments":    "judgment_id",
        "defense_documents":  "doc_id",
    }

    for field in list_fields:
        if field not in new:
            continue

        existing  = merged.get(field, [])
        id_key    = id_keys[field]
        seen_ids  = {item.get(id_key) for item in existing if item.get(id_key)}

        for item in new.get(field, []):
            item_id = item.get(id_key)
            if item_id and item_id in seen_ids:
                continue    # مكرر — تجاهل
            existing.append(item)
            if item_id:
                seen_ids.add(item_id)

        merged[field] = existing

    return merged


def _parse_llm_json(raw: str) -> dict:
    """
    يستخرج JSON من استجابة الـ LLM بشكل آمن،
    يتعامل مع markdown code blocks والمسافات الزائدة.
    """
    import re
    text = raw.strip()

    # إزالة ```json ... ``` إن وُجد
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()

    # إزالة أي نص قبل أول `{`
    brace_start = text.find("{")
    if brace_start > 0:
        text = text[brace_start:]

    return json.loads(text)


def _apply_extracted_to_state(state: AgentState, extracted: dict, doc_id: str) -> dict:
    """
    يحوّل الـ dict المستخلص من الـ LLM إلى Pydantic objects ويدمجها في الـ state.
    يُعيد dict يُمرَّر لـ model_copy.
    """
    def _safe_parse(model_cls, data: dict):
        """يُنشئ Pydantic object بشكل آمن — يتجاهل الحقول غير المعروفة."""
        try:
            return model_cls.model_validate(data)
        except Exception as e:
            logger.warning("⚠️  Failed to parse %s from %s: %s", model_cls.__name__, doc_id, e)
            return None

    updates: dict = {}

    # ── case_meta ─────────────────────────────────────────────────────────────
    meta = extracted.get("case_meta", {})
    if meta:
        if state.case_number is None and meta.get("case_number"):
            updates["case_number"] = meta["case_number"]
        if state.court is None and meta.get("court"):
            updates["court"] = meta["court"]
        if state.jurisdiction is None and meta.get("jurisdiction"):
            updates["jurisdiction"] = meta["jurisdiction"]
        if state.prosecutor_name is None and meta.get("prosecutor_name"):
            updates["prosecutor_name"] = meta["prosecutor_name"]
        # court_level و dates: تحتاج تحويل enum/datetime — يعالجهم Pydantic في AgentState

    # ── lists ─────────────────────────────────────────────────────────────────
    field_map = {
        "defendants":         (Defendant,        "defendants"),
        "charges":            (Charge,           "charges"),
        "incidents":          (CaseIncident,     "incidents"),
        "evidences":          (Evidence,         "evidences"),
        "lab_reports":        (LabReport,        "lab_reports"),
        "witness_statements": (WitnessStatement, "witness_statements"),
        "confessions":        (Confession,       "confessions"),
        "procedural_issues":  (ProceduralIssue,  "procedural_issues"),
        "prior_judgments":    (PriorJudgment,    "prior_judgments"),
        "defense_documents":  (DefenseDocument,  "defense_documents"),
    }

    for json_key, (model_cls, state_field) in field_map.items():
        raw_list = extracted.get(json_key, [])
        if not raw_list:
            continue

        existing   = list(getattr(state, state_field))
        new_parsed = []

        for raw_item in raw_list:
            # تأكد من وجود source_document_id
            if "source_document_id" not in raw_item or not raw_item["source_document_id"]:
                raw_item["source_document_id"] = doc_id

            obj = _safe_parse(model_cls, raw_item)
            if obj is not None:
                new_parsed.append(obj)

        if new_parsed:
            updates[state_field] = existing + new_parsed

    return updates


def _now() -> datetime:
    from datetime import timezone
    return datetime.now(timezone.utc)


def _json_safe(obj) -> Any:
    """
    Recursively converts non-JSON-serializable types (datetime, Enum)
    to their string representations so json.dumps never raises TypeError.
    """
    from enum import Enum as _Enum
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, _Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    return obj


def _safe_dump(pydantic_obj) -> dict:
    """model_dump() + _json_safe — always returns a JSON-serialisable dict."""
    return _json_safe(pydantic_obj.model_dump())


def _state_summary(state: AgentState) -> str:
    """Compact JSON snapshot fed into every Agent's system prompt."""
    return json.dumps(
        {
            "case_id":           state.case_id,
            "case_number":       state.case_number,
            "court":             state.court,
            "defendants":        [d.name for d in state.defendants],
            "charges":           [c.statute for c in state.charges],
            "incident_types":    [i.incident_type for i in state.incidents if i.incident_type],
            "evidence_count":    len(state.evidences),
            "witness_count":     len(state.witness_statements),
            "confession_count":  len(state.confessions),
            "procedural_issues": len(state.procedural_issues),
            "completed_agents":  state.completed_agents,
        },
        ensure_ascii=False,
        indent=2,
    )


def _agent_context(state: AgentState, agent_name: str) -> str:
    """Build the approved context window for a specific agent."""
    base = _state_summary(state)

    extra: Dict[str, Any] = {}

    if agent_name == "procedural_auditor":
        extra = {
            "procedural_issues": [_safe_dump(p) for p in state.procedural_issues],
            "confessions":       [_safe_dump(c) for c in state.confessions],
            "evidences":         [_safe_dump(e) for e in state.evidences],
        }
    elif agent_name == "legal_researcher":
        extra = {
            "charges":           [_safe_dump(c) for c in state.charges],
            "incidents":         [_safe_dump(i) for i in state.incidents],
            "prior_judgments":   [_safe_dump(j) for j in state.prior_judgments],
        }
    elif agent_name == "evidence_analyst":
        extra = {
            "evidences":         [_safe_dump(e) for e in state.evidences],
            "lab_reports":       [_safe_dump(r) for r in state.lab_reports],
            "witness_statements":[_safe_dump(w) for w in state.witness_statements],
            "confessions":       [_safe_dump(c) for c in state.confessions],
            "incidents":         [_safe_dump(i) for i in state.incidents],
            "procedural_audit":  state.agent_outputs.get("procedural_auditor", {}),
        }
    elif agent_name == "defense_analyst":
        extra = {
            "defense_documents": [_safe_dump(d) for d in state.defense_documents],
            "procedural_audit":  state.agent_outputs.get("procedural_auditor", {}),
            "evidence_matrix":   state.agent_outputs.get("evidence_analyst", {}),
            "defendants":        [_safe_dump(d) for d in state.defendants],
        }
    elif agent_name == "judge":
        extra = {
            "procedural_audit":  state.agent_outputs.get("procedural_auditor", {}),
            "legal_research":    state.agent_outputs.get("legal_researcher", {}),
            "evidence_matrix":   state.agent_outputs.get("evidence_analyst", {}),
            "defense_analysis":  state.agent_outputs.get("defense_analyst", {}),
            "charges":           [_safe_dump(c) for c in state.charges],
            "defendants":        [_safe_dump(d) for d in state.defendants],
            "incidents":         [_safe_dump(i) for i in state.incidents],
        }

    return json.dumps({"summary": json.loads(base), **extra}, ensure_ascii=False, indent=2)