from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Prompts.data_ingestion_agent import (
    DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
    DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
    DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
    DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi,
    DATA_INGESTION_AGENT_PROMPT_amr_ihala,
    DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
    DATA_INGESTION_AGENT_PROMPT_sawabiq,
)

from src.Utils.Enums.agents_enums import AgentsEnums, LegalDocType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Empty state template
# ─────────────────────────────────────────────────────────────────

def _empty_extracted() -> dict:
    return {
        "case_meta": {
            "case_number":         None,
            "court":               None,
            "court_level":         None,
            "jurisdiction":        None,
            "filing_date":         None,
            "referral_date":       None,
            "prosecutor_name":     None,
            "referral_order_text": None,
        },
        "defendants":                [],
        "charges":                   [],
        "incidents":                 [],
        "evidences":                 [],
        "lab_reports":               [],
        "witness_statements":        [],
        "confessions":               [],
        "criminal_records":          [],
        "defense_documents":         [],
        "defense_procedural_issues": [],
        "procedural_issues":         [],
    }


# Keys that hold lists and should be merged (not overwritten)
_LIST_KEYS = {
    "defendants", "charges", "incidents", "evidences",
    "lab_reports", "witness_statements", "confessions",
    "criminal_records", "defense_documents",
    "defense_procedural_issues", "procedural_issues",
}

# ─────────────────────────────────────────────────────────────────
#  Routing: keyword sets per document type
#  More specific patterns first
# ─────────────────────────────────────────────────────────────────
_ROUTE_PATTERNS: list[tuple[str, set[str]]] = [
    (
        LegalDocType.AMR_IHALA.value,
        {"amr_ihala", "امر_الاحالة", "أمر_الإحالة", "ihala", "referral",
         "amr", "احالة", "إحالة"},
    ),
    (
        LegalDocType.MAHDAR_DABT.value,
        {"mahdar_dabt", "محضر_الضبط", "mahdar_qesm", "قسم_الشرطة", "dabt",
         "ضبط", "تفتيش", "استدلالات", "jame3", "jame", "استدلال", "mahdar_jame"},
    ),
    (
        LegalDocType.MAHDAR_ISTIJWAB.value,
        {"mahdar_istijwab", "محضر_الاستجواب", "istijwab", "استجواب",
         "tahqiq", "تحقيق", "niyaba", "نيابة"},
    ),
    (
        LegalDocType.AQWAL_SHUHUD.value,
        {"aqwal_shuhud", "أقوال_الشهود", "shuhud", "شهود", "شاهد",
         "aqwal", "شهادة"},
    ),
    (
        LegalDocType.TAQRIR_TIBBI.value,
        {"taqrir_tibbi", "تقرير_طبي", "tibbi", "طبي", "شرعي", "معمل",
         "lab", "taqrir", "تقرير", "forensic", "kimawi", "كيميائي",
         "balisti", "بلستي"},
    ),
    (
        LegalDocType.MOZAKARET_DIFA.value,
        {"mozakeret_difa", "مذكرة_الدفاع", "difa", "دفاع", "mozakera",
         "مذكرة", "defense"},
    ),
    (
        LegalDocType.SAWABIQ.value,
        {"sawabiq", "سوابق", "صحيفة_جنائية", "سجل_جنائي", "criminal_record",
         "sahifa", "صحيفة"},
    ),
]


def _route_stem(stem: str) -> Optional[str]:
    """
    Map a file stem to a LegalDocType value using keyword matching.

    Strategy (in order):
    1. Exact match against the enum value.
    2. Stem contains the enum value as a substring.
    3. Stem shares at least one keyword with the pattern set.

    Returns the LegalDocType value string, or None if no match.
    """
    stem_lower = stem.lower().replace("-", "_").replace(" ", "_")

    for doc_type, keywords in _ROUTE_PATTERNS:
        if stem_lower == doc_type:
            return doc_type
        if doc_type in stem_lower:
            return doc_type
        for kw in keywords:
            if kw in stem_lower:
                return doc_type

    return None


# ─────────────────────────────────────────────────────────────────
#  Text chunking with document-context header
# ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_chars: int,
    overlap: int = 500,
    doc_id: str = "",
) -> List[tuple[int, str]]:
    """
    Split text into (chunk_index, chunk_text) pairs with overlap.

    Each chunk after the first gets an Arabic context header so the model
    knows it is reading a continuation of the same document.
    """
    if len(text) <= max_chars:
        return [(1, text)]

    raw_chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end < len(text):
            search_start = max(start, end - 300)
            for punct in [".\n", ".\r\n", ". ", "!\n", "?\n", "\n\n"]:
                idx = text.rfind(punct, search_start, end)
                if idx != -1:
                    end = idx + len(punct)
                    break

        chunk = text[start:end].strip()
        if chunk:
            raw_chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break
        if start <= 0:
            start = end

    n = len(raw_chunks)
    result: list[tuple[int, str]] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            result.append((1, chunk))
        else:
            header = (
                f"[ملاحظة: هذا النص جزء ({i + 1}/{n}) من المستند '{doc_id}'. "
                f"استمر في الاستخراج من حيث توقف الجزء السابق. "
                f"لا تعيد استخراج ما سبق إلا إن ظهرت معلومة جديدة.]\n\n"
            )
            result.append((i + 1, header + chunk))

    return result


# ─────────────────────────────────────────────────────────────────
#  Schema validation before merging
# ─────────────────────────────────────────────────────────────────

def _validate_extracted(result: dict) -> tuple[bool, str]:
    """
    Sanity check on LLM output before it enters the state.
    Returns (is_valid, reason_if_invalid).

    An empty dict {} is valid — it means the document had nothing to extract
    for this doc-type (e.g. a mahdar_istijwab with no confessions found yet).
    We only reject results that are not dicts at all, or have wrong field types.
    """
    if not isinstance(result, dict):
        return False, "النتيجة ليست قاموساً"

    for key in _LIST_KEYS:
        val = result.get(key)
        if val is not None and not isinstance(val, list):
            return False, f"الحقل '{key}' يجب أن يكون قائمة لكنه {type(val).__name__}"

    cm = result.get("case_meta")
    if cm is not None and not isinstance(cm, dict):
        return False, "case_meta يجب أن يكون قاموساً"

    return True, ""


# ─────────────────────────────────────────────────────────────────
#  DataIngestionAgent
# ─────────────────────────────────────────────────────────────────

class DataIngestionAgent(AgentBase):

    def __init__(self):
        super().__init__(
            "DATA_INGESTION_MODEL",
            "DATA_INGESTION_TEMP",
            [
                DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
                DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
                DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
                DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi,
                DATA_INGESTION_AGENT_PROMPT_amr_ihala,
                DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
                DATA_INGESTION_AGENT_PROMPT_sawabiq,
            ],
        )
        self._prompt_map: dict[str, str] = {
            LegalDocType.AMR_IHALA.value:       DATA_INGESTION_AGENT_PROMPT_amr_ihala,
            LegalDocType.MAHDAR_DABT.value:     DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
            LegalDocType.MAHDAR_ISTIJWAB.value: DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
            LegalDocType.AQWAL_SHUHUD.value:    DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
            LegalDocType.TAQRIR_TIBBI.value:    DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi,
            LegalDocType.MOZAKARET_DIFA.value:  DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
            LegalDocType.SAWABIQ.value:         DATA_INGESTION_AGENT_PROMPT_sawabiq,
        }

    # ── routing ───────────────────────────────────────────────────

    def _get_prompt(self, stem: str) -> Optional[str]:
        """Resolve file stem → prompt string. Returns None if unrecognised."""
        doc_type = _route_stem(stem)
        if doc_type is None:
            return None
        return self._prompt_map.get(doc_type)

    # ── single file processor ─────────────────────────────────────

    def _process_file(
        self,
        path: Path,
        doc_id: str,
        all_extracted: dict,
        file_errors: list,
        processed_docs: list,
        prompt: str,
    ) -> dict:
        """
        Read, chunk, invoke LLM, validate, and merge a single .txt file.
        Always returns the (possibly updated) all_extracted dict.
        """
        # ── read file ────────────────────────────────────────────
        raw_text: Optional[str] = None
        for enc in AgentsEnums.AGENT_INGESTION_TXT_FILES_ENCODING:
            try:
                raw_text = path.read_text(encoding=enc).strip()
                break
            except (UnicodeDecodeError, OSError):
                continue

        if not raw_text:
            msg = f"[{doc_id}] فشل قراءة الملف أو الملف فارغ"
            logger.warning(msg)
            file_errors.append(msg)
            return all_extracted

        # ── chunk ────────────────────────────────────────────────
        max_chars = self.cfg.INGESTION_AGENT_MAX_CHARS
        chunks = chunk_text(raw_text, max_chars, doc_id=doc_id)
        total = len(chunks)
        if total > 1:
            logger.info("[%s] split into %d chunks (%d chars)", doc_id, total, len(raw_text))

        # ── process each chunk ───────────────────────────────────
        any_success = False
        for chunk_idx, chunk_content in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk_idx}" if total > 1 else doc_id
            try:
                response = self._llm_invoke_with_retries(
                    self._llm,
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"النص:\n\n{chunk_content}"),
                    ],
                )
                result = self._parse_llm_json(response.content)

                is_valid, reason = _validate_extracted(result)
                if not is_valid:
                    raise ValueError(f"فشل التحقق: {reason}")

                # Stamp source_document_id on every extracted item
                for key, value_list in result.items():
                    if isinstance(value_list, list):
                        for item in value_list:
                            if isinstance(item, dict) and not item.get("source_document_id"):
                                item["source_document_id"] = chunk_id

                all_extracted = self._merge_extracted(all_extracted, result)
                any_success = True

            except Exception as exc:
                msg = f"[{chunk_id}] فشل المعالجة: {exc}"
                logger.error(msg)
                file_errors.append(msg)

        if any_success:
            processed_docs.append(doc_id)

        return all_extracted

    # ── route + process ───────────────────────────────────────────

    def _route_and_process(
        self,
        path: Path,
        doc_id: str,
        all_extracted: dict,
        file_errors: list,
        processed_docs: list,
    ) -> dict:
        """Resolve prompt for doc_id, then delegate to _process_file."""
        prompt = self._get_prompt(doc_id)

        if prompt is None:
            msg = f"[{doc_id}] نوع المستند غير معروف — تم التخطي"
            logger.warning(msg)
            file_errors.append(msg)
            return all_extracted

        result = self._process_file(
            path, doc_id, all_extracted, file_errors, processed_docs, prompt
        )
        time.sleep(self.cfg.INTER_AGENTS_DELAY)
        return result

    # ── main entry ────────────────────────────────────────────────

    def run(self, state):
        logger.info(
            "DataIngestionAgent starting — %d source(s)",
            len(state.source_documents or []),
        )

        if not state.source_documents:
            logger.warning("No source documents provided")
            return state.model_copy(
                update={
                    "completed_agents": state.completed_agents + ["data_ingestion"],
                    "current_agent":    "data_ingestion",
                    "last_updated":     self._now(),
                }
            )

        all_extracted:  dict = _empty_extracted()
        file_errors:    list = []
        processed_docs: list = []

        for file_path in state.source_documents:
            path = Path(file_path)

            if path.is_dir():
                txt_files = sorted(path.glob("*.txt"))
                if not txt_files:
                    logger.warning("Directory [%s] has no .txt files", path)
                for txt_file in txt_files:
                    all_extracted = self._route_and_process(
                        txt_file, txt_file.stem,
                        all_extracted, file_errors, processed_docs,
                    )

            elif path.suffix.lower() == ".txt":
                all_extracted = self._route_and_process(
                    path, path.stem,
                    all_extracted, file_errors, processed_docs,
                )

            else:
                msg = f"[{path.name}] نوع ملف غير مدعوم: {path.suffix}"
                logger.warning(msg)
                file_errors.append(msg)

        # ── apply to state ────────────────────────────────────────
        state_updates = self._apply_extracted_to_state(state, all_extracted)

        provenance = {
            "processed_at":     self._now().isoformat(),
            "processed_docs":   processed_docs,
            "failed_docs":      file_errors,
            "extracted_counts": {
                k: len(v)
                for k, v in all_extracted.items()
                if isinstance(v, list)
            },
            "graph_status": (
                "entities_extracted" if processed_docs else "no_documents_processed"
            ),
        }

        logger.info(
            "Ingestion complete — processed=%d  failed=%d  counts=%s",
            len(processed_docs),
            len(file_errors),
            provenance["extracted_counts"],
        )

        return state.model_copy(
            update={
                **state_updates,
                "completed_agents": state.completed_agents + ["data_ingestion"],
                "current_agent":    "data_ingestion",
                "errors":           state.errors + file_errors,
                "last_updated":     self._now(),
            }
        )