from __future__ import annotations

import logging
import time , json
from pathlib import Path
from typing import List, Optional, Any

from ..agent_base import AgentBase
from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.data_ingestion_agent.di_enums import DI_enums, DI_LegalDocType
from src.agents.data_ingestion_agent.data_ingestion_prompt import (
    DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
    DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
    DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
    DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi,
    DATA_INGESTION_AGENT_PROMPT_amr_ihala,
    DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
    DATA_INGESTION_AGENT_PROMPT_sawabiq,
)


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
        "criminal_proceedings":      [],
        "defense_documents":         [],
    }

# ─────────────────────────────────────────────────────────────────
#  Routing: keyword sets per document type
# ─────────────────────────────────────────────────────────────────
def _route_stem(stem: str) -> Optional[str]:
    """
    Map a file stem to a DI_LegalDocType value using keyword matching.

    Strategy (in order):
    1. Exact match against the enum value.
    2. Stem contains the enum value as a substring.
    3. Stem shares at least one keyword with the pattern set.

    Returns the DI_LegalDocType value string, or None if no match.
    """
    stem_lower = stem.lower().replace("-", "_").replace(" ", "_")

    for doc_type, keywords in DI_enums._ROUTE_PATTERNS.value:
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
def chunk_text(text: str,max_chars: int,overlap: int = 500,doc_id: str = "") -> List[tuple[int, str]]:
    """Split text into (chunk_index, chunk_text) pairs with overlap."""

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
        return False, "The result is not Dict type"

    for key in DI_enums._LIST_KEYS.value:
        val = result.get(key)
        if val is not None and not isinstance(val, list):
            return False, f"the field '{key}' must to be list but it's {type(val).__name__}"

    cm = result.get("case_meta")
    if cm is not None and not isinstance(cm, dict):
        return False, "case_meta must be Dict type"

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
            DI_LegalDocType.AMR_IHALA.value:       DATA_INGESTION_AGENT_PROMPT_amr_ihala,
            DI_LegalDocType.MAHDAR_DABT.value:     DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
            DI_LegalDocType.MAHDAR_ISTIJWAB.value: DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
            DI_LegalDocType.AQWAL_SHUHUD.value:    DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
            DI_LegalDocType.TAQRIR_TIBBI.value:    DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi,
            DI_LegalDocType.MOZAKARET_DIFA.value:  DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
            DI_LegalDocType.SAWABIQ.value:         DATA_INGESTION_AGENT_PROMPT_sawabiq,
        }

    # Dedup helpers
    def _dedup_by_keys(self,existing: list[dict], new_items: list[dict], *keys: str) -> list[dict]:
        """
        Append items from new_items that don't already exist in existing,
        using `keys` as the identity fields for deduplication.

        If none of the keys are present in an item, the item is always appended.
        """
        def fingerprint(item: dict) -> tuple:
            return tuple(
                json.dumps(item.get(k), ensure_ascii=False) if isinstance(item.get(k), (dict, list))
                else item.get(k)
                for k in keys
            )

        seen: set[tuple] = set()
        for item in existing:
            fp = fingerprint(item)
            if any(v is not None for v in fp):
                seen.add(fp)

        result = list(existing)
        for item in new_items:
            if not isinstance(item, dict):
                continue
            fp = fingerprint(item)
            has_key = any(v is not None for v in fp)
            if has_key and fp in seen:
                continue
            result.append(item)
            seen.add(fp)

        return result

    def _merge_list(self,existing: list[dict], incoming: list[dict], *dedup_keys: str) -> list[dict]:
        if not incoming:
            return existing
        if dedup_keys:
            return self._dedup_by_keys(existing, incoming, *dedup_keys)
        return existing + incoming

    # _merge_extracted
    def _merge_extracted(self, base: dict, incoming: dict) -> dict:
        """
        Merge `incoming` extraction result into `base`.
        List fields are appended with deduplication where meaningful keys exist.
        Scalar fields (case_meta) are merged with first-write-wins.
        """
        merged = dict(base)

        # ── case_meta: first-write-wins per sub-key ───────────────────────────────
        if "case_meta" in incoming and isinstance(incoming["case_meta"], dict):
            base_meta = merged.get("case_meta") or {}
            for k, v in incoming["case_meta"].items():
                if v is not None and base_meta.get(k) is None:
                    base_meta[k] = v
            merged["case_meta"] = base_meta

        # ── lists: merge with dedup ───────────────────────────────────────────────
        for key, dedup_keys in DI_enums.LIST_DEDUP.value.items():
            if key in incoming:
                val = incoming[key]
                if isinstance(val, dict):
                    val = [val]
                if isinstance(val, list):
                    merged[key] = self._merge_list(
                        merged.get(key, []),
                        val,
                        *dedup_keys,
                    )

        return merged

    # _apply_extracted_to_state
    def _apply_extracted_to_state(self, state: Any, extracted: dict) -> dict:
        """
        Convert the raw `extracted` dict (from _merge_extracted) to the
        keyword arguments needed for state.model_copy(update={...}).

        Pydantic models inside lists are constructed here so AgentState
        receives typed objects rather than raw dicts.
        """

        def safe_make(cls, data: dict) -> Any:
            try:
                return cls(**{k: v for k, v in data.items() if v is not None})
            except Exception as exc:
                logger.warning("Could not construct %s from %s: %s", cls.__name__, data, exc)
                return None

        def make_list(cls, raw: list[dict]) -> list:
            return [obj for d in raw if isinstance(d, dict) if (obj := safe_make(cls, d)) is not None]

        updates: dict = {}

        # ── case_meta scalars ─────────────────────────────────────────────────────
        meta = extracted.get("case_meta") or {}
        for field in (DI_enums.CASE_META_ATTRIBUTES.value):
            val = meta.get(field)
            if val is not None:
                updates[field] = val

        # ── entity lists ──────────────────────────────────────────────────────────

        for raw_key, (cls, state_field) in DI_enums.ENTITIES_MAPPING.value.items():
            raw_list = extracted.get(raw_key, [])
            if raw_list:
                existing = getattr(state, state_field, [])
                new_objs = make_list(cls, raw_list)
                updates[state_field] = existing + new_objs

        return updates

    # ── routing ───────────────────────────────────────────────────
    def _get_prompt(self, stem: str) -> Optional[str]:
        """Resolve file stem → prompt string. Returns None if unrecognised."""
        doc_type = _route_stem(stem)
        if doc_type is None:
            return None
        return self._prompt_map.get(doc_type)

    # ── single file processor ─────────────────────────────────────
    def _process_file(self,path: Path,doc_id: str,all_extracted: dict,file_errors: list,processed_docs: list,prompt: str) -> dict:
        """
        Read, chunk, invoke LLM, validate, and merge a single .txt file.
        Always returns the (possibly updated) all_extracted dict.
        """
        # ── read file ────────────────────────────────────────────
        raw_text: Optional[str] = None
        for enc in DI_enums.AGENT_INGESTION_TXT_FILES_ENCODING.value:
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
    def _route_and_process(self,path: Path,doc_id: str,all_extracted: dict,file_errors: list,processed_docs: list) -> dict:
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

        return self._empty_update(state, "data_ingestion", {
            **state_updates,
            "errors": file_errors,
        })