import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Prompts.data_ingestion_agent import DATA_INGESTION_AGENT_PROMPT
from src.Graph.graph_helpers import (
    _parse_llm_json,
    _merge_extracted,
    _apply_extracted_to_state,
    _now,
)
from src.Utils.agents_enums import AgentsEnums

logger = logging.getLogger(__name__)


EMPTY_EXTRACTED = lambda: {
    "case_meta": {
        "case_number":     None,
        "court":           None,
        "court_level":     None,
        "jurisdiction":    None,
        "filing_date":     None,
        "referral_date":   None,
        "prosecutor_name": None,
    },
    "defendants":         [],
    "charges":            [],
    "incidents":          [],
    "evidences":          [],
    "lab_reports":        [],
    "witness_statements": [],
    "confessions":        [],
    "procedural_issues":  [],
    "prior_judgments":    [],
    "defense_documents":  [],
}


class DataIngestionAgent(AgentBase):

    def __init__(self):
        super().__init__(
            "DATA_INGESTION_MODEL",
            "DATA_INGESTION_TEMP",
            DATA_INGESTION_AGENT_PROMPT,
            llm_provider="llama",
        )

    # ── file processor ────────────────────────────────────────────────

    def _process_file(
        self,
        path: Path,
        doc_id: str,
        all_extracted: dict,
        file_errors: list,
        processed_docs: list,
    ) -> dict:
        """
        Process a single .txt file and merge its extractions into
        all_extracted.  Always returns a dict (never None).
        """
        raw_text: str | None = None

        for enc in AgentsEnums.AGENT_INGESTION_TXT_FILES_ENCODING.value:
            try:
                raw_text = path.read_text(encoding=enc).strip()
                break
            except (UnicodeDecodeError, OSError):
                continue

        if raw_text is None:
            logger.warning("Failed to decode [%s] with any supported encoding", doc_id)
            file_errors.append(f"[{doc_id}] فشل قراءة بأي ترميز")
            return all_extracted

        if not raw_text:
            logger.warning("File [%s] is empty", doc_id)
            file_errors.append(f"[{doc_id}] الملف فارغ")
            return all_extracted

        if len(raw_text) > self.cfg.INGESTION_AGENT_MAX_CHARS:
            logger.warning(
                "File [%s] truncated from %d → %d chars",
                doc_id, len(raw_text), self.cfg.INGESTION_AGENT_MAX_CHARS,
            )
            raw_text = raw_text[: self.cfg.INGESTION_AGENT_MAX_CHARS]

        try:
            response = self._llm_invoke_with_retries(
                self._llm,
                [
                    SystemMessage(content=self.prompt),
                    HumanMessage(content=f"اسم الملف: {doc_id}\n\nالنص:\n{raw_text}"),
                ],
            )
            result = _parse_llm_json(response.content)
            if not isinstance(result, dict):
                raise ValueError(f"non-dict result: {type(result)}")

            merged = _merge_extracted(all_extracted, result)
            processed_docs.append(doc_id)
            logger.debug("Processed [%s] successfully", doc_id)
            return merged

        except Exception as e:
            logger.error("Failed to process [%s]: %s", doc_id, e)
            file_errors.append(f"[{doc_id}] فشل: {e}")
            return all_extracted

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        logger.info("DataIngestionAgent starting...")

        if not state.source_documents:
            logger.warning("No source documents provided")
            return state.model_copy(
                update={
                    "agent_outputs": {
                        **state.agent_outputs,
                        "data_ingestion": {"graph_status": "no_documents_provided"},
                    },
                    "completed_agents": state.completed_agents + ["data_ingestion"],
                    "current_agent":    "data_ingestion",
                    "last_updated":     _now(),
                }
            )

        all_extracted   = EMPTY_EXTRACTED()
        file_errors:    list = []
        processed_docs: list = []
        doc_id = "__unknown__"

        for file_path in state.source_documents:
            path   = Path(file_path)
            doc_id = path.name

            if path.is_dir():
                for txt_file in sorted(path.glob("*.txt")):
                    all_extracted = self._process_file(
                        txt_file, txt_file.name,
                        all_extracted, file_errors, processed_docs,
                    )
            elif path.suffix.lower() != ".txt":
                logger.warning(
                    "Unsupported file type [%s]: %s", doc_id, path.suffix
                )
                file_errors.append(
                    f"[{doc_id}] نوع ملف غير مدعوم: {path.suffix}"
                )
            else:
                all_extracted = self._process_file(
                    path, doc_id,
                    all_extracted, file_errors, processed_docs,
                )

        state_updates = _apply_extracted_to_state(
            state, all_extracted, doc_id="__merged__"
        )

        provenance = {
            "processed_at":   _now().isoformat(),
            "processed_docs": processed_docs,
            "failed_docs":    file_errors,
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
            "Ingestion complete — processed=%d  failed=%d",
            len(processed_docs), len(file_errors),
        )
        logger.debug("Extraction stats: %s", provenance["extracted_counts"])

        return state.model_copy(
            update={
                **state_updates,
                "agent_outputs": {
                    **state.agent_outputs,
                    "data_ingestion": provenance,
                },
                "completed_agents": state.completed_agents + ["data_ingestion"],
                "current_agent":    "data_ingestion",
                "errors":           state.errors + file_errors,
                "last_updated":     _now(),
            }
        )