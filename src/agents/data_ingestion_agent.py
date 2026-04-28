from pathlib import Path
import logging
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from .agent_base import AgentBase
from src.Prompts.data_ingestion_agent import DATA_INGESTION_AGENT_PROMPT
from src.Graph.graph_helpers import _parse_llm_json , _merge_extracted , _apply_extracted_to_state  , _now
from src.Utils.agents_enums import AgentsEnums

EMPTY_EXTRACTED = lambda: {
    "case_meta": {}, "defendants": [], "charges": [], "incidents": [],
    "evidences": [], "lab_reports": [], "witness_statements": [],
    "confessions": [], "procedural_issues": [], "prior_judgments": [],
    "defense_documents": [],
}

class DataIngestionAgent(AgentBase):
    def __init__(self):
        super().__init__("DATA_INGESTION_MODEL", "DATA_INGESTION_TEMP", DATA_INGESTION_AGENT_PROMPT)

    def _process_file(self, path, doc_id, all_extracted, file_errors, processed_docs):
        for enc in AgentsEnums.AGENT_INGESTION_TXT_FILES_ENCODING.value:
            try:
                raw_text = path.read_text(encoding=enc).strip()
                break
            except (UnicodeDecodeError, OSError):
                continue
        else:
            logger.warning("Failed to decode file [%s] with any supported encoding", doc_id)
            file_errors.append(f"[{doc_id}] فشل قراءة بأي ترميز")
            return

        if not raw_text:
            logger.warning("File [%s] is empty", doc_id)
            file_errors.append(f"[{doc_id}] الملف فارغ")
            return
        
        if len(raw_text) > self.cfg.INGESTION_AGENT_MAX_CHARS:
            raw_text = raw_text[:self.cfg.INGESTION_AGENT_MAX_CHARS]

        try:
            response = self._llm_invoke_with_retries(
                self.oss_llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=f"اسم الملف: {doc_id}\n\nالنص:\n{raw_text}")]
            )
            result = _parse_llm_json(response.content)
            if not isinstance(result, dict):
                raise ValueError(f"non-dict result: {type(result)}")
            all_extracted.update(_merge_extracted(all_extracted, result))
            processed_docs.append(doc_id)
            logger.debug("Successfully processed file [%s]", doc_id)
        except Exception as e:
            logger.error("Failed to process file [%s]: %s", doc_id, e)
            file_errors.append(f"[{doc_id}] فشل: {e}")

    def run(self, state):
        logger.info("Starting data ingestion...")
        all_extracted = EMPTY_EXTRACTED()
        file_errors, processed_docs = [], []

        for file_path in state.source_documents:
            path = Path(file_path)
            doc_id = path.name

            if path.is_dir():
                for txt_file in sorted(path.glob("*.txt")):
                    self._process_file(txt_file, txt_file.name, all_extracted, file_errors, processed_docs)
            else:
                if path.suffix.lower() != ".txt":
                    logger.warning("Unsupported file type [%s]: %s", doc_id, path.suffix)
                    file_errors.append(f"[{doc_id}] نوع ملف غير مدعوم: {path.suffix}")
                    continue
                self._process_file(path, doc_id, all_extracted, file_errors, processed_docs)

        state_updates = _apply_extracted_to_state(state, all_extracted, doc_id="__merged__")

        provenance = {
            "processed_at": _now().isoformat(),
            "processed_docs": processed_docs,
            "failed_docs": file_errors,
            "extracted_counts": {k: len(v) for k, v in all_extracted.items() if isinstance(v, list)},
            "graph_status": "entities_extracted" if processed_docs else "no_documents_processed",
        }
        logger.info("Data ingestion complete. Processed %d docs, Failed %d docs", len(processed_docs), len(file_errors))
        logger.debug("Extraction stats: %s", provenance["extracted_counts"])

        return state.model_copy(update={
            **state_updates,
            "agent_outputs": {**state.agent_outputs, "data_ingestion": provenance},
            "completed_agents": state.completed_agents + ["data_ingestion"],
            "current_agent": "data_ingestion",
            "errors": state.errors + file_errors,
            "last_updated": _now(),
        })
