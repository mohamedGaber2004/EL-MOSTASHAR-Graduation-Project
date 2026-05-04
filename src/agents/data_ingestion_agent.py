import logging, time
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Prompts.data_ingestion_agent import (
    DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
    DATA_INGESTION_AGENT_PROMPT_amr_ihala,
    DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
    DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
    DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
    DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi
)
from src.Graph.graph_helpers import (
    _parse_llm_json,
    _merge_extracted,
    _apply_extracted_to_state,
    _now,
)
from src.Utils.agents_enums import AgentsEnums, LegalDocType

logger = logging.getLogger(__name__)



EMPTY_EXTRACTED = lambda: {
    "case_meta": {
        "case_id":             None,
        "case_number":         None,
        "court":               None,
        "court_level":         None,
        "jurisdiction":        None,
        "filing_date":         None,
        "referral_date":       None,
        "prosecutor_name":     None,
        "referral_order_text": None
    },
    "defendants":         [],
    "charges":            [],
    "incidents":          [],
    "evidences":          [],
    "lab_reports":        [],
    "witness_statements": [],
    "confessions":        [],
    "prior_judgments":    [],
    "defense_documents":  [],
}

def chunk_text(text: str, max_chars: int, overlap: int = 500) -> List[str]:
    """
    Split text into chunks with overlap to avoid cutting sentences.
    
    Args:
        text: The full text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Look for sentence endings within the last 200 chars of the chunk
            search_start = max(start, end - 200)
            sentence_end = text.rfind('.', search_start, end)
            
            if sentence_end == -1:
                # Try other sentence endings
                for punct in ['!', '?', '\n\n']:
                    sentence_end = text.rfind(punct, search_start, end)
                    if sentence_end != -1:
                        break
            
            if sentence_end != -1 and sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
        elif start <= 0:
            start = end
    
    return chunks

class DataIngestionAgent(AgentBase):

    def __init__(self):
        super().__init__(
            "DATA_INGESTION_MODEL",
            "DATA_INGESTION_TEMP",
            [
                DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
                DATA_INGESTION_AGENT_PROMPT_amr_ihala,
                DATA_INGESTION_AGENT_PROMPT_mozakeret_difa,
                DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
                DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
                DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi
            ],
        )
        self.prompt_map = {
            LegalDocType.AMR_IHALA.value:        DATA_INGESTION_AGENT_PROMPT_amr_ihala,
            LegalDocType.MAHDAR_DABT.value:      DATA_INGESTION_AGENT_PROMPT_mahdar_dabt,
            LegalDocType.MAHDAR_ISTIJWAB.value:  DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab,
            LegalDocType.AQWAL_SHUHUD.value:     DATA_INGESTION_AGENT_PROMPT_aqual_shuhud,
            LegalDocType.TAQRIR_TIBBI.value:     DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi,
            LegalDocType.MOZAKARET_DIFA.value:   DATA_INGESTION_AGENT_PROMPT_mozakeret_difa
        }

    # ── file processor ────────────────────────────────────────────────────────

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
        Process a single .txt file and merge its extractions into
        all_extracted. Always returns a dict (never None).
        """
        raw_text: str | None = None

        for enc in AgentsEnums.AGENT_INGESTION_TXT_FILES_ENCODING:
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

        # Chunk the text if it's too long
        if len(raw_text) > self.cfg.INGESTION_AGENT_MAX_CHARS:
            logger.info(
                "File [%s] chunked: %d chars → %d chunks",
                doc_id, len(raw_text), 
                len(chunk_text(raw_text, self.cfg.INGESTION_AGENT_MAX_CHARS))
            )
            text_chunks = chunk_text(raw_text, self.cfg.INGESTION_AGENT_MAX_CHARS)
        else:
            text_chunks = [raw_text]

        # Process each chunk and merge results
        chunk_results = []
        for i, chunk in enumerate(text_chunks):
            try:
                chunk_id = f"{doc_id}_chunk_{i+1}" if len(text_chunks) > 1 else doc_id
                
                response = self._llm_invoke_with_retries(
                    self._llm,
                    [SystemMessage(content=prompt), HumanMessage(content=f"The text: {chunk}")]
                )
                result = _parse_llm_json(response.content)

                if not isinstance(result, dict):
                    raise ValueError(f"non-dict result: {type(result)}")

                # Stamp source_document_id on each item
                for key, value_list in result.items():
                    if isinstance(value_list, list):
                        for item in value_list:
                            if isinstance(item, dict) and not item.get("source_document_id"):
                                item["source_document_id"] = chunk_id

                chunk_results.append(result)

            except Exception as e:
                logger.error("Failed to process chunk %d of [%s]: %s", i+1, doc_id, e)
                file_errors.append(f"[{doc_id}] فشل معالجة الجزء {i+1}: {e}")
                continue

        # Merge all chunk results
        merged_result = all_extracted
        for chunk_result in chunk_results:
            merged_result = _merge_extracted(merged_result, chunk_result)

        if chunk_results:  # Only mark as processed if at least one chunk succeeded
            processed_docs.append(doc_id)

        return merged_result

    # ── router ────────────────────────────────────────────────────────────────

    def _route_file(self, stem: str) -> str | None:
        """
        Route file stem to the correct prompt using enum-based matching.
        
        Args:
            stem: File stem (name without extension)
            
        Returns:
            Prompt string or None if no match found
        """
        stem_lower = stem.lower()
        
        # Exact match first
        if stem_lower in self.prompt_map:
            return self.prompt_map[stem_lower]
        
        # Partial match as fallback
        for key in self.prompt_map:
            if key in stem_lower:
                return self.prompt_map[key]
        
        return None

    def _route_and_process(
        self,
        path: Path,
        doc_id: str,
        all_extracted: dict,
        file_errors: list,
        processed_docs: list,
    ) -> dict:
        """
        Match file stem to the correct prompt, then process.
        Always returns the (possibly updated) all_extracted dict.
        """
        selected_prompt = self._route_file(doc_id)

        if not selected_prompt:
            logger.warning("No prompt mapping found for [%s]. Skipping.", doc_id)
            file_errors.append(f"[{doc_id}] لا يوجد Prompt مخصص لهذا الملف")
            return all_extracted

        result = self._process_file(
            path, doc_id, all_extracted, file_errors, processed_docs, selected_prompt
        )

        time.sleep(self.cfg.INTER_AGENTS_DELAY)
        return result

    # ── main entry ────────────────────────────────────────────────────────────

    def run(self, state):
        logger.info("DataIngestionAgent starting...")

        if not state.source_documents:
            logger.warning("No source documents provided")
            return state.model_copy(
                update={
                    "completed_agents": state.completed_agents + ["data_ingestion"],
                    "current_agent":    "data_ingestion",
                    "last_updated":     _now(),
                }
            )

        all_extracted   = EMPTY_EXTRACTED()
        file_errors:    list = []
        processed_docs: list = []

        for file_path in state.source_documents:
            path = Path(file_path)

            if path.is_dir():
                for txt_file in sorted(path.glob("*.txt")):
                    all_extracted = self._route_and_process(  # ← مسك الـ return
                        txt_file,
                        txt_file.stem,          # ← stem بدل name (بدون .txt)
                        all_extracted,
                        file_errors,
                        processed_docs,
                    )

            elif path.suffix.lower() == ".txt":
                all_extracted = self._route_and_process(      # ← مسك الـ return
                    path,
                    path.stem,                  # ← stem بدل name
                    all_extracted,
                    file_errors,
                    processed_docs,
                )

            else:
                logger.warning("Unsupported file type [%s]: %s", path.name, path.suffix)
                file_errors.append(f"[{path.name}] نوع ملف غير مدعوم: {path.suffix}")

        state_updates = _apply_extracted_to_state(state, all_extracted)

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
                "completed_agents": state.completed_agents + ["data_ingestion"],
                "current_agent":    "data_ingestion",
                "errors":           state.errors + file_errors,
                "last_updated":     _now(),
            }
        )