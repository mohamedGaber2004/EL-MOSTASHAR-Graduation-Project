import json
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from .agent_base import AgentBase
from src.LLMs import  get_llm_oss
from src.Graph.graph_helpers import ( _parse_llm_json,
    _apply_extracted_to_state, _now,_merge_extracted
)
from src.Prompts import (
    DATA_INGESTION_AGENT_PROMPT
)

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
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            file_errors.append(f"[{doc_id}] قراءة فاشلة: {e}")
            return

        if not raw_text:
            file_errors.append(f"[{doc_id}] الملف فارغ")
            return

        self.log(f"📄 {doc_id} ({len(raw_text)} chars)")

        try:
            response = self._llm_invoke_with_retries(
                get_llm_oss(self.model_name, self.temperature),
                [SystemMessage(content=self.prompt), HumanMessage(content=f"اسم الملف: {doc_id}\n\nالنص:\n{raw_text}")]
            )
            result = _parse_llm_json(response.content)
            if not isinstance(result, dict):
                raise ValueError(f"non-dict result: {type(result)}")
            all_extracted.update(_merge_extracted(all_extracted, result))
            processed_docs.append(doc_id)
            self.log(f"✅ {doc_id}")
        except Exception as e:
            file_errors.append(f"[{doc_id}] فشل: {e}")
            self.log(f"❌ {doc_id}: {e}", "error")


    def run(self, state):
        all_extracted = EMPTY_EXTRACTED()
        file_errors, processed_docs = [], []

        for file_path in state.source_documents:
            path = Path(file_path)
            doc_id = path.name

            if path.is_dir():
                for txt_file in sorted(path.glob("*.txt")):
                    self._process_file(txt_file, txt_file.name, all_extracted, file_errors, processed_docs)
            else:
                self._process_file(path, doc_id, all_extracted, file_errors, processed_docs)

        state_updates = _apply_extracted_to_state(state, all_extracted, doc_id="__merged__")

        provenance = {
            "processed_at": _now().isoformat(),
            "processed_docs": processed_docs,
            "failed_docs": file_errors,
            "extracted_counts": {k: len(v) for k, v in all_extracted.items() if isinstance(v, list)},
            "graph_status": "entities_extracted" if processed_docs else "no_documents_processed",
        }

        return state.model_copy(update={
            **state_updates,
            "agent_outputs": {**state.agent_outputs, "data_ingestion": provenance},
            "completed_agents": state.completed_agents + ["data_ingestion"],
            "current_agent": "data_ingestion",
            "errors": state.errors + file_errors,
            "last_updated": _now(),
        })
