from .agent_base import AgentBase
from src.LLMs import get_llm_qwn
import json
from langchain_core.messages import HumanMessage, SystemMessage
from src.Prompts.data_ingestion_agent import DATA_INGESTION_AGENT_PROMPT
from src.Graph.graph_helpers import (
    _merge_extracted, _parse_llm_json, _apply_extracted_to_state, _now
)
from pathlib import Path

class DataIngestionAgent(AgentBase):
    def __init__(self):
        super().__init__("DATA_INGESTION_MODEL", "DATA_INGESTION_TEMP", DATA_INGESTION_AGENT_PROMPT)

    def run(self, state):
        all_extracted = {
            "case_meta": {}, "defendants": [], "charges": [], "incidents": [],
            "evidences": [], "lab_reports": [], "witness_statements": [],
            "confessions": [], "procedural_issues": [], "prior_judgments": [],
            "defense_documents": [],
        }
        file_errors, processed_docs = [], []
        for file_path in state.source_documents:
            doc_id = file_path.split("/")[-1]
            path = Path(file_path)
            self.log(f"  📄 Reading: {doc_id}")
            try:
                if path.is_dir():
                    txt_files = sorted(path.glob("*.txt"))
                    if not txt_files:
                        err = f"[{doc_id}] المجلد لا يحتوي على ملفات .txt"
                        self.log(err, "warning"); file_errors.append(err); continue
                    for txt_file in txt_files:
                        try:
                            raw_text = txt_file.read_text(encoding="utf-8")
                        except OSError as e:
                            err = f"[{txt_file.name}] قراءة فاشلة: {e}"
                            self.log(err, "error"); file_errors.append(err); continue
                        if not raw_text.strip():
                            err = f"[{txt_file.name}] الملف فارغ"
                            self.log(err, "warning"); file_errors.append(err); continue
                        self.log(f"  ✅ Read {len(raw_text)} chars from {txt_file.name}")
                        self._extract_file(txt_file.name, raw_text, all_extracted, file_errors, processed_docs)
                    continue
                else:
                    raw_text = path.read_text(encoding="utf-8")
            except OSError as e:
                err = f"[{doc_id}] قراءة فاشلة: {e}"
                self.log(err, "error"); file_errors.append(err); continue
            if not raw_text.strip():
                err = f"[{doc_id}] الملف فارغ"
                self.log(err, "warning"); file_errors.append(err); continue
            self.log(f"  ✅ Read {len(raw_text)} chars from {doc_id}")
            self._extract_file(doc_id, raw_text, all_extracted, file_errors, processed_docs)
        self.log("🔗 Applying extracted entities to state ...")
        state_updates = _apply_extracted_to_state(state, all_extracted, doc_id="__merged__")
        provenance = {
            "processed_at": _now().isoformat(),
            "processed_docs": processed_docs,
            "failed_docs": file_errors,
            "extracted_counts": {field: len(all_extracted.get(field, [])) for field in [
                "defendants", "charges", "incidents", "evidences", "lab_reports",
                "witness_statements", "confessions", "procedural_issues",
                "prior_judgments", "defense_documents",
            ]},
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

    def _extract_file(self, txt_doc_id, raw_text, all_extracted, file_errors, processed_docs):
        prompt = f"اسم الملف: {txt_doc_id}\n\nالنص:\n{raw_text}"
        try:
            response = self._llm_invoke_with_retries(
                get_llm_qwn(self.model_name, self.temperature),
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)]
            )
            result = _parse_llm_json(response.content)
            if not isinstance(result, dict):
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
                    self.log(err, "warning"); file_errors.append(err); return
            all_extracted.update(_merge_extracted(all_extracted, result))
            processed_docs.append(txt_doc_id)
            self.log(f"  ✅ Done: {txt_doc_id}")
        except json.JSONDecodeError as e:
            err = f"[{txt_doc_id}] JSON parsing فاشل — {e}"
            self.log(err, "warning"); file_errors.append(err)
        except Exception as e:
            err = f"[{txt_doc_id}] LLM call فاشل — {e}"
            self.log(err, "error"); file_errors.append(err)
