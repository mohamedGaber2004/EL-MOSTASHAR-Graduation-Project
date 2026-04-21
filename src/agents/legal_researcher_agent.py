import json
from pathlib import Path
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
 
from .agent_base import AgentBase
from src.LLMs import get_llm_qwn
from src.Graph.graph_helpers import (
    _retrieve_for_charge, _build_fallback_package, _parse_llm_json
)
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Vectorstore.vector_store_builder import load_vector_store
from src.Prompts import (
    LEGAL_RESEARCHER_AGENT_PROMPT
)
 
 
# ─────────────────────────────────────────────
# Legal Researcher Agent
# ─────────────────────────────────────────────
class LegalResearcherAgent(AgentBase):
    def __init__(self):
        super().__init__("LEGAL_RESEARCHER_MODEL", "LEGAL_RESEARCHER_TEMP", LEGAL_RESEARCHER_AGENT_PROMPT)
 
    def run(self, state):
        self.log(f"📚 الباحث القانوني — {len(state.charges)} تهمة")
        if not state.charges:
            return self._empty_update(state, "legal_researcher", {
                "research_packages": [], "procedural_principles": [], "relevant_cassation_rulings": [],
            })
 
        cfg = self.cfg
        embeddings = HuggingFaceEmbeddings(model_name=getattr(cfg, "EMBEDDING_MODEL", None))
        vs = load_vector_store(str(Path(getattr(cfg, "FAISS_INDEX_PATH", "legal_faiss_index"))), embeddings)
 
        kg = None
        if getattr(cfg, "NEO4J_URI", None):
            try:
                kg = LegalKnowledgeGraph(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD)
            except Exception as e:
                self.log(f"⚠️ KG غير متاح: {e}", "warning")
 
        try:
            all_contexts = [_retrieve_for_charge(c, vs, kg) for c in state.charges]
            case_summary = json.dumps({
                "defendants": [d.name for d in state.defendants],
                "charges":    [c.statute for c in state.charges],
                "incidents":  [i.incident_description for i in state.incidents if i.incident_description],
            }, ensure_ascii=False)
            prompt = (
                f"معطيات القضية:\n{case_summary}\n\n"
                f"السياق المسترجع:\n{json.dumps(all_contexts, ensure_ascii=False)}\n\n"
                "ابنِ حزمة بحث لكل تهمة من هذه المعطيات فقط."
            )
            response = self._llm_invoke_with_retries(
                get_llm_qwn(self.model_name, self.temperature),
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            legal_package = _parse_llm_json(response.content)
        except Exception as e:
            self.log(f"❌ LLM فشل: {e}", "error")
            legal_package = _build_fallback_package(all_contexts)
        finally:
            if kg:
                try: kg.close()
                except Exception: pass
 
        return self._empty_update(state, "legal_researcher", legal_package)