from .agent_base import AgentBase
import json
from src.LLMs import get_llm_qwn
from langchain_core.messages import HumanMessage, SystemMessage
from src.Prompts.legal_researcher_agent import LEGAL_RESEARCHER_AGENT_PROMPT
from src.Graph.graph_helpers import _retrieve_for_charge, _build_fallback_package, _parse_llm_json, _now
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Vectorstore.vector_store_builder import search, load_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

class LegalResearcherAgent(AgentBase):
    def __init__(self):
        super().__init__("LEGAL_RESEARCHER_MODEL", "LEGAL_RESEARCHER_TEMP", LEGAL_RESEARCHER_AGENT_PROMPT)

    def run(self, state):
        self.log(f"📚 LegalResearcher — {len(state.charges)} charge(s)")
        if not state.charges:
            return state.model_copy(update={
                "agent_outputs": {**state.agent_outputs, "legal_researcher": {
                    "research_packages": [], "procedural_principles": [], "relevant_cassation_rulings": [],
                }},
                "completed_agents": state.completed_agents + ["legal_researcher"],
                "current_agent": "legal_researcher",
                "last_updated": _now(),
            })
        cfg = self.cfg
        index_path = Path(getattr(cfg, 'FAISS_INDEX_PATH', 'legal_faiss_index'))
        embeddings = HuggingFaceEmbeddings(model_name=getattr(cfg, 'EMBEDDING_MODEL', None))
        vs = load_vector_store(str(index_path), embeddings)
        kg = None
        if getattr(cfg, "NEO4J_URI", None):
            try:
                kg = LegalKnowledgeGraph(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD)
            except Exception as e:
                self.log(f"⚠️ Could not initialize LegalKnowledgeGraph: {e}", "warning")
        try:
            all_contexts = []
            for charge in state.charges:
                self.log(f"  🔎 Charge: {charge.statute}")
                ctx = _retrieve_for_charge(charge, vs, kg)
                all_contexts.append(ctx)
            case_summary = json.dumps({
                "defendants": [d.name for d in state.defendants],
                "charges": [c.statute for c in state.charges],
                "incidents": [i.incident_description for i in state.incidents if i.incident_description],
            }, ensure_ascii=False, indent=2)
            prompt = f"""
معطيات القضية:
{case_summary}
المعطيات المسترجعة من قاعدة المعرفة (FAISS + Knowledge Graph):
{json.dumps(all_contexts, ensure_ascii=False, indent=2)}
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
            response = self._llm_invoke_with_retries(
                get_llm_qwn(self.model_name, self.temperature),
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)]
            )
            legal_package = _parse_llm_json(response.content)
            self.log("✅ LegalResearcher done")
        except Exception as e:
            self.log(f"❌ LLM failed: {e}", "error")
            legal_package = _build_fallback_package(all_contexts)
        finally:
            if kg:
                try:
                    kg.close()
                except Exception:
                    pass
        return state.model_copy(update={
            "agent_outputs": {**state.agent_outputs, "legal_researcher": legal_package},
            "completed_agents": state.completed_agents + ["legal_researcher"],
            "current_agent": "legal_researcher",
            "last_updated": _now(),
        })
