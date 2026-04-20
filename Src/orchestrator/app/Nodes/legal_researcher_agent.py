from .agent_base import AgentBase
import json, httpx, os
from app.llms import get_llm_qwn
from langchain_core.messages import HumanMessage, SystemMessage
from app.Prompts.legal_researcher_agent import LEGAL_RESEARCHER_AGENT_PROMPT
from app.Graph.graph_helpers import _build_fallback_package, _parse_llm_json, _now

class LegalResearcherAgent(AgentBase):
    def __init__(self):
        super().__init__("LEGAL_RESEARCHER_MODEL", "LEGAL_RESEARCHER_TEMP", LEGAL_RESEARCHER_AGENT_PROMPT)

    def _query_kg_context(self, charge) -> dict:
        """Call kg-service to retrieve structured legal context based strictly on the charge."""
        try:
            import re
            from app.Utils import ServiceCommunicator, reg
            from app.config import get_settings
            from app.Graph.graph_helpers import _lawcode_to_law_id
            
            # 1. Parse article numbers from the charge statute string using shared Regex
            # E.g. "م 274 عقوبات" -> ["274"]
            article_numbers = list(re.findall(reg._ANY_ARTICLE_RE.value, charge.statute or ""))
            
            # 2. Parse Law ID
            law_id = _lawcode_to_law_id(charge)
            
            # 3. Create keyword fallback
            keywords = [k for k in [charge.statute, charge.description] if k]
            
            # 4. Fetch strictly from KG
            kg_url = get_settings().KG_SERVICE_URL
            client = ServiceCommunicator("Orchestrator->KG", base_url=kg_url, timeout=30.0)
            
            payload = {
                "law_id": law_id,
                "article_numbers": article_numbers,
                "keywords": keywords,
            }
            
            return client.post("/query/charge-context", json_payload=payload)
            
        except Exception as e:
            self.log(f"⚠️ kg-service call failed: {e}", "warning")
            return {}

    def _retrieve_cassation(self, query: str, top_k: int = 3) -> list:
        """Call retrieval-service to get relevant Na2d (cassation) chunks."""
        try:
            from app.Utils import ServiceCommunicator
            from app.config import get_settings
            
            retrieval_url = get_settings().RETRIEVAL_SERVICE_URL
            client = ServiceCommunicator("Orchestrator->Retrieval", base_url=retrieval_url, timeout=30.0)
            data = client.post("/retrieve", json_payload={"query": query, "top_k": top_k, "doc_type": "ruling"})
            return data.get("chunks", [])
        except Exception as e:
            self.log(f"⚠️ retrieval-service call (cassation) failed: {e}", "warning")
            return []

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

        all_contexts = []
        try:
            for charge in state.charges:
                self.log(f"  🔎 Charge: {charge.statute}")
                
                # Primary Context: knowledge graph only
                kg_context = self._query_kg_context(charge)
                
                # Secondary Context: Na2d rulings only
                query = f"{charge.statute} {charge.description or ''}".strip()
                cassation_chunks = self._retrieve_cassation(query, top_k=3)
                
                # Format exactly as expected by the LLM minus the raw FAISS statute chunks
                all_contexts.append({
                    "charge_statute": charge.statute,
                    "kg_context": kg_context,
                    "cassation_docs": cassation_chunks
                })

            case_summary = json.dumps({
                "defendants": [d.name for d in state.defendants],
                "charges": [c.statute for c in state.charges],
                "incidents": [i.incident_description for i in state.incidents if i.incident_description],
            }, ensure_ascii=False, indent=2)

            prompt = f"""
معطيات القضية:
{case_summary}
المعطيات المسترجعة من قاعدة المعرفة:
{json.dumps(all_contexts, ensure_ascii=False, indent=2)}
لكل تهمة ابنِ حزمة بحث قانوني من هذه المعطيات فقط — لا تخترع معلومة.
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

        return state.model_copy(update={
            "agent_outputs": {**state.agent_outputs, "legal_researcher": legal_package},
            "completed_agents": state.completed_agents + ["legal_researcher"],
            "current_agent": "legal_researcher",
            "last_updated": _now(),
        })
