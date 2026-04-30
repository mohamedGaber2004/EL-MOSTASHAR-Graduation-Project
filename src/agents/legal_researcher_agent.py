import json , logging
from langchain_core.messages import HumanMessage, SystemMessage
from .agent_base import AgentBase
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Prompts.legal_researcher_agent import LEGAL_RESEARCHER_AGENT_PROMPT,EXPECTED_OUTPUT_SCHEMA
from src.Graph.graph_helpers import (
    _retrieve_for_charge, 
    _build_fallback_package, 
    _parse_llm_json
)


logger = logging.getLogger(__name__)
class LegalResearcherAgent(AgentBase):

    def __init__(self,kg: LegalKnowledgeGraph,vector_store):
        super().__init__("LEGAL_RESEARCHER_MODEL","LEGAL_RESEARCHER_TEMP",LEGAL_RESEARCHER_AGENT_PROMPT)
        self.kg = kg
        self.vs = vector_store

    # ── context retrieval ──────────────────────────────────────────────

    def _retrieve_all_contexts(self, charges: list) -> tuple[list, list]:

        contexts: list = []
        failed: list = []

        for charge in charges:
            try:
                ctx = _retrieve_for_charge(charge, self.vs, self.kg)
                contexts.append(ctx)
            except Exception as e:
                failed.append(charge.statute)
                contexts.append({
                    "charge":  charge.statute,
                    "error":   str(e),
                    "context": [],
                })

        return contexts, failed

    # ── prompt builder ─────────────────────────────────────────────────

    def _build_prompt(self, state, all_contexts: list) -> str:
        case_summary = json.dumps({
            "defendants": [d.name for d in state.defendants],
            "charges":    [c.statute for c in state.charges],
            "incidents":  [
                i.incident_description
                for i in state.incidents
                if i.incident_description
            ],
        }, ensure_ascii=False, indent=2)

        return (
            "## معطيات القضية:\n"
            f"{case_summary}\n\n"

            "## السياق المسترجع من قاعدة البيانات والـ Vector Store:\n"
            f"{json.dumps(all_contexts, ensure_ascii=False, indent=2)}\n\n"

            "## التعليمات:\n"
            "ابنِ حزمة بحث قانوني لكل تهمة من المعطيات أعلاه فقط.\n"
            "لكل تهمة: حدد أركانها، نطاق عقوبتها، المواد ذات الصلة، "
            "ومبادئ النقض المرتبطة.\n"
            "استند فقط إلى ما في السياق المسترجع — لا تخترع مواد أو أحكام.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ─────────────────────────────────────────────────────

    def run(self, state):
        logger.info("Starting legal research for %d charges", len(state.charges or []))
        if not state.charges:
            logger.info("No charges found. Returning empty research package.")
            return self._empty_update(state, "legal_researcher", {
                "research_packages":          [],
                "procedural_principles":      [],
                "relevant_cassation_rulings": [],
            })

        all_contexts, failed_charges = self._retrieve_all_contexts(state.charges)

        if len(failed_charges) == len(state.charges):
            logger.warning("All charges failed during context retrieval. Using fallback.")
            legal_package = _build_fallback_package(all_contexts)
            legal_package["_meta"] = {"source": "fallback", "failed_charges": failed_charges}
            return self._empty_update(state, "legal_researcher", legal_package)

        prompt = self._build_prompt(state, all_contexts)
        logger.debug("Prompt built successfully. Invoking LLM...")

        try:
            response = self._llm_invoke_with_retries(
                self.qwen_llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            legal_package = _parse_llm_json(response.content)

        except Exception as e:
            legal_package = _build_fallback_package(all_contexts)

        if not isinstance(legal_package, dict):
            logger.error("LLM legal research failed to parse as valid dict. Using fallback.")
            legal_package = _build_fallback_package(all_contexts)
        else:
            logger.info("Legal research complete. Generated packages for %d charges.", len(legal_package.get("research_packages", [])))

        legal_package["_meta"] = {
            "charges_count":   len(state.charges),
            "failed_charges":  failed_charges,
            "contexts_count":  len(all_contexts),
            "kg_connected":    self.kg is not None,
            "vs_connected":    self.vs is not None,
        }

        return self._empty_update(state, "legal_researcher", legal_package)