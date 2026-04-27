import json

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.LLMs import get_llm_qwn
from src.Graph.graph_helpers import (
    _retrieve_for_charge, _build_fallback_package, _parse_llm_json
)
from src.Graphstore.KG_builder import LegalKnowledgeGraph
from src.Prompts.legal_researcher_agent import LEGAL_RESEARCHER_AGENT_PROMPT,EXPECTED_OUTPUT_SCHEMA


class LegalResearcherAgent(AgentBase):
    """
    يبحث في الـ KG والـ VectorStore عن المواد والمبادئ
    المرتبطة بكل تهمة ويبني حزمة بحث قانوني.

    الـ KG والـ embeddings يُحقنان من الخارج — لا يُنشآن داخل run().
    """

    def __init__(
        self,
        kg: LegalKnowledgeGraph,
        vector_store,                   # FAISS vector store جاهز
    ):
        super().__init__(
            "LEGAL_RESEARCHER_MODEL",
            "LEGAL_RESEARCHER_TEMP",
            LEGAL_RESEARCHER_AGENT_PROMPT,
        )
        self.kg = kg
        self.vs = vector_store

    # ── context retrieval ──────────────────────────────────────────────

    def _retrieve_all_contexts(self, charges: list) -> tuple[list, list]:
        """
        يسترجع السياق لكل تهمة.

        Returns
        -------
        contexts      : list — السياق لكل تهمة
        failed_charges: list — التهم التي فشل استرجاعها
        """
        contexts: list = []
        failed: list = []

        for charge in charges:
            try:
                ctx = _retrieve_for_charge(charge, self.vs, self.kg)
                contexts.append(ctx)
            except Exception as e:
                failed.append(charge.statute)
                # أضف placeholder بدل ما تسقط التهمة
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
        # حالة لا توجد تهم
        if not state.charges:
            return self._empty_update(state, "legal_researcher", {
                "research_packages":          [],
                "procedural_principles":      [],
                "relevant_cassation_rulings": [],
            })

        # 1. استرجاع السياق — كل تهمة منفصلة مع error handling
        all_contexts, failed_charges = self._retrieve_all_contexts(state.charges)

        # 2. لو كل التهم فشلت — fallback مباشرة
        if len(failed_charges) == len(state.charges):
            legal_package = _build_fallback_package(all_contexts)
            legal_package["_meta"] = {"source": "fallback", "failed_charges": failed_charges}
            return self._empty_update(state, "legal_researcher", legal_package)

        # 3. بناء الـ prompt واستدعاء الـ LLM
        prompt = self._build_prompt(state, all_contexts)

        try:
            response = self._llm_invoke_with_retries(
                get_llm_qwn(self.model_name, self.temperature),
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            legal_package = _parse_llm_json(response.content)

        except Exception as e:
            legal_package = _build_fallback_package(all_contexts)

        # 4. تحقق من صحة الـ output
        if not isinstance(legal_package, dict):
            legal_package = _build_fallback_package(all_contexts)

        # 5. أضف metadata
        legal_package["_meta"] = {
            "charges_count":   len(state.charges),
            "failed_charges":  failed_charges,
            "contexts_count":  len(all_contexts),
            "kg_connected":    self.kg is not None,
            "vs_connected":    self.vs is not None,
        }

        return self._empty_update(state, "legal_researcher", legal_package)