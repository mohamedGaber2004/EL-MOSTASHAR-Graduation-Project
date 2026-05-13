import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Graph.graph_helpers import _parse_llm_json
from src.Utils.agent_output_utils import clean_agent_output
from src.Prompts.evidence_analyst_agent import (
    EVIDENCE_ANALYST_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)


class EvidenceAnalystAgent(AgentBase):
    """
    Evaluates all evidence, excludes items flagged as nullified by the
    ProceduralAuditorAgent, and returns a weighted evidence matrix.
    """

    def __init__(self):
        super().__init__("EVIDENCE_SCORING_MODEL","EVIDENCE_SCORING_TEMP",EVIDENCE_ANALYST_AGENT_PROMPT)

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_invalidated_ids(self, state) -> list[str]:
        """
        Collect evidence IDs that were nullified by the procedural audit.
        Deduplicates while preserving insertion order.
        """
        invalidated: list[str] = []
        audit = state.agent_outputs.get("procedural_auditor", {})

        for violation in audit.get("violations", []):
            if violation.get("nullity") is True:
                invalidated.extend(violation.get("affected_evidence_ids", []))

        for nullity in audit.get("critical_nullities", []):
            if isinstance(nullity, dict):
                invalidated.extend(nullity.get("affected_evidence_ids", []))

        unique = list(dict.fromkeys(invalidated))
        if unique:
            logger.warning("Evidence excluded due to procedural nullity: %s", unique)
        return unique

    def _build_case_context(self, state, invalidated_ids: list[str]) -> str:
        evidences = []
        for e in state.evidences or []:
            e_id = getattr(e, "evidence_id", None) or getattr(e, "id", None)
            evidences.append({
                "evidence_id": e_id,
                "type":        getattr(e, "evidence_type", None),
                "description": getattr(e, "description", None),
                "source":      getattr(e, "source", None),
                "invalidated": e_id in invalidated_ids,
            })

        incidents = [
            {
                "incident_id": getattr(i, "incident_id", None),
                "description": getattr(i, "incident_description", None),
                "date":        getattr(i, "date", None),
            }
            for i in (state.incidents or [])
        ]

        lab_reports = [
            {
                "report_id": getattr(r, "report_id", None),
                "findings":  getattr(r, "findings", None),
            }
            for r in (getattr(state, "lab_reports", None) or [])
        ]

        witness_statements = [
            {
                "witness_id": getattr(w, "witness_id", None),
                "statement":  getattr(w, "statement", None),
            }
            for w in (getattr(state, "witness_statements", None) or [])
        ]

        return json.dumps(
            {
                "evidences":          evidences,
                "incidents":          incidents,
                "lab_reports":        lab_reports,
                "witness_statements": witness_statements,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _build_prompt(self, state, invalidated_ids: list[str]) -> str:
        context = self._build_case_context(state, invalidated_ids)

        invalidated_note = (
            f"⚠️ الأدلة الباطلة التالية يجب استبعادها تماماً: {invalidated_ids}\n"
            "الدليل الباطل لا يُعتد به حتى لو كان مؤيَّداً بأدلة أخرى.\n\n"
            if invalidated_ids
            else "لا توجد أدلة باطلة مُحددة من المدقق الإجرائي.\n\n"
        )

        return (
            "## بيانات الأدلة والوقائع:\n"
            f"{context}\n\n"

            f"{invalidated_note}"

            "## التعليمات:\n"
            "لكل واقعة:\n"
            "1. حدد الأدلة الداعمة وقيّم قوة كل منها مع التسبيب.\n"
            "2. رصد التناقضات بين الأدلة.\n"
            "3. حدد درجة الإثبات الإجمالية للواقعة.\n"
            "4. لا تستخدم الأدلة الباطلة في بناء قناعتك.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        evidences_count = len(state.evidences or [])
        logger.info("EvidenceAnalystAgent starting — %d piece(s) of evidence", evidences_count)

        if not evidences_count:
            logger.info("No evidence — returning empty matrix")
            return self._empty_update(state, "evidence_analyst", {
                "evidence_matrix":           [],
                "invalidated_evidence_used": [],
                "overall_proof_assessment":  "لا توجد أدلة للتحليل",
            })

        invalidated_ids = self._extract_invalidated_ids(state)

        prompt   = self._build_prompt(state, invalidated_ids)
        response = self._llm_invoke_with_retries(
            self._llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        evidence_matrix = clean_agent_output(_parse_llm_json(response.content))

        if not isinstance(evidence_matrix, dict):
            logger.error("Evidence analysis — LLM response could not be parsed as dict")
            evidence_matrix = {
                "evidence_matrix":           [],
                "invalidated_evidence_used": invalidated_ids,
                "overall_proof_assessment":  "فشل التحليل",
                "raw_response":              str(evidence_matrix),
            }
        else:
            logger.debug(
                "Evidence analysis complete — overall proof: %s",
                evidence_matrix.get("overall_proof_assessment", "N/A"),
            )

        evidence_matrix["_meta"] = {
            "total_evidences":   evidences_count,
            "invalidated_count": len(invalidated_ids),
            "invalidated_ids":   invalidated_ids,
        }

        return self._empty_update(state, "evidence_analyst", evidence_matrix)