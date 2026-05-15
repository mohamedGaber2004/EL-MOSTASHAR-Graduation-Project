import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Prompts.evidence_analyst_agent import (
    EVIDENCE_ANALYST_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)


class EvidenceAnalystAgent(AgentBase):
    """
    Evaluates all evidence, excludes items flagged as nullified by the
    ProceduralAuditorAgent, and returns a weighted evidence matrix.

    INPUT : evidences[], lab_reports[], charges[], procedural_issues[]
    OUTPUT: evidence_scores[], chain_of_custody_issues[]
    """

    def __init__(self):
        super().__init__("EVIDENCE_SCORING_MODEL", "EVIDENCE_SCORING_TEMP", EVIDENCE_ANALYST_AGENT_PROMPT)

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_invalidated_ids(self, state) -> list[str]:
        """
        Collect evidence IDs that were nullified — read directly from
        state.procedural_issues (typed list) instead of agent_outputs.
        """
        invalidated: list[str] = []

        for issue in (state.procedural_issues or []):
            # ProceduralIssue has no affected_evidence_ids field yet;
            # derive nullified flag from nullity_type being set
            source_doc = getattr(issue, "source_document_id", None)
            nullity    = getattr(issue, "nullity_type", None)
            if nullity and source_doc:
                invalidated.append(source_doc)

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

        charges = [
            {
                "statute":     getattr(c, "statute", None),
                "description": getattr(c, "description", None),
            }
            for c in (state.charges or [])
        ]

        lab_reports = [
            {
                "report_id": getattr(r, "report_id", None),
                "findings":  getattr(r, "findings", None),
            }
            for r in (state.lab_reports or [])
        ]

        procedural_issues = [
            {
                "procedure_type":  getattr(p, "procedure_type", None),
                "issue_description": getattr(p, "issue_description", None),
                "nullity_type":    getattr(p, "nullity_type", None),
                "source_document_id": getattr(p, "source_document_id", None),
            }
            for p in (state.procedural_issues or [])
        ]

        return json.dumps(
            {
                "evidences":        evidences,
                "charges":          charges,
                "lab_reports":      lab_reports,
                "procedural_issues": procedural_issues,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _build_prompt(self, state, invalidated_ids: list[str]) -> str:
        context = self._build_case_context(state, invalidated_ids)

        invalidated_note = (
            f"⚠️ الأدلة الباطلة التالية يجب استبعادها تماماً: {invalidated_ids}\n"
            "الدليل الباطل لا يُعتد به ولا يُستأنس به ولو بشكل غير مباشر.\n\n"
            if invalidated_ids
            else "لا توجد أدلة باطلة مُحددة من المدقق الإجرائي.\n\n"
        )

        return (
            "## بيانات الأدلة والتهم والمشكلات الإجرائية:\n"
            f"{context}\n\n"

            f"{invalidated_note}"

            "## التعليمات:\n"
            "لكل دليل:\n"
            "1. قيّم قوته ومدى قبوله مع التسبيب.\n"
            "2. رصد التناقضات بينه وبين الأدلة الأخرى.\n"
            "3. حدد مشكلات سلسلة الحيازة إن وُجدت.\n"
            "4. حدد درجة الإثبات الإجمالية لكل تهمة.\n"
            "5. لا تستخدم الأدلة الباطلة في بناء قناعتك.\n\n"

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
                "evidence_scores":           [],
                "chain_of_custody_issues":   [],
                "overall_proof_assessment":  "لا توجد أدلة للتحليل",
            })

        invalidated_ids = self._extract_invalidated_ids(state)
        prompt          = self._build_prompt(state, invalidated_ids)

        response = self._llm_invoke_with_retries(
            self._llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        result = self.clean_agent_output(self._parse_llm_json(response.content))

        if not isinstance(result, dict):
            logger.error("Evidence analysis — LLM response could not be parsed as dict")
            result = {
                "evidence_scores":          [],
                "chain_of_custody_issues":  [],
                "overall_proof_assessment": "فشل التحليل",
                "raw_response":             str(result),
            }
        else:
            logger.debug(
                "Evidence analysis complete — overall proof: %s",
                result.get("overall_proof_assessment", "N/A"),
            )

        result["_meta"] = {
            "total_evidences":   evidences_count,
            "invalidated_count": len(invalidated_ids),
            "invalidated_ids":   invalidated_ids,
        }

        # write typed fields directly onto state
        base_state = self._empty_update(state, "evidence_analyst", result)
        return base_state.model_copy(update={
            "evidence_scores":         result.get("evidence_scores", []),
            "chain_of_custody_issues": result.get("chain_of_custody_issues", []),
        })