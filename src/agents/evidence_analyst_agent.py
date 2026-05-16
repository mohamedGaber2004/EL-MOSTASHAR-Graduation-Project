import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Any, Dict
from .agent_base import AgentBase
from src.Graph.states_and_schemas.state import AgentState
from src.Prompts.evidence_analyst_agent import (
    EVIDENCE_ANALYST_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)


class EvidenceAnalystAgent(AgentBase):
    """
    Evaluates all evidence, excludes items flagged as nullified by the
    ProceduralAuditorAgent, and returns a weighted evidence matrix.

    INPUT : evidences[], lab_reports[], charges[], procedural_audit[]
    OUTPUT: evidence_scores[], chain_of_custody_issues[]
    """

    def __init__(self):
        super().__init__("EVIDENCE_SCORING_MODEL", "EVIDENCE_SCORING_TEMP", EVIDENCE_ANALYST_AGENT_PROMPT)

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_invalidated_ids(self, state) -> list[str]:
        """
        Collect evidence IDs that were nullified — read from
        state.procedural_audit.violations (ProceduralAuditResult).
        """
        invalidated: list[str] = []

        # FIX: state.procedural_audit is a ProceduralAuditResult object, not a list.
        #      violations is the list of ProceduralIssue objects inside it.
        audit = getattr(state, "procedural_audit", None)
        violations = audit.violations if audit else []

        for issue in violations:
            source_doc = getattr(issue, "source_document_id", None)
            nullity    = getattr(issue, "nullity_type", None)
            if nullity and source_doc:
                invalidated.append(source_doc)

        unique = list(dict.fromkeys(invalidated))
        if unique:
            logger.warning("Evidence excluded due to procedural nullity: %s", unique)
        return unique

    def _build_case_context(self, state: AgentState, invalidated_ids: list[str]) -> str:
        evidences = [
            {
                "evidence_id": getattr(e, "evidence_id", None) or getattr(e, "id", None),
                "type":        getattr(e, "evidence_type", None),
                "description": getattr(e, "description", None),
                "source":      getattr(e, "source", None),
                "invalidated": (getattr(e, "evidence_id", None) or getattr(e, "id", None)) in invalidated_ids,
            }
            for e in (state.evidences or [])
        ]

        lab_reports = [
            {
                "report_id": getattr(r, "report_id", None),
                "findings":  getattr(r, "findings", None),
            }
            for r in (state.lab_reports or [])
        ]

        # ── charges: brief label only — analyst doesn't need full charge dump ─
        charges = [
            {
                "statute":     getattr(c, "statute", None),
                "description": getattr(c, "description", None),
            }
            for c in (state.charges or [])
        ]

        # ── incidents: location/time context for chain-of-custody reasoning ───
        incidents = [
            {
                "description": getattr(i, "incident_description", None),
                "date":        getattr(i, "incident_date", None),
                "location":    getattr(i, "location", None),
            }
            for i in (state.incidents or [])
        ]

        audit      = state.procedural_audit
        violations = audit.violations if audit else []
        procedural_issues = [
            {
                "procedure_type":     p.procedure_type,
                "issue_description":  p.issue_description,
                "nullity_type":       p.nullity_type,
                "nullity_effect":     p.nullity_effect,
                "source_document_id": p.source_document_id,
            }
            for p in violations
        ]

        return json.dumps(
            {
                "charges":          charges,
                "incidents":        incidents,
                "evidences":        evidences,
                "lab_reports":      lab_reports,
                "procedural_issues": procedural_issues,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _build_prompt(self, state: AgentState, invalidated_ids: list[str]) -> str:
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

    # ── Agents utils ───────────────────────────────────────────
    def remove_nulls(self,obj: Any) -> Any:
        """
        Recursively remove keys with None/null values from dicts and lists.
        """
        if isinstance(obj, dict):
            return {k: self.remove_nulls(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self.remove_nulls(v) for v in obj if v is not None]
        else:
            return obj
        
    def clean_agent_output(self,output: Dict) -> Dict:
        """
        Clean agent output dict by removing nulls and empty fields.
        """
        cleaned = self.remove_nulls(output)
        if isinstance(cleaned, dict):
            return {k: v for k, v in cleaned.items() if v not in (None, [], {}, "")}
        return cleaned
    
    def _extract_scores(self, result: dict) -> list:
        """
        LLM returns evidence_matrix (nested), we need a flat list for AgentState.
        """
        scores = []
        for incident in result.get("evidence_matrix", []):
            for ev in incident.get("supporting_evidence", []):
                scores.append({
                    "evidence_id":   ev.get("evidence_id"),
                    "strength":      ev.get("strength"),
                    "admissibility": ev.get("admissibility"),
                    "proof_degree":  ev.get("proof_degree"),
                    "reasoning":     ev.get("strength_reason"),
                    "type":          ev.get("type"),
                    "description":   ev.get("description"),
                    "strength_reason": ev.get("strength_reason")
                })
        return scores
    
    # ── main entry ────────────────────────────────────────────────────
    def run(self, state):
        evidences_count = len(state.evidences or [])
        logger.info("EvidenceAnalystAgent starting — %d piece(s) of evidence", evidences_count)

        if not evidences_count:
            logger.info("No evidence — returning empty matrix")
            return self._empty_update(state, "evidence_analyst", {
                "evidence_scores":         [],
                "chain_of_custody_issues": [],
                "completed_agents":        ["evidence_analyst"],
            })

        invalidated_ids = self._extract_invalidated_ids(state)
        prompt          = self._build_prompt(state, invalidated_ids)

        try:
            response = self._llm_invoke_with_retries(
                self._llm,
                [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
            )
            result = self.clean_agent_output(self._parse_agent_json(response.content))
        except Exception as e:
            logger.error("EvidenceAnalystAgent LLM invocation failed: %s", e)
            result = {}

        if not isinstance(result, dict) or not result:
            logger.error("Evidence analysis — LLM response could not be parsed as dict — using fallback")
            result = {
                "evidence_matrix":          [],
                "chain_of_custody_issues":  [],
                "overall_proof_assessment": "فشل التحليل",
            }

        scores         = self._extract_scores(result)
        custody_issues = result.get("chain_of_custody_issues", [])

        logger.info(
            "Evidence analysis complete — %d score(s), %d custody issue(s)",
            len(scores),
            len(custody_issues),
        )

        return self._empty_update(state, "evidence_analyst", {
            "evidence_scores":         scores,
            "chain_of_custody_issues": custody_issues,
            "completed_agents":        ["evidence_analyst"],
        })