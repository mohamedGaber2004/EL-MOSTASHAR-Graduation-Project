import json
import logging
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from ..agent_base.agent_base import AgentBase
from src.agents.judge_agent.judge_agent_output_model import FinalJudgment
from src.agents.judge_agent.judge_agent_prompt import JUDGE_AGENT_PROMPT

logger = logging.getLogger(__name__)


class JudgeAgent(AgentBase):

    def __init__(self):
        super().__init__("JUDGE_MODEL", "JUDGE_TEMP", JUDGE_AGENT_PROMPT)

    # ── context builder ───────────────────────────────────────────────
    def _build_judicial_context(self, state) -> dict:

        audit      = state.procedural_audit
        violations = audit.violations or []

        # ── procedural ────────────────────────────────────────────────────────
        procedural_nullities = [
            {
                "procedure_source":  p.procedure_source,
                "procedure_type":    p.procedure_type,
                "issue_description": p.issue_description,
                "warrant_present":   p.warrant_present,
                "conducting_officer":p.conducting_officer,
                "nullity_type":      p.nullity_type,
                "effect_on_void_act":        getattr(p, "effect_on_void_act", None),
                "effect_on_prior_acts":      getattr(p, "effect_on_prior_acts", None),
                "effect_on_subsequent_acts": getattr(p, "effect_on_subsequent_acts", None),
                "article_basis":     p.article_basis,
                "source_document_id":p.source_document_id,
            }
            for p in violations
            if p.nullity_type
        ]

        invalidated_ids = [
            p.source_document_id
            for p in violations
            if p.nullity_type and p.source_document_id
        ]

        # ── evidence ──────────────────────────────────────────────────────────
        evidence_scores = [self._safe_dump(s) for s in (state.evidence_scores or [])]
        chain_issues    = state.chain_of_custody_issues or []

        # ── confessions & witnesses ───────────────────────────────────────────
        confession_assessments     = [self._safe_dump(c) for c in (state.confession_assessments or [])]
        witness_credibility_scores = [self._safe_dump(w) for w in (state.witness_credibility_scores or [])]

        # ── defense & prosecution ─────────────────────────────────────────────
        defense_arguments     = [self._safe_dump(a) for a in (state.defense_arguments or [])]
        prosecution_arguments = [self._safe_dump(a) for a in (state.prosecution_arguments or [])]
        prosecution_theory    = (
            self._safe_dump(state.prosecution_theory) if state.prosecution_theory else {}
        )

        # ── legal research ────────────────────────────────────────────────────
        case_articles      = state.case_articles or []
        applied_principles = [self._safe_dump(p) for p in (state.applied_principles or [])]

        # ── sentencing inputs ─────────────────────────────────────────────────
        aggravating_factors   = state.aggravating_factors or []
        mitigating_factors    = state.mitigating_factors or []
        applicable_article_17 = state.applicable_article_17
        charge_conviction_map = state.charge_conviction_map or {}
        civil_claim           = self._safe_dump(state.civil_claim) if state.civil_claim else None
        civil_award_suggested = state.civil_award_suggested

        # ── case basics ───────────────────────────────────────────────────────
        case_basics = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "court":       state.court,
            "court_level": state.court_level,
            "defendants":  [getattr(d, "name", str(d)) for d in (state.defendants or [])],
            "charges":     [getattr(c, "statute", str(c)) for c in (state.charges or [])],
        }

        return {
            "case_basics":                case_basics,
            # ── procedural layer ──────────────────────────────────────────
            "procedural_nullities":       procedural_nullities,
            "critical_nullities":         audit.critical_nullities,
            "overall_audit_assessment":   audit.overall_assessment,
            "excluded_defense_claims":    [
                {"claim": c.claim, "reason": c.rejection_reason}
                for c in (audit.excluded_defense_claims or [])
            ],
            "invalidated_evidence":       invalidated_ids,
            # ── evidence layer ────────────────────────────────────────────
            "evidence_scores":            evidence_scores,
            "chain_of_custody_issues":    chain_issues,
            # ── credibility layer ─────────────────────────────────────────
            "confession_assessments":     confession_assessments,
            "witness_credibility_scores": witness_credibility_scores,
            # ── argument layer ────────────────────────────────────────────
            "prosecution_theory":         prosecution_theory,
            "prosecution_arguments":      prosecution_arguments,
            "defense_arguments":          defense_arguments,
            # ── legal layer ───────────────────────────────────────────────
            "case_articles":              case_articles,
            "applied_principles":         applied_principles,
            # ── sentencing layer ──────────────────────────────────────────
            "aggravating_factors":        aggravating_factors,
            "mitigating_factors":         mitigating_factors,
            "applicable_article_17":      applicable_article_17,
            "charge_conviction_map":      charge_conviction_map,
            "civil_claim":                civil_claim,
            "civil_award_suggested":      civil_award_suggested,
        }

    # ── prompt builder ────────────────────────────────────────────────
    def _build_prompt(self, ctx: dict) -> str:
        return self.prompt.format(
            context=json.dumps(
                ctx, ensure_ascii=False, indent=2,
                default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o),
            )
        )

    # ── main entry ────────────────────────────────────────────────────
    def run(self, state):
        logger.info("JudgeAgent starting — issuing verdict")

        required = [
            "procedural_auditor", "evidence_analyst", "confession_validity",
            "witness_credibility", "legal_researcher", "prosecution_analyst",
            "defense_analyst", "sentencing",
        ]
        missing = [a for a in required if a not in (state.completed_agents or [])]
        if missing:
            logger.warning("Some upstream agents not yet complete: %s", missing)

        ctx    = self._build_judicial_context(state)
        prompt = self._build_prompt(ctx)

        response = self._llm_invoke_with_retries(
            self._llm,
            [HumanMessage(content=prompt)],
        )

        judgment = self._parse_agent_json(response.content)

        if not isinstance(judgment, dict):
            logger.error("Judge LLM response could not be parsed — issuing safe acquittal")
            judgment = {
                "verdict":             "براءة",
                "total_prison_years":  0,
                "total_prison_months": 0,
                "total_fine_amount":   0.0,
                "operative_text":      "فشل إصدار الحكم — براءة احتياطية",
                "preamble":            None,
                "established_facts":   None,
                "per_charge_rulings":  [],
                "confidence_score":    0.0,
                "raw_response":        str(judgment),
            }

        logger.info(
            "Verdict issued: %s (confidence=%.2f) - Years: %s, Months: %s",
            judgment.get("verdict", ""),
            float(judgment.get("confidence_score", 0.0)),
            judgment.get("total_prison_years", 0),
            judgment.get("total_prison_months", 0),
        )

        # تم ربط المخرجات بنموذج FinalJudgment الجديد
        return self._empty_update(state, "judge", {
            "suggested_verdict": FinalJudgment(
                verdict             = judgment.get("verdict"),
                total_prison_years  = judgment.get("total_prison_years", 0),
                total_prison_months = judgment.get("total_prison_months", 0),
                total_fine_amount   = float(judgment.get("total_fine_amount", 0.0)),
                operative_text      = judgment.get("operative_text"),
                per_charge_rulings  = judgment.get("per_charge_rulings", []),
                confidence_score    = float(judgment.get("confidence_score", 0.0)),
            ),
            "verdict_preamble":  judgment.get("preamble"),
            "verdict_facts":     judgment.get("established_facts"),
            "verdict_reasoning": "\n\n".join(
                r.get("reasoning", "")
                for r in judgment.get("per_charge_rulings", [])
            ) or None,
            "verdict_operative": judgment.get("operative_text"),
        })