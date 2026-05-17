import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..agent_base import AgentBase
from src.Graph.state import AgentState
from src.agents.defense_analyst_agent.defense_analysis_prompt import (
    DEFENSE_ANALYSIS_AGENT_PROMPT,
    EXPECTED_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)

class DefenseAnalystAgent(AgentBase):
    """
    Synthesises defense documents into formal and substantive arguments.

    INPUT : defense_documents[], procedural_issues[], confession_assessments[],
            witness_credibility_scores[], evidence_scores[], applied_principles[],
            prosecution_theory
    OUTPUT: defense_arguments[]
    """

    def __init__(self):
        super().__init__("DEFENSE_AGENT_MODEL", "DEFENSE_AGENT_TEMP", DEFENSE_ANALYSIS_AGENT_PROMPT)

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_defense_data(self, state) -> dict:
        formal, substantive, principles = [], [], []
        any_alibi = False

        for doc in state.defense_documents or []:
            formal.extend(getattr(doc, "formal_defenses", []))
            substantive.extend(getattr(doc, "substantive_defenses", []))
            principles.extend(getattr(doc, "supporting_principles", []))
            if getattr(doc, "alibi_claimed", False):
                any_alibi = True

        return {
            "formal_defenses":       formal,
            "substantive_defenses":  substantive,
            "supporting_principles": principles,
            "alibi_claimed":         any_alibi,
        }

    def _build_prior_agents_context(self, state: AgentState) -> dict:
        audit      = state.procedural_audit
        violations = audit.violations or []

        nullities = [
            {
                "procedure_source":  p.procedure_source,
                "procedure_type":    p.procedure_type,
                "issue_description": p.issue_description,
                "warrant_present":   p.warrant_present,
                "conducting_officer":p.conducting_officer,
                "nullity_type":      p.nullity_type,
                "nullity_effect":    p.nullity_effect,
                "article_basis":     p.article_basis,
            }
            for p in violations
            if p.nullity_type
        ]

        invalidated_ids = [
            p.source_document_id
            for p in violations
            if p.nullity_type and p.source_document_id
        ]

        return {
            # ── procedural layer ──────────────────────────────────────────
            "procedural_nullities":    nullities,
            "critical_nullities":      audit.critical_nullities or [],
            "overall_audit_assessment":audit.overall_assessment,
            "excluded_defense_claims": [
                {"claim": c.claim, "reason": c.reason}
                for c in (audit.excluded_defense_claims or [])
            ],
            "invalidated_evidence":    invalidated_ids,
            # ── evidence layer ────────────────────────────────────────────
            "evidence_scores":         [self._safe_dump(s) for s in (state.evidence_scores or [])],
            "chain_of_custody_issues": state.chain_of_custody_issues or [],
            # ── credibility layer ─────────────────────────────────────────
            "confession_assessments":     [self._safe_dump(c) for c in (state.confession_assessments or [])],
            "witness_credibility_scores": [self._safe_dump(w) for w in (state.witness_credibility_scores or [])],
            # ── legal layer ───────────────────────────────────────────────
            "case_articles": [                                   # ← was missing
                a.get("article_number", str(a)) if isinstance(a, dict) else a
                for a in (state.case_articles or [])
            ],
            "applied_principles": [self._safe_dump(p) for p in (state.applied_principles or [])],
            # ── prosecution layer ─────────────────────────────────────────
            "prosecution_theory":    self._safe_dump(state.prosecution_theory) if state.prosecution_theory else {},
            "prosecution_arguments": [self._safe_dump(a) for a in (state.prosecution_arguments or [])],  # ← was missing
        }

    def _build_prompt(self, defense_data: dict, prior_context: dict, state) -> str:
        case_basics = json.dumps(
            {
                "defendants": [getattr(d, "name", str(d)) for d in (state.defendants or [])],
                "charges":    [getattr(c, "statute", str(c)) for c in (state.charges or [])],
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            "## معطيات القضية الأساسية:\n"
            f"{case_basics}\n\n"

            "## دفوع المتهم:\n"
            f"{json.dumps(defense_data, ensure_ascii=False, indent=2)}\n\n"

            "## البطلانيات الإجرائية المؤكدة:\n"
            f"{json.dumps(prior_context['procedural_nullities'], ensure_ascii=False, indent=2)}\n\n"

            "## مصفوفة الأدلة:\n"
            f"{json.dumps(prior_context['evidence_scores'], ensure_ascii=False, indent=2)}\n\n"

            "## مشكلات سلسلة الحيازة:\n"
            f"{json.dumps(prior_context['chain_of_custody_issues'], ensure_ascii=False, indent=2)}\n\n"

            "## تقييم الاعترافات:\n"
            f"{json.dumps(prior_context['confession_assessments'], ensure_ascii=False, indent=2)}\n\n"

            "## موثوقية الشهود:\n"
            f"{json.dumps(prior_context['witness_credibility_scores'], ensure_ascii=False, indent=2)}\n\n"

            "## المبادئ القانونية المطبقة:\n"
            f"{json.dumps(prior_context['applied_principles'], ensure_ascii=False, indent=2)}\n\n"

            "## نظرية الاتهام:\n"
            f"{json.dumps(prior_context['prosecution_theory'], ensure_ascii=False, indent=2)}\n\n"

            f"## أدلة مستبعدة بالبطلان: {prior_context['invalidated_evidence']}\n\n"

            "## التعليمات:\n"
            "لكل دفع شكلي وموضوعي:\n"
            "1. حدد سنده القانوني الدقيق (المادة + القانون).\n"
            "2. قيّم قوته بناءً على الأدلة والبطلانيات — مع التسبيب.\n"
            "3. اربطه بأي بطلان إجرائي مؤكد إن وجد.\n"
            "4. حدد الأدلة التي تدحضه إن وجدت.\n"
            "5. قيّم الـ alibi في ضوء مصفوفة الأدلة.\n"
            "6. استثمر ضعف نظرية الاتهام ومشكلات سلسلة الحيازة.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        docs_count = len(state.defense_documents or [])
        logger.info("DefenseAnalystAgent starting — %d defense document(s)", docs_count)

        if not docs_count:
            logger.info("No defense documents — returning empty output")
            return self._empty_update(state, "defense_analyst", {
                "defense_arguments":        [],
                "overall_defense_strength": "لا توجد دفوع",
                "critical_defense_points":  [],
            })

        defense_data  = self._extract_defense_data(state)
        prior_context = self._build_prior_agents_context(state)
        logger.debug("Defense data extracted — alibi claimed: %s", defense_data["alibi_claimed"])

        prompt   = self._build_prompt(defense_data, prior_context, state)
        response = self._llm_invoke_with_retries(
            self._llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        defense_analysis = self._parse_agent_json(response.content)

        if not isinstance(defense_analysis, dict):
            logger.error("Defense analysis — LLM response could not be parsed as dict")
            defense_analysis = {
                "defense_arguments":        [],
                "overall_defense_strength": "فشل التحليل",
                "critical_defense_points":  prior_context["critical_nullities"],
                "raw_response":             str(defense_analysis),
            }
        else:
            logger.debug(
                "Defense analysis complete — overall strength: %s",
                defense_analysis.get("overall_defense_strength", "N/A"),
            )

        defense_analysis["_meta"] = {
            "docs_count":        docs_count,
            "formal_count":      len(defense_data["formal_defenses"]),
            "substantive_count": len(defense_data["substantive_defenses"]),
            "nullities_count":   len(prior_context["procedural_nullities"]),
            "alibi_claimed":     defense_data["alibi_claimed"],
        }

        return self._empty_update(state, "defense_analyst", {
            "defense_arguments": defense_analysis.get("defense_arguments", []),
        })