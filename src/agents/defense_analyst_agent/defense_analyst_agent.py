import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..agent_base.agent_base import AgentBase
from src.Graph.state import AgentState
from src.agents.defense_analyst_agent.defense_analyst_output_model import DefenseAnalysis
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
    OUTPUT: defense_arguments[]  (List[DefenseArgument])
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
        audit = state.procedural_audit
        violations = []
        critical_nullities = []
        overall_assessment = ""
        excluded_claims = []

        if audit:
            violations = audit.violations or []
            critical_nullities = audit.critical_nullities or []
            overall_assessment = audit.overall_assessment or ""
            excluded_claims = audit.excluded_defense_claims or []

        nullities = [
            {
                "procedure_source":   p.procedure_source   or "",
                "procedure_type":     p.procedure_type     or "",
                "issue_description":  p.issue_description  or "",
                "warrant_present":    p.warrant_present     if p.warrant_present is not None else False,
                "conducting_officer": p.conducting_officer or "",
                "nullity_type":       p.nullity_type       or "",
                "nullity_effect":     p.nullity_effect     or "",
                "article_basis":      p.article_basis      or "",
            }
            for p in violations
            if p.nullity_type
        ]

        invalidated_ids = [
            p.source_document_id
            for p in violations
            if p.nullity_type and p.source_document_id
        ]

        def safe_list(items):
            """اتحول كل item لـ dict نظيف من غير None values"""
            result = []
            for item in (items or []):
                dumped = self._safe_dump(item)
                if isinstance(dumped, dict):
                    cleaned = {k: (v if v is not None else "") for k, v in dumped.items()}
                    result.append(cleaned)
                else:
                    result.append(str(dumped) if dumped is not None else "")
            return result

        return {
            "procedural_nullities":       nullities,
            "critical_nullities":         [str(n) for n in critical_nullities if n is not None],
            "overall_audit_assessment":   overall_assessment,
            "excluded_defense_claims":    [
                {
                    "claim":  c.claim  or "",
                    "reason": c.reason or "",
                }
                for c in excluded_claims
            ],
            "invalidated_evidence":       [str(i) for i in invalidated_ids if i is not None],
            "evidence_scores":            safe_list(state.evidence_scores),
            "chain_of_custody_issues":    [str(i) for i in (state.chain_of_custody_issues or []) if i is not None],
            "confession_assessments":     safe_list(state.confession_assessments),
            "witness_credibility_scores": safe_list(state.witness_credibility_scores),
            "case_articles": [
                a.get("article_number", str(a)) if isinstance(a, dict) else str(a)
                for a in (state.case_articles or [])
                if a is not None
            ],
            "applied_principles":         safe_list(state.applied_principles),
            "prosecution_theory":         (
                {k: (v if v is not None else "") for k, v in self._safe_dump(state.prosecution_theory).items()}
                if state.prosecution_theory else {}
            ),
            "prosecution_arguments":      safe_list(state.prosecution_arguments),
        }

    def _build_prompt(self, defense_data: dict, prior_context: dict, state) -> str:
        # ── clean defense_data من None ──────────────────────────────
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean(i) for i in obj if i is not None]
            return obj if obj is not None else ""

        defense_data  = clean(defense_data)
        prior_context = clean(prior_context)
        
        
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

    # ── mapping ───────────────────────────────────────────────────────

    @staticmethod
    def _map_to_defense_arguments(analysis: dict) -> list:
        """
        Flatten the LLM's structured JSON into List[DefenseArgument]-compatible dicts.
        Each formal/substantive defense becomes one DefenseArgument entry.
        """
        args = []

        for item in analysis.get("formal_defenses", []):
            args.append({
                "argument_type":        "شكلي",
                "description":          item.get("defense"),
                "legal_basis":          item.get("legal_basis"),
                "strength":             item.get("strength"),
                "strength_reasoning":   item.get("strength_reasoning"),
                "linked_nullity":       item.get("linked_nullity"),
                "countered_by_evidence": [],
                "notes":                item.get("notes"),
            })

        for item in analysis.get("substantive_defenses", []):
            args.append({
                "argument_type":         "موضوعي",
                "description":           item.get("defense"),
                "legal_basis":           item.get("legal_basis"),
                "strength":              item.get("strength"),
                "strength_reasoning":    item.get("strength_reasoning"),
                "linked_nullity":        None,
                "countered_by_evidence": item.get("countered_by_evidence", []),
                "notes":                 item.get("notes"),
            })

        return args

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state):
        docs_count = len(state.defense_documents or [])
        logger.info("DefenseAnalystAgent starting — %d defense document(s)", docs_count)

        if not docs_count:
            logger.info("No defense documents — returning empty output")
            return self._empty_update(state, "defense_analyst", {
                "defense_arguments": [],
            })

        defense_data  = self._extract_defense_data(state)
        prior_context = self._build_prior_agents_context(state)
        logger.debug("Defense data extracted — alibi claimed: %s", defense_data["alibi_claimed"])

        prompt   = self._build_prompt(defense_data, prior_context, state)
        response = self._llm_invoke_with_retries(
            self._llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        analysis = self._parse_agent_json(response.content)

        if not isinstance(analysis, dict):
            logger.error("DefenseAnalystAgent — LLM response could not be parsed as dict")
            return self._empty_update(state, "defense_analyst", {
                "defense_arguments": [],
            })

        logger.debug(
            "Defense analysis complete — strength: %s | formal: %d | substantive: %d",
            analysis.get("overall_defense_strength", "N/A"),
            len(analysis.get("formal_defenses", [])),
            len(analysis.get("substantive_defenses", [])),
        )

        analysis = self._parse_agent_json(response.content)
        defense_analysis = DefenseAnalysis(**analysis)

        return self._empty_update(state, "defense_analyst", {
            "defense_arguments": [defense_analysis.model_dump()],
        })