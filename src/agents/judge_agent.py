import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Prompts.judge_agent import JUDGE_AGENT_PROMPT, EXPECTED_OUTPUT_SCHEMA
from src.Graph.states_and_schemas.Agents_output_models import FinalJudgment

logger = logging.getLogger(__name__)


class JudgeAgent(AgentBase):
    """
    Produces the final structured Arabic judgment.

    INPUT : prosecution_theory, prosecution_arguments[], defense_arguments[],
            evidence_scores[], confession_assessments[], witness_credibility_scores[],
            procedural_issues[], applied_principles[], case_articles[],
            aggravating_factors[], mitigating_factors[], applicable_article_17,
            charge_conviction_map, civil_claim, civil_award_suggested
    OUTPUT: suggested_verdict (FinalJudgment), verdict_preamble, verdict_facts,
            verdict_reasoning, verdict_operative
    """

    def __init__(self):
        super().__init__("JUDGE_MODEL", "JUDGE_TEMP", JUDGE_AGENT_PROMPT)

    # ── context builder ───────────────────────────────────────────────

    def _build_judicial_context(self, state) -> dict:

        # ── procedural — read from typed list ─────────────────────────
        procedural_nullities = [
            {
                "procedure_type":    getattr(p, "procedure_type", None),
                "issue_description": getattr(p, "issue_description", None),
                "nullity_type":      getattr(p, "nullity_type", None),
                "article_basis":     getattr(p, "article_basis", None),
                "source_document_id": getattr(p, "source_document_id", None),
            }
            for p in (state.procedural_issues or [])
            if getattr(p, "nullity_type", None)
        ]

        invalidated_ids = [
            p["source_document_id"]
            for p in procedural_nullities
            if p.get("source_document_id")
        ]

        # ── evidence ──────────────────────────────────────────────────
        evidence_scores  = [self._safe_dump(s) for s in (state.evidence_scores or [])]
        chain_issues     = state.chain_of_custody_issues or []

        # ── confessions & witnesses ───────────────────────────────────
        confession_assessments     = [self._safe_dump(c) for c in (state.confession_assessments or [])]
        witness_credibility_scores = [self._safe_dump(w) for w in (state.witness_credibility_scores or [])]

        # ── defense & prosecution ─────────────────────────────────────
        defense_arguments     = [self._safe_dump(a) for a in (state.defense_arguments or [])]
        prosecution_arguments = [self._safe_dump(a) for a in (state.prosecution_arguments or [])]
        prosecution_theory    = (
            self._safe_dump(state.prosecution_theory) if state.prosecution_theory else {}
        )

        # ── legal research ────────────────────────────────────────────
        case_articles      = state.case_articles or []
        applied_principles = [self._safe_dump(p) for p in (state.applied_principles or [])]

        # ── sentencing inputs ─────────────────────────────────────────
        aggravating_factors   = state.aggravating_factors or []
        mitigating_factors    = state.mitigating_factors or []
        applicable_article_17 = state.applicable_article_17
        charge_conviction_map = state.charge_conviction_map or {}
        civil_claim           = self._safe_dump(state.civil_claim) if state.civil_claim else None
        civil_award_suggested = state.civil_award_suggested

        # ── case basics ───────────────────────────────────────────────
        case_basics = {
            "case_id":     state.case_id,
            "case_number": state.case_number,
            "court":       state.court,
            "court_level": state.court_level,
            "defendants":  [getattr(d, "name", str(d)) for d in (state.defendants or [])],
            "charges":     [getattr(c, "statute", str(c)) for c in (state.charges or [])],
        }

        return {
            "case_basics":                 case_basics,
            "procedural_nullities":        procedural_nullities,
            "invalidated_evidence":        invalidated_ids,
            "evidence_scores":             evidence_scores,
            "chain_of_custody_issues":     chain_issues,
            "confession_assessments":      confession_assessments,
            "witness_credibility_scores":  witness_credibility_scores,
            "prosecution_theory":          prosecution_theory,
            "prosecution_arguments":       prosecution_arguments,
            "defense_arguments":           defense_arguments,
            "case_articles":               case_articles,
            "applied_principles":          applied_principles,
            "aggravating_factors":         aggravating_factors,
            "mitigating_factors":          mitigating_factors,
            "applicable_article_17":       applicable_article_17,
            "charge_conviction_map":       charge_conviction_map,
            "civil_claim":                 civil_claim,
            "civil_award_suggested":       civil_award_suggested,
        }

    # ── prompt builder ────────────────────────────────────────────────

    def _build_prompt(self, ctx: dict) -> str:
        nullities_count   = len(ctx["procedural_nullities"])
        invalidated_count = len(ctx["invalidated_evidence"])
        principles_count  = len(ctx["applied_principles"])

        return (
            "## ملف القضية الكامل للمداولة:\n"
            f"{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"

            "## مبادئ إصدار الحكم:\n"
            "1. الشك يُفسَّر لصالح المتهم — إن وُجد شك جدي في أي تهمة تكون البراءة.\n"
            "2. الدليل الباطل لا يُعتد به ولا يُستأنس به ولو بشكل غير مباشر.\n"
            "3. رُد على كل دفع من الدفوع المُثارة صراحةً في الحكم.\n"
            "4. قيّم كل دليل على حدة قبل الحكم الإجمالي.\n"
            f"5. يوجد {nullities_count} بطلان إجرائي و{invalidated_count} دليل مستبعد"
            " — ضعها في الاعتبار عند تقدير الأدلة.\n"
            f"6. يوجد {principles_count} مبدأ من محكمة النقض — استشهد بها عند التسبيب"
            " إن كانت وثيقة الصلة.\n"
            "7. استند إلى charge_conviction_map من عميل التقدير كنقطة انطلاق.\n"
            "8. راعِ aggravating_factors و mitigating_factors وأثر المادة 17 عند تحديد العقوبة.\n"
            "9. أصدر الحكم في أربعة أقسام: ديباجة → وقائع → أسباب → منطوق.\n\n"

            "## التعليمات:\n"
            "أصدر حكماً مسبباً يناقش كل تهمة على حدة، "
            "ويرد على كل دفع، ويوضح أثر البطلانيات على منظومة الإثبات، "
            "ويتضمن الفصل في الدعوى المدنية إن وُجدت.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    def _build_reasoning_trace(self, judgment: dict) -> list[dict]:
        trace = []
        for ruling in judgment.get("charges_rulings", []):
            trace.append({
                "agent":     "judge",
                "charge":    ruling.get("charge"),
                "ruling":    ruling.get("ruling"),
                "reasoning": ruling.get("ruling_reasoning", ""),
                "doubt":     ruling.get("doubt_noted", False),
            })
        trace.append({
            "agent":     "judge",
            "charge":    "إجمالي",
            "ruling":    judgment.get("verdict"),
            "reasoning": judgment.get("verdict_reasoning", ""),
        })
        return trace

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
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        judgment = self._parse_llm_json(response.content)

        if not isinstance(judgment, dict):
            logger.error("Judge LLM response could not be parsed — issuing safe acquittal")
            judgment = {
                "verdict":           "براءة",
                "verdict_reasoning": "فشل إصدار الحكم — براءة احتياطية",
                "confidence_score":  0.0,
                "charges_rulings":   [],
                "raw_response":      str(judgment),
            }

        reasoning_trace = self._build_reasoning_trace(judgment)
        logger.info(
            "Verdict issued: %s (confidence=%.2f)",
            judgment.get("verdict", ""),
            float(judgment.get("confidence_score", 0.0)),
        )

        base_state = self._empty_update(state, "judge", judgment)
        return base_state.model_copy(update={
            "suggested_verdict": judgment.get("verdict"),
            "verdict_preamble":  judgment.get("verdict_preamble"),
            "verdict_facts":     judgment.get("verdict_facts"),
            "verdict_reasoning": judgment.get("verdict_reasoning"),
            "verdict_operative": judgment.get("verdict_operative"),
        })