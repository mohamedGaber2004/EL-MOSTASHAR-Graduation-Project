import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.Graph.graph_helpers import _parse_llm_json, _now
from src.Prompts.judge_agent import JUDGE_AGENT_PROMPT, EXPECTED_OUTPUT_SCHEMA
from src.Utils.agents_enums import AgentsEnums

logger = logging.getLogger(__name__)


class JudgeAgent(AgentBase):
    """Issues the final verdict after receiving the outputs of all preceding agents."""

    def __init__(self):
        super().__init__("JUDGE_MODEL","JUDGE_TEMP",JUDGE_AGENT_PROMPT)

    # ── context builder ───────────────────────────────────────────────

    def _build_judicial_context(self, state) -> dict:
        outputs = state.agent_outputs

        # ── procedural ────────────────────────────────────────────────
        procedural    = outputs.get("procedural_auditor", {})
        nullities     = [
            v for v in procedural.get("violations", [])
            if v.get("nullity") is True
        ]

        # ── evidence ──────────────────────────────────────────────────
        evidence_out    = outputs.get("evidence_analyst", {})
        evidence_matrix = evidence_out.get("evidence_matrix", [])
        invalidated_ids = evidence_out.get("_meta", {}).get("invalidated_ids", [])
        overall_proof   = evidence_out.get("overall_proof_assessment", "غير متاح")

        # ── defense ───────────────────────────────────────────────────
        defense_out          = outputs.get("defense_analyst", {})
        formal_defenses      = defense_out.get("formal_defenses", [])
        substantive_defenses = defense_out.get("substantive_defenses", [])
        defense_strength     = defense_out.get("overall_defense_strength", "غير متاح")
        critical_points      = defense_out.get("critical_defense_points", [])

        # ── legal research (includes cassation rulings now) ────────────
        legal_out     = outputs.get("legal_researcher", {})
        research_pkgs = legal_out.get("research_packages", [])
        cassation     = legal_out.get("relevant_cassation_rulings", [])

        # ── case basics ───────────────────────────────────────────────
        case_basics = {
            "case_id":    state.case_id,
            "defendants": [getattr(d, "name", str(d)) for d in (state.defendants or [])],
            "charges":    [getattr(c, "statute", str(c)) for c in (state.charges or [])],
        }

        return {
            "case_basics":              case_basics,
            "procedural_nullities":     nullities,
            "invalidated_evidence":     invalidated_ids,
            "evidence_matrix":          evidence_matrix,
            "overall_proof":            overall_proof,
            "formal_defenses":          formal_defenses,
            "substantive_defenses":     substantive_defenses,
            "defense_strength":         defense_strength,
            "critical_defense_points":  critical_points,
            "legal_research":           research_pkgs,
            "cassation_rulings":        cassation,       # ← new
        }

    # ── prompt builder ────────────────────────────────────────────────

    def _build_prompt(self, judicial_context: dict) -> str:
        nullities_count   = len(judicial_context["procedural_nullities"])
        invalidated_count = len(judicial_context["invalidated_evidence"])
        cassation_count   = len(judicial_context["cassation_rulings"])

        return (
            "## ملف القضية الكامل للمداولة:\n"
            f"{json.dumps(judicial_context, ensure_ascii=False, indent=2)}\n\n"

            "## مبادئ إصدار الحكم:\n"
            "1. الشك يُفسَّر لصالح المتهم — إن وُجد شك جدي في أي تهمة تكون البراءة.\n"
            "2. الدليل الباطل لا يُعتد به ولا يُستأنس به ولو بشكل غير مباشر.\n"
            "3. رُد على كل دفع من الدفوع المُثارة صراحةً في الحكم.\n"
            "4. قيّم كل دليل على حدة قبل الحكم الإجمالي.\n"
            f"5. يوجد {nullities_count} بطلان إجرائي و{invalidated_count} دليل مستبعد "
            "— ضعها في الاعتبار عند تقدير الأدلة.\n"
            f"6. يوجد {cassation_count} مبدأ من محكمة النقض — استشهد بها عند التسبيب "
            "إن كانت وثيقة الصلة.\n\n"

            "## التعليمات:\n"
            "أصدر حكماً مسبباً يناقش كل تهمة على حدة، "
            "ويرد على كل دفع، ويوضح أثر البطلانيات على منظومة الإثبات.\n\n"

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

        required = ["procedural_auditor", "evidence_analyst",
                    "defense_analyst", "legal_researcher"]
        missing  = [a for a in required if a not in state.completed_agents]
        if missing:
            logger.warning("Some upstream agents not yet complete: %s", missing)

        judicial_context = self._build_judicial_context(state)
        prompt           = self._build_prompt(judicial_context)

        response = self._llm_invoke_with_retries(
            self._llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        judgment = _parse_llm_json(response.content)

        if not isinstance(judgment, dict):
            logger.error("Judge LLM response could not be parsed — issuing safe acquittal")
            judgment = {
                "verdict":           "براءة",
                "verdict_reasoning": "فشل إصدار الحكم — براءة احتياطية",
                "confidence_score":  0.0,
                "charges_rulings":   [],
                "raw_response":      str(judgment),
            }

        verdict         = judgment.get("verdict", "")
        reasoning_trace = self._build_reasoning_trace(judgment)

        logger.info(
            "Verdict issued: %s (confidence=%.2f)",
            verdict,
            float(judgment.get("confidence_score", 0.0)),
        )

        # _empty_update handles agent_outputs, completed_agents,
        # current_agent, and last_updated — we pass verdict-specific
        # fields via a subsequent model_copy to keep _empty_update clean.
        base_state = self._empty_update(state, "judge", judgment)

        return base_state.model_copy(
            update={
                "suggested_verdict": verdict,
                "confidence_score":  float(judgment.get("confidence_score", 0.0)),
                "reasoning_trace":   reasoning_trace,
            }
        )