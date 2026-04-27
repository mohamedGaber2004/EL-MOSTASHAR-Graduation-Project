import json
from langchain_core.messages import HumanMessage, SystemMessage

from .agent_base import AgentBase
from src.LLMs import get_llm_qwn
from src.Graph.graph_helpers import _parse_llm_json
from src.Prompts.defense_analysis_agent import DEFENSE_ANALYSIS_AGENT_PROMPT , EXPECTED_OUTPUT_SCHEMA


# ── Output Schema ──────────────────────────────────────────────────────



class DefenseAnalystAgent(AgentBase):
    """
    يحلل دفوع المتهم (شكلية وموضوعية) في ضوء:
    - البطلانيات الإجرائية من ProceduralAuditorAgent
    - مصفوفة الأدلة من EvidenceAnalystAgent
    ويُقيّم كل دفع بالاستناد إلى الـ LLM فعلاً.
    """

    def __init__(self):
        super().__init__(
            "DEFENSE_ANALYST_MODEL",
            "DEFENSE_ANALYST_TEMP",
            DEFENSE_ANALYSIS_AGENT_PROMPT,
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _extract_defense_data(self, state) -> dict:
        """يستخرج بيانات الدفوع من state.defense_documents"""
        formal, substantive, principles = [], [], []
        any_alibi = False

        for doc in (state.defense_documents or []):
            formal.extend(getattr(doc, "formal_defenses", []))
            substantive.extend(getattr(doc, "substantive_defenses", []))
            principles.extend(getattr(doc, "supporting_principles", []))
            if getattr(doc, "alibi_claimed", False):
                any_alibi = True

        return {
            "formal_defenses":      formal,
            "substantive_defenses": substantive,
            "supporting_principles": principles,
            "alibi_claimed":        any_alibi,
        }

    def _build_prior_agents_context(self, state) -> dict:
        """
        يجمع النتائج ذات الصلة من الـ agents السابقة فقط —
        بدل إرسال كل الـ state.
        """
        # من ProceduralAuditorAgent
        procedural = state.agent_outputs.get("procedural_auditor", {})
        nullities = [
            {
                "violation":   v.get("violation_type"),
                "article":     v.get("article"),
                "nullity_type": v.get("nullity_type"),
                "effect":      v.get("effect_on_evidence"),
            }
            for v in procedural.get("violations", [])
            if v.get("nullity") is True
        ]
        critical_nullities = procedural.get("critical_nullities", [])

        # من EvidenceAnalystAgent
        evidence_output = state.agent_outputs.get("evidence_analyst", {})
        evidence_matrix = evidence_output.get("evidence_matrix", [])
        overall_proof   = evidence_output.get("overall_proof_assessment", "غير متاح")
        invalidated_ids = evidence_output.get("_meta", {}).get("invalidated_ids", [])

        return {
            "procedural_nullities":  nullities,
            "critical_nullities":    critical_nullities,
            "evidence_matrix":       evidence_matrix,
            "overall_proof":         overall_proof,
            "invalidated_evidence":  invalidated_ids,
        }

    def _build_prompt(self, defense_data: dict, prior_context: dict, state) -> str:
        # بيانات المتهمين والتهم للسياق
        case_basics = json.dumps({
            "defendants": [
                getattr(d, "name", str(d)) for d in (state.defendants or [])
            ],
            "charges": [
                getattr(c, "statute", str(c)) for c in (state.charges or [])
            ],
        }, ensure_ascii=False, indent=2)

        return (
            "## معطيات القضية الأساسية:\n"
            f"{case_basics}\n\n"

            "## دفوع المتهم:\n"
            f"{json.dumps(defense_data, ensure_ascii=False, indent=2)}\n\n"

            "## نتائج المدقق الإجرائي (بطلانيات مؤكدة):\n"
            f"{json.dumps(prior_context['procedural_nullities'], ensure_ascii=False, indent=2)}\n\n"

            "## مصفوفة الأدلة من محلل الأدلة:\n"
            f"{json.dumps(prior_context['evidence_matrix'], ensure_ascii=False, indent=2)}\n\n"

            f"## تقييم الإثبات الإجمالي: {prior_context['overall_proof']}\n\n"

            f"## أدلة مستبعدة بالبطلان: {prior_context['invalidated_evidence']}\n\n"

            "## التعليمات:\n"
            "لكل دفع شكلي وموضوعي:\n"
            "1. حدد سنده القانوني الدقيق (المادة + القانون).\n"
            "2. قيّم قوته بناءً على الأدلة والبطلانيات — مع التسبيب.\n"
            "3. اربطه بأي بطلان إجرائي مؤكد إن وجد.\n"
            "4. حدد الأدلة التي تدحضه إن وجدت.\n"
            "5. قيّم الـ alibi في ضوء مصفوفة الأدلة.\n\n"

            "## صيغة الإجابة (JSON فقط — بلا مقدمة ولا شرح خارج JSON):\n"
            f"{EXPECTED_OUTPUT_SCHEMA}"
        )

    # ── main entry ─────────────────────────────────────────────────────

    def run(self, state):
        docs_count = len(state.defense_documents or [])

        # حالة لا توجد مستندات دفاع
        if not docs_count:
            return self._empty_update(state, "defense_analyst", {
                "formal_defenses":       [],
                "substantive_defenses":  [],
                "alibi_analysis":        {"claimed": False},
                "supporting_principles": [],
                "overall_defense_strength": "لا توجد دفوع",
                "critical_defense_points":  [],
            })

        # 1. استخرج بيانات الدفوع
        defense_data   = self._extract_defense_data(state)
        prior_context  = self._build_prior_agents_context(state)

        # 2. بناء الـ prompt
        prompt = self._build_prompt(defense_data, prior_context, state)

        # 3. استدعاء الـ LLM وحفظ النتيجة  ← الإصلاح الأهم
        response = self._llm_invoke_with_retries(
            get_llm_qwn(self.model_name, self.temperature),
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )
        defense_analysis = _parse_llm_json(response.content)

        # 4. تحقق من صحة الـ output
        if not isinstance(defense_analysis, dict):
            defense_analysis = {
                "formal_defenses":          [],
                "substantive_defenses":     [],
                "alibi_analysis":           {"claimed": defense_data["alibi_claimed"]},
                "supporting_principles":    defense_data["supporting_principles"],
                "overall_defense_strength": "فشل التحليل",
                "critical_defense_points":  prior_context["critical_nullities"],
                "raw_response":             str(defense_analysis),
            }

        # 5. أضف metadata
        defense_analysis["_meta"] = {
            "docs_count":          docs_count,
            "formal_count":        len(defense_data["formal_defenses"]),
            "substantive_count":   len(defense_data["substantive_defenses"]),
            "nullities_count":     len(prior_context["procedural_nullities"]),
            "alibi_claimed":       defense_data["alibi_claimed"],
        }

        return self._empty_update(state, "defense_analyst", defense_analysis)