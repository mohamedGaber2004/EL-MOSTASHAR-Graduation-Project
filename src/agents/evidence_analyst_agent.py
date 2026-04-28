import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from .agent_base import AgentBase
from src.Graph.graph_helpers import _parse_llm_json
from src.Utils.agent_output_utils import clean_agent_output
from src.Prompts.evidence_analyst_agent import EVIDENCE_ANALYST_AGENT_PROMPT , EXPECTED_OUTPUT_SCHEMA


class EvidenceAnalystAgent(AgentBase):
    """
    يحلل الأدلة في ضوء نتائج المدقق الإجرائي،
    يستبعد الأدلة الباطلة، ويبني مصفوفة إثبات لكل واقعة.
    """

    def __init__(self):
        super().__init__("EVIDENCE_ANALYST_MODEL","EVIDENCE_ANALYST_TEMP",EVIDENCE_ANALYST_AGENT_PROMPT)

    # ── helpers ────────────────────────────────────────────────────────

    def _extract_invalidated_ids(self, state) -> list[str]:
        """
        يستخرج معرفات الأدلة الباطلة من تقرير المدقق الإجرائي.
        يتوافق مع الـ schema الفعلي لـ ProceduralAuditorAgent.
        """
        invalidated: list[str] = []
        audit = state.agent_outputs.get("procedural_auditor", {})

        for violation in audit.get("violations", []):
            if violation.get("nullity") is True:
                affected = violation.get("affected_evidence_ids", [])
                invalidated.extend(affected)

        # أضف أيضاً الأدلة المرتبطة بـ critical_nullities إن وجدت
        for nullity in audit.get("critical_nullities", []):
            if isinstance(nullity, dict):
                invalidated.extend(nullity.get("affected_evidence_ids", []))

        unique = list(dict.fromkeys(invalidated))  # إزالة التكرار مع الحفاظ على الترتيب
        if unique:
            logger.warning(f"Excluded evidence due to invalidity: {unique}")
        return unique

    def _build_case_context(self, state, invalidated_ids: list[str]) -> str:
        """
        يبني السياق الضروري فقط — بدل إرسال كل الـ state.
        """
        # الأدلة — مع تمييز الباطل منها
        evidences = []
        for e in (state.evidences or []):
            e_id = getattr(e, "evidence_id", None) or getattr(e, "id", None)
            evidences.append({
                "evidence_id":  e_id,
                "type":         getattr(e, "evidence_type", None),
                "description":  getattr(e, "description", None),
                "source":       getattr(e, "source", None),
                "invalidated":  e_id in invalidated_ids,
            })

        # الوقائع
        incidents = [
            {
                "incident_id":   getattr(i, "incident_id", None),
                "description":   getattr(i, "incident_description", None),
                "date":          getattr(i, "date", None),
            }
            for i in (state.incidents or [])
        ]

        # تقارير المعمل والشهود — مصادر إثبات إضافية
        lab_reports = [
            {"report_id": getattr(r, "report_id", None), "findings": getattr(r, "findings", None)}
            for r in (getattr(state, "lab_reports", None) or [])
        ]
        witness_statements = [
            {"witness_id": getattr(w, "witness_id", None), "statement": getattr(w, "statement", None)}
            for w in (getattr(state, "witness_statements", None) or [])
        ]

        return json.dumps({
            "evidences":          evidences,
            "incidents":          incidents,
            "lab_reports":        lab_reports,
            "witness_statements": witness_statements,
        }, ensure_ascii=False, indent=2)

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

    # ── main entry ─────────────────────────────────────────────────────

    def run(self, state):
        evidences_count = len(state.evidences or [])
        logger.info("Starting evidence analysis... Total pieces of evidence: %d", evidences_count)

        # حالة لا توجد أدلة
        if not evidences_count:
            logger.info("No evidence found. Returning empty matrix.")
            return self._empty_update(state, "evidence_analyst", {
                "evidence_matrix":          [],
                "invalidated_evidence_used": [],
                "overall_proof_assessment":  "لا توجد أدلة للتحليل",
            })

        # 1. استخرج الأدلة الباطلة من تقرير المدقق الإجرائي
        invalidated_ids = self._extract_invalidated_ids(state)

        # 2. بناء الـ prompt
        prompt = self._build_prompt(state, invalidated_ids)
        logger.debug("Prompt built for evidence analysis. Invalidated IDs: %s", invalidated_ids)

        # 3. استدعاء الـ LLM
        logger.debug("Invoking LLM for evidence analysis...")
        response = self._llm_invoke_with_retries(
            self.lama_llm,
            [SystemMessage(content=self.prompt), HumanMessage(content=prompt)],
        )

        # 4. تحليل الرد
        evidence_matrix = clean_agent_output(_parse_llm_json(response.content))

        # 5. تحقق من صحة الـ output
        if not isinstance(evidence_matrix, dict):
            logger.error("LLM evidence evaluation failed to parse as valid dict.")
            evidence_matrix = {
                "evidence_matrix":           [],
                "invalidated_evidence_used": invalidated_ids,
                "overall_proof_assessment":  "فشل التحليل",
                "raw_response":              str(evidence_matrix),
            }
        else:
            logger.debug("Evidence analysis complete. Overall proof: %s", evidence_matrix.get("overall_proof_assessment", "N/A"))

        # 6. أضف metadata
        evidence_matrix["_meta"] = {
            "total_evidences":    evidences_count,
            "invalidated_count":  len(invalidated_ids),
            "invalidated_ids":    invalidated_ids,
        }

        return self._empty_update(state, "evidence_analyst", evidence_matrix)