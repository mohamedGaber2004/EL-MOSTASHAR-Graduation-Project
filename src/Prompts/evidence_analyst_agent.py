EVIDENCE_ANALYST_AGENT_PROMPT = """أنت محلل أدلة. قيّم قوة كل دليل وصلته بالوقائع والتهم.
لاحظ التناقضات. استبعد الأدلة الباطلة إجرائيًا. لا تُصدر حكمًا نهائيًا.
أجب بـ JSON فقط."""

EXPECTED_OUTPUT_SCHEMA = """
{
  "evidence_matrix": [
    {
      "incident_id":        "معرف الواقعة",
      "incident_summary":   "وصف مختصر للواقعة",
      "supporting_evidence": [
        {
          "evidence_id":     "معرف الدليل",
          "type":            "نوع الدليل (مادي / شهادة / تقرير / اعتراف / ...)",
          "description":     "وصف الدليل",
          "strength":        "قوي | متوسط | ضعيف",
          "strength_reason": "سبب التقييم",
          "admissibility":   "مقبول | مستبعد",
          "proof_degree":    "قاطع | كافٍ | غير كافٍ | منعدم"
        }
      ],
      "contradictions": [
        {
          "between":     ["evidence_id_1", "evidence_id_2"],
          "description": "وصف التناقض"
        }
      ],
      "proof_reasoning": "تسبيب درجة الإثبات"
    }
  ],
  "chain_of_custody_issues": [
    "وصف مشكلة سلسلة الحيازة — دليل X لم يُختم / فُقد / انقطعت سلسلة حيازته"
  ],
  "invalidated_evidence_used": ["evidence_ids استُبعدت بسبب البطلان"],
  "overall_proof_assessment":  "تقييم عام لمنظومة الأدلة"
}
"""