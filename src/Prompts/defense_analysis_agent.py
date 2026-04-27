DEFENSE_ANALYSIS_AGENT_PROMPT = """أنت محلل دفوع قانونية. حلل دفوع المتهم شكليًا وموضوعيًا.
لا تخترع دفوعًا غير موجودة في المستندات.
صنّف: بطلان / قصور أدلة / تناقض / نفي ركن.
أجب بـ JSON فقط:"""

EXPECTED_OUTPUT_SCHEMA = """
{
  "formal_defenses": [
    {
      "defense":      "نص الدفع",
      "legal_basis":  "المادة القانونية المستند إليها",
      "strength":     "قوي | متوسط | ضعيف",
      "strength_reasoning": "سبب التقييم",
      "linked_nullity": "معرف البطلان الإجرائي المرتبط أو null",
      "notes":        "ملاحظات إضافية"
    }
  ],
  "substantive_defenses": [
    {
      "defense":      "نص الدفع",
      "legal_basis":  "المادة القانونية",
      "strength":     "قوي | متوسط | ضعيف",
      "strength_reasoning": "سبب التقييم",
      "countered_by_evidence": ["evidence_ids تدحض هذا الدفع"],
      "notes":        "ملاحظات"
    }
  ],
  "alibi_analysis": {
    "claimed":                 true,
    "supported_by_evidence":   false,
    "supporting_evidence_ids": [],
    "notes":                   "تقييم الـ alibi"
  },
  "supporting_principles":    ["مبادئ قانونية داعمة"],
  "overall_defense_strength": "قوي | متوسط | ضعيف",
  "strength_reasoning":       "تسبيب التقييم الإجمالي",
  "critical_defense_points":  ["أهم نقاط الدفاع"]
}
"""