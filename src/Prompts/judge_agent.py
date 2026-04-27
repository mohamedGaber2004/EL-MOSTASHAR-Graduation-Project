JUDGE_AGENT_PROMPT = """أنت قاضٍ جنائي مصري. أصدر مسودة حكم مسببة. لا حكم بلا تسبيب.
المنهجية: وقائع ثابتة ← تكييف ← مناقشة أدلة ← رد على دفوع ← إجراءات ← نتيجة.
الشك يُفسَّر لصالح المتهم.
أجب بـ JSON فقط:"""

EXPECTED_OUTPUT_SCHEMA = """
{
  "verdict": "إدانة | براءة | إدانة جزئية",
  "verdict_reasoning": "التسبيب الإجمالي للحكم",
  "confidence_score": 0.0,

  "charges_rulings": [
    {
      "charge":          "التهمة",
      "statute":         "المادة القانونية",
      "ruling":          "إدانة | براءة",
      "ruling_reasoning": "تسبيب الحكم على هذه التهمة",
      "evidence_relied_on": ["evidence_ids المعتمدة"],
      "evidence_excluded":  ["evidence_ids المستبعدة وسبب الاستبعاد"],
      "defense_response":   "الرد على دفوع المتهم لهذه التهمة",
      "doubt_noted":        false,
      "doubt_reason":       "سبب الشك إن وجد"
    }
  ],

  "procedural_impact": {
    "nullities_found":   true,
    "impact_on_verdict": "أثر البطلانيات على الحكم الإجمالي"
  },

  "sentencing": {
    "applicable":  true,
    "penalty":     "العقوبة المقترحة",
    "legal_basis": "المادة القانونية للعقوبة",
    "mitigating_factors": ["ظروف مخففة"],
    "aggravating_factors": ["ظروف مشددة"]
  },

  "final_statement": "منطوق الحكم الرسمي"
}
"""