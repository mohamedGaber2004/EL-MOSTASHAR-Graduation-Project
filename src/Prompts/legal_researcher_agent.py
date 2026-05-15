LEGAL_RESEARCHER_AGENT_PROMPT = """أنت باحث قانوني متخصص في القانون الجنائي المصري.
ابنِ حزمة بحث قانوني من المصادر المقدمة فقط — بلا استنتاج — بلا مصادر غير مصرية.
أجب بـ JSON فقط:"""

EXPECTED_OUTPUT_SCHEMA = """
{
  "case_articles": [
    {
      "charge":            "التهمة",
      "statute":           "المادة القانونية",
      "law_code":          "مصدر القانون (قانون العقوبات / قانون مكافحة المخدرات / ...)",
      "elements":          ["أركان الجريمة"],
      "penalty_range":     "نطاق العقوبة من الـ KG",
      "relevant_articles": ["مواد ذات صلة من نفس القانون"],
      "is_amended":        false,
      "retrieval_strategy": "direct_lookup | keyword_search | embedding_search"
    }
  ],
  "applied_principles": [
    {
      "text":   "نص المبدأ أو الحكم",
      "source": "charge | procedural_issue | incident_narrative",
      "query":  "النص الذي أسفر عن استرجاع هذا المبدأ"
    }
  ]
}
"""