LEGAL_RESEARCHER_AGENT_PROMPT = """أنت باحث قانوني متخصص في القانون الجنائي المصري.
ابنِ حزمة بحث قانوني من المصادر المقدمة فقط — بلا استنتاج — بلا مصادر غير مصرية.
أجب بـ JSON فقط:"""

EXPECTED_OUTPUT_SCHEMA = """
{
  "research_packages": [
    {
      "charge":               "التهمة",
      "statute":              "المادة القانونية",
      "elements":             ["أركان الجريمة"],
      "penalty_range":        "نطاق العقوبة من الـ KG",
      "relevant_articles":    ["مواد ذات صلة"],
      "cassation_principles": ["مبادئ نقض ذات صلة"],
      "kg_source":            true
    }
  ],
  "procedural_principles": ["مبادئ إجرائية عامة"],
  "relevant_cassation_rulings": ["أحكام نقض مرتبطة"]
}
"""