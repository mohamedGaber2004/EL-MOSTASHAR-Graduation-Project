DATA_INGESTION_AGENT_PROMPT = """
أنت خبير استخلاص البيانات من الوثائق القانونية المصرية.
استخرج المعلومات من النص كما هي فقط — بدون استنتاج أو تفسير.

أرجع VALID JSON ONLY بهذا الهيكل بالضبط:

{
  "case_meta": {"case_number": null, "court": null, "court_level": null, "prosecutor_name": null},
  "defendants": [],
  "charges": [],
  "incidents": [],
  "evidences": [],
  "lab_reports": [],
  "witness_statements": [],
  "confessions": [],
  "procedural_issues": [],
  "prior_judgments": [],
  "defense_documents": []
}

أمثلة الحقول الأساسية:
- defendant: {"name": "أحمد", "age": 35, "prior_record": false}
- charge: {"statute": "م 274 عقوبات", "description": "السرقة"}
- confession: {"defendant_name": "محمد", "stage": "أمام النيابة", "voluntary": true}

إذا لم توجد معلومة → ضع null أو []
أجب بـ JSON فقط ← بدون نص إضافي
"""

case_clssifier_prompt = """
صنّف هذا المستند القانوني إلى واحد من الأنواع التالية:
amr_ihalah | mahdar_dabt | taqrir_tibbi | aqwal_shuhud | mozakart_difa |
hukm_sabiq | taqrir_lab | other

أجب بكلمة واحدة فقط.

النص:
{text}
"""