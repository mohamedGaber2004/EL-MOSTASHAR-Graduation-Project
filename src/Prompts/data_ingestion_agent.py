DATA_INGESTION_AGENT_PROMPT_amr_ihala = """أنت محلل قانوني. استخرج من نص "أمر الإحالة" المعلومات وأجب بـ JSON فقط.

القيم المقبولة:
court_level: "محكمة الجنح"|"محكمة الجنايات"|"محكمة الاستئناف"|"محكمة النقض"|"المحكمة العليا"|"المحكمة الاقتصادية"|"المحكمة العسكرية"|"محكمة الأحداث"
law_code: "قانون العقوبات"|"قانون مكافحة المخدرات"|"قانون الأسلحة والذخائر"|"قانون آخر"
incident_type: "قتل عمد"|"ضرب أفضى إلى موت"|"حيازة مخدرات"|"اتجار في مخدرات"|"أخرى"

قواعد:
- استخرج ما هو مذكور صراحةً فقط — ما لم يُذكر → null أو []
- حقل referral_order_text: ضع فيه النص الكامل لأمر الإحالة أو ملخصه الوافي ليكون مرجعاً للقاضي.
- استنتج age من سنة الميلاد المذكورة وسنة القضية.
- التواريخ بصيغة ISO 8601 فقط (مثال: 2024-03-01).

يجب أن يحتوي الـ JSON على المفاتيح التالية (التزم بهذا الهيكل تماماً ولا تضف مفاتيح غيرها):
{
  "case_meta": {
    "case_number": "...", 
    "court": "...", 
    "court_level": "...", 
    "referral_date": "...", 
    "prosecutor_name": "...", 
    "referral_order_text": "..."
  },
  "defendants": [{
    "name": "...", 
    "gender": "...",
    "age": 0, 
    "occupation": "...", 
    "address": "...",
    "complicity_role": "..."
  }],
  "charges": [{
    "description": "...", 
    "law_code": "...", 
    "incident_type": "...",
    "attempt_flag": false,
    "linked_defendant_names": ["..."]
  }]
}"""


DATA_INGESTION_AGENT_PROMPT_mahdar_dabt = """أنت محلل قانوني. استخرج من نص "محضر الضبط والتفتيش" المعلومات وأجب بـ JSON فقط.

القيم المقبولة:
evidence_type: "دليل مادي"|"دليل مستندي"|"أخرى"
validity_status: "سليم"|"مشكوك فيه"|"باطل"
incident_type: "ضرب وجرح" (اخترها إذا كان النص يقول اعتداء بالضرب) | "قتل عمد" | "سرقة"

قواعد:
- استخرج ما هو مذكور صراحةً فقط — ما لم يُذكر → null أو []
- التواريخ والأوقات بصيغة ISO 8601 فقط (مثال: "2024-01-10T02:00:00").
- ركز بشدة على إثبات ما إذا كان الضبط تم بإذن قضائي أم لا (حالة التلبس) في حقل seizure_warrant_present (true/false).
- استخرج اسم المجني عليه إن وُجد.

يجب أن يحتوي الـ JSON على المفاتيح التالية (التزم بهذا الهيكل تماماً ولا تضف مفاتيح غيرها):
{
  "incidents": [{
    "incident_type": "...",
    "incident_date": "...", 
    "incident_location": "...", 
    "incident_description": "...", 
    "perpetrator_names": ["..."], 
    "victim_names": ["..."]
  }],
  "evidences": [{
    "evidence_type": "...", 
    "description": "...", 
    "seizure_date": "...", 
    "seizure_location": "...", 
    "seized_by": "...", 
    "seizure_warrant_present": false, 
    "linked_defendant_name": "..."
  }],
  "procedural_issues": [{
    "procedure_type": "ضبط",
    "issue_description": "...", 
    "warrant_present": false,
    "conducting_officer": "...",
    "nullity_type": "..."
  }]
}"""


DATA_INGESTION_AGENT_PROMPT_mahdar_istijwab = """أنت محلل قانوني. استخرج من نص "محضر الاستجواب أو التحقيقات" المعلومات وأجب بـ JSON فقط.

قواعد:
- استخرج ما هو مذكور صراحةً فقط — ما لم يُذكر → null أو []
- التواريخ بصيغة ISO 8601 فقط.
- استخرج نص اعتراف المتهم أو إنكاره بدقة (هل دافع عن نفسه أم اعترف بالجريمة؟).
- ركز جداً في الملاحظات: هل تم الاستجواب في حضور محامٍ؟ وهل ادعى المتهم تعرضه لضغوط أو إكراه؟

يجب أن يحتوي الـ JSON على المفتاح التالي (التزم بهذا الهيكل تماماً ولا تضف مفاتيح غيرها):
{
  "confessions": [{
    "defendant_name": "...", 
    "text": "...", 
    "confession_date": "...", 
    "confession_stage": "تحقيق",
    "legal_counsel_present": true,
    "coercion_claimed": false,
    "voluntary": true,
    "key_admissions": ["..."]
  }]
}"""



DATA_INGESTION_AGENT_PROMPT_aqual_shuhud = """أنت محلل قانوني. استخرج من نص "أقوال الشهود" المعلومات وأجب بـ JSON فقط.

القيم المقبولة:
witness_type: "شاهد إثبات"|"شاهد نفي"|"مجني عليه"|"خبير"|"شاهد عيان"
witness_relation: "قريب"|"زوجة"|"صديق"|"غريب"|"غير محدد"
relation_to_defendant: "قريب" (اخترها إذا كان الشاهد زوج أو زوجة) | "صديق" | "عدو" 

قواعد:
- استخرج ما هو مذكور صراحةً فقط — ما لم يُذكر → null أو []
- التواريخ بصيغة ISO 8601 فقط.
- لخص شهادة الشاهد بوضوح في حقل statement_summary (مثال: "شهد بأنه رأى المتهم يضرب المجني عليه" أو "شهدت بأن المتهم كان في المنزل").

يجب أن يحتوي الـ JSON على المفتاح التالي (التزم بهذا الهيكل تماماً ولا تضف مفاتيح غيرها):
{
  "witness_statements": [{
    "witness_name": "...", 
    "witness_type": "...", 
    "relation_to_defendant": "...", 
    "statement_summary": "...", 
    "statement_date": "...",
    "was_sworn_in": true,
    "presence_at_scene": true,
    "key_facts_mentioned": ["..."]
  }]
}"""


DATA_INGESTION_AGENT_PROMPT_taqrir_tibbi = """أنت محلل قانوني. استخرج من نص "التقرير الطبي أو الفني أو المعملي" المعلومات وأجب بـ JSON فقط.

قواعد:
- استخرج ما هو مذكور صراحةً فقط — ما لم يُذكر → null أو []
- التواريخ بصيغة ISO 8601 فقط.
- ركز على استخراج "النتيجة النهائية" للتقرير بدقة (سبب الوفاة، نوع الأداة المستخدمة، المطابقة).
- اربط التقرير باسم المتوفى أو المجني عليه.

يجب أن يحتوي الـ JSON على المفتاح التالي (التزم بهذا الهيكل تماماً ولا تضف مفاتيح غيرها):
{
  "lab_reports": [{
    "report_type": "طبي شرعي", 
    "examination_date": "...", 
    "examiner_name": "...", 
    "result": "...", 
    "linked_defendant_name": "..."
  }]
}"""


DATA_INGESTION_AGENT_PROMPT_mozakeret_defa3 = """أنت محلل قانوني. استخرج من نص "مذكرة الدفاع" المعلومات وأجب بـ JSON فقط.

القيم المقبولة:
nullity_type: "بطلان مطلق"|"بطلان نسبي"|"لا بطلان"

قواعد:
- استخرج ما هو مذكور صراحةً فقط — ما لم يُذكر → null أو []
- استخرج الدفوع الشكلية (مثل بطلان الاعتراف، بطلان القبض) وضعها في procedural_issues واربطها بالسند القانوني المذكور (أرقام المواد أو مبادئ النقض).
- استخرج الدفوع الموضوعية والطلبات الختامية وضعها في defense_documents.

يجب أن يحتوي الـ JSON على المفاتيح التالية (التزم بهذا الهيكل تماماً ولا تضف مفاتيح غيرها):
{
  "defense_documents": [{
    "submitted_by": "...", 
    "formal_defenses": ["..."], 
    "substantive_defenses": ["..."], 
    "supporting_principles": ["..."],
    "alibi_claimed": false
  }], 
  "procedural_issues": [{
    "issue_description": "...", 
    "article_basis": "...", 
    "nullity_type": "..."
  }]
}"""