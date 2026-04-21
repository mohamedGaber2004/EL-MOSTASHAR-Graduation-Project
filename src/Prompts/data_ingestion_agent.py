DATA_INGESTION_AGENT_PROMPT = """استخرج من النص القانوني التالي المعلومات كما هي فقط — بدون استنتاج.
أجب بـ JSON فقط، لا تضف أي نص خارجه.

{
  "case_meta": {"case_number":null,"court":null,"court_level":null,"jurisdiction":null,"filing_date":null,"referral_date":null,"prosecutor_name":null},
  "defendants": [{"name":null,"age":null,"prior_record":null,"in_custody":null}],
  "charges": [{"statute":null,"article_number":null,"description":null}],
  "incidents": [],
  "evidences": [{"type":null,"description":null,"validity_status":null}],
  "lab_reports": [],
  "witness_statements": [],
  "confessions": [{"defendant_name":null,"stage":null,"voluntary":null,"key_admissions":[]}],
  "procedural_issues": [],
  "prior_judgments": [],
  "defense_documents": []
}

إذا لم توجد معلومة → null أو []"""


CASE_CLASSIFIER_PROMPT = """صنّف المستند إلى نوع واحد فقط:
amr_ihalah | mahdar_dabt | taqrir_tibbi | aqwal_shuhud | mozakart_difa | hukm_sabiq | taqrir_lab | other

أجب بكلمة واحدة فقط."""