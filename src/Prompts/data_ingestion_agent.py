DATA_INGESTION_AGENT_PROMPT = """استخرج من النص القانوني التالي المعلومات كما هي فقط — بدون استنتاج.
أجب بـ JSON فقط، لا تضف أي نص خارجه.

{
  "case_meta": {"case_number":null,"court":null,"court_level":null,"jurisdiction":null,
                "filing_date":null,"referral_date":null,"prosecutor_name":null},
  "defendants": [{"name":null,"age":null,"prior_record":null,"in_custody":null}],
  "charges": [{"statute":null,"article_number":null,"description":null}],
  "incidents": [{"incident_id":null,"incident_description":null,"date":null}],
  "evidences": [{"evidence_id":null,"type":null,"description":null,
                 "source":null,"validity_status":null}],
  "lab_reports": [{"report_id":null,"findings":null}],
  "witness_statements": [{"witness_id":null,"statement":null}],
  "confessions": [{"defendant_name":null,"stage":null,"voluntary":null,"key_admissions":[]}],
  "procedural_issues": [{"issue_id":null,"description":null}],
  "prior_judgments": [],
  "defense_documents": [{"formal_defenses":[],"substantive_defenses":[],
                         "supporting_principles":[],"alibi_claimed":false}]
}

إذا لم توجد معلومة → null أو []"""