"""
وكيل تحليل الأدلة (EVIDENCE ANALYST AGENT)
محلل متخصص في تقييم القوة الإثباتية للأدلة
"""

EVIDENCE_ANALYST_AGENT_PROMPT = """
═══════════════════════════════════════════════════════════════════════════════
وكيل تحليل الأدلة (EVIDENCE ANALYST AGENT)
═══════════════════════════════════════════════════════════════════════════════

الدور: تحليل القوة الإثباتية للأدلة دون إصدار حكم نهائي

═══════════════════════════════════════════════════════════════════════════════
القواعد الجوهرية الملزمة
═══════════════════════════════════════════════════════════════════════════════

أنت وكيل تحليل الأدلة المتخصص في تقييم قوة الإثبات.

القواعد المطلقة:
• تحليل قوة الأدلة فقط (لا إصدار أحكام نهائية)
• تقييم الاتساق، التعاضد، والتناقضات
• عدم الحكم على صحة الوقائع أو براءة/إدانة المتهم

═══════════════════════════════════════════════════════════════════════════════
المدخلات من الرسم البياني المعرفي
═══════════════════════════════════════════════════════════════════════════════

ستتلقى من قاعدة البيانات:

• عقد الأدلة (Evidence Nodes):
  - الأدلة المادية (سلاح، أحراز)
  - المستندات الرسمية
  - التقارير الفنية (طب شرعي، معامل)

• عقد الشهود (Witness Nodes):
  - أقوال الشهود
  - أقوال المتهم
  - تقارير الخبراء

• عقد الوقائع (Fact Nodes):
  - الوقائع المادية المدعاة
  - الأحداث الزمنية

═══════════════════════════════════════════════════════════════════════════════
آلية التحليل
═══════════════════════════════════════════════════════════════════════════════

1. الربط (LINKING):
   - ربط الأدلة المادية بالوقائع
   - ربط الشهادات بالوقائع

2. التقييم (EVALUATION):
   - الاتساق الداخلي
   - التعاضد والتأييد
   - التناقضات

3. الفصل (SEPARATION):
   - الأدلة المؤيدة
   - الأدلة المعارضة

═══════════════════════════════════════════════════════════════════════════════
مقياس قوة الدليل (0.0 - 1.0)
═══════════════════════════════════════════════════════════════════════════════

• 0.9 - 1.0: دليل قوي جداً
• 0.7 - 0.89: دليل قوي
• 0.5 - 0.69: دليل متوسط
• 0.3 - 0.49: دليل ضعيف
• 0.1 - 0.29: دليل ضعيف جداً

═══════════════════════════════════════════════════════════════════════════════
صيغة المخرجات المطلوبة (JSON)
═══════════════════════════════════════════════════════════════════════════════

{
  "fact": "الواقعة",
  "fact_id": "معرّف الواقعة",
  "supporting_evidence": [
    {
      "evidence_id": "E12",
      "evidence_type": "دليل مادي",
      "evidence_description": "وصف",
      "strength": 0.85,
      "reasoning": "أساس التقييم",
      "corroborated_by": ["E13"]
    }
  ],
  "contradicting_evidence": [
    {
      "evidence_id": "W07",
      "evidence_type": "شهادة",
      "contradiction_type": "تناقض مكاني",
      "contradiction_details": "التفاصيل"
    }
  ],
  "internal_contradictions": [],
  "missing_links": [],
  "overall_assessment": {
    "prosecution_case_strength": 0.75,
    "defense_case_strength": 0.40,
    "key_strengths": [],
    "key_weaknesses": []
  }
}

═══════════════════════════════════════════════════════════════════════════════
محظورات صارمة
═══════════════════════════════════════════════════════════════════════════════

ممنوع منعاً باتاً:

✗ إصدار حكم نهائي بالبراءة أو الإدانة
✗ تجاوز دورك كمحلل إلى دور القاضي
✗ اختلاق أدلة غير موجودة في الرسم البياني
✗ تجاهل الأدلة المعارضة أو التقليل من شأنها تعسفياً
✗ إعطاء وزن للأدلة الباطلة إجرائياً
✗ التحيز لطرف على حساب الآخر

═══════════════════════════════════════════════════════════════════════════════
"""


class EvidenceAnalystAgentConfig:
    """إعدادات وكيل تحليل الأدلة"""
    
    AGENT_NAME = "وكيل تحليل الأدلة"
    AGENT_ROLE = "Analyze probative value without final judgment"
    
    # أنواع الأدلة
    EVIDENCE_TYPES = {
        "PHYSICAL": "دليل مادي",
        "TESTIMONY": "شهادة",
        "FORENSIC_REPORT": "تقرير فني",
        "DOCUMENT": "مستند رسمي"
    }
    
    # أنواع التناقضات
    CONTRADICTION_TYPES = {
        "TEMPORAL": "تناقض زمني",
        "SPATIAL": "تناقض مكاني",
        "SUBSTANTIVE": "تناقض موضوعي"
    }
    
    # مقياس قوة الدليل
    STRENGTH_SCALE = {
        "VERY_STRONG": (0.9, 1.0, "قوي جداً"),
        "STRONG": (0.7, 0.89, "قوي"),
        "MEDIUM": (0.5, 0.69, "متوسط"),
        "WEAK": (0.3, 0.49, "ضعيف"),
        "VERY_WEAK": (0.1, 0.29, "ضعيف جداً")
    }
    
    @staticmethod
    def get_prompt():
        """الحصول على نص التعليمات الكامل"""
        return EVIDENCE_ANALYST_AGENT_PROMPT
    
    @staticmethod
    def get_strength_label(strength_value):
        """الحصول على تصنيف قوة الدليل"""
        for category, (min_val, max_val, label) in EvidenceAnalystAgentConfig.STRENGTH_SCALE.items():
            if min_val <= strength_value <= max_val:
                return label
        return "غير محدد"


class EvidenceAnalysis:
    """نموذج تحليل الأدلة"""
    
    def __init__(self, fact, fact_id):
        self.fact = fact
        self.fact_id = fact_id
        self.supporting_evidence = []
        self.contradicting_evidence = []
        self.internal_contradictions = []
        self.missing_links = []
        self.overall_assessment = {}
    
    def add_supporting_evidence(self, evidence_id, evidence_type, description, 
                                strength, reasoning, corroborated_by=None):
        """إضافة دليل مؤيد"""
        self.supporting_evidence.append({
            "evidence_id": evidence_id,
            "evidence_type": evidence_type,
            "evidence_description": description,
            "strength": strength,
            "reasoning": reasoning,
            "corroborated_by": corroborated_by or []
        })
    
    def add_contradicting_evidence(self, evidence_id, evidence_type, 
                                   contradiction_type, details):
        """إضافة دليل معارض"""
        self.contradicting_evidence.append({
            "evidence_id": evidence_id,
            "evidence_type": evidence_type,
            "contradiction_type": contradiction_type,
            "contradiction_details": details
        })
    
    def add_internal_contradiction(self, evidence_id, contradiction):
        """إضافة تناقض داخلي"""
        self.internal_contradictions.append({
            "evidence_id": evidence_id,
            "contradiction": contradiction
        })
    
    def add_missing_link(self, link):
        """إضافة ثغرة في سلسلة الإثبات"""
        self.missing_links.append(link)
    
    def set_overall_assessment(self, prosecution_strength, defense_strength,
                              strengths, weaknesses):
        """تعيين التقييم الإجمالي"""
        self.overall_assessment = {
            "prosecution_case_strength": prosecution_strength,
            "defense_case_strength": defense_strength,
            "key_strengths": strengths,
            "key_weaknesses": weaknesses
        }
    
    def to_dict(self):
        """تحويل التحليل إلى قاموس"""
        return {
            "fact": self.fact,
            "fact_id": self.fact_id,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "internal_contradictions": self.internal_contradictions,
            "missing_links": self.missing_links,
            "overall_assessment": self.overall_assessment
        }


def get_evidence_analyst_prompt():
    """دالة مساعدة للحصول على prompt وكيل تحليل الأدلة"""
    return EVIDENCE_ANALYST_AGENT_PROMPT


if __name__ == "__main__":
    print("=" * 80)
    print(EvidenceAnalystAgentConfig.AGENT_NAME)
    print("=" * 80)
    print(f"الدور: {EvidenceAnalystAgentConfig.AGENT_ROLE}")
    print(f"\nأنواع الأدلة ({len(EvidenceAnalystAgentConfig.EVIDENCE_TYPES)}):")
    for key, value in EvidenceAnalystAgentConfig.EVIDENCE_TYPES.items():
        print(f"  • {key}: {value}")
    print(f"\nمقياس قوة الدليل:")
    for category, (min_val, max_val, label) in EvidenceAnalystAgentConfig.STRENGTH_SCALE.items():
        print(f"  • {label}: {min_val} - {max_val}")
