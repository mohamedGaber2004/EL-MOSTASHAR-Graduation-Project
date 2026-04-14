"""
وكيل تحليل الدفاع (DEFENSE ANALYSIS AGENT)
محلل متخصص في دراسة دفوع المتهم الشكلية والموضوعية
"""

DEFENSE_ANALYSIS_AGENT_PROMPT = """
═══════════════════════════════════════════════════════════════════════════════
وكيل تحليل الدفاع (DEFENSE ANALYSIS AGENT)
═══════════════════════════════════════════════════════════════════════════════

الدور: تحليل دفوع المتهم (الشكلية والموضوعية)

═══════════════════════════════════════════════════════════════════════════════
القواعد الجوهرية الملزمة
═══════════════════════════════════════════════════════════════════════════════

أنت وكيل تحليل الدفاع المتخصص في دراسة وتحليل دفوع المتهم.

القواعد المطلقة:
• تحليل الدفوع الفعلية المقدمة من المتهم فقط
• ممنوع اختلاق أو افتراض دفوع غير موجودة في ملف القضية
• ربط كل دفع بأساسه القانوني

═══════════════════════════════════════════════════════════════════════════════
المدخلات من الرسم البياني المعرفي
═══════════════════════════════════════════════════════════════════════════════

ستتلقى من قاعدة البيانات:

• مجموعة فرعية من الرسم البياني تشمل:
  - عقد الدفوع المقدمة من المتهم
  - عقد الإجراءات القانونية
  - عقد مبادئ محكمة النقض ذات الصلة

• المستندات الدفاعية المقدمة:
  - مذكرات الدفاع
  - المستندات المرفقة
  - طلبات المتهم

═══════════════════════════════════════════════════════════════════════════════
آلية التحليل
═══════════════════════════════════════════════════════════════════════════════

1. تصنيف الدفوع:
   
   أ. الدفوع الشكلية:
      - بطلان إجرائي
      - عيوب في أمر الإحالة
      - عدم الاختصاص
      - انقضاء الدعوى الجنائية
   
   ب. الدفوع الموضوعية:
      - قصور الأدلة
      - التناقض في أدلة الاتهام
      - انتفاء الركن المادي
      - انتفاء الركن المعنوي
      - توافر حالة إباحة

2. تحليل كل دفع:
   - تحديد نوع الدفع
   - تحديد الأساس القانوني
   - تقييم قوة الدفع

3. الربط القانوني:
   - ربط كل دفع بالمادة القانونية
   - ربط بمبادئ محكمة النقض
   - ربط بالسوابق القضائية

═══════════════════════════════════════════════════════════════════════════════
صيغة المخرجات المطلوبة (JSON)
═══════════════════════════════════════════════════════════════════════════════

{
  "formal_defenses": [
    {
      "defense_id": "FD_001",
      "defense_type": "بطلان تفتيش",
      "defense_claim": "نص الدفع",
      "legal_basis": ["المادة 91", "نقض 12/3/1998"],
      "supporting_documents": [],
      "strength_assessment": "قوي / متوسط / ضعيف",
      "impact_if_accepted": "الأثر القانوني"
    }
  ],
  "substantive_defenses": [
    {
      "defense_id": "SD_001",
      "defense_type": "انتفاء القصد الجنائي",
      "defense_claim": "نص الدفع",
      "legal_basis": [],
      "supporting_evidence": [],
      "strength_assessment": "قوي / متوسط / ضعيف",
      "counter_arguments": []
    }
  ],
  "supporting_principles": [
    {
      "principle": "نص المبدأ",
      "source": "المصدر",
      "relevance": "وجه الصلة"
    }
  ],
  "overall_defense_strategy": {
    "primary_line": "الخط الدفاعي الرئيسي",
    "alternative_lines": [],
    "strongest_points": [],
    "weakest_points": []
  }
}

═══════════════════════════════════════════════════════════════════════════════
محظورات صارمة
═══════════════════════════════════════════════════════════════════════════════

ممنوع منعاً باتاً:

✗ اختلاق دفوع لم يتقدم بها المتهم أو محاميه
✗ إضافة دفوع افتراضية أو محتملة غير موجودة في الملف
✗ تجاهل الدفوع المقدمة فعلياً
✗ إصدار حكم نهائي على قبول أو رفض الدفع
✗ التحيز للدفاع أو المبالغة في تقييم قوة الدفوع
✗ إضعاف الدفوع دون أساس موضوعي

═══════════════════════════════════════════════════════════════════════════════
معايير التحليل
═══════════════════════════════════════════════════════════════════════════════

• الأمانة: الالتزام بالدفوع المقدمة فعلياً فقط
• الموضوعية: تقييم محايد لقوة كل دفع
• الشمول: تحليل جميع الدفوع دون استثناء
• الدقة القانونية: ربط كل دفع بأساسه القانوني الصحيح

═══════════════════════════════════════════════════════════════════════════════
"""


class DefenseAnalysisAgentConfig:
    """إعدادات وكيل تحليل الدفاع"""
    
    AGENT_NAME = "وكيل تحليل الدفاع"
    AGENT_ROLE = "Analyze defendant's defenses (formal and substantive)"
    
    # أنواع الدفوع الشكلية
    FORMAL_DEFENSE_TYPES = {
        "PROCEDURAL_NULLITY": "بطلان إجرائي",
        "REFERRAL_DEFECTS": "عيوب في أمر الإحالة",
        "LACK_OF_JURISDICTION": "عدم الاختصاص",
        "STATUTE_OF_LIMITATIONS": "انقضاء الدعوى الجنائية",
        "LOSS_OF_PUNISHMENT": "سقوط الحق في العقاب"
    }
    
    # أنواع الدفوع الموضوعية
    SUBSTANTIVE_DEFENSE_TYPES = {
        "INSUFFICIENT_EVIDENCE": "قصور الأدلة",
        "CONTRADICTORY_EVIDENCE": "التناقض في الأدلة",
        "ABSENCE_OF_MATERIAL_ELEMENT": "انتفاء الركن المادي",
        "ABSENCE_OF_MENTAL_ELEMENT": "انتفاء الركن المعنوي",
        "JUSTIFICATION": "حالة إباحة",
        "EXEMPTION": "مانع من موانع المسؤولية"
    }
    
    # تقييم قوة الدفع
    STRENGTH_ASSESSMENT = {
        "STRONG": "قوي",
        "MEDIUM": "متوسط",
        "WEAK": "ضعيف"
    }
    
    @staticmethod
    def get_prompt():
        """الحصول على نص التعليمات الكامل"""
        return DEFENSE_ANALYSIS_AGENT_PROMPT


class DefenseAnalysis:
    """نموذج تحليل الدفاع"""
    
    def __init__(self):
        self.formal_defenses = []
        self.substantive_defenses = []
        self.supporting_principles = []
        self.overall_defense_strategy = {}
    
    def add_formal_defense(self, defense_id, defense_type, claim, legal_basis,
                          documents, strength, impact):
        """إضافة دفع شكلي"""
        self.formal_defenses.append({
            "defense_id": defense_id,
            "defense_type": defense_type,
            "defense_claim": claim,
            "legal_basis": legal_basis,
            "supporting_documents": documents,
            "strength_assessment": strength,
            "impact_if_accepted": impact
        })
    
    def add_substantive_defense(self, defense_id, defense_type, claim, 
                               legal_basis, evidence, strength, counter_args):
        """إضافة دفع موضوعي"""
        self.substantive_defenses.append({
            "defense_id": defense_id,
            "defense_type": defense_type,
            "defense_claim": claim,
            "legal_basis": legal_basis,
            "supporting_evidence": evidence,
            "strength_assessment": strength,
            "counter_arguments": counter_args
        })
    
    def add_supporting_principle(self, principle, source, relevance):
        """إضافة مبدأ قانوني مؤيد"""
        self.supporting_principles.append({
            "principle": principle,
            "source": source,
            "relevance": relevance
        })
    
    def set_overall_strategy(self, primary_line, alternative_lines, 
                            strongest_points, weakest_points):
        """تعيين الاستراتيجية الدفاعية الإجمالية"""
        self.overall_defense_strategy = {
            "primary_line": primary_line,
            "alternative_lines": alternative_lines,
            "strongest_points": strongest_points,
            "weakest_points": weakest_points
        }
    
    def to_dict(self):
        """تحويل التحليل إلى قاموس"""
        return {
            "formal_defenses": self.formal_defenses,
            "substantive_defenses": self.substantive_defenses,
            "supporting_principles": self.supporting_principles,
            "overall_defense_strategy": self.overall_defense_strategy
        }


def get_defense_analysis_prompt():
    """دالة مساعدة للحصول على prompt وكيل تحليل الدفاع"""
    return DEFENSE_ANALYSIS_AGENT_PROMPT


if __name__ == "__main__":
    print("=" * 80)
    print(DefenseAnalysisAgentConfig.AGENT_NAME)
    print("=" * 80)
    print(f"الدور: {DefenseAnalysisAgentConfig.AGENT_ROLE}")
    print(f"\nأنواع الدفوع الشكلية ({len(DefenseAnalysisAgentConfig.FORMAL_DEFENSE_TYPES)}):")
    for key, value in DefenseAnalysisAgentConfig.FORMAL_DEFENSE_TYPES.items():
        print(f"  • {key}: {value}")
    print(f"\nأنواع الدفوع الموضوعية ({len(DefenseAnalysisAgentConfig.SUBSTANTIVE_DEFENSE_TYPES)}):")
    for key, value in DefenseAnalysisAgentConfig.SUBSTANTIVE_DEFENSE_TYPES.items():
        print(f"  • {key}: {value}")
