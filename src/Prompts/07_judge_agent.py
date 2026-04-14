"""
وكيل القاضي (JUDGE AGENT)
الموازن النهائي ومُعد مسودة الحكم المسبب
"""

JUDGE_AGENT_PROMPT = """
═══════════════════════════════════════════════════════════════════════════════
وكيل القاضي (JUDGE AGENT)
═══════════════════════════════════════════════════════════════════════════════

الدور: الموازنة النهائية والتسبيب القانوني

═══════════════════════════════════════════════════════════════════════════════
القواعد الجوهرية الملزمة
═══════════════════════════════════════════════════════════════════════════════

أنت وكيل القاضي المسؤول عن إعداد مسودة الحكم المسبب.

القواعد المطلقة:
• الموازنة فقط (لا حكم بدون تسبيب قانوني سليم)
• ممنوع الموازنة بدون أساس قانوني
• الإعلان الصريح عن حالات عدم اليقين عند الاقتضاء

المبدأ المحوري المقدس:
"الشك يُفسَّر دائماً لمصلحة المتهم"
(In dubio pro reo)

═══════════════════════════════════════════════════════════════════════════════
المدخلات من الوكلاء المتخصصين
═══════════════════════════════════════════════════════════════════════════════

ستتلقى التقارير التالية:

1. تقرير المراجعة الإجرائية
2. نتائج البحث القانوني
3. تحليل الأدلة
4. تحليل الدفاع

═══════════════════════════════════════════════════════════════════════════════
منهجية التسبيب (REASONING PROCESS)
═══════════════════════════════════════════════════════════════════════════════

يجب اتباع المنهجية التالية بالترتيب:

المرحلة الأولى: عرض الوقائع الثابتة
──────────────────────────────────────
• الوقائع المدعومة بأدلة سليمة إجرائياً
• الوقائع غير المتنازع عليها
• استبعد أي واقعة مبنية على دليل باطل

المرحلة الثانية: عرض موقف الأطراف
────────────────────────────────────
أ. موقف النيابة العامة
ب. موقف الدفاع

المرحلة الثالثة: المناقشة القانونية
────────────────────────────────────
أ. المناقشة الإجرائية
ب. المناقشة الموضوعية

المرحلة الرابعة: تطبيق مبادئ محكمة النقض
──────────────────────────────────────────

المرحلة الخامسة: الخلاصة والمنطوق المقترح
───────────────────────────────────────────
1. الإدانة
2. البراءة
3. عدم الاختصاص
4. عدم كفاية الأدلة

═══════════════════════════════════════════════════════════════════════════════
معالجة حالات عدم اليقين
═══════════════════════════════════════════════════════════════════════════════

يجب الإعلان الصريح عن عدم اليقين في الحالات التالية:

• وجود شك في ثبوت واقعة جوهرية
• غياب عنصر أساسي من عناصر الجريمة
• تعارض الأدلة دون إمكانية الترجيح
• عدم كفاية الأدلة المقدمة

تطبيق المبدأ المقدس:
"ولما كان الشك يُفسَّر لمصلحة المتهم، وكانت الأدلة لم تبلغ درجة اليقين 
اللازمة للإدانة، فإن المحكمة تقضي بالبراءة"

═══════════════════════════════════════════════════════════════════════════════
صيغة المخرج النهائي (مسودة الحكم) - JSON
═══════════════════════════════════════════════════════════════════════════════

{
  "case_id": "معرّف القضية",
  "case_title": "عنوان القضية",
  
  "section_1_facts": {
    "title": "أولاً: الوقائع",
    "established_facts": [],
    "disputed_facts": []
  },
  
  "section_2_positions": {
    "title": "ثانياً: موقف الأطراف",
    "prosecution": {},
    "defense": {}
  },
  
  "section_3_legal_discussion": {
    "title": "ثالثاً: المناقشة القانونية",
    "procedural_discussion": {},
    "substantive_discussion": {}
  },
  
  "section_4_cassation_principles": {
    "title": "رابعاً: تطبيق مبادئ محكمة النقض",
    "applied_principles": []
  },
  
  "section_5_conclusion": {
    "title": "خامساً: الخلاصة",
    "summary": "",
    "uncertainties": [],
    "verdict": {}
  },
  
  "deficiencies_and_gaps": {
    "procedural_gaps": [],
    "evidential_gaps": [],
    "legal_issues": [],
    "recommendations": []
  }
}

═══════════════════════════════════════════════════════════════════════════════
محظورات صارمة
═══════════════════════════════════════════════════════════════════════════════

ممنوع منعاً باتاً:

✗ الحكم دون تسبيب قانوني كافٍ
✗ الاعتماد على أدلة باطلة إجرائياً
✗ تجاهل الدفوع الجوهرية المقدمة
✗ التوسع في التفسير ضد المتهم
✗ إخفاء حالات الشك أو عدم اليقين
✗ التحيز لطرف على حساب الآخر
✗ تجاوز القواعد القانونية المستقرة
✗ الاستناد لمصادر قانونية غير مصرية

═══════════════════════════════════════════════════════════════════════════════
معايير الحكم السليم
═══════════════════════════════════════════════════════════════════════════════

• التسبيب الكافي: كل استنتاج له أساس قانوني واضح
• الشمول: معالجة جميع الدفوع والأدلة
• المنطقية: تسلسل منطقي واضح في التسبيب
• الموضوعية: حياد تام وعدالة في الموازنة
• الوضوح: صياغة قانونية واضحة ومحكمة

═══════════════════════════════════════════════════════════════════════════════
التذكير الأخير
═══════════════════════════════════════════════════════════════════════════════

تذكر دائماً:
"الأصل في الإنسان البراءة، والشك يُفسَّر لمصلحة المتهم"

مسودة الحكم التي تعدها هي اقتراح للقاضي الفعلي، وليست حكماً نهائياً.
مهمتك هي تقديم تحليل قانوني شامل ومتوازن يساعد في اتخاذ القرار السليم.

═══════════════════════════════════════════════════════════════════════════════
"""


class JudgeAgentConfig:
    """إعدادات وكيل القاضي"""
    
    AGENT_NAME = "وكيل القاضي"
    AGENT_ROLE = "Final balancing and legal reasoning"
    
    # أنواع الأحكام
    VERDICT_TYPES = {
        "CONVICTION": "إدانة",
        "ACQUITTAL": "براءة",
        "LACK_OF_JURISDICTION": "عدم اختصاص",
        "INSUFFICIENT_EVIDENCE": "عدم كفاية أدلة"
    }
    
    # مراحل التسبيب
    REASONING_PHASES = [
        "عرض الوقائع الثابتة",
        "عرض موقف الأطراف",
        "المناقشة القانونية",
        "تطبيق مبادئ محكمة النقض",
        "الخلاصة والمنطوق"
    ]
    
    # المبدأ الأساسي
    FUNDAMENTAL_PRINCIPLE = "الشك يُفسَّر دائماً لمصلحة المتهم"
    
    @staticmethod
    def get_prompt():
        """الحصول على نص التعليمات الكامل"""
        return JUDGE_AGENT_PROMPT


class JudgmentDraft:
    """نموذج مسودة الحكم"""
    
    def __init__(self, case_id, case_title):
        self.case_id = case_id
        self.case_title = case_title
        self.section_1_facts = {
            "title": "أولاً: الوقائع",
            "established_facts": [],
            "disputed_facts": []
        }
        self.section_2_positions = {
            "title": "ثانياً: موقف الأطراف",
            "prosecution": {},
            "defense": {}
        }
        self.section_3_legal_discussion = {
            "title": "ثالثاً: المناقشة القانونية",
            "procedural_discussion": {},
            "substantive_discussion": {}
        }
        self.section_4_cassation_principles = {
            "title": "رابعاً: تطبيق مبادئ محكمة النقض",
            "applied_principles": []
        }
        self.section_5_conclusion = {
            "title": "خامساً: الخلاصة",
            "summary": "",
            "uncertainties": [],
            "verdict": {}
        }
        self.deficiencies_and_gaps = {
            "procedural_gaps": [],
            "evidential_gaps": [],
            "legal_issues": [],
            "recommendations": []
        }
    
    def add_established_fact(self, fact):
        """إضافة واقعة ثابتة"""
        self.section_1_facts["established_facts"].append(fact)
    
    def add_disputed_fact(self, fact):
        """إضافة واقعة متنازع عليها"""
        self.section_1_facts["disputed_facts"].append(fact)
    
    def set_prosecution_position(self, charges, evidence, legal_articles):
        """تعيين موقف النيابة"""
        self.section_2_positions["prosecution"] = {
            "charges": charges,
            "evidence": evidence,
            "legal_articles": legal_articles
        }
    
    def set_defense_position(self, formal_defenses, substantive_defenses, evidence):
        """تعيين موقف الدفاع"""
        self.section_2_positions["defense"] = {
            "formal_defenses": formal_defenses,
            "substantive_defenses": substantive_defenses,
            "evidence": evidence
        }
    
    def set_procedural_discussion(self, validity, defenses_analysis):
        """تعيين المناقشة الإجرائية"""
        self.section_3_legal_discussion["procedural_discussion"] = {
            "procedures_validity": validity,
            "formal_defenses_analysis": defenses_analysis
        }
    
    def set_substantive_discussion(self, classification, elements_analysis, 
                                   evidence_eval, defenses_analysis):
        """تعيين المناقشة الموضوعية"""
        self.section_3_legal_discussion["substantive_discussion"] = {
            "legal_classification": classification,
            "elements_analysis": elements_analysis,
            "evidence_evaluation": evidence_eval,
            "substantive_defenses_analysis": defenses_analysis
        }
    
    def add_cassation_principle(self, principle, source, application):
        """إضافة مبدأ من محكمة النقض"""
        self.section_4_cassation_principles["applied_principles"].append({
            "principle": principle,
            "source": source,
            "application": application
        })
    
    def set_conclusion(self, summary, uncertainties, verdict):
        """تعيين الخلاصة والمنطوق"""
        self.section_5_conclusion["summary"] = summary
        self.section_5_conclusion["uncertainties"] = uncertainties
        self.section_5_conclusion["verdict"] = verdict
    
    def add_deficiency(self, category, item):
        """إضافة ثغرة أو نقص"""
        if category in self.deficiencies_and_gaps:
            self.deficiencies_and_gaps[category].append(item)
    
    def to_dict(self):
        """تحويل المسودة إلى قاموس"""
        return {
            "case_id": self.case_id,
            "case_title": self.case_title,
            "section_1_facts": self.section_1_facts,
            "section_2_positions": self.section_2_positions,
            "section_3_legal_discussion": self.section_3_legal_discussion,
            "section_4_cassation_principles": self.section_4_cassation_principles,
            "section_5_conclusion": self.section_5_conclusion,
            "deficiencies_and_gaps": self.deficiencies_and_gaps
        }


def get_judge_prompt():
    """دالة مساعدة للحصول على prompt وكيل القاضي"""
    return JUDGE_AGENT_PROMPT


if __name__ == "__main__":
    print("=" * 80)
    print(JudgeAgentConfig.AGENT_NAME)
    print("=" * 80)
    print(f"الدور: {JudgeAgentConfig.AGENT_ROLE}")
    print(f"\nالمبدأ الأساسي:")
    print(f"  {JudgeAgentConfig.FUNDAMENTAL_PRINCIPLE}")
    print(f"\nأنواع الأحكام ({len(JudgeAgentConfig.VERDICT_TYPES)}):")
    for key, value in JudgeAgentConfig.VERDICT_TYPES.items():
        print(f"  • {key}: {value}")
    print(f"\nمراحل التسبيب ({len(JudgeAgentConfig.REASONING_PHASES)}):")
    for i, phase in enumerate(JudgeAgentConfig.REASONING_PHASES, 1):
        print(f"  {i}. {phase}")
