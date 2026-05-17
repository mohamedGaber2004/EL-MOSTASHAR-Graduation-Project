from enum import Enum
from typing import (
    Any , 
    Dict ,  
    Optional , 
    Tuple
)

class norm_regu(Enum) : 
    TOPICS_MAP = {
        "قتل":    ["قتل", "قاتل"],
        "سرقة":   ["سرقة", "سارق"],
        "مخدرات": ["مخدر", "مخدرات"],
        "أسلحة":  ["سلاح", "أسلحة"],
        "إرهاب":  ["إرهاب", "إرهابي"],
    }
    NORMALIZING_ARABIC_LITTERS = [("أ","ا"),("إ","ا"),("آ","ا"),("ى","ي"),("ئ","ي"),("ة","ه"),("ؤ","و")]
    
    DELETION_KEYWORDS = ["يلغى","ألغي","يحذف","حذف","إلغاء"]
    ADDITION_KEYWORDS = ["تضاف","يضاف","أضيف","إضافة","جديدة","جديد"]
    
    _LAW_REGISTRY: Dict[str, Tuple[str, Optional[str], Optional[Any]]] = {
        "penal_code":            ("قانون العقوبات",                          "1937-07-31", None),
        "money_laundering":      ("قانون مكافحة غسل الأموال",               "2002-05-15", None),
        "weapons_ammunition":    ("قانون الأسلحة والذخيرة",                 "1954-09-22", "weapon"),
        "criminal_constitution": ("الدستور الجنائي",                        None,         None),
        "anti_drugs":            ("قانون مكافحة المخدرات",                  "1960-05-01", "drug"),
        "criminal_procedure":    ("قانون الإجراءات الجنائية",               "1950-10-18", None),
        "anti_terror":           ("قانون مكافحة الإرهاب",                   "2015-08-16", None),
        "emergency_law":         ("قانون الطوارئ",                          "1958-01-01", None),
        "cybercrime":            ("قانون مكافحة جرائم تقنية المعلومات",     "2018-07-17", None),
    }

