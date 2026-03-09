from enum import Enum
import re
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
    ADDITION_KEYWORDS = ["تضاف","يضاف","أضيف","إضافة","جديدة","جديد"]
    DELETION_KEYWORDS = ["يلغى","ألغي","يحذف","حذف","إلغاء"]

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

    FOLDER_TO_LAW_KEY: Dict[str, str] = {
        "3okobat":         "penal_code",
        "8asl_amwal":      "money_laundering",
        "Asle7a":          "weapons_ammunition",
        "dostor_gena2y":   "criminal_constitution",
        "drugs":           "anti_drugs",
        "egra2at_gena2ya": "criminal_procedure",
        "erhab":           "anti_terror",
        "taware2":         "emergency_law",
        "technology":      "cybercrime",
    }

    LAW_KEY_TO_TITLE: Dict[str, str] = {
        "penal_code":            "قانون العقوبات",
        "money_laundering":      "قانون مكافحة غسل الأموال",
        "weapons_ammunition":    "قانون الأسلحة والذخيرة",
        "criminal_constitution": "الدستور الجنائي",
        "anti_drugs":            "قانون مكافحة المخدرات",
        "criminal_procedure":    "قانون الإجراءات الجنائية",
        "anti_terror":           "قانون مكافحة الإرهاب",
        "emergency_law":         "قانون الطوارئ",
        "cybercrime":            "قانون مكافحة جرائم تقنية المعلومات",
    }

    BOOK_TITLE_MAP: Dict[str, str] = {
        "mbade2_ultra_clean"        :"مبادئ محكمة النقض",
        "1762919079675_ultra_clean" : "مبادئ محكمة النقض - المجلد الثاني",
        "2019-2018_المستحدث"        :"المستحدث من المبادئ 2018-2019",
    }

    CREATION_OF_SCHEMA = [
                "CREATE CONSTRAINT law_id_unique        IF NOT EXISTS FOR (l:Law)          REQUIRE l.law_id IS UNIQUE",
                "CREATE CONSTRAINT article_id_unique    IF NOT EXISTS FOR (a:Article)       REQUIRE a.article_id IS UNIQUE",
                "CREATE CONSTRAINT penalty_id_unique    IF NOT EXISTS FOR (p:Penalty)       REQUIRE p.penalty_id IS UNIQUE",
                "CREATE CONSTRAINT definition_id_unique IF NOT EXISTS FOR (d:Definition)    REQUIRE d.definition_id IS UNIQUE",
                "CREATE CONSTRAINT topic_id_unique      IF NOT EXISTS FOR (t:Topic)         REQUIRE t.topic_id IS UNIQUE",
                "CREATE CONSTRAINT schedule_id_unique   IF NOT EXISTS FOR (s:Schedule)      REQUIRE s.schedule_id IS UNIQUE",
                "CREATE CONSTRAINT entry_id_unique      IF NOT EXISTS FOR (e:ScheduleEntry) REQUIRE e.entry_id IS UNIQUE",
                "CREATE CONSTRAINT substance_id_unique  IF NOT EXISTS FOR (sub:Substance)   REQUIRE sub.substance_id IS UNIQUE",
                "CREATE CONSTRAINT amendment_id_unique  IF NOT EXISTS FOR (am:Amendment)    REQUIRE am.amendment_id IS UNIQUE",
                "CREATE INDEX supersedes_date_idx IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.amendment_date)",
            ]

    NODE_LABELS = ["Law","Article","Penalty","Definition","Topic","Schedule","ScheduleEntry","Substance","Amendment"]
    RELATIONSHIPS = ["CONTAINS","REFERENCES","HAS_PENALTY","DEFINES","TAGGED_WITH","HAS_SCHEDULE","CONTAINS_ENTRY","DEFINES_SUBSTANCE","HAS_AMENDMENT","AMENDED_BY","SUPERSEDES"]

