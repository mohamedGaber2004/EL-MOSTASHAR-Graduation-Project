from enum import Enum
from src.Utils import VerdictType


class AgentsEnums (Enum):
    AGENT_INVOKATION_ERRORS  = ("429", "rate_limit", "rate_limit_exceeded", "tokens per minute", "please try again")
    AGENT_INGESTION_TXT_FILES_ENCODING = ("utf-8", "cp1256", "latin-1")

    CRIMINAL_PROCEDURE_LAW_ID = "egyptian_criminal_procedure"

    PROCEDURAL_ARTICLE_NUMBERS = [
    "41",   # الدستور — اشتراطات القبض
    "45",   # الدستور — اشتراطات التفتيش
    "54",   # الإجراءات — القبض بغير إذن قضائي
    "55",   # الإجراءات — مدة الاحتجاز قبل العرض على النيابة
    "91",   # الإجراءات — الضبط والإحضار
    "92",   # الإجراءات — الحبس الاحتياطي
    "135",  # الإجراءات — الاستجواب وإخطار النيابة خلال 24 ساعة
    "136",  # الإجراءات — حق المتهم في الاستعانة بمحامٍ
    "302",  # الإجراءات — الاعتراف وشروط صحته
    "333",  # الإجراءات — البطلان وأثره
    ]

    VERDICT_MAP = {
        "إدانة":         VerdictType.CONVICTION,
        "براءة":         VerdictType.ACQUITTAL,
        "إدانة جزئية":  VerdictType.PARTIAL_GUILTY,
        "guilty":        VerdictType.CONVICTION,
        "acquittal":     VerdictType.ACQUITTAL,
        "partial_guilty": VerdictType.PARTIAL_GUILTY,
    }