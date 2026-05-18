from enum import Enum

class LR_Enums(Enum):
    _CHARGE_ANN_QUERY = """
        CALL db.index.vector.queryNodes('article_embeddings', $k, $query_vector)
        YIELD node AS article, score
        RETURN
            article.article_id     AS article_id,
            article.article_number AS article_number,
            article.law_id         AS law_id,
            article.text           AS text,
            score
        ORDER BY score DESC
    """

    @staticmethod
    def law_code_to_kg_id(law_code: str) -> str:
        mapping = {
            "قانون العقوبات":               "penal_code",
            "قانون الإجراءات الجنائية":     "criminal_procedure",
            "قانون مكافحة المخدرات":        "anti_drugs",
            "قانون مكافحة الإرهاب":         "anti_terror",
            "قانون الأسلحة والذخائر":       "weapons_ammunition",
            "قانون مكافحة جرائم المعلومات": "cybercrime",
            "قانون مكافحة غسيل الأموال":    "money_laundering",
            "قانون الطوارئ":                "emergency_law",
            "الدستور الجنائي":              "criminal_constitution"
        }
        return mapping.get(law_code, law_code)