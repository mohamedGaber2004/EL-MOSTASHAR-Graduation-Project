from __future__ import annotations

from enum import Enum


class IndexName(Enum):
    ARTICLES     = "article_embeddings"
    TABLE_CHUNKS = "table_chunk_embeddings"


class SimilarityFunction(Enum):
    COSINE = "cosine"


class EmbedNodeLabel(Enum):
    ARTICLE     = "Article"
    TABLE_CHUNK = "TableChunk"


class EmbedIdProperty(Enum):
    ARTICLE_ID = "article_id"
    CHUNK_ID   = "chunk_id"


# =============================================================================
# Retrieval helper enum
# =============================================================================

class LR_Enums(Enum):
    """
    Misc. retrieval constants/helpers: the raw ANN Cypher query for article
    embeddings, and the Arabic law-name → law_id mapping used as a fallback
    by LegalRetriever._resolve_law_id() when a name isn't found among the
    live :Law.title values pulled from the graph.
    """

    _CHARGE_ANN_QUERY = f"""
        CALL db.index.vector.queryNodes('{IndexName.ARTICLES.value}', $k, $query_vector)
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
            "الدستور الجنائي":              "criminal_constitution",
        }
        return mapping.get(law_code, law_code)