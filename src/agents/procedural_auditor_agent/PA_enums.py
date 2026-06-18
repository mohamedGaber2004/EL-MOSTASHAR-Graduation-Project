from enum import Enum


class PA_Enums(Enum):
    _PROCEDURAL_LAW_ANN_QUERY = """
        CALL db.index.vector.queryNodes('article_embeddings', $k, $query_vector)
        YIELD node AS article, score
        WHERE article.law_id = $law_id
        RETURN
            article.article_id     AS article_id,
            article.article_number AS article_number,
            article.law_id         AS law_id,
            article.text           AS text,
            score
        ORDER BY score DESC
    """

    CRIMINAL_PROCEDURE_LAW_ID = "criminal_procedure"

class PipelineFatalityLevel(str, Enum):
    NONE     = "none"       # الإجراءات سليمة أو فيها عيوب قابلة للتصحيح
    PARTIAL  = "partial"    # بطلان مطلق في دليل معين — يُستبعد الدليل ويكمل
    FATAL    = "fatal"      # الدعوى ساقطة أو المحكمة غير مختصة — يذهب للقاضي مباشرة