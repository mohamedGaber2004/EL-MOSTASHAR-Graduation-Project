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