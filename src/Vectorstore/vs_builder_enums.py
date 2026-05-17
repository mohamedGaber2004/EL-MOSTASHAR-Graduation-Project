from enum import Enum

class ChunkType(str, Enum):
    """Granularity of a document chunk stored in the index."""
    ARTICLE   = "article"
    PARAGRAPH = "paragraph"
    SENTENCE  = "sentence"
    TABLE     = "table"
    HEADER    = "header"


class DocType(str, Enum):
    """High-level document category."""
    LAW        = "law"
    REGULATION = "regulation"
    JUDGMENT   = "judgment"
    CONTRACT   = "contract"
    OTHER      = "other"