from enum import Enum


class NodeLabel(Enum):
    LAW         = "Law"
    ARTICLE     = "Article"
    PENALTY     = "Penalty"
    DEFINITION  = "Definition"
    TOPIC       = "Topic"
    AMENDMENT   = "Amendment"
    TABLE       = "Table"
    TABLE_CHUNK = "TableChunk"

    NODE_LABELS = ["Law","Article","Penalty","Definition","Topic","Table","Substance","Amendment"]
    RELATIONSHIPS = ["CONTAINS","REFERENCES","HAS_PENALTY","DEFINES","TAGGED_WITH","HAS_TABLE","DEFINES_SUBSTANCE","HAS_AMENDMENT","AMENDED_BY","SUPERSEDES"]


class RelType(str, Enum):
    """
    Every Neo4j relationship type created or queried by :class:`LegalKnowledgeGraph`.
    """
    # Law → Article / Table / Amendment
    CONTAINS      = "CONTAINS"
    HAS_TABLE     = "HAS_TABLE"
    HAS_AMENDMENT = "HAS_AMENDMENT"

    # Article relationships
    HAS_PENALTY   = "HAS_PENALTY"
    DEFINES       = "DEFINES"
    TAGGED_WITH   = "TAGGED_WITH"
    REFERENCES    = "REFERENCES"
    AMENDED_BY    = "AMENDED_BY"
    SUPERSEDES    = "SUPERSEDES"

    # Table → TableChunk
    HAS_CHUNK     = "HAS_CHUNK"


class ArticleTag(str, Enum):
    """
    Version-tag suffixes appended to stable article IDs when creating
    amended / added article versions.
    """
    ADDED    = "added"
    AMENDED  = "amended"
    PREAMBLE = "preamble"


class ReferenceType(str, Enum):
    """Annotation for cross-article :attr:`RelType.REFERENCES` edges."""
    DIRECT = "direct"


class Language(str, Enum):
    """ISO language codes used in node properties."""
    ARABIC = "ar"

class KgSchemas(Enum):
    CREATION_OF_SCHEMA = [
        "CREATE CONSTRAINT law_id_unique        IF NOT EXISTS FOR (l:Law)       REQUIRE l.law_id IS UNIQUE",
        "CREATE CONSTRAINT article_id_unique    IF NOT EXISTS FOR (a:Article)    REQUIRE a.article_id IS UNIQUE",
        "CREATE CONSTRAINT penalty_id_unique    IF NOT EXISTS FOR (p:Penalty)    REQUIRE p.penalty_id IS UNIQUE",
        "CREATE CONSTRAINT definition_id_unique IF NOT EXISTS FOR (d:Definition) REQUIRE d.definition_id IS UNIQUE",
        "CREATE CONSTRAINT topic_id_unique      IF NOT EXISTS FOR (t:Topic)      REQUIRE t.topic_id IS UNIQUE",
        "CREATE CONSTRAINT table_id_unique      IF NOT EXISTS FOR (t:Table)      REQUIRE t.table_id IS UNIQUE",
        "CREATE CONSTRAINT substance_id_unique  IF NOT EXISTS FOR (s:Substance)  REQUIRE s.substance_id IS UNIQUE",
        "CREATE CONSTRAINT amendment_id_unique  IF NOT EXISTS FOR (am:Amendment) REQUIRE am.amendment_id IS UNIQUE",
        "CREATE INDEX supersedes_date_idx IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.amendment_date)",
    ]