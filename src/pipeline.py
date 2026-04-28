from __future__ import annotations

from src.Config.log_config import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.Chunking.chunking import get_chunks, get_na2d_chunks
from src.Vectorstore.vector_store_builder import build_vector_store, load_vector_store
from src.Config import get_settings
from src.Graph.state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    data_path: Path
    na2d_data_path: Path
    faiss_index_path: Path
    embedding_model: str
    # Optional placeholders for other services (Neo4j, LLM keys)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None


class PipelineManager:
    """Lightweight pipeline manager with lazy initialization of heavy components."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._components: Dict[str, Any] = {}
        self._initialized = False
        logger.debug("PipelineManager initialized with config: %s", config)

    def init_components(self) -> None:
        if self._initialized:
            return

        logger.info("Pipeline loading chunks...")
        law_chunks = get_chunks()
        na2d_chunks = get_na2d_chunks()

        all_chunks = law_chunks + na2d_chunks.get("rulings", []) + na2d_chunks.get("principles", [])
        logger.debug("Loaded %d total chunks for vectorstore", len(all_chunks))

        index_path = Path(self.config.faiss_index_path)
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)

        if index_path.exists():
            logger.info("Loading existing FAISS index from %s", index_path)
            vs = load_vector_store(str(index_path), embeddings)
        else:
            logger.info("Building FAISS index (this may take a while)...")
            vs = build_vector_store(all_chunks, embeddings, index_path=str(index_path))

        self._components["vectorstore"] = vs
        self._components["law_chunks"] = law_chunks
        self._components["na2d_chunks"] = na2d_chunks

        self._initialized = True
        logger.info("Pipeline components initialization complete.")

    def get_vectorstore(self) -> FAISS:
        if not self._initialized:
            logger.debug("get_vectorstore called but not initialized. Initializing now...")
            self.init_components()
        return self._components["vectorstore"]

    def run_case(self, state: AgentState) -> AgentState:
        """Run a case through the graph. This method currently delegates to Graph modules.

        Note: Graph builder modules may have import-time side effects; they should be refactored
        to avoid display/IO at import time. See repo TODOs.
        """
        logger.info("Starting run_case for case_id: %s", state.case_id)
        # Lazy initialize vectorstore (and chunks) to ensure resources are ready
        self.init_components()

        # The actual LangGraph orchestration lives in src.Graph.graph_builder
        logger.debug("Importing build_legal_graph and invoking graph...")
        from src.Graph.graph_builder import build_legal_graph

        graph = build_legal_graph()
        final_state = graph.invoke({
            "case_id": state.case_id,
            "source_documents": state.source_documents,
        })

        logger.info("run_case completed for case_id: %s", state.case_id)
        return AgentState(**final_state)


def default_pipeline_manager() -> PipelineManager:
    settings = get_settings()
    cfg = PipelineConfig(
        data_path=Path(settings.DataPath),
        na2d_data_path=Path(getattr(settings, 'na2d_data_path', getattr(settings, 'NA2DPath', ''))),
        faiss_index_path=Path(getattr(settings, 'FAISS_INDEX_PATH', 'legal_faiss_index')),
        embedding_model=getattr(settings, 'EMBEDDING_MODEL', None),
        neo4j_uri=getattr(settings, 'NEO4J_URI', None),
        neo4j_user=getattr(settings, 'NEO4J_USER', None),
        neo4j_password=getattr(settings, 'NEO4J_PASSWORD', None),
    )
    logger.debug("Default PipelineManager setup complete.")
    return PipelineManager(cfg)
