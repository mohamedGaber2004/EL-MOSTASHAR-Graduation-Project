from src.Chunking.chunking import get_chunks
from src.Graphstore.KG_builder import run_KG
from src.Vectorstore.vector_store_builder import get_retriever



docs = get_chunks()
kg = run_KG()
ret = get_retriever()

