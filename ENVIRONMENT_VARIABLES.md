# Environment Variables Documentation

## gateway/.env
- ORCHESTRATOR_URL: URL for orchestrator service (default: http://orchestrator:8000)
- INGESTION_SERVICE_URL: URL for ingestion service (default: http://ingestion-service:8004)

## orchestrator/.env
- LLM_SERVICE_URL: URL for LLM service (default: http://llm-service:8001)
- RETRIEVAL_SERVICE_URL: URL for retrieval service (default: http://retrieval-service:8002)
- KG_SERVICE_URL: URL for knowledge graph service (default: http://kg-service:8003)
- INGESTION_SERVICE_URL: URL for ingestion service (default: http://ingestion-service:8004)

## llm_service/.env
- GROQ_API_KEY: API key for Groq LLM
- HUGGINGFACE_API_TOKEN: API token for HuggingFace
- REASONING_MODEL: Model name for reasoning
- REASONING_MODEL_TEMP: Temperature for reasoning model
- ARABIC_MODEL: Model name for Arabic
- ARABIC_MODEL_TEMP: Temperature for Arabic model
- ASSISTANT_MODEL: Model name for assistant
- ASSISTANT_MODEL_TEMP: Temperature for assistant model

## retrieval_service/.env
- HUGGINGFACE_API_TOKEN: API token for HuggingFace
- EMBEDDING_MODEL: Embedding model name
- FAISS_INDEX_PATH: Path to FAISS index inside container

## kg_service/.env
- NEO4J_URI: URI for Neo4j database
- NEO4J_USERNAME: Username for Neo4j
- NEO4J_PASSWORD: Password for Neo4j

## ingestion_service/.env
- DataPath: Path to unstructured data
- na2d_data_path: Path to na2d data

> **Note:** Never commit secrets (API keys, passwords) to version control. Use environment variables for all sensitive configuration.
