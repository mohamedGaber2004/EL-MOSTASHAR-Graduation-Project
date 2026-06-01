# 📋 Egyptian Legal Multi-Agent System (EL-MOSTASHAR)

## Executive Summary

The **Egyptian Legal Multi-Agent System** (EL-MOSTASHAR) is an AI-powered platform designed to automate legal case analysis and decision-making in the Egyptian judicial system. Built with LangChain, LangGraph, Neo4j, and multiple LLM providers, this system provides comprehensive multi-agent legal analysis, hybrid information retrieval, and case intelligence.

---

## 🎯 Core Purpose

This system provides:
- **Multi-Agent Case Processing**: 10 specialized agents for different aspects of legal analysis
- **Intelligent Retrieval**: Hybrid retrieval combining knowledge graph, semantic search, and lexical search
- **Case Invocation**: Single endpoint to process legal cases with document analysis
- **Production-Ready APIs**: RESTful endpoints for integration with external systems

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Case Invocation** | Submit cases with documents and get comprehensive analysis results |
| **Multi-Agent Pipeline** | 10 specialized agents working together: data ingestion, procedural auditing, legal research, evidence analysis, defense analysis, confession validity, witness credibility, prosecution analysis, sentencing, and judgment |
| **Knowledge Graph Retrieval** | Hybrid retrieval combining Neo4j vector search + BM25 ranking + graph expansion |
| **Vector Store Retrieval** | FAISS dense vectors with BM25 sparse search and RRF fusion for optimal results |
| **Intelligent Agents** | Data Ingestion, Procedural Auditor, Legal Researcher, Evidence Analyst, Defense Analyst, Confession Validity, Witness Credibility, Prosecution Analyst, Sentencing, Judge |
| **Document Chunking** | Advanced text preprocessing and segmentation for Arabic legal documents |
| **Multi-Model Support** | Integration with OpenAI, Google Gemini, Groq, Cerebras, Mistral, Cohere |
| **Production-Ready** | Docker containerization, environment configuration, comprehensive logging |

---

## 📊 Technology Stack

### Backend
- **FastAPI** - Async REST API framework
- **Python** (3.11+) - Core language
- **Uvicorn** - ASGI application server

### AI & Orchestration
- **LangChain** - LLM framework and orchestration
- **LangGraph** - Multi-agent workflow execution
- **Sentence-Transformers** - Arabic semantic embeddings

### Data & Storage
- **Neo4j** - Knowledge graph database for legal documents
- **FAISS** - Vector similarity search (CPU)
- **BM25** - Hybrid ranking algorithm

### LLM Providers
- OpenAI GPT models
- Google Gemini
- Groq LPU
- Cerebras
- Mistral AI
- Cohere

### DevOps
- **Docker** & **Docker Compose** - Containerization
- **Python-dotenv** - Environment management

---

## 🏗️ Project Structure

```
src/
├── agents/                      # 10 specialized agents
│   ├── data_ingestion_agent/
│   ├── procedural_auditor_agent/
│   ├── legal_research_agent/
│   ├── evidence_analyst_agent/
│   ├── defense_analyst_agent/
│   ├── confessoin_validity_agent/
│   ├── witness_credibility_agent/
│   ├── prosecution_analyst_agent/
│   ├── sentencing_agent/
│   └── judge_agent/
├── routers/                     # API endpoints
│   ├── case_router.py           # Case invocation
│   ├── kg_retriever_router.py   # KG retrieval
│   ├── vs_retriever_router.py   # Vector store retrieval
│   ├── kg_router.py             # KG management
│   ├── vs_router.py             # Vector store management
│   ├── data_ingestion_router.py # Data ingestion
│   └── chunking_router.py       # Chunking services
├── Graph/                       # Multi-agent graph builder
│   ├── graph_builder.py         # LangGraph construction
│   └── state.py                 # AgentState definition
├── retriever/                   # Retrieval systems
│   ├── kg_retriever/            # Knowledge graph retriever
│   └── vs_retriever/            # Vector store retriever
├── Graphstore/                  # Neo4j graph builder
├── Vectorstore/                 # FAISS index builder
├── Chunking/                    # Document preprocessing
├── Config/                      # Configuration & logging
└── LLMs/                        # LLM provider configs
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Neo4j database
- LLM API keys (OpenAI, Google, etc.)

### Installation

```bash
# Clone repository
git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git
cd EL-MOSTASHAR-Graduation-Project

# Setup environment
cp .env.example .env
# Edit .env with your API keys and database credentials

# Install dependencies
pip install -r requirements.txt
# OR using uv
uv sync

# Start the application
python main.py

# Access API docs
# Interactive: http://127.0.0.1:8000/docs
# ReDoc: http://127.0.0.1:8000/redoc
```

### Docker Setup

```bash
docker-compose up -d
```

---

## 🔌 Core API Endpoints

### Case Invocation (Main Entry Point)

**POST** `/cases/invoke_case`
- Submit case documents and receive comprehensive AI analysis
- Parameters: `case_id`, `files`
- Returns: Full case analysis with all agent outputs

### Knowledge Graph Retrieval APIs

**POST** `/kg/retriever/retrieve` - Hybrid legal retrieval
- Query parameters: `question`, `k` (result count), `threshold`
- Returns: Relevant articles with sources

**POST** `/kg/retriever/index/setup` - Create vector indexes
**POST** `/kg/retriever/index/embed` - Embed nodes
**POST** `/kg/retriever/index/reindex` - Re-embed all articles
**POST** `/kg/retriever/index/rebuild` - Full index rebuild

### Vector Store Retrieval APIs

**POST** `/vs/retriever/dense` - Semantic search via FAISS
**POST** `/vs/retriever/sparse` - Lexical search via BM25
**POST** `/vs/retriever/hybrid` - RRF fusion of dense + sparse

For complete API reference, see [PROFESSIONAL_DOCS/API_REFERENCE.md](./PROFESSIONAL_DOCS/API_REFERENCE.md) or visit `/docs`

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📞 Contact

**Email**: mg7357432@gmail.com

---

**Last Updated**: June 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅
