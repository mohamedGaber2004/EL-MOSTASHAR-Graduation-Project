# 👨‍💻 Development Guide

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- Code editor (VS Code recommended)
- Neo4j instance

### Clone & Setup
```bash
git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git
cd EL-MOSTASHAR-Graduation-Project

python3.11 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## Project Structure

```
src/
├── agents/                    # 10 specialized agents
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
├── routers/                   # API endpoints
│   ├── case_router.py
│   ├── kg_retriever_router.py
│   ├── vs_retriever_router.py
│   ├── kg_router.py
│   ├── vs_router.py
│   ├── data_ingestion_router.py
│   └── chunking_router.py
├── Graph/
│   ├── graph_builder.py       # LangGraph pipeline
│   └── state.py               # AgentState
├── retriever/
│   ├── kg_retriever/
│   └── vs_retriever/
├── Graphstore/                # Neo4j builder
├── Vectorstore/               # FAISS builder
├── Chunking/                  # Document processing
├── Config/                    # Configuration
└── LLMs/                      # LLM models
```

---

## Adding a New Agent

### 1. Create Agent Structure
```bash
mkdir -p src/agents/my_new_agent
touch src/agents/my_new_agent/{my_agent.py,my_agent_prompt.py,my_agent_output_model.py}
```

### 2. Define Output Model
```python
# src/agents/my_new_agent/my_agent_output_model.py
from pydantic import BaseModel

class MyAgentOutput(BaseModel):
    analysis_result: str
    confidence_score: float
    recommendations: list[str]
```

### 3. Create Agent Class
```python
# src/agents/my_new_agent/my_agent.py
from src.agents.agent_base.agent_base import AgentBase
from .my_agent_output_model import MyAgentOutput

class MyNewAgent(AgentBase):
    def __init__(self):
        super().__init__()
        self.agent_type = "my_new_agent"
    
    def run(self, state):
        # Implement agent logic
        result = MyAgentOutput(
            analysis_result="...",
            confidence_score=0.95,
            recommendations=[...]
        )
        state.my_new_agent_output = result
        return state
```

### 4. Register in Graph Builder
```python
# src/Graph/graph_builder.py
from src.agents.my_new_agent.my_agent import MyNewAgent

agents = {
    ...
    "my_new_agent": MyNewAgent(),
}

# Add edges
builder.add_edge("previous_agent", "my_new_agent")
builder.add_edge("my_new_agent", "next_agent")
```

---

## Adding a New Endpoint

### 1. Create Router
```python
# src/routers/my_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

my_router = APIRouter(prefix="/my_feature", tags=["My Feature"])

class MyRequest(BaseModel):
    query: str

class MyResponse(BaseModel):
    result: str

@my_router.post("/endpoint")
async def my_endpoint(req: MyRequest) -> MyResponse:
    try:
        result = "process: " + req.query
        return MyResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Register Router
```python
# main.py
from src.routers.my_router import my_router

app.include_router(my_router)
```

---

## Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=src
```

### Example Test
```python
# tests/test_agents.py
def test_data_ingestion_agent():
    from src.agents.data_ingestion_agent.data_ingestion_agent import DataIngestionAgent
    agent = DataIngestionAgent()
    # Test implementation
```

---

## Code Standards

### Style Guide
- Follow PEP 8
- Use type hints
- Max line length: 100 characters
- Docstrings for all functions

### Example
```python
def process_document(content: str, language: str = "ar") -> dict[str, any]:
    """
    Process a legal document.
    
    Args:
        content: Document text content
        language: Language code (default: "ar" for Arabic)
    
    Returns:
        Processed document data
    
    Raises:
        ValueError: If content is empty
    """
    if not content:
        raise ValueError("Content cannot be empty")
    
    # Implementation
    return {}
```

---

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Debugger
```python
import pdb; pdb.set_trace()  # Breakpoint
```

### Print LangGraph State
```python
from src.Graph.state import AgentState
state = AgentState(...)
print(state.model_dump())
```

---

## Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(param: str) -> str:
    # Cached results
    return compute(param)
```

### Async Operations
```python
import asyncio

async def process_batch(items: list):
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_func, items))
```

---

## Database Operations

### Neo4j Queries
```python
from src.Graphstore.KG_builder import LegalKnowledgeGraph

graph = LegalKnowledgeGraph(...)
stats = graph.get_statistics()
```

### FAISS Operations
```python
from src.Vectorstore.vector_store_builder import load_vector_store

vs = load_vector_store("na2d_faiss_index", embeddings)
results = vs.similarity_search("query", k=5)
```

---

## Logging

### Configure Logging
```python
# src/Config/log_config.py
import logging

logger = logging.getLogger(__name__)
logger.info("Application started")
```

### Log Levels
- DEBUG: Detailed info for debugging
- INFO: General informational messages
- WARNING: Warning messages
- ERROR: Error messages
- CRITICAL: Critical failures

---

## Version Control

### Commit Message Format
```
[TYPE] Brief description

Detailed explanation if needed

- Bullet point 1
- Bullet point 2

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Refactoring

---

## Documentation

### Code Documentation
- Add docstrings to all public functions
- Use type hints
- Include usage examples

### Update Docs
- Edit relevant .md files in PROFESSIONAL_DOCS/
- Keep API_REFERENCE.md in sync
- Update ARCHITECTURE.md for major changes

---

## Common Development Tasks

### Add LLM Provider
```python
# src/LLMs/NEW_PROVIDER_MODEL.py
from langchain_community.llms import NewProvider

class NewProviderModel:
    def __init__(self):
        self.model = NewProvider(api_key="...")
```

### Update Prompt
```python
# src/agents/agent_name/agent_name_prompt.py
AGENT_PROMPT = """
Your prompt here...
"""
```

### Modify State
```python
# src/Graph/state.py
class AgentState(TypedDict):
    new_field: str
    new_field_value: list[str]
```

---

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Neo4j Errors
```bash
# Check Neo4j connection
docker ps  # Verify container running
docker logs neo4j  # Check logs
```

### FAISS Errors
```bash
# Rebuild index
curl -X POST http://localhost:8000/vs/build
```

---

## Contributing

1. Create feature branch
2. Make changes
3. Add/update tests
4. Update documentation
5. Commit with descriptive message
6. Push to branch
7. Create pull request

---

**Last Updated**: June 2024  
**Version**: 1.0.0
