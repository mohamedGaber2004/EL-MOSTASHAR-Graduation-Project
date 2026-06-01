# 🛠️ Development Guide

## Table of Contents
1. [Development Environment](#development-environment)
2. [Code Structure](#code-structure)
3. [Adding New Features](#adding-new-features)
4. [Testing](#testing)
5. [Code Style & Standards](#code-style--standards)
6. [Contributing](#contributing)
7. [Debugging](#debugging)

---

## Development Environment

### Setting Up Development Environment

```bash
# Clone and navigate to repository
git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git
cd EL-MOSTASHAR-Graduation-Project

# Create virtual environment with dev dependencies
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

### IDE Setup (VS Code)

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true
  }
}
```

---

## Code Structure

### Directory Organization

```
src/
├── agents/              # AI agents implementation
│   ├── base_agent.py
│   ├── data_agent.py
│   ├── analysis_agent.py
│   └── __init__.py
├── Chunking/            # Document chunking
│   ├── chunker.py
│   └── __init__.py
├── Config/              # Configuration management
│   ├── config.py
│   ├── log_config.py
│   └── __init__.py
├── Graph/               # Graph construction
│   ├── graph_builder.py
│   ├── state.py
│   └── __init__.py
├── Graphstore/          # Knowledge graph operations
│   ├── KG_builder.py
│   ├── kg_enums.py
│   └── __init__.py
├── LLMs/                # LLM provider configs
│   ├── providers.py
│   └── __init__.py
├── Prompts/             # Prompt templates
│   ├── templates.py
│   └── __init__.py
├── retriever/           # Retrieval systems
│   ├── kg_retriever/
│   └── vs_retriever/
├── routers/             # API endpoints
│   ├── case_router.py
│   ├── data_ingestion_router.py
│   └── ...
├── Vectorstore/         # Vector store operations
│   ├── vectorstore.py
│   └── __init__.py
├── Utils/               # Utility functions
│   ├── helpers.py
│   └── __init__.py
└── pipeline.py          # Processing pipelines
```

---

## Adding New Features

### Adding a New API Endpoint

1. **Create router file** in `src/routers/`:

```python
# src/routers/new_feature_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/new_feature", tags=["New Feature"])

class NewFeatureRequest(BaseModel):
    param1: str
    param2: int = 10

@router.post("/process")
async def process(request: NewFeatureRequest):
    """Process new feature request"""
    try:
        result = await perform_processing(request.param1, request.param2)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_processing(param1: str, param2: int):
    """Implementation details"""
    pass
```

2. **Register router** in `main.py`:

```python
from src.routers.new_feature_router import router as new_feature_router

app.include_router(new_feature_router)
```

### Adding a New Agent

1. **Create agent file** in `src/agents/`:

```python
# src/agents/new_agent.py
from src.agents.base_agent import BaseAgent

class NewAgent(BaseAgent):
    def __init__(self, name: str = "NewAgent"):
        super().__init__(name)
    
    async def execute(self, input_data: dict) -> dict:
        """Execute agent task"""
        # Implementation
        return {"status": "completed", "result": "..."}
```

2. **Integrate** with graph:

```python
# In src/Graph/graph_builder.py
from src.agents.new_agent import NewAgent

new_agent = NewAgent()
# Add to workflow graph
```

### Adding a New Data Processor

1. **Create processor** in `src/Chunking/`:

```python
# src/Chunking/new_processor.py
class NewProcessor:
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def process(self, data):
        """Process data"""
        # Implementation
        return processed_data
```

---

## Testing

### Unit Testing

```python
# tests/test_chunking.py
import pytest
from src.Chunking.chunker import DocumentChunker

class TestDocumentChunker:
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=512)
    
    def test_chunk_creation(self, chunker):
        text = "Sample text " * 100
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        assert all(len(c) <= 512 for c in chunks)
    
    def test_empty_input(self, chunker):
        chunks = chunker.chunk("")
        assert len(chunks) == 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_chunking.py

# Run with verbose output
pytest -v

# Run and show print statements
pytest -s
```

### Test Coverage Report

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Open report
open htmlcov/index.html
```

---

## Code Style & Standards

### Code Formatting with Black

```bash
# Format all Python files
black src/

# Check without formatting
black --check src/
```

### Linting with Flake8

```bash
# Check code style
flake8 src/

# Ignore specific rules
flake8 src/ --ignore=E501,W503
```

### Type Checking with MyPy

```bash
# Type check code
mypy src/

# With strict mode
mypy --strict src/
```

### Python Style Guide

```python
# ✓ Good
def process_document(doc: str, chunk_size: int = 512) -> List[str]:
    """
    Process document into chunks.
    
    Args:
        doc: Input document text
        chunk_size: Maximum chunk size
    
    Returns:
        List of document chunks
    """
    chunks = []
    # Implementation
    return chunks

# ✗ Bad
def processDocument(doc, chunk_size=512):
    # Process document
    chunks = []
    return chunks
```

---

## Contributing

### Contribution Workflow

1. **Fork and clone** the repository
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes** and commit: `git commit -m "Add new feature"`
4. **Push** to branch: `git push origin feature/new-feature`
5. **Create Pull Request**

### Commit Message Standards

```
# Good commit messages
[FEATURE] Add document ingestion endpoint
[FIX] Fix vector similarity calculation
[DOCS] Update API documentation
[REFACTOR] Reorganize chunking logic
[TEST] Add unit tests for retrieval

# Format
[TYPE] Brief description

Body (optional):
- Detailed explanation
- Implementation notes
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Functions have docstrings
- [ ] Tests pass (pytest)
- [ ] Code is tested (>80% coverage)
- [ ] Documentation is updated
- [ ] No hardcoded values or secrets
- [ ] Performance impact considered
- [ ] Backwards compatibility maintained

---

## Debugging

### Enabling Debug Mode

```bash
# Set debug flag
export APP_DEBUG=True

# Run with debug output
python main.py --debug

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Viewing Logs

```bash
# View application logs
tail -f logs/app.log

# Filter by level
tail -f logs/app.log | grep ERROR

# View Docker logs
docker-compose logs -f app
docker-compose logs -f neo4j
```

### Using Python Debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use breakpoint() (Python 3.7+)
breakpoint()

# Commands:
# n - Next line
# s - Step into
# c - Continue
# p variable_name - Print variable
```

### Debugging with VS Code

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Main",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

---

## Database Debugging

### Neo4j Cypher Queries

```bash
# Access Neo4j browser
# Open: http://localhost:7474/browser

# Test connection
:help

# Show databases
SHOW DATABASES;

# Query data
MATCH (n) RETURN COUNT(n);

# Clear data (careful!)
MATCH (n) DETACH DELETE n;
```

### Vector Store Debugging

```python
# Test FAISS
import faiss
import numpy as np

# Create index
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Add vectors
vectors = np.random.random((100, 384)).astype('float32')
index.add(vectors)

# Search
query = np.random.random((1, 384)).astype('float32')
distances, indices = index.search(query, 5)
```

---

## Performance Profiling

```python
# Use cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
perform_analysis()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Performance Optimization Tips

1. **Database Queries**: Use indices and limit results
2. **Vector Searches**: Pre-filter before similarity search
3. **Caching**: Cache frequently accessed data
4. **Async Processing**: Use async/await for I/O operations
5. **Batch Operations**: Process in batches when possible

---

**Happy coding! For questions, refer to the project documentation or reach out to the team.**
