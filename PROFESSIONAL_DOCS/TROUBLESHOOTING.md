# 🐛 Troubleshooting Guide

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Runtime Issues](#runtime-issues)
3. [Database Issues](#database-issues)
4. [API Issues](#api-issues)
5. [Performance Issues](#performance-issues)
6. [Data & Processing Issues](#data--processing-issues)
7. [Deployment Issues](#deployment-issues)

---

## Installation Issues

### Issue: Python Version Incompatibility

**Error**: `Python 3.10 is not supported`

**Solution**:
```bash
# Install Python 3.11 or 3.12
# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv

# macOS
brew install python@3.11

# Verify version
python3.11 --version

# Create virtual environment with correct version
python3.11 -m venv venv
```

---

### Issue: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'langchain'`

**Solution**:
```bash
# Reinstall all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or use uv (faster)
pip install uv
uv sync

# Verify installation
pip list | grep langchain
```

---

### Issue: CUDA/GPU Not Detected

**Error**: `CUDA is not available` (if using GPU)

**Solution**:
```bash
# For CPU-only (default)
pip install faiss-cpu

# For GPU support
pip install faiss-gpu

# Verify
python -c "import faiss; print(faiss.get_num_gpus())"
```

---

## Runtime Issues

### Issue: Port Already in Use

**Error**: `Address already in use: ('0.0.0.0', 8000)`

**Solution**:
```bash
# Find process using port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
python main.py --port 8001

# Or change in docker-compose.yml
# ports:
#   - "8001:8000"
```

---

### Issue: API Not Responding

**Error**: `ConnectionError: Failed to establish connection`

**Solution**:
```bash
# Check if API is running
ps aux | grep main.py

# Restart application
pkill -f main.py
python main.py

# Check for errors
tail -f logs/app.log

# Verify port
curl http://localhost:8000/
```

---

### Issue: Import Errors

**Error**: `ImportError: cannot import name 'X' from 'module'`

**Solution**:
```bash
# Update PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Or run from project root
cd /path/to/project
python main.py

# Check module locations
python -c "import sys; print('\n'.join(sys.path))"
```

---

## Database Issues

### Issue: Neo4j Connection Failed

**Error**: `Connection refused: ('localhost', 7687)`

**Solution**:
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker-compose restart neo4j

# View logs
docker-compose logs neo4j

# Test connection directly
cypher-shell -a bolt://localhost:7687 -u neo4j -p password

# Check port
lsof -i :7687
```

---

### Issue: Authentication Failed

**Error**: `401 Unauthorized` from database

**Solution**:
```bash
# Reset Neo4j password
docker-compose exec neo4j bin/neo4j-admin set-initial-password newpassword

# Update .env file
NEO4J_PASSWORD=newpassword

# Restart containers
docker-compose restart

# Verify credentials in config
cat src/Config/config.py | grep NEO4J
```

---

### Issue: Database Out of Memory

**Error**: `OutOfMemoryError: Java heap space`

**Solution**:
```bash
# Increase Neo4j memory in docker-compose.yml
environment:
  - NEO4J_dbms_memory_heap_maxSize=4G
  - NEO4J_dbms_memory_pagecache_size=2G

# Restart
docker-compose restart neo4j

# Clear old data if needed
docker-compose exec neo4j bin/neo4j-admin database drop neo4j
```

---

### Issue: Query Timeout

**Error**: `TimeoutError: Query took too long`

**Solution**:
```bash
# Add indexes for frequently queried fields
# In Neo4j Browser:
CREATE INDEX idx_case_id FOR (c:Case) ON (c.caseId);
CREATE INDEX idx_judge_id FOR (j:Judge) ON (j.judgeId);

# Verify indexes
SHOW INDEXES;

# Profile slow queries
PROFILE MATCH (c:Case) WHERE c.status = 'ACTIVE' RETURN c;
```

---

## API Issues

### Issue: 404 Not Found

**Error**: `404 Client Error: Not Found`

**Solution**:
```bash
# Check endpoint exists
curl http://localhost:8000/docs  # Interactive docs

# Verify correct URL
# Should be: http://localhost:8000/case/analyze
# Not: http://localhost:8000/analyze/case

# Check route registration in main.py
grep include_router main.py
```

---

### Issue: 422 Validation Error

**Error**: `422 Unprocessable Entity`

**Solution**:
```bash
# Check request format
# Correct:
curl -X POST http://localhost:8000/case/analyze \
  -H "Content-Type: application/json" \
  -d '{"case_id": "CASE_001"}'

# View validation errors in response
# Usually indicates missing or wrong type fields

# Check API docs for expected format
curl http://localhost:8000/docs
```

---

### Issue: 500 Internal Server Error

**Error**: `500 Internal Server Error`

**Solution**:
```bash
# Check application logs
tail -f logs/app.log

# Restart with debug mode
export APP_DEBUG=True
python main.py

# Check error stack trace
# Find the specific line causing issue

# Common causes:
# - Database not connected
# - LLM API key invalid
# - Missing configuration
```

---

## Performance Issues

### Issue: Slow API Response

**Error**: Response takes >1000ms

**Solution**:
```bash
# Profile the code
python -m cProfile -s cumulative main.py

# Check database query performance
# In Neo4j Browser: PROFILE [your_query]

# Increase workers
python main.py --workers 8

# Enable caching
# In code: use @cache decorator
```

---

### Issue: High Memory Usage

**Error**: Memory grows continuously

**Solution**:
```bash
# Monitor memory
watch -n 1 'free -h'
docker stats

# Check for memory leaks
# Use memory_profiler
pip install memory-profiler
python -m memory_profiler main.py

# Limit batch size in processing
# In config: BATCH_SIZE = 10 (instead of 100)

# Clear cache periodically
# Add: gc.collect() in long-running operations
```

---

### Issue: High CPU Usage

**Error**: CPU constantly at 100%

**Solution**:
```bash
# Check process
top -p $(pgrep -f main.py)

# Profile CPU
python -m cProfile main.py > stats.txt

# Optimize hot paths
# Usually: vector operations, graph queries

# Use vectorization
# Replace loops with numpy operations

# Increase worker threads
APP_WORKERS=8
```

---

## Data & Processing Issues

### Issue: Document Upload Fails

**Error**: `Failed to process document`

**Solution**:
```bash
# Check file format
file test_document.pdf

# Check file size (< 50MB recommended)
ls -lh test_document.pdf

# Verify encoding
chardet test_document.pdf

# Try smaller file first
# PDF might be corrupted

# Check logs for specifics
tail -f logs/app.log | grep ERROR
```

---

### Issue: Low Retrieval Quality

**Error**: Search results not relevant

**Solution**:
```bash
# Improve queries
# Instead of: "property"
# Use: "property ownership rights disputes"

# Adjust threshold
# Lower threshold for more results
# Higher threshold for quality

# Check embedding quality
# Verify vector embeddings are generated

# Rebuild indices
# Recreate FAISS indices with better configuration

# Add more training data
# Improve embeddings with domain-specific data
```

---

### Issue: Entity Extraction Not Working

**Error**: No entities extracted from documents

**Solution**:
```bash
# Verify LLM API keys
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Check prompt templates
cat src/Prompts/templates.py

# Test extraction with simple text
python -c "
from src.agents.analysis_agent import AnalysisAgent
agent = AnalysisAgent()
result = agent.extract_entities('Simple test text')
print(result)
"

# Check if LLM provider is working
# Test with direct API call first
```

---

## Deployment Issues

### Issue: Container Won't Start

**Error**: `docker-compose up` fails

**Solution**:
```bash
# Check image exists
docker images | grep el-mostashar

# View detailed error
docker-compose up (without -d)

# Check configuration
docker-compose config

# Verify environment variables
cat .env

# Check logs
docker-compose logs app
```

---

### Issue: Services Can't Communicate

**Error**: `Connection refused` between containers

**Solution**:
```bash
# Check network
docker network ls
docker network inspect el-mostashar-prod

# Use service names (not localhost)
# In container: NEO4J_URI=bolt://neo4j:7687
# Not: NEO4J_URI=bolt://localhost:7687

# Verify services are running
docker-compose ps

# Check service logs
docker-compose logs neo4j
```

---

### Issue: Persistent Data Lost After Restart

**Error**: Data disappears when container restarts

**Solution**:
```bash
# Check volumes in docker-compose.yml
volumes:
  - /data/neo4j:/var/lib/neo4j/data
  - /data/faiss:/app/faiss_indices

# Create host directories
mkdir -p /data/neo4j /data/faiss

# Set permissions
chmod 755 /data/neo4j /data/faiss

# Verify volumes
docker volume ls
docker inspect el-mostashar_neo4j-data
```

---

## Advanced Debugging

### Enable Debug Logging

```python
# In src/Config/config.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Interactive Debugging

```bash
# Use pdb
python -m pdb main.py

# Commands
# n - next line
# s - step into function
# c - continue
# p variable_name - print variable
# l - list code
# h - help
```

### Remote Debugging

```python
# For VS Code
# Add to .vscode/launch.json
{
  "configurations": [
    {
      "name": "Python: Remote Debug",
      "type": "python",
      "request": "attach",
      "port": 5678,
      "host": "localhost"
    }
  ]
}

# Start debug server
python -m debugpy --listen 5678 main.py
```

---

## Getting Help

### Diagnostic Information

```bash
#!/bin/bash
# Save diagnostic info

echo "=== System Info ===" > diagnostics.txt
uname -a >> diagnostics.txt

echo "\n=== Python ===" >> diagnostics.txt
python --version >> diagnostics.txt
pip list >> diagnostics.txt

echo "\n=== Docker ===" >> diagnostics.txt
docker version >> diagnostics.txt
docker-compose ps >> diagnostics.txt

echo "\n=== Logs ===" >> diagnostics.txt
docker-compose logs >> diagnostics.txt

echo "\n=== Environment ===" >> diagnostics.txt
env | grep -E "OPENAI|GOOGLE|GROQ|NEO4J" >> diagnostics.txt

# Share safely
cat diagnostics.txt
```

### Support Resources

- **GitHub Issues**: Report bugs
- **Documentation**: Check all .md files
- **Logs**: Check `logs/` directory
- **API Docs**: Visit `/docs` endpoint

---

**If issues persist, collect diagnostic information and check the logs carefully for specific error messages.**
