# 📦 Installation and Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Docker Setup](#docker-setup)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Verification & Testing](#verification--testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11 | 3.12 |
| RAM | 4GB | 8GB+ |
| Storage | 10GB | 20GB+ |
| CPU | 2 cores | 4+ cores |
| OS | Linux/macOS/Windows | Linux (Ubuntu 22.04+) |

### Required Software

- **Python 3.11+**: [Download](https://www.python.org/downloads/)
- **Git**: [Download](https://git-scm.com/)
- **Docker & Docker Compose** (optional): [Download](https://www.docker.com/)
- **Neo4j** (database): [Download](https://neo4j.com/download/)

### API Keys Required

Before setup, obtain these API keys:
- **OpenAI**: [Get API Key](https://platform.openai.com/api-keys)
- **Google Gemini**: [Get API Key](https://makersuite.google.com/app/apikey)
- **Groq**: [Get API Key](https://console.groq.com/keys)

---

## Development Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git

# Navigate to project directory
cd EL-MOSTASHAR-Graduation-Project

# Verify repository structure
ls -la
```

### Step 2: Create Virtual Environment

```bash
# Using Python venv
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

#### Option A: Using pip

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Verify installation
pip list | grep -E "fastapi|langchain|neo4j|faiss"
```

#### Option B: Using uv (Recommended - Faster)

```bash
# Install uv package manager
pip install uv

# Install dependencies using uv
uv sync

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### Step 4: Verify Installation

```bash
# Test Python
python --version  # Should output 3.11+

# Test FastAPI
python -c "import fastapi; print(fastapi.__version__)"

# Test LangChain
python -c "import langchain; print(langchain.__version__)"

# Test Neo4j driver
python -c "import neo4j; print(neo4j.__version__)"
```

---

## Docker Setup

### Step 1: Install Docker

```bash
# On Ubuntu
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Verify installation
docker --version
docker-compose --version

# Test Docker
docker run hello-world
```

### Step 2: Build Application Image

```bash
# Navigate to project directory
cd EL-MOSTASHAR-Graduation-Project

# Build the Docker image
docker build -t el-mostashar:latest .

# Verify build
docker images | grep el-mostashar
```

### Step 3: Configure Docker Compose

```bash
# Copy example compose file if needed
# docker-compose.yml should already exist

# Verify configuration
cat docker-compose.yml

# Review services defined:
# - app: FastAPI application
# - neo4j: Graph database
```

### Step 4: Start Services

```bash
# Start all services in the background
docker-compose up -d

# Check service status
docker-compose ps

# Expected output:
# NAME                        COMMAND                  SERVICE      STATUS
# el-mostashar-app-1         python main.py          app          Up
# el-mostashar-neo4j-1       /startup/docker_entrypoint.sh  neo4j  Up

# View logs
docker-compose logs -f app

# View Neo4j logs
docker-compose logs -f neo4j
```

### Step 5: Verify Services

```bash
# Test API endpoint
curl http://localhost:8000/

# Expected response:
# {"message":"Welcome to the Graduation Project API!"}

# Test Neo4j connection
curl http://localhost:7474/browser/

# Should return Neo4j browser interface
```

---

## Environment Configuration

### Step 1: Create .env File

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

### Step 2: Configure LLM Providers

```bash
# Edit .env file with your API keys

# OpenAI Configuration
OPENAI_API_KEY="sk-..."
OPENAI_MODEL_NAME="gpt-4"

# Google Gemini Configuration
GOOGLE_API_KEY="..."
GOOGLE_MODEL_NAME="gemini-pro"

# Groq Configuration
GROQ_API_KEY="..."
GROQ_MODEL_NAME="mixtral-8x7b-32768"

# Alternative: Use environment variables
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export GROQ_API_KEY="..."
```

### Step 3: Configure Database

```bash
# Neo4j Connection
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_secure_password"

# Vector Store Configuration
FAISS_INDEX_PATH="./data/faiss_indices"
VECTOR_DIMENSION=384

# For Docker:
NEO4J_URI="bolt://neo4j:7687"
```

### Step 4: Configure Application

```bash
# Application Settings
APP_HOST="0.0.0.0"
APP_PORT=8000
APP_DEBUG=False
APP_WORKERS=4

# Logging
LOG_LEVEL="INFO"
LOG_FORMAT="json"

# LangSmith (Optional - for debugging)
LANGSMITH_TRACING=False
LANGSMITH_API_KEY="..."
LANGSMITH_PROJECT="..."
```

### Step 5: Example .env File

```bash
# ============== LLM Providers ==============
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL_NAME=gpt-4-turbo
OPENAI_TEMPERATURE=0.7

GOOGLE_API_KEY=your-google-key
GOOGLE_MODEL_NAME=gemini-pro

GROQ_API_KEY=your-groq-key
GROQ_MODEL_NAME=mixtral-8x7b-32768

# ============== Database ==============
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j_password

# ============== Vector Store ==============
FAISS_INDEX_PATH=./data/faiss_indices
VECTOR_DIMENSION=384
VECTOR_MODEL_NAME=all-MiniLM-L6-v2

# ============== Application ==============
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=False
APP_WORKERS=4

# ============== Logging ==============
LOG_LEVEL=INFO
LOG_FORMAT=json

# ============== LangSmith (Optional) ==============
LANGSMITH_TRACING=False
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

---

## Database Setup

### Step 1: Neo4j Setup (Local)

```bash
# Download Neo4j Community Edition
# From: https://neo4j.com/download-center/

# Start Neo4j
# On macOS/Linux:
./neo4j-community-5.x/bin/neo4j start

# On Windows:
neo4j-community-5.x/bin/neo4j.bat start

# Access Neo4j Browser
# Open: http://localhost:7474/browser

# Default credentials:
# Username: neo4j
# Password: neo4j (change on first login)
```

### Step 2: Neo4j Setup (Docker)

```bash
# Start Neo4j container
docker run --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest

# Connect to Neo4j
# Browser: http://localhost:7474/browser
# Bolt: bolt://localhost:7687
```

### Step 3: Initialize Neo4j Schema

```bash
# Create indexes for performance
# In Neo4j Browser, execute:

CREATE INDEX idx_case_id FOR (c:Case) ON (c.caseId);
CREATE INDEX idx_judge_id FOR (j:Judge) ON (j.judgeId);
CREATE INDEX idx_party_id FOR (p:Party) ON (p.partyId);
CREATE INDEX idx_law_id FOR (l:Law) ON (l.lawId);

# Verify indexes
SHOW INDEXES;

# Clear any existing data (careful!)
MATCH (n) DETACH DELETE n;
```

### Step 4: Verify Database Connection

```bash
# Test connection from Python
python -c "
from neo4j import GraphDatabase

URI = 'bolt://localhost:7687'
USERNAME = 'neo4j'
PASSWORD = 'your_password'

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
session = driver.session()

result = session.run('RETURN 1')
print('Database connection successful!')
session.close()
"
```

---

## Verification & Testing

### Step 1: Start the Application

```bash
# Using Python directly
python main.py

# Or using Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Press CTRL+C to quit
```

### Step 2: Test API Endpoints

```bash
# Test root endpoint
curl http://localhost:8000/
# Expected: {"message":"Welcome to the Graduation Project API!"}

# Access API documentation
# Open browser: http://localhost:8000/docs
# This shows interactive Swagger UI for all endpoints

# Access ReDoc documentation
# Open browser: http://localhost:8000/redoc
```

### Step 3: Test Core Functionality

```bash
# Test ingestion endpoint
curl -X POST http://localhost:8000/ingest/documents \
  -F "document=@test_document.pdf"

# Test chunking
curl -X POST http://localhost:8000/chunk/process \
  -H "Content-Type: application/json" \
  -d '{"document_id": "test-001"}'

# Test KG query
curl -X POST http://localhost:8000/kg/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all cases"}'
```

### Step 4: Run Unit Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

### Step 5: Check System Health

```bash
# Create a simple health check script
cat > health_check.py << 'EOF'
import requests
import sys

def check_api():
    try:
        response = requests.get('http://localhost:8000/')
        assert response.status_code == 200
        print("✓ API is running")
        return True
    except:
        print("✗ API is not accessible")
        return False

def check_database():
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver('bolt://localhost:7687',
                                     auth=('neo4j', 'password'))
        driver.verify_connectivity()
        print("✓ Database is connected")
        return True
    except:
        print("✗ Database connection failed")
        return False

if check_api() and check_database():
    print("\n✓ All systems operational!")
    sys.exit(0)
else:
    print("\n✗ Some systems are not ready")
    sys.exit(1)
EOF

python health_check.py
```

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in docker-compose.yml
# ports:
#   - "8001:8000"
```

### Issue: Neo4j Connection Failed

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker-compose restart neo4j

# Check logs
docker-compose logs neo4j

# Verify connection string
# Should be: bolt://localhost:7687 (local)
# or: bolt://neo4j:7687 (Docker)
```

### Issue: Missing Dependencies

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or clear cache and reinstall
pip cache purge
pip install -r requirements.txt
```

### Issue: API Not Responding

```bash
# Check if API is running
ps aux | grep main.py

# View application logs
tail -f logs/app.log

# Restart application
pkill -f main.py
python main.py
```

### Issue: CUDA/GPU Not Detected

```bash
# For CPU-only mode (default)
# FAISS is already configured for CPU

# For GPU support (optional)
pip install faiss-gpu

# Test GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Quick Diagnostics

```bash
#!/bin/bash
# Create diagnostics script

echo "=== System Information ==="
python --version
pip --version

echo "\n=== Python Packages ==="
pip list | grep -E "fastapi|langchain|neo4j|torch|faiss"

echo "\n=== Port Status ==="
lsof -i :8000 || echo "Port 8000 is free"
lsof -i :7687 || echo "Port 7687 is free"

echo "\n=== Docker Status ==="
docker-compose ps

echo "\n=== Environment Variables ==="
env | grep -E "OPENAI|GOOGLE|GROQ|NEO4J" | grep -v PASS

echo "\n=== API Connectivity ==="
curl -s http://localhost:8000/ || echo "API not responding"
```

---

## Next Steps After Installation

1. ✅ Verify all systems are operational
2. 📚 Read [USAGE_GUIDE.md](./USAGE_GUIDE.md) for system usage
3. 🔌 Check [API_REFERENCE.md](./API_REFERENCE.md) for endpoint documentation
4. 🏗️ Review [ARCHITECTURE.md](./ARCHITECTURE.md) to understand system design
5. 📝 Explore [SYSTEM_COMPONENTS.md](./SYSTEM_COMPONENTS.md) for component details

---

## Support & Resources

### Useful Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Python Documentation](https://docs.python.org/3.11/)

### Getting Help
- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
- Review application logs in `logs/` directory
- Check Docker Compose logs: `docker-compose logs -f`

---

**Installation complete! You're now ready to start using the Egyptian Legal Multi-Agent System.**
