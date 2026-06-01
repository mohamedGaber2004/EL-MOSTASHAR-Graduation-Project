# 🚀 Installation & Setup Guide

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.11 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk**: 20GB free space

### Required Software
- Git
- Docker & Docker Compose (for containerized setup)
- Python 3.11+
- pip or uv package manager

### External Services
- Neo4j database (local or remote)
- LLM API keys (at least one of):
  - OpenAI API key
  - Google Gemini API key
  - Groq API key
  - Others (Cerebras, Mistral, Cohere)

---

## Step 1: Clone Repository

```bash
git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git
cd EL-MOSTASHAR-Graduation-Project
```

---

## Step 2: Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Arabic Embedding Model
ARABIC_NATIVE_EMBEDDING_MODEL=sentence-transformers/arabic-MiniLM-L6-v2

# LangSmith Tracing (Optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your_project_name

# Application
DEBUG=false
LOG_LEVEL=INFO
```

---

## Step 3: Install Dependencies

### Option A: Using pip

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

---

## Step 4: Setup Neo4j Database

### Local Setup (Docker)

```bash
# Run Neo4j in Docker
docker run -d \
  --name neo4j-instance \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest

# Wait for Neo4j to start (30-60 seconds)
sleep 60

# Verify connection
curl http://localhost:7474
```

### Existing Neo4j Instance

Update `.env` with your Neo4j connection details:
```bash
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

---

## Step 5: Initialize Vector Store & Knowledge Graph

Once the application is running, call these endpoints:

### Build Vector Store
```bash
curl -X POST http://localhost:8000/vs/build
```

### Setup Knowledge Graph Indexes
```bash
curl -X POST http://localhost:8000/kg/retriever/index/setup
```

### Build Knowledge Graph
```bash
curl -X POST http://localhost:8000/kg/build?drop_existing=true
```

### Embed Nodes
```bash
curl -X POST http://localhost:8000/kg/retriever/index/embed
```

---

## Step 6: Start Application

### Local Development

```bash
# Activate virtual environment (if using pip)
source venv/bin/activate

# Run application
python main.py

# Application starts at http://localhost:8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Step 7: Verify Installation

### Check API Health
```bash
curl http://localhost:8000/
# Expected response:
# {"message": "Welcome to the Graduation Project API!"}
```

### Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test Case Invocation
```bash
curl -X POST http://localhost:8000/cases/invoke_case \
  -F "case_id=TEST_CASE_001" \
  -F "files=@path/to/case_document.pdf"
```

---

## Step 8: Download/Prepare Legal Corpus

The system requires a legal corpus. Prepare one of:

1. **Use Example Data** (if provided)
   ```bash
   # Copy example data to expected location
   cp -r Datasets/Example/* Datasets/
   ```

2. **Ingest Your Own Data**
   ```bash
   # Place documents in Datasets/User_Cases/
   mkdir -p Datasets/User_Cases
   # Copy your legal documents
   ```

---

## Troubleshooting

### Issue: Neo4j Connection Failed
```
Error: Could not connect to Neo4j
```
**Solution**:
- Verify Neo4j is running: `docker ps`
- Check URI and credentials in `.env`
- Ensure firewall allows port 7687

### Issue: LLM API Key Invalid
```
Error: Invalid API key for OpenAI
```
**Solution**:
- Verify API key in `.env`
- Ensure key has correct permissions
- Check key hasn't expired

### Issue: Vector Store Not Loaded
```
Error: Vector store not loaded. Call POST /vs/build first
```
**Solution**:
- Run: `curl -X POST http://localhost:8000/vs/build`
- Wait for completion
- Load index: `curl -X POST http://localhost:8000/vs/load`

### Issue: Out of Memory
```
Error: Cannot allocate memory
```
**Solution**:
- Reduce batch size in embedding calls
- Use different embedding model (smaller)
- Increase system RAM or reduce document count

---

## Configuration Reference

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| NEO4J_URI | bolt://localhost:7687 | Neo4j connection URI |
| NEO4J_USERNAME | neo4j | Neo4j username |
| NEO4J_PASSWORD | required | Neo4j password |
| ARABIC_NATIVE_EMBEDDING_MODEL | sentence-transformers/arabic-MiniLM-L6-v2 | Embedding model |
| LANGSMITH_TRACING | false | Enable LangSmith tracing |
| DEBUG | false | Debug mode |
| LOG_LEVEL | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## System Requirements by Component

### FastAPI Server
- RAM: 2GB minimum
- CPU: 2 cores minimum
- Disk: 500MB

### Neo4j Database
- RAM: 4GB minimum
- Disk: 10GB minimum
- CPU: 2 cores minimum

### Vector Store (FAISS)
- RAM: 4GB minimum (for 10k documents)
- Disk: 2GB minimum
- CPU: 2 cores minimum

### LLM Processing
- RAM: 8GB recommended
- Network: Stable internet connection

---

## Post-Installation Checks

✅ Checklist for successful installation:
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `.env` file configured
- [ ] Neo4j running and accessible
- [ ] Vector store built
- [ ] Knowledge graph built
- [ ] API responds at http://localhost:8000/
- [ ] Swagger docs accessible
- [ ] Test case processes successfully

---

## Next Steps

1. **Read [USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Learn how to use the system
2. **Review [API_REFERENCE.md](./API_REFERENCE.md)** - Explore all endpoints
3. **Check [ARCHITECTURE.md](./ARCHITECTURE.md)** - Understand system design
4. **Test case invocation** - Submit your first case

---

## Support

For issues or questions:
- Email: mg7357432@gmail.com
- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- Review logs: `docker-compose logs`

---

**Last Updated**: June 2024  
**Version**: 1.0.0
