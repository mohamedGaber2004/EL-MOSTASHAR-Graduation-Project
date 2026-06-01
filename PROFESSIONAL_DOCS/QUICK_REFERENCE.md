# 🚀 Quick Reference Guide

## Start Here - Navigation Guide

### 📖 Documentation Files Overview

```
📚 PROFESSIONAL_DOCS/
├── README.md ⭐ START HERE
│   └─ Overview & Quick Start
├── PROJECT_OVERVIEW.md
│   └─ Detailed project description
├── ARCHITECTURE.md
│   └─ System design & structure
├── INSTALLATION_AND_SETUP.md
│   └─ Setup instructions
├── USAGE_GUIDE.md
│   └─ How to use the system
├── API_REFERENCE.md
│   └─ API endpoints documentation
├── DEVELOPMENT_GUIDE.md
│   └─ Development guidelines
├── DEPLOYMENT.md
│   └─ Production deployment
├── TROUBLESHOOTING.md
│   └─ Problem solving
└── DOCUMENTATION_INDEX.md
    └─ Full navigation guide
```

---

## Quick Navigation by Task

### 🆕 First Time? 
1. Read: **README.md** (15 min)
2. Then: **INSTALLATION_AND_SETUP.md** (40 min)
3. Finally: **USAGE_GUIDE.md** (25 min)

### 🔍 Want to Understand Architecture?
1. **ARCHITECTURE.md** - System design (35 min)
2. **PROJECT_OVERVIEW.md** - Project context (30 min)

### 💻 Ready to Develop?
1. **DEVELOPMENT_GUIDE.md** - Setup & guidelines (25 min)
2. **API_REFERENCE.md** - API endpoints (30 min)
3. **ARCHITECTURE.md** - System design (35 min)

### 🚀 Need to Deploy?
1. **DEPLOYMENT.md** - Deployment steps (30 min)
2. **INSTALLATION_AND_SETUP.md** - Configuration (40 min)

### 🐛 Something Broken?
1. **TROUBLESHOOTING.md** - Common issues (20 min)
2. **INSTALLATION_AND_SETUP.md** - Verification (20 min)

### 👥 Presenting to Others?
1. Use: **DOCUMENTATION_INDEX.md** for structure
2. Mix content from: All documents as needed
3. Prep time: 2-4 hours

---

## Most Useful Commands

```bash
# Install system
git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git
cd EL-MOSTASHAR-Graduation-Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start system
python main.py

# Or with Docker
docker-compose up -d

# Access API
# Browser: http://localhost:8000/docs
# API: http://localhost:8000
# Neo4j: http://localhost:7474/browser
```

---

## Key Endpoints Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/case/analyze` | POST | Analyze case |
| `/ingest/documents` | POST | Upload document |
| `/retrieval/hybrid` | POST | Search |
| `/kg/query` | POST | Graph query |

---

## Key Configuration Variables

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Application
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=False
```

---

## Tech Stack at a Glance

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Database**: Neo4j
- **Vector Store**: FAISS
- **Orchestration**: LangGraph
- **LLMs**: OpenAI, Google, Groq, etc.
- **Deployment**: Docker

---

## File Locations

```
Project Root:
├── main.py (Entry point)
├── src/ (Source code)
│   ├── agents/ (AI agents)
│   ├── routers/ (API endpoints)
│   ├── Config/ (Configuration)
│   └── ...
├── PROFESSIONAL_DOCS/ (📍 You are here)
│   └── All .md files
├── docker-compose.yml (Docker setup)
└── requirements.txt (Dependencies)
```

---

## Common Workflows

### Upload & Analyze a Case
```bash
curl -X POST http://localhost:8000/ingest/documents \
  -F "document=@case.pdf" \
  -F "case_id=CASE_001"

curl -X POST http://localhost:8000/case/analyze \
  -H "Content-Type: application/json" \
  -d '{"case_id": "CASE_001"}'
```

### Search Cases
```bash
curl -X POST http://localhost:8000/retrieval/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "property rights", "k": 5}'
```

### Query Database
```bash
curl -X POST http://localhost:8000/kg/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (c:Case) RETURN c LIMIT 10"}'
```

---

## Documentation Quick Stats

| Metric | Value |
|--------|-------|
| Total Files | 10 markdown docs |
| Total Size | 172 KB |
| Total Lines | 5,500+ lines |
| Reading Time | ~9 hours |
| Code Examples | 50+ |
| Diagrams | 20+ |

---

## When to Read Each Document

| Document | When | Time |
|----------|------|------|
| README | First visit | 15 min |
| PROJECT_OVERVIEW | Understanding goal | 30 min |
| ARCHITECTURE | Deep dive | 35 min |
| INSTALLATION_AND_SETUP | Setting up | 40 min |
| USAGE_GUIDE | Using system | 25 min |
| API_REFERENCE | Building with API | 30 min |
| DEVELOPMENT_GUIDE | Coding features | 25 min |
| DEPLOYMENT | Going live | 30 min |
| TROUBLESHOOTING | Something wrong | 20 min |

---

## Pro Tips 💡

1. **Search within docs**: Use Ctrl+F (Cmd+F on Mac)
2. **Use table of contents**: Jump to sections quickly
3. **Code examples**: Copy and modify for your use
4. **API Docs**: Visit `/docs` for interactive testing
5. **Logs**: Check `logs/` folder for debugging

---

## Getting Help

1. **Search documentation** - Most answers are here
2. **Check logs** - See `logs/app.log`
3. **Try troubleshooting** - See TROUBLESHOOTING.md
4. **Review API docs** - Visit http://localhost:8000/docs

---

## Next Steps

- [ ] Read **README.md** (15 min)
- [ ] Follow **INSTALLATION_AND_SETUP.md** (40 min)
- [ ] Try examples in **USAGE_GUIDE.md** (25 min)
- [ ] Review **ARCHITECTURE.md** for understanding (35 min)
- [ ] Bookmark **API_REFERENCE.md** for reference

---

## Document Locations

All documentation files are in:
```
PROFESSIONAL_DOCS/
├── README.md
├── PROJECT_OVERVIEW.md
├── ARCHITECTURE.md
├── INSTALLATION_AND_SETUP.md
├── USAGE_GUIDE.md
├── API_REFERENCE.md
├── DEVELOPMENT_GUIDE.md
├── DEPLOYMENT.md
├── TROUBLESHOOTING.md
├── DOCUMENTATION_INDEX.md
└── QUICK_REFERENCE.md (you are here)
```

---

**Start with README.md → Follow your use case above → Reference as needed!**

Last Updated: June 2024 | Version: 1.0.0
