# 🚀 Deployment Guide

## Production Deployment

### Prerequisites
- Docker & Docker Compose installed
- Neo4j instance available
- All LLM API keys configured
- Domain name (optional)

---

## Docker Deployment

### Build Image
```bash
docker build -t el-mostashar:1.0 .
```

### Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Configuration via Environment
```bash
# .env file for docker-compose
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=secure_password
OPENAI_API_KEY=your_key
```

---

## Network Configuration

### Ports
- **8000**: FastAPI application (HTTP)
- **7687**: Neo4j Bolt protocol
- **7474**: Neo4j Browser (optional)

### Expose API
```yaml
# docker-compose.yml
services:
  app:
    ports:
      - "8000:8000"
```

---

## Performance Tuning

### Resource Allocation
```yaml
services:
  app:
    resources:
      limits:
        memory: 8G
        cpus: '4'
      reservations:
        memory: 4G
        cpus: '2'
```

### Neo4j Optimization
```yaml
environment:
  NEO4J_dbms_memory_heap_initial__size: 4G
  NEO4J_dbms_memory_heap_max__size: 8G
  NEO4J_dbms_memory_pagecache_size: 4G
```

---

## Monitoring & Logging

### Container Logs
```bash
docker-compose logs -f app
docker-compose logs -f neo4j
```

### Health Check
```bash
curl http://localhost:8000/
```

### Metrics
```bash
# System resources
docker stats

# Application info
curl http://localhost:8000/kg/statistics
```

---

## Scaling

### Horizontal Scaling
- Run multiple API containers behind load balancer
- Share Neo4j instance across containers
- Use Redis for distributed caching (future)

### Load Balancing
```yaml
# nginx configuration example
upstream el_mostashar {
  server app1:8000;
  server app2:8000;
  server app3:8000;
}

server {
  listen 80;
  location / {
    proxy_pass http://el_mostashar;
  }
}
```

---

## Backup & Recovery

### Database Backup
```bash
# Backup Neo4j
docker exec neo4j-container neo4j-admin dump --database neo4j --to /backups/neo4j.dump
```

### Vector Store Backup
```bash
# Backup FAISS index
cp -r na2d_faiss_index /backups/
```

---

## Security

### Environment Secrets
- Never commit `.env` file
- Use secrets management system
- Rotate API keys regularly

### API Hardening
```yaml
environment:
  CORS_ORIGINS: "['https://yourdomain.com']"
  ALLOWED_HOSTS: "yourdomain.com"
```

### Firewall Rules
```bash
# Allow only HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp  # SSH

# Block direct Neo4j access
ufw deny 7687/tcp
```

---

## SSL/TLS Configuration

### Using Let's Encrypt
```bash
# Generate certificate
certbot certonly --standalone -d yourdomain.com

# Update nginx
ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
```

---

## Continuous Deployment

### GitHub Actions Example
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build image
        run: docker build -t el-mostashar:latest .
      - name: Push to registry
        run: docker push registry.example.com/el-mostashar:latest
      - name: Deploy
        run: |
          ssh user@server 'cd /app && docker-compose pull && docker-compose up -d'
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs app

# Verify image
docker images el-mostashar

# Rebuild
docker-compose build --no-cache
```

### Neo4j connection failed
```bash
# Test connection
docker exec app curl http://neo4j:7687

# Check Neo4j logs
docker-compose logs neo4j
```

### Out of memory
```bash
# Increase container memory
docker-compose down
# Edit docker-compose.yml
docker-compose up -d
```

---

## Production Checklist

- [ ] All API keys configured
- [ ] Neo4j instance running and backed up
- [ ] Vector store built and loaded
- [ ] SSL/TLS certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring tools set up
- [ ] Backup strategy in place
- [ ] Load balancer configured
- [ ] Health checks enabled
- [ ] Logging aggregation set up

---

## Support

Email: mg7357432@gmail.com

---

**Last Updated**: June 2024  
**Version**: 1.0.0
