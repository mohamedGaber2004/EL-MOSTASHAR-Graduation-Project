# 🚀 Deployment Guide: Egyptian Legal Multi-Agent System

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Preparation](#environment-preparation)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Pre-Deployment Checklist

### Code Quality
- [ ] All tests passing (pytest coverage >80%)
- [ ] Code formatted with Black
- [ ] Linting passed (Flake8)
- [ ] Type checking passed (MyPy)
- [ ] Security scan completed
- [ ] Dependencies up to date

### Documentation
- [ ] README complete and accurate
- [ ] API documentation updated
- [ ] Architecture documented
- [ ] Environment variables documented
- [ ] Database schema documented
- [ ] Deployment runbook created

### Infrastructure
- [ ] Database provisioned
- [ ] Secrets management configured
- [ ] Load balancer configured
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] SSL/TLS certificates ready

### Testing
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Load testing completed
- [ ] Security testing completed
- [ ] Disaster recovery tested

---

## Environment Preparation

### 1. Server Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
  python3.11 \
  python3-pip \
  docker.io \
  docker-compose \
  git \
  curl \
  wget

# Verify installations
python3 --version
docker --version
docker-compose --version
```

### 2. Configure System Resources

```bash
# Set up swap (if needed)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw allow 8000/tcp # API

# Check resources
free -h
df -h
```

### 3. Create Application User

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash appuser

# Grant Docker access
sudo usermod -aG docker appuser

# Switch to user
su - appuser
```

---

## Docker Deployment

### 1. Build Production Image

```bash
# Navigate to project
cd /home/appuser/el-mostashar

# Build image with version tag
docker build -t el-mostashar:1.0.0 \
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  .

# Tag for registry
docker tag el-mostashar:1.0.0 registry.example.com/el-mostashar:latest
```

### 2. Push to Registry

```bash
# Login to registry
docker login registry.example.com

# Push image
docker push registry.example.com/el-mostashar:latest

# Verify
docker images registry.example.com/el-mostashar
```

### 3. Configure Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: registry.example.com/el-mostashar:latest
    ports:
      - "8000:8000"
    environment:
      - APP_DEBUG=False
      - APP_WORKERS=4
      - LOG_LEVEL=INFO
    depends_on:
      - neo4j
    volumes:
      - /data/logs:/app/logs
      - /data/cache:/app/cache
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:6.1.0
    ports:
      - "7687:7687"
    environment:
      NEO4J_AUTH: ${NEO4J_AUTH}
      NEO4J_dbms_memory_heap_maxSize: 4G
    volumes:
      - /data/neo4j:/var/lib/neo4j/data
    restart: always

networks:
  default:
    name: el-mostashar-prod
```

### 4. Deploy with Docker Compose

```bash
# Set environment
export NEO4J_AUTH=neo4j:${SECURE_PASSWORD}

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Verify
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f app
```

---

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: el-mostashar
  namespace: legal-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: el-mostashar
  template:
    metadata:
      labels:
        app: el-mostashar
    spec:
      containers:
      - name: api
        image: registry.example.com/el-mostashar:latest
        ports:
        - containerPort: 8000
        env:
        - name: APP_DEBUG
          value: "False"
        - name: NEO4J_URI
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: neo4j_uri
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace legal-ai

# Create configmap
kubectl create configmap app-config \
  --from-literal=neo4j_uri=bolt://neo4j:7687 \
  -n legal-ai

# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=neo4j_password=${NEO4J_PASSWORD} \
  -n legal-ai

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get pods -n legal-ai
kubectl describe deployment el-mostashar -n legal-ai
```

---

## Cloud Deployment

### AWS Deployment

```bash
# 1. Create EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.large \
  --key-name production-key \
  --security-groups app-security-group

# 2. Configure RDS for Neo4j (or use Neo4j Aura)
aws rds create-db-instance \
  --db-instance-identifier el-mostashar-neo4j \
  --db-instance-class db.t3.large

# 3. Configure S3 for backups
aws s3 mb s3://el-mostashar-backups

# 4. Deploy using Elastic Beanstalk
eb create el-mostashar-prod
eb deploy
```

### GCP Deployment

```bash
# 1. Create Cloud Run service
gcloud run deploy el-mostashar \
  --image gcr.io/project/el-mostashar:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2

# 2. Configure Cloud SQL for Neo4j
gcloud sql instances create el-mostashar-neo4j \
  --database-version MYSQL_8_0 \
  --tier db-custom-4-16384

# 3. Set up Cloud Storage
gsutil mb gs://el-mostashar-backups
```

---

## Post-Deployment Verification

### 1. Health Checks

```bash
# Test API endpoint
curl http://localhost:8000/

# Check database connectivity
curl -X POST http://localhost:8000/kg/query \
  -H "Content-Type: application/json" \
  -d '{"query": "RETURN 1"}'

# Monitor system health
docker-compose logs app | grep ERROR
```

### 2. Performance Verification

```bash
# Load test
ab -n 100 -c 10 http://localhost:8000/

# Monitor resource usage
docker stats

# Check database performance
# In Neo4j Browser: PROFILE MATCH (n) RETURN COUNT(n);
```

### 3. Security Verification

```bash
# Check SSL/TLS configuration
openssl s_client -connect localhost:443

# Verify secrets are not exposed
grep -r "password\|api_key\|secret" src/ --include="*.py"

# Check for vulnerabilities
pip install safety
safety check
```

---

## Monitoring & Maintenance

### 1. Set Up Monitoring

```bash
# Install monitoring stack
docker run -d \
  -p 9090:9090 \
  -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Monitor endpoints
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### 2. Configure Logging

```bash
# Set up centralized logging
docker run -d \
  -p 5601:5601 \
  docker.elastic.co/kibana/kibana:8.0.0

# Send logs to ELK stack
# Update docker-compose to stream logs
```

### 3. Backup Strategy

```bash
# Backup database daily
0 2 * * * /usr/local/bin/backup-neo4j.sh

# Backup Neo4j
neo4j-admin backup --to-path=/backups --database=neo4j

# Upload to cloud
aws s3 sync /backups s3://el-mostashar-backups/
```

### 4. Update & Patch Management

```bash
# Create update script
#!/bin/bash
# Pull latest image
docker pull registry.example.com/el-mostashar:latest

# Stop current container
docker-compose down

# Start new version
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs app

# Verify configuration
docker-compose config

# Test image locally
docker run -it el-mostashar:latest bash
```

### Database Connection Issues

```bash
# Test Neo4j connectivity
docker-compose exec neo4j bin/cypher-shell -u neo4j -p password

# Check port availability
netstat -an | grep 7687

# View Neo4j logs
docker-compose logs neo4j
```

### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check database slow queries
# In Neo4j Browser:
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Query executor");

# Optimize indexes
SHOW INDEXES;
```

---

**Deployment complete! Monitor the system continuously and keep backups up to date.**
