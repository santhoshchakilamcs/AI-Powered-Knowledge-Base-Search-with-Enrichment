# Deployment Guide

## Local Development

### Prerequisites
- Python 3.8+
- OpenAI API key
- 2GB+ RAM
- 1GB+ disk space

### Setup
```bash
# Clone and setup
git clone <your-repo>
cd RAG_1
chmod +x setup.sh run.sh
./setup.sh

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
./run.sh
```

## Production Deployment

### Option 1: Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/chroma_db data/uploads

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Create docker-compose.yml
```yaml
version: '3.8'

services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db
      - UPLOAD_DIRECTORY=/app/data/uploads
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

#### Deploy
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Cloud Deployment (AWS)

#### EC2 Deployment

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium (2 vCPU, 4GB RAM)
   - Storage: 20GB EBS
   - Security Group: Allow ports 22, 80, 443, 8000

2. **Connect and Setup**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip nginx

# Clone repository
git clone <your-repo>
cd RAG_1

# Setup application
./setup.sh
```

3. **Configure Environment**
```bash
# Edit .env
nano .env
# Add your OPENAI_API_KEY and other settings
```

4. **Setup Systemd Service**
```bash
sudo nano /etc/systemd/system/rag-app.service
```

```ini
[Unit]
Description=RAG Knowledge Base Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/RAG_1
Environment="PATH=/home/ubuntu/RAG_1/venv/bin"
ExecStart=/home/ubuntu/RAG_1/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rag-app
sudo systemctl start rag-app
sudo systemctl status rag-app
```

5. **Configure Nginx Reverse Proxy**
```bash
sudo nano /etc/nginx/sites-available/rag-app
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/rag-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

6. **Setup SSL with Let's Encrypt**
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Option 3: Heroku Deployment

1. **Create Procfile**
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

2. **Create runtime.txt**
```
python-3.11.0
```

3. **Deploy**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key

# Deploy
git push heroku main

# Open app
heroku open
```

### Option 4: Google Cloud Run

1. **Create Dockerfile** (same as Docker option)

2. **Deploy**
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project your-project-id

# Build and deploy
gcloud run deploy rag-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key
```

### Option 5: DigitalOcean App Platform

1. **Create app.yaml**
```yaml
name: rag-knowledge-base
services:
- name: web
  github:
    repo: your-username/your-repo
    branch: main
  run_command: uvicorn app.main:app --host 0.0.0.0 --port 8080
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xs
  envs:
  - key: OPENAI_API_KEY
    value: ${OPENAI_API_KEY}
    type: SECRET
  http_port: 8080
```

2. **Deploy via Dashboard or CLI**
```bash
doctl apps create --spec app.yaml
```

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key

### Optional (with defaults)
- `CHROMA_PERSIST_DIRECTORY`: Vector DB storage path
- `UPLOAD_DIRECTORY`: Document upload path
- `LLM_MODEL`: GPT model to use (default: gpt-4-turbo-preview)
- `LLM_TEMPERATURE`: Temperature for generation (default: 0.1)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200)
- `TOP_K_RESULTS`: Number of results to retrieve (default: 5)

## Monitoring

### Health Check
```bash
curl http://your-domain.com/api/health
```

### Logs
```bash
# Systemd
sudo journalctl -u rag-app -f

# Docker
docker-compose logs -f

# Heroku
heroku logs --tail

# Cloud Run
gcloud run logs read rag-app --limit 50
```

### Metrics to Monitor
- API response time
- Document upload success rate
- Search query latency
- OpenAI API usage and costs
- Disk space (for vector DB and uploads)
- Memory usage
- Error rates

## Backup and Recovery

### Backup Vector Database
```bash
# Local
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/chroma_db/

# To S3
aws s3 cp data/chroma_db/ s3://your-bucket/backups/chroma_db/ --recursive
```

### Backup Uploaded Documents
```bash
# Local
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz data/uploads/

# To S3
aws s3 cp data/uploads/ s3://your-bucket/backups/uploads/ --recursive
```

### Restore
```bash
# From local backup
tar -xzf chroma_backup_20240101.tar.gz -C data/

# From S3
aws s3 cp s3://your-bucket/backups/chroma_db/ data/chroma_db/ --recursive
```

## Scaling

### Vertical Scaling
- Increase server resources (CPU, RAM)
- Use faster storage (SSD)
- Optimize chunk size and retrieval parameters

### Horizontal Scaling
- Use load balancer (Nginx, AWS ALB)
- Deploy multiple application instances
- Use shared storage for vector DB (S3, EFS)
- Consider managed vector DB (Pinecone, Weaviate Cloud)

### Database Scaling
- ChromaDB can handle millions of vectors
- For larger scale, consider:
  - Pinecone (managed, scalable)
  - Weaviate (self-hosted or cloud)
  - Milvus (distributed vector DB)

## Security Checklist

- [ ] Use HTTPS (SSL/TLS)
- [ ] Add authentication (JWT, OAuth)
- [ ] Implement rate limiting
- [ ] Restrict CORS origins
- [ ] Validate file uploads
- [ ] Sanitize user inputs
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Monitor for suspicious activity
- [ ] Implement API key rotation
- [ ] Use firewall rules
- [ ] Enable logging and auditing

## Cost Optimization

### OpenAI API Costs
- Use GPT-3.5-turbo for lower costs
- Implement caching for frequent queries
- Set max_tokens limits
- Monitor usage with OpenAI dashboard

### Infrastructure Costs
- Use auto-scaling
- Implement request caching
- Optimize chunk size to reduce embeddings
- Use spot instances (AWS, GCP)
- Set up billing alerts

## Troubleshooting

### Application Won't Start
```bash
# Check logs
sudo journalctl -u rag-app -n 50

# Check Python environment
source venv/bin/activate
python -c "import app.main"

# Check dependencies
pip list
```

### High Memory Usage
- Reduce chunk size
- Limit concurrent requests
- Increase server RAM
- Monitor with `htop` or `top`

### Slow Queries
- Check OpenAI API latency
- Optimize TOP_K_RESULTS
- Add caching layer
- Use faster embedding model

### Vector DB Issues
- Check disk space
- Verify permissions on data directory
- Rebuild index if corrupted
- Check ChromaDB logs

## Maintenance

### Regular Tasks
- Monitor API costs
- Review error logs
- Update dependencies
- Backup data
- Check disk space
- Review rating statistics
- Update documents

### Updates
```bash
# Pull latest code
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart rag-app
```

## Support

For issues and questions:
- Check logs first
- Review documentation
- Search existing issues
- Create new issue with details
- Include error messages and logs

