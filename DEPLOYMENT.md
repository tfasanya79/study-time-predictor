# Deployment Guide - Study Time Predictor

## Overview

This guide covers deploying the Study Time Predictor from development to production environments. The application is built with Flask and can be deployed using various methods.

---

## Development Deployment (Current Setup)

### Current Status
✅ **Already Running**: Your app is currently running in development mode at:
- http://localhost:5000
- http://172.20.100.237:5000

### Development Features
- Flask development server
- SQLite database
- Debug mode enabled
- Hot reload on file changes
- Detailed error pages

---

## Production Deployment Options

### Option 1: Simple Production Server (Recommended for MVP)

#### 1. Prepare Environment Variables
Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env`:
```env
SECRET_KEY=your-secure-secret-key-here-32-characters-min
FLASK_ENV=production
FLASK_DEBUG=0
DATABASE_URL=sqlite:///study_predictor.db
APP_HOST=0.0.0.0
APP_PORT=5000
```

#### 2. Install Gunicorn (Production WSGI Server)
```bash
pip install gunicorn
```

#### 3. Run with Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 run:app
```

### Option 2: Docker Deployment

#### 1. Create Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ml_models data

# Initialize database and train model
RUN python setup.py

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "run:app"]
```

#### 2. Create docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secret-key-here
    volumes:
      - ./data:/app/data
      - ./ml_models:/app/ml_models
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - web
    restart: unless-stopped
```

#### 3. Build and Run
```bash
docker-compose up -d
```

### Option 3: Cloud Deployment

#### Heroku Deployment

1. **Create Procfile**:
```
web: gunicorn run:app
```

2. **Create runtime.txt**:
```
python-3.12.0
```

3. **Deploy**:
```bash
# Install Heroku CLI
heroku create your-app-name
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### Railway Deployment

1. **Connect GitHub repository**
2. **Configure environment variables**
3. **Deploy with one click**

#### DigitalOcean App Platform

1. **Connect repository**
2. **Configure build settings**:
   - Build command: `python setup.py`
   - Run command: `gunicorn --bind 0.0.0.0:$PORT run:app`

---

## Database Migration (SQLite → PostgreSQL)

### For Larger Scale Production

#### 1. Install PostgreSQL Driver
```bash
pip install psycopg2-binary
```

#### 2. Update Requirements
Add to `requirements.txt`:
```
psycopg2-binary==2.9.7
```

#### 3. Update Environment Variables
```env
DATABASE_URL=postgresql://username:password@localhost/study_predictor
```

#### 4. Migration Script
```python
# migrate_to_postgres.py
import sqlite3
import psycopg2
from urllib.parse import urlparse

def migrate_sqlite_to_postgres():
    # Export from SQLite
    sqlite_conn = sqlite3.connect('study_predictor.db')
    
    # Import to PostgreSQL
    postgres_url = os.environ['DATABASE_URL']
    postgres_conn = psycopg2.connect(postgres_url)
    
    # Migration logic here
    # (Implementation would depend on specific needs)
```

---

## Performance Optimization

### 1. Application Level

#### Optimize Model Loading
```python
# In app/__init__.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_predictor():
    return StudyTimePredictor()
```

#### Database Connection Pooling
```python
# For PostgreSQL
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 300,
    'pool_pre_ping': True
}
```

### 2. Web Server Configuration

#### Nginx Configuration (nginx.conf)
```nginx
upstream app_server {
    server web:5000;
}

server {
    listen 80;
    client_max_body_size 4G;

    location / {
        proxy_set_header Host $http_host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_pass http://app_server;
    }

    location /static/ {
        alias /app/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### 3. Static File Serving

#### Option A: CDN (Recommended)
Upload static files to AWS S3/CloudFront or similar

#### Option B: Nginx Static Serving
Configure Nginx to serve static files directly

---

## Monitoring and Logging

### 1. Application Monitoring

#### Add Logging
```python
# In run.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

#### Health Check Endpoint
```python
# In app/routes/main.py
@bp.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}
```

### 2. System Monitoring

#### Docker Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1
```

#### Monitoring Tools
- **Prometheus + Grafana**: Comprehensive metrics
- **Sentry**: Error tracking
- **Uptime Robot**: Availability monitoring

---

## Security Considerations

### 1. Environment Variables
Never commit sensitive data to version control:
```bash
# Add to .gitignore
.env
*.db
ml_models/*.h5
ml_models/*.pkl
```

### 2. Flask Security Headers
```python
# In app/__init__.py
from flask_talisman import Talisman

def create_app():
    app = Flask(__name__)
    Talisman(app, force_https=False)  # Set True in production
    # ... rest of app setup
```

### 3. Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@bp.route('/api/predict')
@limiter.limit("10 per minute")
def predict():
    # ... prediction logic
```

---

## Backup and Recovery

### 1. Database Backups
```bash
# SQLite backup
cp study_predictor.db study_predictor_backup_$(date +%Y%m%d).db

# PostgreSQL backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql
```

### 2. Model Backups
```bash
# Backup trained models
tar -czf ml_models_backup_$(date +%Y%m%d).tar.gz ml_models/
```

### 3. Automated Backups
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d)
cp study_predictor.db "backups/db_backup_$DATE.db"
tar -czf "backups/models_backup_$DATE.tar.gz" ml_models/
```

---

## SSL/HTTPS Setup

### 1. Let's Encrypt (Free SSL)
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Nginx HTTPS Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # ... rest of configuration
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

---

## Scaling Considerations

### Horizontal Scaling
- **Load Balancer**: Nginx, HAProxy, or cloud load balancers
- **Multiple App Instances**: Run multiple Gunicorn workers
- **Database Scaling**: Read replicas, connection pooling

### Vertical Scaling
- **CPU**: More cores for ML model inference
- **Memory**: Larger datasets and model caching
- **Storage**: SSD for faster database operations

### Microservices Architecture (Future)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │────│  Prediction API  │────│  Model Service  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐             │
         └──────────────│   Data Service   │─────────────┘
                        └──────────────────┘
```

---

## Troubleshooting Common Issues

### 1. Model Loading Errors
```bash
# Retrain model if corrupted
python -c "from ml_models.predictor import train_model; train_model()"
```

### 2. Database Connection Issues
```bash
# Reset database
rm study_predictor.db
python setup.py
```

### 3. Permission Issues
```bash
# Fix file permissions
chmod +x run.py
chmod -R 755 app/
```

### 4. Memory Issues
```bash
# Monitor memory usage
ps aux | grep python
free -h

# Reduce model complexity if needed
```

---

## Maintenance Tasks

### Regular Maintenance Checklist
- [ ] **Weekly**: Check application logs
- [ ] **Weekly**: Monitor disk space and performance
- [ ] **Monthly**: Update dependencies
- [ ] **Monthly**: Backup database and models
- [ ] **Quarterly**: Review security settings
- [ ] **Quarterly**: Performance optimization review

### Update Dependencies
```bash
# Check outdated packages
pip list --outdated

# Update requirements
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

---

## Support and Resources

### Getting Help
1. Check application logs first
2. Review this deployment guide
3. Search GitHub issues
4. Create new issue with detailed error information

### Useful Commands
```bash
# Check application status
ps aux | grep gunicorn

# View logs
tail -f logs/app.log

# Test endpoints
curl http://localhost:5000/health

# Database size
du -h study_predictor.db
```

This deployment guide provides multiple pathways from development to production, allowing you to choose the approach that best fits your infrastructure and scaling needs.