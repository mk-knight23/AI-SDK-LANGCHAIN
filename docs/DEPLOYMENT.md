# Deployment Guide

**AI-SDK-LANGCHAIN** - Production Deployment Instructions

---

## Environment Setup

### Required Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-openai-...

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://host:6379

# Application
NODE_ENV=production
API_BASE_URL=https://api.langchain.com
```

---

## Platform Deployments

### Vercel (Frontend)

```bash
cd frontend
vercel --prod
```

### Railway (Backend)

```bash
cd backend
railway up
```

### Fly.io (Full Stack)

```bash
fly deploy
```

---

## Docker Deployment

```bash
docker-compose up -d
```

---

## Health Check

```bash
curl https://api.langchain.com/health
```

---

For more details, see the project README.
