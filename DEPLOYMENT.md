# Deployment Guide

## Railway Deployment Steps

### Prerequisites

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

### Deploy Backend Service

```bash
cd backend

# Create new service or link to existing
railway link

# Deploy
railway up

# View logs
railway logs
```

### Deploy Frontend Service

```bash
cd frontend

# Create new service or link to existing
railway link

# Deploy
railway up

# View logs
railway logs
```

### Alternative: Using Railway Dashboard

1. Go to https://railway.app/dashboard
2. Create a new project
3. Add services:
   - Service 1: Deploy from GitHub repo, set Dockerfile path to `frontend/Dockerfile`
   - Service 2: Deploy from GitHub repo, set Dockerfile path to `backend/Dockerfile`
4. Add environment variables as needed
5. Deploy

### Environment Variables

Set these in Railway dashboard for each service:

**Frontend Service:**
- `NEXT_PUBLIC_API_URL` - URL of backend service

**Backend Service:**
- `ALLOWED_ORIGINS` - URL of frontend service (for CORS)

### Verify Deployment

Once deployed, verify:

1. Frontend displays "Hello World"
2. Backend `/health` endpoint returns `{"status": "healthy"}`
