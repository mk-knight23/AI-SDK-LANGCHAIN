# VentureGraph

**Project:** VentureGraph (Startup ecosystem intelligence SaaS)
**Tech Stack:** Next.js 15 + FastAPI
**Deployment Target:** Railway

## MANDATORY WORKFLOW

1. Superpowers → Brainstorm → Plan → TDD
2. ECC → /plan → /tdd → /code-review → /security-scan
3. UI/UX Pro Max → Apply design system
4. Claude-Tips → /dx:handoff before end of session

## AGENTS TO USE

- /architect for system design
- /tdd-guide for test-first implementation
- /security-reviewer before API key usage

## Project Structure

```
frontend/     # Next.js 15 + React 19 + TypeScript
backend/      # FastAPI + Python 3.12
.github/      # CI/CD workflows
```

## Development Commands

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload
```

## API Endpoints

- GET `/` - API info
- GET `/health` - Health check

## Deployment

Railway deployment:
```bash
cd frontend && railway up
cd backend && railway up
```

## Environment Variables

- `NEXT_PUBLIC_API_URL` - Frontend API URL
- `ALLOWED_ORIGINS` - Backend CORS origins
