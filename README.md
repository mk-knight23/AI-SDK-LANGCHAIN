# LangChain Framework SDK

Startup ecosystem intelligence platform.

## Tech Stack

- **Frontend:** Next.js 15 + React 19 + TypeScript
- **Backend:** FastAPI + Python 3.12
- **Deployment:** Railway

## Project Structure

```
.
├── frontend/          # Next.js 15 application
│   ├── app/          # App router
│   ├── Dockerfile
│   └── package.json
├── backend/          # FastAPI application
│   ├── main.py       # Main application
│   ├── Dockerfile
│   └── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml    # GitHub Actions CI
```

## Development

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on http://localhost:3000

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs on http://localhost:8000

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment

### Railway Deployment

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

3. Create project and services:
   ```bash
   railway init
   ```

4. Deploy frontend:
   ```bash
   cd frontend
   railway link
   railway up
   ```

5. Deploy backend:
   ```bash
   cd backend
   railway link
   railway up
   ```

### Environment Variables

Create a `.env` file in each directory:

**frontend/.env:**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**backend/.env:**
```
ALLOWED_ORIGINS=http://localhost:3000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |

## CI/CD

GitHub Actions workflow runs on push/PR to main:
- Builds frontend
- Tests backend imports
