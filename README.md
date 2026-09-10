# AI Interview Coach

**AI Interview Coach** is a production-quality, fully role-agnostic interview preparation system built on **IBM watsonx.ai** and **IBM Granite** foundation models as part of the IBM SkillsBuild · Edunet · AICTE internship programme.

---

## What it does

- Conducts adaptive multi-turn mock interviews for **any job role** (free-form text — no hardcoded role list)
- Generates questions using **IBM Granite** via IBM watsonx.ai, grounded in a RAG knowledge base
- Evaluates each answer across six dimensions: overall, technical, relevance, clarity, communication, completeness
- Persists sessions in **SQLite** so the full interview survives across API requests
- Produces a final **performance report** aggregating scores, strengths, weaknesses, and improvement suggestions
- Serves a polished single-page UI at `/`

---

## Architecture

```
Browser / API client
        │
        ▼
FastAPI (app/main.py)
        │
        ├─ POST /interview/session            ← start session, get first question
        ├─ POST /interview/session/{id}/answer ← submit answer, get evaluation + next question
        ├─ GET  /interview/session/{id}/summary ← performance report
        ├─ POST /interview/question            ← legacy stateless question
        ├─ POST /interview/evaluate            ← legacy stateless evaluation
        └─ GET  /health                        ← liveness check (public)
        │
        ▼
InterviewEngine (app/interview_engine.py)
        │
        ├─ InterviewerAgent  → IBM Granite (question generation, adaptive)
        ├─ EvaluatorAgent    → IBM Granite (structured JSON evaluation)
        └─ RAGEngine         → FAISS index over data/knowledge/ (context injection)
        │
        ▼
SessionStore (app/session_store.py) — SQLite at data/sessions.db
```

---

## Prerequisites

- Python 3.12+
- IBM watsonx.ai account with:
  - `IBM_API_KEY`
  - `IBM_PROJECT_ID`

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd AI-Interview-Coach

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and fill in IBM_API_KEY and IBM_PROJECT_ID

# 5. Run
uvicorn app.main:app --reload
```

Open your browser at **http://localhost:8000** to use the interview UI.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IBM_API_KEY` | ✅ | — | IBM Cloud API key |
| `IBM_PROJECT_ID` | ✅ | — | IBM watsonx.ai project ID |
| `IBM_REGION` | | `jp-tok` | IBM Cloud region |
| `MODEL_ID` | | `ibm/granite-4-h-small` | IBM Granite model ID |
| `RAG_DOCUMENTS_DIR` | | `data/knowledge` | Knowledge base directory |
| `RAG_VECTOR_DIR` | | `data/vector_store` | Pre-built FAISS index directory |
| `API_KEY` | | _(disabled)_ | Optional bearer token to protect all API endpoints |

---

## API

### Session-based (recommended)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/interview/session` | Create session, get first question |
| `POST` | `/interview/session/{id}/answer` | Submit answer, get evaluation + next question |
| `GET` | `/interview/session/{id}/summary` | Get full performance report |

### Stateless (legacy)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/interview/question` | Generate a single question |
| `POST` | `/interview/evaluate` | Evaluate a single answer |
| `GET` | `/health` | Liveness check |

Interactive API docs: **http://localhost:8000/docs**

---

## Running tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 66 tests must pass. Live IBM tests (`test_ibm_connection`, `test_evaluator_agent`, `test_interviewer_agent`, `test_interview_flow`) require valid IBM credentials in `.env`.

---

## Technology stack

| Component | Technology |
|---|---|
| LLM | IBM Granite via IBM watsonx.ai (`ibm_watsonx_ai` SDK) |
| RAG | FAISS + sentence-transformers (`all-MiniLM-L6-v2`) |
| API | FastAPI + Uvicorn |
| Session persistence | SQLite (stdlib `sqlite3`) |
| Frontend | Vanilla HTML/CSS/JS SPA (no framework) |
| Testing | pytest |

---

## Project structure

```
app/              FastAPI application, engine, session store, routes, schemas, config
agents/           InterviewerAgent, EvaluatorAgent (IBM Granite via utils/llm.py)
rag/              RAGEngine, VectorStore, embeddings, document loader
evaluation/       Performance report builder
utils/            LLM utility (single IBM watsonx.ai call path)
data/knowledge/   Interview knowledge base (6 topic files)
data/vector_store/ Pre-built FAISS index
static/           Single-page UI (index.html)
tests/            66 pytest tests (mocked + live IBM)
docs/             Architecture and project specification
```

---

*Built with IBM Bob — IBM SkillsBuild · Edunet · AICTE internship project*
