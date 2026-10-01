# DocuLens

An AI-assisted document summarization web app. Paste text or upload a `.txt`/`.md`
file, pick a summary length, and get a concise summary plus extracted keywords.

It works **fully offline** out of the box using an extractive summarizer (word-frequency
sentence scoring, zero ML dependencies). Set `OPENAI_API_KEY` to enable abstractive
AI summaries via any OpenAI-compatible chat completions endpoint — the app falls back
to the extractive summarizer automatically if the LLM call fails.

## Architecture

```
┌─────────────┐      POST /api/summarize       ┌──────────────────┐
│  Frontend   │  ─────────────────────────────▶ │    Backend       │
│ React + TS  │                                │  FastAPI (Py)    │
│ (Vite)      │  ◀───────────────────────────── │                  │
└─────────────┘      JSON: summary, keywords    │  extractive NLP  │
                                                   │  └─▶ optional  │
                                                   │  OpenAI-compatible│
                                                   │  chat completions │
                                                   └──────────────────┘
```

- **Summarization strategy**: extractive baseline always runs offline. If
  `OPENAI_API_KEY` is set, the backend first tries an abstractive summary via the
  configured chat completions endpoint; any failure (network, auth, bad response)
  falls back to the extractive result. The response's `mode` field tells you which
  was used.
- **Keywords**: top content words by frequency (stopwords filtered).

## Tech Stack

| Layer    | Tech                                                  |
|----------|-------------------------------------------------------|
| Backend  | Python 3.12, FastAPI, Uvicorn, httpx, pytest          |
| Frontend | React 18, TypeScript, Vite, plain CSS (no UI library) |
| Serving  | Docker, docker-compose, nginx (frontend prod image)   |

## Local Dev Setup

**Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev        # serves on http://localhost:5173, proxies /api to :8000
```

Run the backend tests:

```bash
cd backend
.venv/bin/python -m pytest -v
```

## Docker Setup

```bash
# from the project root
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000 (`/api/health`, `/api/summarize`)

To enable AI summaries, export `OPENAI_API_KEY` (and optionally
`OPENAI_BASE_URL` / `OPENAI_MODEL`) before `docker compose up`.

## API Reference

### `GET /api/health`

```json
{ "status": "ok", "version": "1.0.0", "llm_enabled": false }
```

### `POST /api/summarize`

Multipart form fields:

| Field    | Type             | Required | Description                                      |
|----------|------------------|----------|--------------------------------------------------|
| `text`   | string           | *        | Raw document text                                |
| `file`   | `.txt` / `.md`   | *        | File upload (takes precedence if both are given)  |
| `length` | `short\|medium\|detailed` | no | Summary length (default `medium`)          |

\* At least one of `text` or `file` must be provided.

Response:

```json
{
  "summary": "…",
  "mode": "extractive",
  "length": "medium",
  "keywords": ["evaluation", "models", "software"],
  "sentences_used": 6,
  "char_count": 1234,
  "word_count": 210
}
```

`mode` is `"abstractive"` when the LLM produced the summary, `"extractive"` otherwise.

## Environment Variables

| Variable          | Default                    | Description                                        |
|-------------------|----------------------------|----------------------------------------------------|
| `OPENAI_API_KEY`  | *(unset)*                  | Enables abstractive summaries when set             |
| `OPENAI_BASE_URL` | `https://api.openai.com`   | Base URL of an OpenAI-compatible API               |
| `OPENAI_MODEL`    | `gpt-4o-mini`              | Chat model used for abstractive summaries          |
| `VITE_API_URL`    | *(empty → relative `/api`)*| Backend base URL baked into the frontend build     |

Copy `backend/.env.example` to `backend/.env` for local development.

## License

MIT — see `LICENSE` (to be added) for details. Free to use, modify, and distribute.
