# Federated Learning CKD - Professional Demo Console

This project now uses the **existing central server** as an official-looking federated experimentation console.

## What is now implemented

- Professional frontend at `/demo` with:
  - polished enterprise-style layout
  - loading animations and live log stream
  - comparison table + animated bar comparison chart
  - model code visibility panel
  - persisted run history panel
- Dataset upload directly to database (`SQLite`) via `POST /api/upload_dataset`.
- Dataset configuration (`target_column`) and automatic compatible model list by task type.
- Federated run orchestration with local-vs-federated-vs-centralized comparison.
- Run artifacts (summary + per-model rows) persisted in database tables:
  - `experiment_runs`
  - `experiment_rows`
- Model code generation endpoint with `openai()` integration + deterministic fallback template.
- Model ping endpoint for querying trained global model in memory.

## Main files

- `federated_ckd/central_server/central_server.py`
  - frontend (`/demo`)
  - dataset APIs
  - model code API
  - run/status/history APIs
  - ping API
- `federated_ckd/fl_core.py`
  - algorithm registry
  - dataset prep
  - non-IID partitioning
  - aggregation
  - evaluation

## Run commands

```bash
pip install fastapi uvicorn pandas numpy scikit-learn psycopg2-binary
uvicorn federated_ckd.central_server.central_server:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://localhost:8000/demo`

## API flow

1. Upload dataset to DB: `POST /api/upload_dataset`
2. Configure target + task models: `POST /api/configure_dataset`
3. Preview model code: `GET /api/model_code`
4. Run comparison: `POST /api/run`
5. Track progress: `GET /api/status`
6. Inspect persisted history: `GET /api/runs`, `GET /api/run/{run_id}`
7. Ping model: `POST /api/ping`

## Note on `openai()`

- If `OPENAI_API_KEY` and OpenAI SDK are available, model code prompt is sent using `openai()` wrapper.
- If unavailable, the system still works via deterministic local template fallback.
