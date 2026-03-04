from __future__ import annotations

import io
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from federated_ckd.fl_core import (
    MODEL_REGISTRY,
    aggregate_models,
    evaluate,
    model_code_template,
    models_for_task,
    partition_non_iid,
    prepare_dataset_from_df,
)

CENTRAL_DB = {
    "host": "central-db.postgres.database.azure.com",
    "database": "centraldb",
    "user": "adminuser",
    "password": "Phase2ckd",
    "sslmode": "require",
}

SQLITE_PATH = "federated_ckd/central_server/demo_store.db"

app = FastAPI(title="Federated CKD Central Server")
RUN_STATE: Dict[str, Any] = {"running": False, "logs": [], "result": None, "models": {}}


class UploadDatasetRequest(BaseModel):
    filename: str
    csv_text: str


class ConfigureDatasetRequest(BaseModel):
    dataset_id: int
    target_column: str


class ExperimentRequest(BaseModel):
    dataset_id: int
    model_keys: List[str]
    nodes: int = 3
    alpha: float = 0.8
    test_size: float = 0.25
    random_seed: int = 42


class PingRequest(BaseModel):
    run_id: str
    model_key: str
    features: List[float]


def openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "# OPENAI_API_KEY not set. Using local generated template.\n"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.responses.create(model="gpt-4o-mini", input=prompt, temperature=0.1)
        return resp.output_text
    except Exception as exc:
        return f"# OpenAI generation unavailable ({exc}). Using local generated template.\n"


def db_conn() -> sqlite3.Connection:
    return sqlite3.connect(SQLITE_PATH, check_same_thread=False)


def init_store():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            csv_blob BLOB NOT NULL,
            target_column TEXT,
            task_type TEXT,
            model_keys TEXT,
            row_count INTEGER,
            col_count INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id TEXT PRIMARY KEY,
            dataset_id INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            task TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            nodes INTEGER NOT NULL,
            alpha REAL NOT NULL,
            test_size REAL NOT NULL,
            random_seed INTEGER NOT NULL,
            all_gain_positive INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            error_text TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_key TEXT NOT NULL,
            model_label TEXT NOT NULL,
            avg_local REAL NOT NULL,
            best_local REAL NOT NULL,
            federated REAL NOT NULL,
            centralized REAL NOT NULL,
            gain_vs_local REAL NOT NULL,
            FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
        )
        """
    )
    conn.commit()
    conn.close()


def add_log(message: str):
    RUN_STATE["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def metric_name_for_task(task: str) -> str:
    return "Accuracy" if task == "classification" else "R2"


def persist_run_start(run_id: str, req: ExperimentRequest, ds: Dict[str, Any], task: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO experiment_runs
        (run_id,dataset_id,dataset_name,task,metric_name,nodes,alpha,test_size,random_seed,all_gain_positive,status,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            req.dataset_id,
            ds["name"],
            task,
            metric_name_for_task(task),
            req.nodes,
            req.alpha,
            req.test_size,
            req.random_seed,
            0,
            "running",
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def persist_run_finish(run_id: str, rows: List[Dict[str, Any]], all_gain_positive: bool, error_text: str | None = None):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM experiment_rows WHERE run_id=?", (run_id,))
    for row in rows:
        cur.execute(
            """
            INSERT INTO experiment_rows
            (run_id,model_key,model_label,avg_local,best_local,federated,centralized,gain_vs_local)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                row["model_key"],
                row["model_label"],
                row["avg_local"],
                row["best_local"],
                row["federated"],
                row["centralized"],
                row["gain_vs_local"],
            ),
        )

    cur.execute(
        """
        UPDATE experiment_runs
        SET all_gain_positive=?, status=?, finished_at=?, error_text=?
        WHERE run_id=?
        """,
        (1 if all_gain_positive else 0, "failed" if error_text else "completed", datetime.utcnow().isoformat(), error_text, run_id),
    )
    conn.commit()
    conn.close()


def run_aggregation():
    conn = psycopg2.connect(**CENTRAL_DB)
    cursor = conn.cursor()
    models = []
    for table in ["hospitala", "hospitalb"]:
        cursor.execute(f"SELECT model_data FROM {table} ORDER BY updated_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            import pickle

            models.append(pickle.loads(row[0]))
    if len(models) < 2:
        print("Missing data from one or more hospitals.")
        return

    avg_coef = np.mean([m["model"].coef_ for m in models], axis=0)
    avg_intercept = np.mean([m["model"].intercept_ for m in models], axis=0)
    global_assets = models[0]
    global_assets["model"].coef_ = avg_coef
    global_assets["model"].intercept_ = avg_intercept

    cursor.execute("TRUNCATE TABLE aggregated")
    cursor.execute(
        "INSERT INTO aggregated (model_data) VALUES (%s)",
        (psycopg2.Binary(__import__("pickle").dumps(global_assets)),),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_dataset(dataset_id: int) -> Dict[str, Any]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,name,csv_blob,target_column,task_type,model_keys,row_count,col_count,uploaded_at FROM datasets WHERE id=?",
        (dataset_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": row[0],
        "name": row[1],
        "csv_blob": row[2],
        "target_column": row[3],
        "task_type": row[4],
        "model_keys": row[5],
        "row_count": row[6],
        "col_count": row[7],
        "uploaded_at": row[8],
    }


def run_experiment_job(req: ExperimentRequest):
    RUN_STATE["running"] = True
    RUN_STATE["logs"] = []
    RUN_STATE["result"] = None
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    RUN_STATE["models"][run_id] = {}

    rows: List[Dict[str, Any]] = []
    try:
        ds = get_dataset(req.dataset_id)
        if not ds["target_column"] or not ds["task_type"]:
            raise ValueError("Dataset not configured. Please set target column first.")

        df = pd.read_csv(io.BytesIO(ds["csv_blob"]))
        X_train, X_test, y_train, y_test, task = prepare_dataset_from_df(
            df, ds["target_column"], req.test_size, req.random_seed
        )
        persist_run_start(run_id, req, ds, task)

        partitions = partition_non_iid(y_train, task, req.nodes, req.alpha, req.random_seed)
        metric_name = metric_name_for_task(task)

        for model_key in req.model_keys:
            if model_key not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_key}")
            if MODEL_REGISTRY[model_key]["task"] != task:
                raise ValueError(f"Model {model_key} is incompatible with {task}")

            add_log(f"Starting model: {model_key}")
            local_models = []
            local_primary = []
            sample_counts = []

            for i, idx in enumerate(partitions, start=1):
                if len(idx) == 0:
                    continue

                X_local = X_train[idx]
                y_local = y_train[idx]
                if task == "classification" and len(np.unique(y_local)) < 2:
                    missing_pool = np.where(y_train != y_local[0])[0]
                    if len(missing_pool) > 0:
                        extra = np.random.choice(missing_pool, size=1, replace=False)
                        X_local = np.vstack([X_local, X_train[extra]])
                        y_local = np.concatenate([y_local, y_train[extra]])

                model = MODEL_REGISTRY[model_key]["factory"]()
                model.fit(X_local, y_local)
                local_metric = evaluate(task, model, X_test, y_test)
                local_models.append(model)
                local_primary.append(local_metric.primary)
                sample_counts.append(len(idx))
                add_log(f"Node {i}/{req.nodes} trained {model_key}; {metric_name}={local_metric.primary:.4f}")
                time.sleep(0.15)

            global_model = aggregate_models(local_models, sample_counts)
            global_metric = evaluate(task, global_model, X_test, y_test)
            centralized_model = MODEL_REGISTRY[model_key]["factory"]()
            centralized_model.fit(X_train, y_train)
            centralized_metric = evaluate(task, centralized_model, X_test, y_test)

            RUN_STATE["models"][run_id][model_key] = global_model
            avg_local = float(np.mean(local_primary))
            row = {
                "model_key": model_key,
                "model_label": MODEL_REGISTRY[model_key]["label"],
                "avg_local": round(avg_local, 4),
                "best_local": round(float(np.max(local_primary)), 4),
                "federated": round(global_metric.primary, 4),
                "centralized": round(centralized_metric.primary, 4),
                "gain_vs_local": round(global_metric.primary - avg_local, 4),
            }
            rows.append(row)
            add_log(f"Central aggregation and comparison done for {model_key}")

        all_gain_positive = all(r["gain_vs_local"] > 0 for r in rows)
        RUN_STATE["result"] = {
            "run_id": run_id,
            "dataset_id": req.dataset_id,
            "task": task,
            "metric_name": metric_name,
            "rows": rows,
            "all_gain_positive": all_gain_positive,
        }
        persist_run_finish(run_id, rows=rows, all_gain_positive=all_gain_positive)
        add_log("Experiment completed")
    except Exception as exc:
        RUN_STATE["result"] = {"error": str(exc)}
        persist_run_finish(run_id, rows=rows, all_gain_positive=False, error_text=str(exc))
        add_log(f"ERROR: {exc}")
    finally:
        RUN_STATE["running"] = False


@app.on_event("startup")
def startup_event():
    init_store()


@app.get("/", response_class=HTMLResponse)
def index():
    return demo_ui()


@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    return """
<!doctype html>
<html>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Federated AI Operations Console</title>
<style>
:root{--bg:#081021;--bg2:#121d33;--card:#0f1b30;--line:#284163;--text:#dbe7ff;--muted:#8fa7ca;--ok:#16a34a;--warn:#f59e0b;--bad:#ef4444;--brand:#3b82f6;--brand2:#7c3aed}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,Segoe UI,Arial;background:radial-gradient(circle at 20% -20%,#1e3a8a55,transparent 40%),radial-gradient(circle at 90% 0,#7c3aed33,transparent 30%),var(--bg);color:var(--text)}
header{padding:22px 28px;border-bottom:1px solid var(--line);background:#091327cc;backdrop-filter:blur(8px);position:sticky;top:0;z-index:5}
header h1{margin:0;font-size:22px}
header p{margin:6px 0 0;color:var(--muted)}
.wrap{padding:20px;display:grid;grid-template-columns:1.2fr 1fr;gap:16px}
.card{background:linear-gradient(180deg,#12213a,#0c182c);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:0 8px 26px #00000055;animation:slideIn .35s ease}
@keyframes slideIn{from{transform:translateY(8px);opacity:.6}to{transform:translateY(0);opacity:1}}
.step{font-size:12px;display:inline-block;border:1px solid #355682;border-radius:99px;padding:4px 8px;color:#a9c4eb;margin-bottom:8px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
label{font-size:12px;color:var(--muted)}
input,select,button,textarea{width:100%;margin-top:6px;background:#0a1426;border:1px solid #36557e;color:var(--text);border-radius:10px;padding:10px}
button{background:linear-gradient(90deg,var(--brand),var(--brand2));border:none;font-weight:700;cursor:pointer;transition:.2s}
button:hover{transform:translateY(-1px)}
button:active{transform:translateY(0)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#060d1a;border:1px solid #2e4667;border-radius:10px;padding:10px;max-height:240px;overflow:auto;white-space:pre-wrap}
.table{width:100%;border-collapse:collapse;overflow:hidden;border-radius:8px}
.table th,.table td{border:1px solid #2b4466;padding:8px;text-align:left}
.table th{background:#112443}
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}
.k{background:#0a172b;border:1px solid #2a4568;border-radius:10px;padding:10px}
.k b{font-size:20px}
.status{margin-top:8px;min-height:20px;color:#9bc5ff}
.spin{display:inline-block;width:12px;height:12px;border:2px solid #93c5fd;border-top-color:transparent;border-radius:999px;animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.chart{display:flex;gap:10px;align-items:flex-end;min-height:200px;padding:10px;background:#081327;border:1px solid #2a4568;border-radius:10px}
.barGroup{flex:1;display:flex;flex-direction:column;align-items:center;gap:8px}
.bars{display:flex;gap:5px;align-items:flex-end;height:150px}
.bar{width:18px;border-radius:6px 6px 0 0;transition:height .7s ease}
.bar.local{background:#64748b}.bar.fed{background:#3b82f6}.bar.cent{background:#8b5cf6}
.legend{font-size:11px;color:var(--muted)}
.badge{font-size:11px;padding:4px 8px;border:1px solid #355682;border-radius:99px;color:#b8d3f5}
@media (max-width:1050px){.wrap{grid-template-columns:1fr}.row,.kpi{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Federated AI Operations Console <span class='badge'>Professional Demo</span></h1>
  <p>Upload datasets, auto-map compatible models, run federated comparisons, and inspect persisted run history.</p>
</header>
<div class='wrap'>
  <div class='card'>
    <div class='step'>STEP 1</div>
    <h3>Upload dataset to database</h3>
    <input id='csv' type='file' accept='.csv'/>
    <button onclick='uploadDataset()'>Upload</button>
    <div id='upload_status' class='status'></div>

    <div class='step'>STEP 2</div>
    <h3>Configure task and compatible model list</h3>
    <div class='row'>
      <div><label>Dataset</label><select id='dataset_select'></select></div>
      <div><label>Target column</label><input id='target_col' placeholder='e.g. classification'/></div>
    </div>
    <button onclick='configureDataset()'>Configure</button>
    <div id='model_box' class='status'></div>

    <div class='step'>STEP 3</div>
    <h3>Run federated model comparison</h3>
    <div class='row'>
      <div><label>Nodes</label><input id='nodes' type='number' value='3' min='2' max='10'/></div>
      <div><label>Alpha (non-IID)</label><input id='alpha' type='number' value='0.8' step='0.1'/></div>
    </div>
    <button onclick='runExperiment()'>Run Comparison</button>
    <div id='run_state' class='status'></div>
  </div>

  <div class='card'>
    <h3>Live backend log stream</h3>
    <pre id='logs' class='mono'>No run yet.</pre>
    <h3>Model code visibility</h3>
    <select id='code_model_select' onchange='loadModelCode()'></select>
    <pre id='model_code' class='mono'>Select a configured model.</pre>
  </div>

  <div class='card' style='grid-column:1 / span 2;'>
    <h3>Comparison results</h3>
    <div class='kpi'>
      <div class='k'><div>Latest Run ID</div><b id='k_run'>—</b></div>
      <div class='k'><div>Metric</div><b id='k_metric'>—</b></div>
      <div class='k'><div>Federated Better?</div><b id='k_gain'>—</b></div>
    </div>
    <div id='results'>No results yet.</div>
    <div id='chart' class='chart'></div>
  </div>

  <div class='card' style='grid-column:1 / span 2;'>
    <h3>Persisted experiment history (from DB)</h3>
    <div id='history'>No runs yet.</div>
  </div>

  <div class='card' style='grid-column:1 / span 2;'>
    <h3>Ping trained global model</h3>
    <div class='row'>
      <div><label>Run ID</label><input id='ping_run_id' placeholder='auto-filled from latest run'/></div>
      <div><label>Model key</label><input id='ping_model' placeholder='e.g. logistic_regression'/></div>
    </div>
    <textarea id='ping_features' rows='3' placeholder='Comma-separated numeric feature values'></textarea>
    <button onclick='pingModel()'>Ping</button>
    <pre id='ping_out' class='mono'>No ping yet.</pre>
  </div>
</div>

<script>
let poller = null;
let configuredModels = [];

function esc(s){return (s??'').toString().replace(/[&<>\"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]))}

async function refreshDatasets(){
  const res = await fetch('/api/datasets');
  const data = await res.json();
  dataset_select.innerHTML = data.map(d=>`<option value='${d.id}'>#${d.id} ${esc(d.name)} (${d.row_count}×${d.col_count})</option>`).join('');
}

async function refreshRuns(){
  const res = await fetch('/api/runs?limit=8');
  const data = await res.json();
  if(!data.length){history.innerHTML='No runs yet.'; return;}
  history.innerHTML = `<table class='table'><tr><th>Run</th><th>Dataset</th><th>Status</th><th>Task</th><th>Metric</th><th>Created</th></tr>`+
    data.map(r=>`<tr><td>${esc(r.run_id)}</td><td>${esc(r.dataset_name)}</td><td>${esc(r.status)}</td><td>${esc(r.task)}</td><td>${esc(r.metric_name)}</td><td>${esc(r.created_at)}</td></tr>`).join('')+`</table>`;
}

async function uploadDataset(){
  const f = csv.files[0];
  if(!f){alert('Choose CSV file first.');return;}
  const csvText = await f.text();
  upload_status.innerHTML = "<span class='spin'></span> uploading and indexing...";
  const res = await fetch('/api/upload_dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:f.name,csv_text:csvText})});
  const j = await res.json();
  upload_status.textContent = j.status ? `Uploaded dataset #${j.dataset_id}` : JSON.stringify(j);
  await refreshDatasets();
}

async function configureDataset(){
  const payload={dataset_id:parseInt(dataset_select.value),target_column:target_col.value};
  const res = await fetch('/api/configure_dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const j = await res.json();
  configuredModels = j.models || [];
  model_box.innerHTML = `<b>Task:</b> ${esc(j.task)}<br><b>Compatible Models:</b><br>`+
    configuredModels.map(m=>`<label><input type='checkbox' name='mdl' value='${m.key}' checked/> ${esc(m.label)} <span class='badge'>${esc(m.key)}</span></label>`).join('<br>');
  code_model_select.innerHTML = configuredModels.map(m=>`<option value='${m.key}'>${esc(m.label)}</option>`).join('');
  if(configuredModels.length) ping_model.value = configuredModels[0].key;
  loadModelCode();
}

async function loadModelCode(){
  const modelKey = code_model_select.value;
  if(!modelKey) return;
  const datasetId = parseInt(dataset_select.value || '0');
  const res = await fetch(`/api/model_code?dataset_id=${datasetId}&model_key=${modelKey}`);
  const j = await res.json();
  model_code.textContent = j.code;
}

async function runExperiment(){
  const keys = Array.from(document.querySelectorAll("input[name='mdl']:checked")).map(x=>x.value);
  if(!keys.length){alert('Select at least one model.');return;}
  const payload={dataset_id:parseInt(dataset_select.value),model_keys:keys,nodes:parseInt(nodes.value),alpha:parseFloat(alpha.value),test_size:0.25,random_seed:42};
  run_state.innerHTML = "<span class='spin'></span> running federated comparison...";
  const res = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!res.ok){run_state.textContent='Failed to start'; return;}
  if(poller) clearInterval(poller);
  poller = setInterval(fetchStatus, 700);
}

function renderChart(rows){
  if(!rows || !rows.length){chart.innerHTML=''; return;}
  let html='';
  rows.forEach(r=>{
    const maxv=Math.max(r.avg_local,r.federated,r.centralized,0.0001);
    const h1=Math.max(8,Math.round((r.avg_local/maxv)*140));
    const h2=Math.max(8,Math.round((r.federated/maxv)*140));
    const h3=Math.max(8,Math.round((r.centralized/maxv)*140));
    html += `<div class='barGroup'><div class='bars'>
      <div class='bar local' style='height:${h1}px' title='Avg Local ${r.avg_local}'></div>
      <div class='bar fed' style='height:${h2}px' title='Federated ${r.federated}'></div>
      <div class='bar cent' style='height:${h3}px' title='Centralized ${r.centralized}'></div>
    </div><div class='legend'>${esc(r.model_label)}</div></div>`;
  });
  chart.innerHTML = html;
}

async function fetchStatus(){
  const s = await (await fetch('/api/status')).json();
  logs.textContent = s.logs.join('\n');
  if(!s.running && s.result){
    run_state.textContent = 'completed';
    if(poller) clearInterval(poller);

    if(s.result.error){
      results.innerHTML = `<b style='color:#fca5a5'>${esc(s.result.error)}</b>`;
      await refreshRuns();
      return;
    }

    k_run.textContent = s.result.run_id;
    k_metric.textContent = s.result.metric_name;
    k_gain.textContent = s.result.all_gain_positive ? 'YES' : 'MIXED';
    ping_run_id.value = s.result.run_id;

    results.innerHTML = `<table class='table'><tr><th>Model</th><th>Avg Local</th><th>Best Local</th><th>Federated</th><th>Centralized</th><th>Gain</th></tr>`+
      s.result.rows.map(r=>`<tr><td>${esc(r.model_label)}</td><td>${r.avg_local}</td><td>${r.best_local}</td><td>${r.federated}</td><td>${r.centralized}</td><td>${r.gain_vs_local}</td></tr>`).join('')+`</table>`+
      `<p><b>${s.result.all_gain_positive ? 'Federated outperformed average local across all selected models.' : 'Federated gains are mixed for this run.'}</b></p>`;

    renderChart(s.result.rows);
    await refreshRuns();
  }
}

async function pingModel(){
  const features = ping_features.value.split(',').map(x=>parseFloat(x.trim())).filter(x=>!Number.isNaN(x));
  const payload={run_id:ping_run_id.value,model_key:ping_model.value,features};
  const res = await fetch('/api/ping',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  ping_out.textContent = JSON.stringify(await res.json(), null, 2);
}

refreshDatasets();
refreshRuns();
</script>
</body>
</html>
"""


@app.post("/api/upload_dataset")
def upload_dataset(req: UploadDatasetRequest):
    if not req.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    raw = req.csv_text.encode("utf-8")
    df = pd.read_csv(io.StringIO(req.csv_text))
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (name,uploaded_at,csv_blob,row_count,col_count) VALUES (?,?,?,?,?)",
        (req.filename, datetime.utcnow().isoformat(), raw, len(df), len(df.columns)),
    )
    ds_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"status": "uploaded", "dataset_id": ds_id, "columns": list(df.columns)}


@app.get("/api/datasets")
def list_datasets():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,name,row_count,col_count,uploaded_at,target_column,task_type,model_keys FROM datasets ORDER BY id DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "name": r[1],
            "row_count": r[2],
            "col_count": r[3],
            "uploaded_at": r[4],
            "target_column": r[5],
            "task_type": r[6],
            "model_keys": r[7].split(",") if r[7] else [],
        }
        for r in rows
    ]


@app.post("/api/configure_dataset")
def configure_dataset(req: ConfigureDatasetRequest):
    ds = get_dataset(req.dataset_id)
    df = pd.read_csv(io.BytesIO(ds["csv_blob"]))
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found in dataset")

    _, _, _, _, task = prepare_dataset_from_df(df, req.target_column, 0.25, 42)
    model_map = models_for_task(task)

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE datasets SET target_column=?, task_type=?, model_keys=? WHERE id=?",
        (req.target_column, task, ",".join(model_map.keys()), req.dataset_id),
    )
    conn.commit()
    conn.close()

    return {
        "status": "configured",
        "dataset_id": req.dataset_id,
        "task": task,
        "models": [{"key": k, "label": v["label"]} for k, v in model_map.items()],
    }


@app.get("/api/model_code")
def model_code(dataset_id: int, model_key: str):
    ds = get_dataset(dataset_id)
    task = ds.get("task_type") or "classification"
    prompt = (
        f"Generate concise sklearn training code for task={task}, model={model_key}. "
        "Return Python only."
    )
    code = openai(prompt) + model_code_template(model_key, task)
    return {"dataset_id": dataset_id, "model_key": model_key, "code": code}


@app.post("/api/run")
def api_run(req: ExperimentRequest):
    if RUN_STATE["running"]:
        raise HTTPException(status_code=409, detail="Experiment already running")
    worker = threading.Thread(target=run_experiment_job, args=(req,), daemon=True)
    worker.start()
    return {"status": "started"}


@app.get("/api/status")
def api_status():
    return {
        "running": RUN_STATE["running"],
        "logs": RUN_STATE["logs"][-300:],
        "result": RUN_STATE["result"],
    }


@app.get("/api/runs")
def list_runs(limit: int = 10):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id,dataset_name,status,task,metric_name,created_at,finished_at,all_gain_positive,error_text
        FROM experiment_runs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (max(1, min(limit, 100)),),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "run_id": r[0],
            "dataset_name": r[1],
            "status": r[2],
            "task": r[3],
            "metric_name": r[4],
            "created_at": r[5],
            "finished_at": r[6],
            "all_gain_positive": bool(r[7]),
            "error_text": r[8],
        }
        for r in rows
    ]


@app.get("/api/run/{run_id}")
def get_run_details(run_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id,dataset_id,dataset_name,status,task,metric_name,nodes,alpha,test_size,random_seed,all_gain_positive,created_at,finished_at,error_text FROM experiment_runs WHERE run_id=?",
        (run_id,),
    )
    header = cur.fetchone()
    if not header:
        conn.close()
        raise HTTPException(status_code=404, detail="Run not found")

    cur.execute(
        "SELECT model_key,model_label,avg_local,best_local,federated,centralized,gain_vs_local FROM experiment_rows WHERE run_id=?",
        (run_id,),
    )
    details = cur.fetchall()
    conn.close()
    return {
        "run_id": header[0],
        "dataset_id": header[1],
        "dataset_name": header[2],
        "status": header[3],
        "task": header[4],
        "metric_name": header[5],
        "nodes": header[6],
        "alpha": header[7],
        "test_size": header[8],
        "random_seed": header[9],
        "all_gain_positive": bool(header[10]),
        "created_at": header[11],
        "finished_at": header[12],
        "error_text": header[13],
        "rows": [
            {
                "model_key": d[0],
                "model_label": d[1],
                "avg_local": d[2],
                "best_local": d[3],
                "federated": d[4],
                "centralized": d[5],
                "gain_vs_local": d[6],
            }
            for d in details
        ],
    }


@app.post("/api/ping")
def ping_model(req: PingRequest):
    model = RUN_STATE["models"].get(req.run_id, {}).get(req.model_key)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found in memory for run/model key")
    x = np.array(req.features, dtype=float).reshape(1, -1)
    prediction = model.predict(x).tolist()
    return {"run_id": req.run_id, "model_key": req.model_key, "prediction": prediction}


if __name__ == "__main__":
    run_aggregation()
