from fastapi import FastAPI, Body
import psycopg2
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from federated_ckd.fl_core import ALGORITHMS, evaluate

# ---------------- APP ----------------
app = FastAPI()

# ---------------- LOAD MODEL ARTIFACTS (GLOBAL) ----------------
with open("initial_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
encoders = artifacts["encoders"]
feature_names = artifacts["feature_names"]

# ---------------- DB CONNECTION (GLOBAL) ----------------
conn = psycopg2.connect(
    host="hospitalb-db.postgres.database.azure.com",
    database="hospitalb",
    user="adminuser",
    password="Phase2ckd",
    port=5432,
    sslmode="require"
)

# ---------------- ADD PATIENT ----------------
@app.post("/add_patient")
def add_patient(data: dict = Body(...)):

    cursor = conn.cursor()

    query = """
        INSERT INTO patient_records (
            age, bp, sg, al, su, rbc, pc, pcc, ba,
            bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
            htn, dm, cad, appet, pe, ane, classification
        )
        VALUES (
            %(age)s, %(bp)s, %(sg)s, %(al)s, %(su)s, %(rbc)s, %(pc)s, %(pcc)s, %(ba)s,
            %(bgr)s, %(bu)s, %(sc)s, %(sod)s, %(pot)s, %(hemo)s, %(pcv)s, %(wc)s, %(rc)s,
            %(htn)s, %(dm)s, %(cad)s, %(appet)s, %(pe)s, %(ane)s, %(classification)s
        )
    """

    cursor.execute(query, data)
    conn.commit()
    cursor.close()

    return {"status": "patient record added successfully"}

# ---------------- LOCAL TRAINING (MANUAL TRIGGER) ----------------
@app.post("/train_local")
def train_local():

    df = pd.read_sql("SELECT * FROM patient_records", conn)

    if len(df) < 5:
        return {"message": "Not enough data to train"}

    # Drop DB-only columns
    df = df.drop(columns=["id", "created_at"])

    # Numeric columns
    numeric_cols = [
        'age','bp','sg','al','su','bgr','bu','sc',
        'sod','pot','hemo','pcv','wc','rc'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Categorical columns
    categorical_cols = [
        'rbc','pc','pcc','ba','htn','dm',
        'cad','appet','pe','ane','classification'
    ]

    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col].astype(str))

    # Split X and y
    X = df.drop(columns=["classification"])
    y = df["classification"]

    # Ensure same column order
    X = X[feature_names]

    # Train locally
    model.fit(X, y)

    # Save updated model
    updated_artifacts = {
        "model": model,
        "encoders": encoders,
        "feature_names": feature_names
    }

    with open("hospitalB_model.pkl", "wb") as f:
        pickle.dump(updated_artifacts, f)

    return {
        "status": "Local training complete",
        "records_used": len(df),
        "model_saved": "hospitalB_model.pkl"
    }


# ---------------- LOCAL TRAINING HELPER FOR ORCHESTRATION ----------------
def train_local_model(X_local, y_local, algorithm_name: str):
    model = ALGORITHMS[algorithm_name]()
    model.fit(X_local, y_local)
    return model


def evaluate_local_model(model, X_test, y_test):
    return evaluate(model, X_test, y_test)
