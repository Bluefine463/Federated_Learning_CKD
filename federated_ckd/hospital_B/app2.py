from fastapi import FastAPI
import psycopg2
import pickle

app2 = FastAPI()

# ---------------- CENTRAL DB CONNECTION ----------------
# This connects ONLY to the Model Hub
CENTRAL_DB_PARAMS = {
    "host": "central-db.postgres.database.azure.com",
    "database": "centraldb",
    "user": "adminuser",
    "password": "Phase2ckd",
    "sslmode": "require"
}

HOSPITAL_ID = "hospitalB" # Change to "hospitalb" for the other node

@app2.post("/push_to_central")
def push_to_central():
    """Reads the local .pkl and pushes it to central-db"""
    try:
        # Load the file created by app.py's /train_local
        with open(f"{HOSPITAL_ID}_model.pkl", "rb") as f:
            model_bytes = f.read()

        conn = psycopg2.connect(**CENTRAL_DB_PARAMS)
        cursor = conn.cursor()
        
        # Insert into the central table designated for this hospital
        cursor.execute(f"INSERT INTO {HOSPITAL_ID} (model_data) VALUES (%s)", (psycopg2.Binary(model_bytes),))
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"status": "Local model pushed to Central DB"}
    except FileNotFoundError:
        return {"error": "Local model file not found. Run /train_local in app.py first."}
    except Exception as e:
        return {"error": str(e)}

@app2.get("/pull_from_central")
def pull_from_central():
    """Downloads the aggregated global model to initial_model.pkl"""
    try:
        conn = psycopg2.connect(**CENTRAL_DB_PARAMS)
        cursor = conn.cursor()
        
        # Get the one record from the aggregated table
        cursor.execute("SELECT model_data FROM aggregated ORDER BY updated_at DESC LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            return {"error": "Aggregated model not found in central-db"}

        # Overwrite the starting model for app.py
        with open("initial_model.pkl", "wb") as f:
            f.write(row[0])
            
        cursor.close()
        conn.close()
        return {"status": "Global model pulled. app.py is now ready for a new training round."}
    except Exception as e:
        return {"error": str(e)}