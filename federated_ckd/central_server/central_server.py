import psycopg2
import pickle
import numpy as np

CENTRAL_DB = {
    "host": "central-db.postgres.database.azure.com",
    "database": "centraldb",
    "user": "adminuser",
    "password": "Phase2ckd",
    "sslmode": "require"
}

def run_aggregation():
    conn = psycopg2.connect(**CENTRAL_DB)
    cursor = conn.cursor()

    # 1. Pull latest from A and B
    models = []
    for table in ["hospitala", "hospitalb"]:
        cursor.execute(f"SELECT model_data FROM {table} ORDER BY updated_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            models.append(pickle.loads(row[0]))

    if len(models) < 2:
        print("Missing data from one or more hospitals.")
        return

    # 2. Federated Averaging Math
    avg_coef = np.mean([m['model'].coef_ for m in models], axis=0)
    avg_intercept = np.mean([m['model'].intercept_ for m in models], axis=0)

    # 3. Create Global Model Object
    global_assets = models[0] # Template
    global_assets['model'].coef_ = avg_coef
    global_assets['model'].intercept_ = avg_intercept

    # 4. Clear 'aggregated' table and store the one true record
    cursor.execute("TRUNCATE TABLE aggregated")
    cursor.execute(
        "INSERT INTO aggregated (model_data) VALUES (%s)",
        (psycopg2.Binary(pickle.dumps(global_assets)),)
    )
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Aggregation Successful: Global model updated in Central DB.")

if __name__ == "__main__":
    run_aggregation()