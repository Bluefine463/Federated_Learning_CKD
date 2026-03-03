import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("kidney.csv")

# 2. Drop ID
if 'id' in df.columns:
    df = df.drop(columns=["id"])

# 3. Strip spaces & replace '?'
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.replace("?", np.nan)

# 4. Explicit numeric columns
numeric_cols = [
    'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Fill missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df = df.ffill().bfill()

# 6. Explicit categorical columns
categorical_cols = [
    'rbc', 'pc', 'pcc', 'ba', 'htn',
    'dm', 'cad', 'appet', 'pe', 'ane', 'classification'
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 7. Split X & y
X = df.drop(columns=["classification"])
y = df["classification"]

# 8. Train model
model = LogisticRegression(max_iter=5000)
model.fit(X, y)

# 9. Save artifacts
artifacts = {
    "model": model,
    "encoders": label_encoders,
    "feature_names": list(X.columns)
}

with open("initial_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ Success! Initial assets saved.")
print(f"Model trained on {len(df)} samples with {X.shape[1]} features.")
