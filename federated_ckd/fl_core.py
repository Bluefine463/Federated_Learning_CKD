from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class RunMetrics:
    primary: float
    secondary: float


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "task": "classification",
        "factory": lambda: LogisticRegression(max_iter=2000, random_state=42),
        "label": "Logistic Regression",
    },
    "sgd_classifier": {
        "task": "classification",
        "factory": lambda: SGDClassifier(loss="log_loss", max_iter=2500, random_state=42),
        "label": "SGD Classifier",
    },
    "mlp_classifier": {
        "task": "classification",
        "factory": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42),
        "label": "MLP Classifier",
    },
    "linear_regression": {
        "task": "regression",
        "factory": lambda: LinearRegression(),
        "label": "Linear Regression",
    },
    "sgd_regressor": {
        "task": "regression",
        "factory": lambda: SGDRegressor(max_iter=2500, random_state=42),
        "label": "SGD Regressor",
    },
    "mlp_regressor": {
        "task": "regression",
        "factory": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42),
        "label": "MLP Regressor",
    },
}


def suggest_task(df: pd.DataFrame, target_col: str) -> str:
    y = df[target_col]
    if y.dtype == object or y.nunique() <= 12:
        return "classification"
    return "regression"


def models_for_task(task: str) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in MODEL_REGISTRY.items() if v["task"] == task}


def prepare_dataset_from_df(df: pd.DataFrame, target_col: str, test_size: float, seed: int):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    data = df.copy()
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = pd.factorize(data[col].astype(str))[0]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    X = data.drop(columns=[target_col]).astype(float).values
    y = data[target_col].values
    task = suggest_task(data, target_col)

    if task == "classification":
        y = pd.factorize(pd.Series(y).astype(str))[0]

    stratify = None
    if task == "classification":
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) > 1 and np.min(counts) >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, task


def partition_non_iid(y: np.ndarray, task: str, n_nodes: int, alpha: float, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    if task == "regression":
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return [x.astype(int) for x in np.array_split(idx, n_nodes)]

    classes = np.unique(y)
    node_indices: List[List[int]] = [[] for _ in range(n_nodes)]
    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_nodes))
        cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        chunks = np.split(class_idx, cuts)
        for node, chunk in enumerate(chunks):
            node_indices[node].extend(chunk.tolist())
    return [np.array(idx, dtype=int) for idx in node_indices]


def _aggregate_linear(local_models: List[Any], sample_counts: List[int]):
    weights = np.array(sample_counts, dtype=float)
    weights = weights / weights.sum()
    global_model = clone(local_models[0])

    if hasattr(local_models[0], "coef_"):
        global_model.coef_ = np.sum([w * m.coef_ for w, m in zip(weights, local_models)], axis=0)
    if hasattr(local_models[0], "intercept_"):
        global_model.intercept_ = np.sum([w * m.intercept_ for w, m in zip(weights, local_models)], axis=0)
    if hasattr(local_models[0], "classes_"):
        global_model.classes_ = local_models[0].classes_
    if hasattr(local_models[0], "n_features_in_"):
        global_model.n_features_in_ = local_models[0].n_features_in_
    return global_model


def _aggregate_mlp(local_models: List[Any], sample_counts: List[int]):
    weights = np.array(sample_counts, dtype=float)
    weights = weights / weights.sum()
    global_model = clone(local_models[0])
    global_model.coefs_ = []
    global_model.intercepts_ = []
    for layer in range(len(local_models[0].coefs_)):
        global_model.coefs_.append(np.sum([w * m.coefs_[layer] for w, m in zip(weights, local_models)], axis=0))
    for layer in range(len(local_models[0].intercepts_)):
        global_model.intercepts_.append(np.sum([w * m.intercepts_[layer] for w, m in zip(weights, local_models)], axis=0))

    for attr in ["n_layers_", "n_outputs_", "out_activation_", "classes_", "n_features_in_", "loss_"]:
        if hasattr(local_models[0], attr):
            setattr(global_model, attr, getattr(local_models[0], attr))
    return global_model


def aggregate_models(local_models: List[Any], sample_counts: List[int]):
    if hasattr(local_models[0], "coefs_") and hasattr(local_models[0], "intercepts_"):
        return _aggregate_mlp(local_models, sample_counts)
    return _aggregate_linear(local_models, sample_counts)


def evaluate(task: str, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> RunMetrics:
    preds = model.predict(X_test)
    if task == "classification":
        return RunMetrics(primary=float(accuracy_score(y_test, preds)), secondary=float(f1_score(y_test, preds, average="macro")))
    return RunMetrics(primary=float(r2_score(y_test, preds)), secondary=float(mean_absolute_error(y_test, preds)))


def model_code_template(model_key: str, task: str) -> str:
    snippets = {
        "logistic_regression": "model = LogisticRegression(max_iter=2000, random_state=42)",
        "sgd_classifier": "model = SGDClassifier(loss='log_loss', max_iter=2500, random_state=42)",
        "mlp_classifier": "model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=800, random_state=42)",
        "linear_regression": "model = LinearRegression()",
        "sgd_regressor": "model = SGDRegressor(max_iter=2500, random_state=42)",
        "mlp_regressor": "model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=800, random_state=42)",
    }
    return f"""# Auto-configured {task} model\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, SGDRegressor\nfrom sklearn.neural_network import MLPClassifier, MLPRegressor\n\n{snippets[model_key]}\nmodel.fit(X_train, y_train)\npred = model.predict(X_test)\n"""
