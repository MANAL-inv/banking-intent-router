import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from src import config
from src.selection import select_one_se
from src.splits import load_splits

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPLIT_NAMES = ["train", "val", "test_in", "test_idoos", "test_oodoos"]

COMPLEXITY = {
    "emb_centroid": 0.5,
    "emb_logreg_C1": 1.0,
    "emb_logreg_C3": 3.0,
    "emb_logreg_C10": 10.0,
    "emb_logreg_C30": 30.0,
    "emb_logreg_C100": 100.0,
    "emb_logreg_C300": 300.0,
    "emb_knn_k1": 500.0,
    "emb_knn_k5": 500.0,
    "emb_knn_k15": 500.0,
}


def _emb_path(split: str):
    return config.DATA_PROCESSED / f"emb_{split}.npy"


def embed_all(force: bool = False):
    splits = load_splits()
    todo = [s for s in SPLIT_NAMES if force or not _emb_path(s).exists()]

    if todo:
        from sentence_transformers import SentenceTransformer

        print(f"loading encoder: {ENCODER_NAME}")
        model = SentenceTransformer(ENCODER_NAME)

        for split in todo:
            texts = splits[split]["text"].tolist()
            t0 = time.perf_counter()
            emb = model.encode(
                texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            np.save(_emb_path(split), emb.astype(np.float32))
            print(f"{split:12s} {emb.shape}  {time.perf_counter() - t0:.1f}s")

    embeddings = {s: np.load(_emb_path(s)) for s in SPLIT_NAMES}
    return embeddings, splits


def candidates() -> dict:
    grid = {}
    for C in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        grid[f"emb_logreg_C{C:g}"] = LogisticRegression(
            C=C,
            max_iter=3000,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
    for k in (1, 5, 15):
        grid[f"emb_knn_k{k}"] = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            weights="distance",
        )
    grid["emb_centroid"] = NearestCentroid(metric="euclidean")
    return grid


def evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def run() -> pd.DataFrame:
    emb, splits = embed_all()

    X_train, y_train = emb["train"], splits["train"]["label"].to_numpy()
    X_val, y_val = emb["val"], splits["val"]["label"].to_numpy()
    X_test, y_test = emb["test_in"], splits["test_in"]["label"].to_numpy()

    print(f"\nembedding dim: {X_train.shape[1]}  train shape: {X_train.shape}\n")

    rows, fitted = [], {}
    for name, model in candidates().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        scores = evaluate(model, X_val, y_val)
        rows.append({"model": name, "fit_seconds": round(fit_s, 2), **scores})
        fitted[name] = model
        print(f"{name:20s} val_macro_f1={scores['macro_f1']:.4f} "
              f"acc={scores['accuracy']:.4f} ({fit_s:.1f}s)")

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best_name, sel_info = select_one_se(results, COMPLEXITY, n_val=len(y_val))
    best = fitted[best_name]

    print(f"\nselection: argmax={sel_info['argmax_model']} ({sel_info['argmax_score']}), "
          f"SE={sel_info['standard_error']}, {sel_info['n_eligible']} within 1 SE "
          f"-> selected {sel_info['selected']}")

    test_scores = evaluate(best, X_test, y_test)
    print(f"TEST in-scope: macro_f1={test_scores['macro_f1']:.4f} "
          f"acc={test_scores['accuracy']:.4f}")

    joblib.dump(best, config.MODELS_DIR / "track_b.joblib")
    (config.MODELS_DIR / "track_b_results.json").write_text(json.dumps({
        "encoder": ENCODER_NAME,
        "embedding_dim": int(X_train.shape[1]),
        "selected": best_name,
        "selection": sel_info,
        "validation": results.to_dict(orient="records"),
        "test_in_scope": test_scores,
    }, indent=2))

    return results