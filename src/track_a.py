import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.splits import load_splits


def make_pipeline(clf, ngram_range=(1, 2), min_df=2) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            stop_words=None,
        )),
        ("clf", clf),
    ])


def candidates() -> dict[str, Pipeline]:
    grid = {}
    for C in (1.0, 3.0, 10.0,30.0, 100.0):
        for cw in (None, "balanced"):
            name = f"logreg_C{C:g}_{'bal' if cw else 'none'}"
            grid[name] = make_pipeline(LogisticRegression(
                C=C,
                class_weight=cw,
                max_iter=2000,
                random_state=config.RANDOM_SEED,
            ))

    grid["linsvc_calibrated"] = make_pipeline(CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, random_state=config.RANDOM_SEED),
        method="sigmoid",
        cv=3,
    ))
    return grid


def evaluate(pipe: Pipeline, texts, y_true) -> dict:
    y_pred = pipe.predict(texts)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def run() -> pd.DataFrame:
    splits = load_splits()
    X_train, y_train = splits["train"]["text"], splits["train"]["label"]
    X_val, y_val = splits["val"]["text"], splits["val"]["label"]
    X_test, y_test = splits["test_in"]["text"], splits["test_in"]["label"]

    rows = []
    fitted = {}

    for name, pipe in candidates().items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        val_scores = evaluate(pipe, X_val, y_val)
        rows.append({"model": name, "fit_seconds": round(fit_s, 2), **val_scores})
        fitted[name] = pipe
        print(f"{name:28s} val_macro_f1={val_scores['macro_f1']:.4f} "
              f"acc={val_scores['accuracy']:.4f} ({fit_s:.1f}s)")

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]

    n_features = len(best.named_steps["tfidf"].vocabulary_)
    print(f"\nselected: {best_name}  |  vocabulary size: {n_features}")

    test_scores = evaluate(best, X_test, y_test)
    print(f"TEST in-scope: macro_f1={test_scores['macro_f1']:.4f} "
          f"acc={test_scores['accuracy']:.4f}")

    joblib.dump(best, config.MODELS_DIR / "track_a.joblib")
    (config.MODELS_DIR / "track_a_results.json").write_text(json.dumps({
        "selected": best_name,
        "n_features": n_features,
        "validation": results.to_dict(orient="records"),
        "test_in_scope": test_scores,
    }, indent=2))

    return results