import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp
from sklearn.metrics import accuracy_score, roc_auc_score

from src import config
from src.splits import load_splits

N_BINS = 15


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "confidence": np.nan, "accuracy": np.nan})
            continue
        c, a = conf[m].mean(), correct[m].mean()
        total += m.mean() * abs(c - a)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "confidence": c, "accuracy": a})
    return total, pd.DataFrame(rows)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    def nll(T):
        scaled = logits / T
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return -log_probs[np.arange(len(y_true)), y_true].mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def track_logits():
    splits = load_splits()
    names = ["val", "test_in", "test_idoos", "test_oodoos"]
    out = {}

    pipe_a = joblib.load(config.MODELS_DIR / "track_a.joblib")
    out["A_tfidf"] = {s: pipe_a.decision_function(splits[s]["text"]) for s in names}

    clf_b = joblib.load(config.MODELS_DIR / "track_b.joblib")
    out["B_minilm"] = {
        s: clf_b.decision_function(np.load(config.DATA_PROCESSED / f"emb_{s}.npy"))
        for s in names
    }

    c_dir = config.MODELS_DIR / "track_c"
    out["C_distilbert"] = {s: np.load(c_dir / f"logits_{s}.npy") for s in names}
    return out, splits


def run():
    logits, splits = track_logits()
    y_val = splits["val"]["label"].to_numpy()
    y_test = splits["test_in"]["label"].to_numpy()

    rows, diagrams = [], {}
    for track, per_split in logits.items():
        T = fit_temperature(per_split["val"].astype(np.float64), y_val)

        L_test = per_split["test_in"].astype(np.float64)
        p_before = softmax(L_test, axis=1)
        p_after = softmax(L_test / T, axis=1)

        ece_before, diag_before = ece(p_before, y_test)
        ece_after, diag_after = ece(p_after, y_test)
        diagrams[track] = (diag_before, diag_after)

        def msp_auroc(scaled_by):
            s_in = softmax(L_test / scaled_by, axis=1).max(axis=1)
            s_ood = softmax(per_split["test_oodoos"].astype(np.float64) / scaled_by,
                            axis=1).max(axis=1)
            s_id = softmax(per_split["test_idoos"].astype(np.float64) / scaled_by,
                           axis=1).max(axis=1)
            y_ood = np.r_[np.ones(len(s_in)), np.zeros(len(s_ood))]
            y_id = np.r_[np.ones(len(s_in)), np.zeros(len(s_id))]
            return (roc_auc_score(y_ood, np.r_[s_in, s_ood]),
                    roc_auc_score(y_id, np.r_[s_in, s_id]))

        ood_b, id_b = msp_auroc(1.0)
        ood_a, id_a = msp_auroc(T)

        rows.append({
            "track": track,
            "temperature": T,
            "mean_conf_before": p_before.max(axis=1).mean(),
            "mean_conf_after": p_after.max(axis=1).mean(),
            "accuracy": accuracy_score(y_test, p_before.argmax(axis=1)),
            "accuracy_after": accuracy_score(y_test, p_after.argmax(axis=1)),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "msp_auroc_ood_before": ood_b, "msp_auroc_ood_after": ood_a,
            "msp_auroc_idoos_before": id_b, "msp_auroc_idoos_after": id_a,
        })
        print(f"{track:14s} T={T:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}  "
              f"mean_conf {p_before.max(axis=1).mean():.3f} -> {p_after.max(axis=1).mean():.3f}  "
              f"acc {accuracy_score(y_test, p_before.argmax(axis=1)):.4f} -> "
              f"{accuracy_score(y_test, p_after.argmax(axis=1)):.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(config.MODELS_DIR / "calibration_results.csv", index=False)
    (config.MODELS_DIR / "temperatures.json").write_text(
        json.dumps({r["track"]: r["temperature"] for _, r in results.iterrows()}, indent=2))
    return results, diagrams