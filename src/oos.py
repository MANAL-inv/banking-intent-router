import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, roc_auc_score

from src import config
from src.splits import load_splits

EVAL_SPLITS = ["val", "test_in", "test_idoos", "test_oodoos"]
TARGET_COVERAGE = 0.95
DEPLOY_SCORE = "energy"


def collect_logits() -> tuple[dict, dict]:
    """(n, 70) logit-like matrix per track per split."""
    splits = load_splits()
    logits = {}

    pipe_a = joblib.load(config.MODELS_DIR / "track_a.joblib")
    logits["A_tfidf"] = {s: pipe_a.decision_function(splits[s]["text"]) for s in EVAL_SPLITS}

    clf_b = joblib.load(config.MODELS_DIR / "track_b.joblib")
    emb = {s: np.load(config.DATA_PROCESSED / f"emb_{s}.npy") for s in EVAL_SPLITS}
    logits["B_minilm"] = {s: clf_b.decision_function(emb[s]) for s in EVAL_SPLITS}

    c_dir = config.MODELS_DIR / "track_c"
    logits["C_distilbert"] = {s: np.load(c_dir / f"logits_{s}.npy") for s in EVAL_SPLITS}

    for track, per_split in logits.items():
        shapes = {s: per_split[s].shape for s in EVAL_SPLITS}
        print(f"{track:14s} {shapes}")

    return logits, splits


def confidence_scores(L: np.ndarray) -> dict[str, np.ndarray]:
    """Higher = more confident it is in-scope. All derived from one logit matrix."""
    p = softmax(L, axis=1)
    p_sorted = np.sort(p, axis=1)
    return {
        "msp": p.max(axis=1),
        "margin": p_sorted[:, -1] - p_sorted[:, -2],
        "neg_entropy": (p * np.log(p + 1e-12)).sum(axis=1),
        "max_logit": L.max(axis=1),
        "energy": logsumexp(L, axis=1),
    }


def evaluate_track(L_by_split: dict, y_test_in: np.ndarray) -> pd.DataFrame:
    scores = {s: confidence_scores(L_by_split[s]) for s in EVAL_SPLITS}
    preds_in = L_by_split["test_in"].argmax(axis=1)

    rows = []
    for score_name in scores["val"]:
        s_val = scores["val"][score_name]
        s_in = scores["test_in"][score_name]
        s_id = scores["test_idoos"][score_name]
        s_ood = scores["test_oodoos"][score_name]

        threshold = float(np.quantile(s_val, 1.0 - TARGET_COVERAGE))
        accepted = s_in >= threshold

        rows.append({
            "score": score_name,
            "auroc_idoos": roc_auc_score(
                np.r_[np.ones(len(s_in)), np.zeros(len(s_id))], np.r_[s_in, s_id]),
            "auroc_oodoos": roc_auc_score(
                np.r_[np.ones(len(s_in)), np.zeros(len(s_ood))], np.r_[s_in, s_ood]),
            "threshold": threshold,
            "coverage_test_in": accepted.mean(),
            "rejected_idoos": (s_id < threshold).mean(),
            "rejected_oodoos": (s_ood < threshold).mean(),
            "acc_on_accepted": accuracy_score(y_test_in[accepted], preds_in[accepted]),
            "acc_all": accuracy_score(y_test_in, preds_in),
        })
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    logits, splits = collect_logits()
    y_test_in = splits["test_in"]["label"].to_numpy()

    frames = []
    for track, per_split in logits.items():
        df = evaluate_track(per_split, y_test_in)
        df.insert(0, "track", track)
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(config.MODELS_DIR / "oos_results.csv", index=False)

    deployed = results[results["score"] == DEPLOY_SCORE]
    print(f"\n=== deployed score: {DEPLOY_SCORE}, target coverage {TARGET_COVERAGE:.0%} ===")
    print(deployed[["track", "threshold", "coverage_test_in", "rejected_idoos",
                    "rejected_oodoos", "acc_all", "acc_on_accepted"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    (config.MODELS_DIR / "oos_thresholds.json").write_text(json.dumps({
        "deploy_score": DEPLOY_SCORE,
        "target_coverage": TARGET_COVERAGE,
        "thresholds": {r["track"]: r["threshold"] for _, r in deployed.iterrows()},
    }, indent=2))

    return results