import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src import config
from src.oos import collect_logits
from src.splits import load_splits

N_LATENCY_SAMPLES = 40


def measure_latency() -> dict[str, float]:
    """Median single-query latency, ms. Serving is one query at a time, not batched."""
    from sentence_transformers import SentenceTransformer

    splits = load_splits()
    probes = splits["test_in"]["text"].sample(N_LATENCY_SAMPLES, random_state=0).tolist()

    pipe_a = joblib.load(config.MODELS_DIR / "track_a.joblib")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    clf_b = joblib.load(config.MODELS_DIR / "track_b.joblib")

    pipe_a.decision_function([probes[0]])
    clf_b.decision_function(encoder.encode([probes[0]], normalize_embeddings=True))

    times_a, times_b = [], []
    for text in probes:
        t0 = time.perf_counter()
        pipe_a.decision_function([text])
        times_a.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        vec = encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        clf_b.decision_function(vec)
        times_b.append((time.perf_counter() - t0) * 1000)

    lat = {"A_tfidf": float(np.median(times_a)), "B_minilm": float(np.median(times_b))}
    print(f"median single-query latency: A={lat['A_tfidf']:.2f} ms  B={lat['B_minilm']:.2f} ms")
    return lat


def run() -> pd.DataFrame:
    logits, splits = collect_logits()
    thresholds = json.loads((config.MODELS_DIR / "oos_thresholds.json").read_text())
    thr_b = thresholds["thresholds"]["B_minilm"]

    lat = measure_latency()
    y_true = splits["test_in"]["label"].to_numpy()

    energy = {
        track: {s: logsumexp(logits[track][s], axis=1)
                for s in ["val", "test_in", "test_idoos", "test_oodoos"]}
        for track in ["A_tfidf", "B_minilm"]
    }
    pred_a = logits["A_tfidf"]["test_in"].argmax(axis=1)
    pred_b = logits["B_minilm"]["test_in"].argmax(axis=1)

    percentiles = np.linspace(0, 100, 41)
    taus = np.quantile(energy["A_tfidf"]["val"], percentiles / 100.0)

    rows = []
    for pct, tau in zip(percentiles, taus):
        keep_a = energy["A_tfidf"]["test_in"] >= tau
        b_ok = energy["B_minilm"]["test_in"] >= thr_b

        accepted = keep_a | (~keep_a & b_ok)
        preds = np.where(keep_a, pred_a, pred_b)
        correct = preds == y_true

        esc = float((~keep_a).mean())
        oos_row = {}
        for name, key in [("idoos", "test_idoos"), ("oodoos", "test_oodoos")]:
            ka = energy["A_tfidf"][key] >= tau
            kb = energy["B_minilm"][key] >= thr_b
            oos_row[f"rejected_{name}"] = float((~ka & ~kb).mean())

        rows.append({
            "escalate_pct": pct,
            "tau": float(tau),
            "escalation_rate": esc,
            "coverage": float(accepted.mean()),
            "acc_all": float(correct.mean()),
            "acc_on_accepted": float(correct[accepted].mean()) if accepted.any() else np.nan,
            "wrong_and_accepted": float((accepted & ~correct).mean()),
            "mean_latency_ms": lat["A_tfidf"] + esc * lat["B_minilm"],
            **oos_row,
        })

    results = pd.DataFrame(rows)
    results.to_csv(config.MODELS_DIR / "cascade_results.csv", index=False)

    show = results.iloc[[0, 10, 20, 30, 40]]
    print("\nescalate%  esc_rate  coverage  acc_all  wrong_acc  latency_ms  rej_id  rej_ood")
    for _, r in show.iterrows():
        print(f"{r['escalate_pct']:8.0f}  {r['escalation_rate']:8.3f}  {r['coverage']:8.3f}  "
              f"{r['acc_all']:7.4f}  {r['wrong_and_accepted']:9.4f}  "
              f"{r['mean_latency_ms']:10.2f}  {r['rejected_idoos']:6.3f}  {r['rejected_oodoos']:7.3f}")

    return results