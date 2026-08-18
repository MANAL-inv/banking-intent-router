import json
import string

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src import config
from src.splits import load_splits

RATES = [0.0, 0.05, 0.10, 0.20]
N_SAMPLE = 600
SEED = 42


def corrupt(text: str, rate: float, rng: np.random.Generator) -> str:
    if rate <= 0:
        return text
    chars = list(text)
    out = []
    for ch in chars:
        if ch == " " or rng.random() >= rate:
            out.append(ch)
            continue
        op = rng.integers(0, 3)
        if op == 0:
            continue                                    # deletion
        elif op == 1:
            out.append(rng.choice(list(string.ascii_lowercase)))   # substitution
        else:
            out.append(ch)
            out.append(ch)                              # duplication
    return "".join(out) or text


def _logits_a(texts):
    pipe = joblib.load(config.MODELS_DIR / "track_a.joblib")
    return pipe.decision_function(texts)


def _logits_b(texts):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    clf = joblib.load(config.MODELS_DIR / "track_b.joblib")
    vecs = encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
    return clf.decision_function(vecs)


def _logits_c(texts, batch_size=64):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    d = config.MODELS_DIR / "track_c" / "distilbert_intent"
    tok = AutoTokenizer.from_pretrained(d)
    mdl = AutoModelForSequenceClassification.from_pretrained(d)
    mdl.eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            chunks.append(mdl(**enc).logits.numpy())
    return np.concatenate(chunks)


TRACK_FNS = {"A_tfidf": _logits_a, "B_minilm": _logits_b, "C_distilbert": _logits_c}


def run() -> pd.DataFrame:
    splits = load_splits()
    thresholds = json.loads((config.MODELS_DIR / "oos_thresholds.json").read_text())

    rng = np.random.default_rng(SEED)
    sample = splits["test_in"].sample(N_SAMPLE, random_state=SEED).reset_index(drop=True)
    y_true = sample["label"].to_numpy()

    variants = {r: [corrupt(t, r, rng) for t in sample["text"]] for r in RATES}
    print("example at each rate:")
    for r in RATES:
        print(f"  {r:.0%}  {variants[r][0][:70]}")

    rows = []
    for track, fn in TRACK_FNS.items():
        thr = thresholds["thresholds"][track]
        for r in RATES:
            L = np.asarray(fn(variants[r]), dtype=np.float64)
            energy = logsumexp(L, axis=1)
            preds = L.argmax(axis=1)
            accepted = energy >= thr
            correct = preds == y_true

            rows.append({
                "track": track,
                "noise_rate": r,
                "coverage": accepted.mean(),
                "acc_all": correct.mean(),
                "acc_on_accepted": correct[accepted].mean() if accepted.any() else np.nan,
                "wrong_and_accepted": (accepted & ~correct).mean(),
            })
            print(f"{track:14s} rate={r:.0%} coverage={accepted.mean():.3f} "
                  f"acc_all={correct.mean():.3f} "
                  f"wrong_accepted={(accepted & ~correct).mean():.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(config.MODELS_DIR / "robustness_results.csv", index=False)
    return results