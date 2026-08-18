import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import streamlit as st
from scipy.special import logsumexp, softmax

from src import config
from src.splits import load_label_map

st.set_page_config(page_title="Banking Intent Router", layout="wide")

EXAMPLES = [
    "my card still hasn't arrived after two weeks",
    "why was I charged extra for the exchange rate",
    "the atm kept my card and didn't give it back",
    "how much has the dow changed today",
    "what's the weather in Lahore tomorrow",
]


@st.cache_resource
def load_shared():
    label_map = load_label_map()
    inv = {v: k for k, v in label_map.items()}
    thresholds = json.loads((config.MODELS_DIR / "oos_thresholds.json").read_text())
    return inv, thresholds


@st.cache_resource
def load_track_a():
    return joblib.load(config.MODELS_DIR / "track_a.joblib")


@st.cache_resource
def load_track_b():
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    clf = joblib.load(config.MODELS_DIR / "track_b.joblib")
    return encoder, clf


@st.cache_resource
def load_track_c():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    d = config.MODELS_DIR / "track_c" / "distilbert_intent"
    tok = AutoTokenizer.from_pretrained(d)
    mdl = AutoModelForSequenceClassification.from_pretrained(d)
    mdl.eval()
    return torch, tok, mdl


def logits_a(text):
    return load_track_a().decision_function([text])[0]


def logits_b(text):
    encoder, clf = load_track_b()
    vec = encoder.encode([text], normalize_embeddings=True)
    return clf.decision_function(vec)[0]


def logits_c(text):
    torch, tok, mdl = load_track_c()
    enc = tok(text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        return mdl(**enc).logits[0].numpy()


TRACKS = {
    "A_tfidf": ("TF-IDF + logistic regression", logits_a),
    "B_minilm": ("Frozen MiniLM + logistic regression", logits_b),
    "C_distilbert": ("Fine-tuned DistilBERT", logits_c),
}


def analyse(text, fn, threshold, inv):
    t0 = time.perf_counter()
    L = np.asarray(fn(text), dtype=np.float64)
    latency_ms = (time.perf_counter() - t0) * 1000

    energy = float(logsumexp(L))
    probs = softmax(L)
    order = np.argsort(probs)[::-1][:3]
    return {
        "abstain": energy < threshold,
        "energy": energy,
        "threshold": threshold,
        "latency_ms": latency_ms,
        "top3": [(inv[int(i)], float(probs[i])) for i in order],
    }


inv, thresholds = load_shared()

st.title("Banking Intent Router")
st.caption(
    f"70 in-scope intents · abstain when energy < threshold "
    f"(set for {thresholds['target_coverage']:.0%} coverage on validation, "
    f"score = {thresholds['deploy_score']})"
)

with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex
    st.divider()
    st.subheader("Test-set results")
    st.table({
        "Track": ["A TF-IDF", "B MiniLM", "C DistilBERT"],
        "macro-F1": [0.887, 0.921, 0.921],
        "AUROC OOD": [0.968, 0.992, 0.989],
        "AUROC ID-OOS": [0.828, 0.901, 0.913],
    })

query = st.text_area("Customer query", key="query", height=90)

if query.strip():
    cols = st.columns(len(TRACKS))
    for col, (key, (label, fn)) in zip(cols, TRACKS.items()):
        with col:
            st.markdown(f"**{label}**")
            try:
                r = analyse(query, fn, thresholds["thresholds"][key], inv)
            except Exception as exc:
                st.error(f"unavailable: {type(exc).__name__}")
                continue

            if r["abstain"]:
                st.warning("ABSTAIN — route to a human")
            else:
                st.success(f"{r['top3'][0][0]}")

            st.caption(
                f"energy {r['energy']:.2f} (threshold {r['threshold']:.2f}) · "
                f"{r['latency_ms']:.0f} ms"
            )
            for name, p in r["top3"]:
                st.progress(min(p, 1.0), text=f"{name} — {p:.1%}")
else:
    st.info("Enter a query, or pick an example from the sidebar.")