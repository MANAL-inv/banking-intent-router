import io
import json

import pandas as pd
import requests

from src import config

BANKING77_CSV = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/{split}.csv"
)
CLINC_JSON = (
    "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
)


def _get(url: str) -> bytes:
    """Download a URL and return raw bytes, failing loudly on any HTTP error."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _normalise(df: pd.DataFrame, intent_vocab: list[str]) -> pd.DataFrame:
    """Return a frame with exactly: text (str), intent (str), label (int)."""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["intent"].map({name: i for i, name in enumerate(intent_vocab)})
    if df["label"].isna().any():
        unknown = sorted(df.loc[df["label"].isna(), "intent"].unique())
        raise ValueError(f"intents not in vocab: {unknown}")
    df["label"] = df["label"].astype(int)
    return df[["text", "intent", "label"]].reset_index(drop=True)


def load_banking77(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = config.DATA_RAW / "banking77_train.csv"
    test_path = config.DATA_RAW / "banking77_test.csv"
    vocab_path = config.DATA_PROCESSED / "intent_vocab.json"

    if train_path.exists() and test_path.exists() and not force_download:
        return pd.read_csv(train_path), pd.read_csv(test_path)

    train_raw = pd.read_csv(io.BytesIO(_get(BANKING77_CSV.format(split="train"))))
    test_raw = pd.read_csv(io.BytesIO(_get(BANKING77_CSV.format(split="test"))))

    train_raw = train_raw.rename(columns={"category": "intent"})
    test_raw = test_raw.rename(columns={"category": "intent"})

    intent_vocab = sorted(train_raw["intent"].unique())

    train_df = _normalise(train_raw, intent_vocab)
    test_df = _normalise(test_raw, intent_vocab)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    vocab_path.write_text(json.dumps(intent_vocab, indent=2))

    print(f"[banking77] train={train_df.shape} test={test_df.shape} intents={len(intent_vocab)}")
    return train_df, test_df


def load_clinc_oos_queries(force_download: bool = False) -> pd.DataFrame:
    """Out-of-scope queries only — used as out-of-domain probes in Phase 6."""
    oos_path = config.DATA_RAW / "clinc_oos_test.csv"

    if oos_path.exists() and not force_download:
        return pd.read_csv(oos_path)

    payload = json.loads(_get(CLINC_JSON).decode("utf-8"))
    frame = pd.DataFrame({"text": [row[0] for row in payload["oos_test"]]})
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.reset_index(drop=True)

    frame.to_csv(oos_path, index=False)
    print(f"[clinc] oos={frame.shape}")
    return frame