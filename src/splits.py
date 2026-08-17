import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_banking77, load_clinc_oos_queries

HELD_OUT_INTENTS = [
    "card_swallowed",
    "contactless_not_working",
    "virtual_card_not_working",
    "card_acceptance",
    "top_up_limits",
    "receiving_money",
    "lost_or_stolen_card",
]

UNKNOWN_LABEL = -1


def build_splits(force: bool = False) -> dict[str, pd.DataFrame]:
    train_full, test_full = load_banking77()
    oos_raw = load_clinc_oos_queries()

    missing = set(HELD_OUT_INTENTS) - set(train_full["intent"].unique())
    if missing:
        raise ValueError(f"held-out intents not present in data: {sorted(missing)}")

    held = set(HELD_OUT_INTENTS)

    pool = train_full[~train_full["intent"].isin(held)].copy()
    test_in = test_full[~test_full["intent"].isin(held)].copy()
    test_idoos = test_full[test_full["intent"].isin(held)].copy()

    kept_intents = sorted(pool["intent"].unique())
    label_map = {name: idx for idx, name in enumerate(kept_intents)}

    pool["label"] = pool["intent"].map(label_map).astype(int)
    test_in["label"] = test_in["intent"].map(label_map).astype(int)
    test_idoos["label"] = UNKNOWN_LABEL

    test_oodoos = oos_raw.assign(intent="__ood__", label=UNKNOWN_LABEL)

    train_df, val_df = train_test_split(
        pool,
        test_size=config.VAL_FRACTION,
        stratify=pool["label"],
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test_in": test_in.reset_index(drop=True),
        "test_idoos": test_idoos.reset_index(drop=True),
        "test_oodoos": test_oodoos.reset_index(drop=True),
    }

    for name, frame in splits.items():
        frame[["text", "intent", "label"]].to_csv(
            config.DATA_PROCESSED / f"{name}.csv", index=False
        )

    (config.DATA_PROCESSED / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (config.DATA_PROCESSED / "held_out_intents.json").write_text(
        json.dumps(sorted(HELD_OUT_INTENTS), indent=2)
    )

    return splits


def load_splits() -> dict[str, pd.DataFrame]:
    names = ["train", "val", "test_in", "test_idoos", "test_oodoos"]
    return {n: pd.read_csv(config.DATA_PROCESSED / f"{n}.csv") for n in names}


def load_label_map() -> dict[str, int]:
    return json.loads((config.DATA_PROCESSED / "label_map.json").read_text())
    