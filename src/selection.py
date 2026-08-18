import math

import pandas as pd


def standard_error(accuracy: float, n: int) -> float:
    """Standard error of a proportion estimated from n samples."""
    return math.sqrt(accuracy * (1.0 - accuracy) / n)


def select_one_se(
    results: pd.DataFrame,
    complexity: dict[str, float],
    n_val: int,
    metric: str = "macro_f1",
) -> tuple[str, dict]:
    """Pick the simplest model whose score is within one SE of the best."""
    best_row = results.loc[results[metric].idxmax()]
    se = standard_error(float(best_row["accuracy"]), n_val)
    threshold = float(best_row[metric]) - se

    eligible = results[results[metric] >= threshold].copy()
    eligible["complexity"] = eligible["model"].map(complexity)
    eligible = eligible.sort_values(["complexity", metric], ascending=[True, False])

    chosen = str(eligible.iloc[0]["model"])
    info = {
        "rule": "one-standard-error",
        "argmax_model": str(best_row["model"]),
        "argmax_score": round(float(best_row[metric]), 4),
        "standard_error": round(se, 4),
        "threshold": round(threshold, 4),
        "n_eligible": int(len(eligible)),
        "selected": chosen,
    }
    return chosen, info