# src/transform.py
"""
Готовит фичи для ML-регрессии stargazers_count из data/processed/sample.csv
(собран через collect.py --sample-size). Не трогает "глубокий" датасет
summary.csv (3 репо) - тот для сравнительного EDA, а не для обучения модели.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "sample.csv"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"

TARGET_COL = "stargazers_count"
LOG_TARGET_COL = "log_stars"

TOP_LANGUAGES = 15
TOP_LICENSES = 8

# watchers_count на GitHub - точный дубликат stargazers_count (утечка цели).
# id/full_name/owner_login/html_url - идентификаторы, а не признаки.
# created_at/updated_at/pushed_at/topics - сырьё, заменяется производными ниже.
# visibility - константа (все репо public в этом сэмпле), нулевая информативность.
DROP_RAW_COLS = [
    "id", "full_name", "owner_login", "html_url", "description",
    "watchers_count", "created_at", "updated_at", "pushed_at", "topics",
    "visibility", "size_kb",
]


def load_sample(path: Path = DEFAULT_SAMPLE_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def bucket_rare(series: pd.Series, top_n: int, other_label: str = "Other", missing_label: str = "Unknown") -> pd.Series:
    filled = series.fillna(missing_label)
    top_values = filled.value_counts().head(top_n).index
    return filled.where(filled.isin(top_values), other_label)


def engineer_features(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    df = df.copy()
    now = now or pd.Timestamp.now(tz="UTC")

    created_at = pd.to_datetime(df["created_at"], utc=True)
    pushed_at = pd.to_datetime(df["pushed_at"], utc=True)
    df["repo_age_days"] = (now - created_at).dt.total_seconds() / 86400
    df["days_since_push"] = (now - pushed_at).dt.total_seconds() / 86400

    df["log_size_kb"] = np.log1p(df["size_kb"].clip(lower=0))
    df["description_length"] = df["description"].fillna("").str.len()
    df["topics_count"] = df["topics"].fillna("").apply(lambda s: len([t for t in s.split(";") if t]))
    df["has_license"] = df["license_key"].notna().astype(int)

    df["language_bucket"] = bucket_rare(df["language"], TOP_LANGUAGES)
    df["license_bucket"] = bucket_rare(df["license_key"], TOP_LICENSES, missing_label="None")
    df["default_branch_bucket"] = df["default_branch"].where(
        df["default_branch"].isin(["main", "master"]), "other"
    )

    for col in ["archived", "disabled", "has_issues", "has_projects", "has_wiki", "has_pages"]:
        df[col] = df[col].astype(int)

    df[LOG_TARGET_COL] = np.log1p(df[TARGET_COL])
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    features = engineer_features(df)
    return features.drop(columns=DROP_RAW_COLS + ["language", "license_key", "default_branch"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ML feature table from the repo sample dataset.")
    p.add_argument("--sample-csv", default=str(DEFAULT_SAMPLE_CSV))
    p.add_argument("--out", default=str(DEFAULT_FEATURES_CSV))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_sample(Path(args.sample_csv))
    features = build_feature_table(df)
    features.to_csv(args.out, index=False)
    print(f"Wrote {len(features)} rows x {features.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
