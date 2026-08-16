# src/train.py
"""
Регрессия: предсказание stargazers_count (в log1p-шкале) по метаданным
репозитория из data/processed/sample.csv. Baseline (LinearRegression) +
основная модель (RandomForestRegressor), сравнение метрик, важность признаков.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import transform

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "stars_regressor.joblib"
DEFAULT_IMPORTANCE_PLOT = PROJECT_ROOT / "data" / "processed" / "feature_importance.png"
DEFAULT_PRED_PLOT = PROJECT_ROOT / "data" / "processed" / "pred_vs_actual.png"

CATEGORICAL_COLS = ["language_bucket", "license_bucket", "default_branch_bucket", "owner_type"]
RANDOM_STATE = 42


def build_pipeline(estimator, feature_cols: list[str]) -> Pipeline:
    categorical = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_log_test: pd.Series, y_log_train: pd.Series) -> dict:
    pred_log = pipeline.predict(X_test)
    y_true = np.expm1(y_log_test)
    # expm1 усиливает даже небольшие ошибки в log-шкале в огромные (для линейной
    # регрессии без регуляризации отдельные экстраполяции взрываются до e^30+),
    # поэтому для звёздной шкалы клипуем предсказание диапазоном обучающих данных.
    pred_log_clipped = np.clip(pred_log, y_log_train.min(), y_log_train.max())
    y_pred = np.expm1(pred_log_clipped)
    return {
        "rmse_log": mean_squared_error(y_log_test, pred_log) ** 0.5,
        "mae_log": mean_absolute_error(y_log_test, pred_log),
        "r2_log": r2_score(y_log_test, pred_log),
        "rmse_stars": mean_squared_error(y_true, y_pred) ** 0.5,
        "medae_stars": median_absolute_error(y_true, y_pred),
        "pred_log": pred_log,
    }


def plot_feature_importance(pipeline: Pipeline, out_path: Path, top_n: int = 20) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances[::-1].plot.barh(ax=ax)
    ax.set_title("RandomForest feature importance (top %d)" % top_n)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(y_log_test: pd.Series, pred_log: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_log_test, pred_log, alpha=0.3, s=10)
    lims = [min(y_log_test.min(), pred_log.min()), max(y_log_test.max(), pred_log.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("actual log1p(stars)")
    ax.set_ylabel("predicted log1p(stars)")
    ax.set_title("RandomForest: predicted vs actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a stargazers_count regressor on the repo sample dataset.")
    p.add_argument("--sample-csv", default=str(transform.DEFAULT_SAMPLE_CSV))
    p.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = transform.load_sample(Path(args.sample_csv))
    features = transform.build_feature_table(df)

    feature_cols = [c for c in features.columns if c not in (transform.TARGET_COL, transform.LOG_TARGET_COL)]
    X = features[feature_cols]
    y_log = features[transform.LOG_TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=args.test_size, random_state=RANDOM_STATE)

    results = {}

    linear = build_pipeline(LinearRegression(), feature_cols)
    linear.fit(X_train, y_train)
    results["linear_regression"] = (linear, evaluate(linear, X_test, y_test, y_train))

    forest = build_pipeline(
        RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        feature_cols,
    )
    forest.fit(X_train, y_train)
    results["random_forest"] = (forest, evaluate(forest, X_test, y_test, y_train))

    print(f"\n{'model':<20}{'RMSE(log)':>12}{'MAE(log)':>12}{'R2(log)':>10}{'RMSE(stars)':>15}{'MedAE(stars)':>15}")
    for name, (_, m) in results.items():
        print(f"{name:<20}{m['rmse_log']:>12.4f}{m['mae_log']:>12.4f}{m['r2_log']:>10.4f}{m['rmse_stars']:>15.1f}{m['medae_stars']:>15.1f}")

    best_name = max(results, key=lambda n: results[n][1]["r2_log"])
    best_pipeline, best_metrics = results[best_name]
    print(f"\nBest model: {best_name} (R2 log-scale = {best_metrics['r2_log']:.4f})")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_out)
    print(f"Saved best model -> {model_out}")

    DEFAULT_IMPORTANCE_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(results["random_forest"][0], DEFAULT_IMPORTANCE_PLOT)
    plot_pred_vs_actual(y_test, results["random_forest"][1]["pred_log"], DEFAULT_PRED_PLOT)
    print(f"Saved plots -> {DEFAULT_IMPORTANCE_PLOT}, {DEFAULT_PRED_PLOT}")


if __name__ == "__main__":
    main()
