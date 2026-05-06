from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_COLUMNS = [
    "Make",
    "Year",
    "Fuel_Type",
    "Engine_Cylinders",
    "Engine_Size_L",
    "Drivetrain",
    "City_MPG",
    "Highway_MPG",
    "Combined_MPG",
    "CO2_Emissions_g_per_mile",
    "EV_Range_miles",
    "Vehicle_Category",
    "Transmission_Type",
]

CATEGORY_TARGET = "Vehicle_Category"
CO2_TARGET = "CO2_Emissions_g_per_mile"
IMPACT_TARGET = "Impact_Level"

NUMERIC_FEATURES = [
    "Year",
    "Engine_Cylinders",
    "Engine_Size_L",
    "City_MPG",
    "Highway_MPG",
    "Combined_MPG",
    "EV_Range_miles",
]

CATEGORICAL_FEATURES = [
    "Make",
    "Fuel_Type",
    "Drivetrain",
    "Transmission_Type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train EV/ICE vehicle category, CO2 regression, and environmental "
            "impact-level models from the cleaned vehicle specs dataset."
        )
    )
    parser.add_argument(
        "--data",
        default="cleaned_EV_vs_ICE_vehicle_specs.csv",
        help="Path to the cleaned CSV dataset.",
    )
    parser.add_argument(
        "--outdir",
        default="model_outputs",
        help="Directory where models, metrics, and plots will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and models.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees for each random forest model.",
    )
    parser.add_argument(
        "--min-category-count",
        type=int,
        default=2,
        help=(
            "Vehicle categories with fewer rows than this are skipped for the "
            "category classifier because they cannot be learned reliably."
        ),
    )
    parser.add_argument(
        "--min-impact-count",
        type=int,
        default=2,
        help="Impact labels with fewer rows than this are skipped for impact classification.",
    )
    parser.add_argument(
        "--drop-fuel-type",
        action="store_true",
        help=(
            "Remove Fuel_Type from model features. Use this for a stricter "
            "technical-spec experiment because fuel type strongly reveals EV/ICE class."
        ),
    )
    parser.add_argument(
        "--drop-make",
        action="store_true",
        help="Remove Make from model features to reduce brand memorization.",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in DATASET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    for column in NUMERIC_FEATURES + [CO2_TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES + [CATEGORY_TARGET]:
        df[column] = df[column].astype(str).str.strip()

    required_for_training = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [CATEGORY_TARGET, CO2_TARGET]
    before = len(df)
    df = df.dropna(subset=required_for_training).copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing required training values.")
    return df


def selected_features(args: argparse.Namespace) -> list[str]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if args.drop_fuel_type:
        features = [feature for feature in features if feature != "Fuel_Type"]
    if args.drop_make:
        features = [feature for feature in features if feature != "Make"]
    return features


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric = [feature for feature in NUMERIC_FEATURES if feature in features]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_category_model(features: list[str], args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def make_regression_model(features: list[str], args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def impact_level(co2_g_per_mile: float) -> str:
    if co2_g_per_mile <= 0:
        return "Zero tailpipe"
    if co2_g_per_mile <= 250:
        return "Low"
    if co2_g_per_mile <= 400:
        return "Medium"
    if co2_g_per_mile <= 550:
        return "High"
    return "Very high"


def add_impact_level(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result[IMPACT_TARGET] = result[CO2_TARGET].apply(impact_level)
    return result


def filter_rare_labels(
    df: pd.DataFrame,
    target: str,
    min_count: int,
    task_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = df[target].value_counts()
    rare = counts[counts < min_count].to_dict()
    if not rare:
        return df.copy(), {}

    print(f"Skipping rare labels for {task_name}: {rare}")
    filtered = df[df[target].isin(counts[counts >= min_count].index)].copy()
    return filtered, {str(label): int(count) for label, count in rare.items()}


def split_train_test(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool]:
    y = df[target]
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df[features],
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
        return X_train, X_test, y_train, y_test, stratify is not None
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            df[features],
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=None,
        )
        return X_train, X_test, y_train, y_test, False


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def report_to_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for label, values in report.items():
        if isinstance(values, dict):
            rows[str(label)] = {str(k): float(v) for k, v in values.items()}
        else:
            rows[str(label)] = {
                "precision": float(values),
                "recall": float(values),
                "f1-score": float(values),
                "support": np.nan,
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), max(5, len(labels) * 1.2)))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance(model: Pipeline, out_csv: Path, out_png: Path, title: str) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(out_csv, index=False)

    top = importance_df.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Random forest importance")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def train_classifier(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task_name: str,
    min_count: int,
    args: argparse.Namespace,
    outdir: Path,
) -> dict[str, Any]:
    filtered_df, skipped_labels = filter_rare_labels(df, target, min_count, task_name)
    if filtered_df[target].nunique() < 2:
        raise ValueError(f"{task_name} needs at least two labels after rare-label filtering.")

    X_train, X_test, y_train, y_test, used_stratify = split_train_test(
        filtered_df,
        features,
        target,
        args,
    )

    model = make_category_model(features, args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = [str(label) for label in model.named_steps["model"].classes_]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_to_dataframe(report).to_csv(outdir / f"{task_name}_classification_report.csv")
    save_confusion_matrix(
        y_test,
        y_pred,
        labels,
        f"{task_name.replace('_', ' ').title()} Confusion Matrix",
        outdir / f"{task_name}_confusion_matrix.png",
    )
    save_feature_importance(
        model,
        outdir / f"{task_name}_feature_importance.csv",
        outdir / f"{task_name}_feature_importance.png",
        f"{task_name.replace('_', ' ').title()} Feature Importance",
    )
    joblib.dump(model, outdir / f"{task_name}_model.joblib")

    return {
        "rows_used": len(filtered_df),
        "rows_skipped_for_rare_labels": int(len(df) - len(filtered_df)),
        "skipped_labels": skipped_labels,
        "target_distribution": filtered_df[target].value_counts().to_dict(),
        "used_stratified_split": used_stratify,
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": report,
        "model_path": str(outdir / f"{task_name}_model.joblib"),
    }


def train_co2_regressor(
    df: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
    outdir: Path,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[CO2_TARGET],
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = make_regression_model(features, args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    residuals = pd.DataFrame(
        {
            "actual_co2_g_per_mile": y_test.to_numpy(),
            "predicted_co2_g_per_mile": y_pred,
            "residual": y_test.to_numpy() - y_pred,
        }
    )
    residuals.to_csv(outdir / "co2_regression_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, s=18)
    lower = float(min(y_test.min(), y_pred.min()))
    upper = float(max(y_test.max(), y_pred.max()))
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Actual CO2 (g/mile)")
    ax.set_ylabel("Predicted CO2 (g/mile)")
    ax.set_title("CO2 Regression: Actual vs Predicted")
    fig.tight_layout()
    fig.savefig(outdir / "co2_regression_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    save_feature_importance(
        model,
        outdir / "co2_regression_feature_importance.csv",
        outdir / "co2_regression_feature_importance.png",
        "CO2 Regression Feature Importance",
    )
    joblib.dump(model, outdir / "co2_regression_model.joblib")

    mse = mean_squared_error(y_test, y_pred)
    return {
        "rows_used": len(df),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, y_pred),
        "model_path": str(outdir / "co2_regression_model.joblib"),
    }


def save_environmental_analysis(df: pd.DataFrame, outdir: Path) -> None:
    analysis_df = add_impact_level(df)
    analysis_df.to_csv(outdir / "dataset_with_impact_level.csv", index=False)

    summary = (
        analysis_df.groupby(["Vehicle_Category", "Fuel_Type"], dropna=False)
        .agg(
            rows=("Make", "size"),
            avg_co2_g_per_mile=(CO2_TARGET, "mean"),
            median_co2_g_per_mile=(CO2_TARGET, "median"),
            avg_combined_mpg=("Combined_MPG", "mean"),
            avg_ev_range_miles=("EV_Range_miles", "mean"),
        )
        .reset_index()
        .sort_values(["Vehicle_Category", "avg_co2_g_per_mile"])
    )
    summary.to_csv(outdir / "environmental_impact_summary.csv", index=False)

    impact_counts = analysis_df[IMPACT_TARGET].value_counts().reindex(
        ["Zero tailpipe", "Low", "Medium", "High", "Very high"],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(impact_counts.index, impact_counts.values)
    ax.set_title("Environmental Impact Level Distribution")
    ax.set_xlabel("Impact level")
    ax.set_ylabel("Vehicle count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(outdir / "impact_level_distribution.png", dpi=160)
    plt.close(fig)

    category_summary = (
        analysis_df.groupby("Vehicle_Category")[CO2_TARGET]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(category_summary.index, category_summary.values)
    ax.set_title("Average CO2 by Vehicle Category")
    ax.set_xlabel("Vehicle category")
    ax.set_ylabel("Average CO2 (g/mile)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(outdir / "avg_co2_by_vehicle_category.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    features = selected_features(args)
    save_environmental_analysis(df, outdir)

    metrics: dict[str, Any] = {
        "data_path": str(data_path),
        "output_directory": str(outdir),
        "rows": len(df),
        "features": features,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "vehicle_category_distribution": df[CATEGORY_TARGET].value_counts().to_dict(),
        "fuel_type_distribution": df["Fuel_Type"].value_counts().to_dict(),
    }

    print("Training vehicle category classifier...")
    metrics["vehicle_category_classifier"] = train_classifier(
        df=df,
        features=features,
        target=CATEGORY_TARGET,
        task_name="vehicle_category",
        min_count=args.min_category_count,
        args=args,
        outdir=outdir,
    )

    print("Training CO2 regression model...")
    metrics["co2_regression"] = train_co2_regressor(
        df=df,
        features=features,
        args=args,
        outdir=outdir,
    )

    print("Training environmental impact classifier...")
    impact_df = add_impact_level(df)
    metrics["impact_level_distribution"] = impact_df[IMPACT_TARGET].value_counts().to_dict()
    metrics["environmental_impact_classifier"] = train_classifier(
        df=impact_df,
        features=features,
        target=IMPACT_TARGET,
        task_name="environmental_impact",
        min_count=args.min_impact_count,
        args=args,
        outdir=outdir,
    )

    save_json(outdir / "metrics.json", metrics)
    print(f"Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
