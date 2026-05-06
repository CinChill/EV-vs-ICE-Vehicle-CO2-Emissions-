from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_EV_vs_ICE_vehicle_specs.csv"
CONFIGURED_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model_outputs"))
MODEL_DIR = CONFIGURED_MODEL_DIR if CONFIGURED_MODEL_DIR.is_absolute() else BASE_DIR / CONFIGURED_MODEL_DIR

CATEGORY_MODEL = "vehicle_category_model.joblib"
CO2_MODEL = "co2_regression_model.joblib"
IMPACT_MODEL = "environmental_impact_model.joblib"

NUMERIC_FIELDS = [
    "Year",
    "Engine_Cylinders",
    "Engine_Size_L",
    "City_MPG",
    "Highway_MPG",
    "Combined_MPG",
    "EV_Range_miles",
]

CATEGORICAL_FIELDS = [
    "Make",
    "Fuel_Type",
    "Drivetrain",
    "Transmission_Type",
]

FEATURE_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

FALLBACK_OPTIONS = {
    "Make": ["Tesla", "Toyota", "Ford", "BMW", "Nissan"],
    "Fuel_Type": ["Electricity", "Regular Gasoline", "Premium Gasoline", "Diesel"],
    "Drivetrain": ["Front-Wheel Drive", "Rear-Wheel Drive", "All-Wheel Drive"],
    "Transmission_Type": ["Automatic", "Manual"],
}

DEFAULT_INPUT = {
    "Make": "Tesla",
    "Year": 2025,
    "Fuel_Type": "Electricity",
    "Engine_Cylinders": 0.0,
    "Engine_Size_L": 0.0,
    "Drivetrain": "All-Wheel Drive",
    "City_MPG": 120.0,
    "Highway_MPG": 112.0,
    "Combined_MPG": 116.0,
    "EV_Range_miles": 310.0,
    "Transmission_Type": "Automatic",
    "Annual_KM": 15000.0,
}

PRESETS = {
    "ev": {
        "Make": "Tesla",
        "Year": 2025,
        "Fuel_Type": "Electricity",
        "Engine_Cylinders": 0.0,
        "Engine_Size_L": 0.0,
        "Drivetrain": "All-Wheel Drive",
        "City_MPG": 120.0,
        "Highway_MPG": 112.0,
        "Combined_MPG": 116.0,
        "EV_Range_miles": 310.0,
        "Transmission_Type": "Automatic",
        "Annual_KM": 15000.0,
    },
    "gas": {
        "Make": "Toyota",
        "Year": 2024,
        "Fuel_Type": "Regular Gasoline",
        "Engine_Cylinders": 4.0,
        "Engine_Size_L": 2.5,
        "Drivetrain": "Front-Wheel Drive",
        "City_MPG": 28.0,
        "Highway_MPG": 39.0,
        "Combined_MPG": 32.0,
        "EV_Range_miles": 0.0,
        "Transmission_Type": "Automatic",
        "Annual_KM": 15000.0,
    },
    "diesel": {
        "Make": "BMW",
        "Year": 2023,
        "Fuel_Type": "Diesel",
        "Engine_Cylinders": 6.0,
        "Engine_Size_L": 3.0,
        "Drivetrain": "Rear-Wheel Drive",
        "City_MPG": 24.0,
        "Highway_MPG": 36.0,
        "Combined_MPG": 29.0,
        "EV_Range_miles": 0.0,
        "Transmission_Type": "Automatic",
        "Annual_KM": 15000.0,
    },
}

app = Flask(__name__)


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


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def get_options() -> dict[str, Any]:
    df = read_dataset()
    options: dict[str, Any] = {
        "categorical": FALLBACK_OPTIONS.copy(),
        "defaults": DEFAULT_INPUT.copy(),
        "presets": PRESETS,
        "year_min": 2015,
        "year_max": 2026,
    }
    if df is None:
        return options

    categorical: dict[str, list[str]] = {}
    for column in CATEGORICAL_FIELDS:
        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )
        categorical[column] = values if values else FALLBACK_OPTIONS.get(column, [])

    defaults = DEFAULT_INPUT.copy()
    for field in NUMERIC_FIELDS:
        defaults[field] = float(pd.to_numeric(df[field], errors="coerce").median())

    for field in CATEGORICAL_FIELDS:
        mode = df[field].dropna().astype(str).mode()
        if not mode.empty:
            defaults[field] = str(mode.iloc[0])

    years = pd.to_numeric(df["Year"], errors="coerce").dropna()
    if not years.empty:
        options["year_min"] = int(years.min())
        options["year_max"] = int(max(years.max(), DEFAULT_INPUT["Year"]))

    options["categorical"] = categorical
    options["defaults"] = defaults
    return options


def model_paths() -> dict[str, Path]:
    return {
        "vehicle_category": MODEL_DIR / CATEGORY_MODEL,
        "co2_regression": MODEL_DIR / CO2_MODEL,
        "environmental_impact": MODEL_DIR / IMPACT_MODEL,
    }


def model_status() -> dict[str, Any]:
    paths = model_paths()
    models = {
        name: {
            "path": str(path),
            "exists": path.exists(),
        }
        for name, path in paths.items()
    }
    missing = [name for name, payload in models.items() if not payload["exists"]]
    return {
        "model_dir": str(MODEL_DIR),
        "ready": not missing,
        "missing": missing,
        "models": models,
        "train_command": "python train_vehicle_models.py --data cleaned_EV_vs_ICE_vehicle_specs.csv --outdir model_outputs",
    }


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    status = model_status()
    if status["missing"]:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(status["missing"])
        )
    return {name: joblib.load(path) for name, path in model_paths().items()}


def parse_float(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field, DEFAULT_INPUT.get(field))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc


def parse_input(payload: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    row: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        row[field] = parse_float(payload, field)
    for field in CATEGORICAL_FIELDS:
        value = str(payload.get(field, DEFAULT_INPUT[field])).strip()
        if not value:
            raise ValueError(f"{field} cannot be empty.")
        row[field] = value

    annual_km = parse_float(payload, "Annual_KM")
    if annual_km < 0:
        raise ValueError("Annual_KM cannot be negative.")
    return pd.DataFrame([row], columns=FEATURE_FIELDS), annual_km


def probability_table(model: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not hasattr(model, "predict_proba"):
        return []

    probabilities = model.predict_proba(frame)[0]
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is None:
        return []

    rows = [
        {"label": str(label), "probability": float(probability)}
        for label, probability in zip(classes, probabilities)
    ]
    return sorted(rows, key=lambda item: item["probability"], reverse=True)


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        options=get_options(),
        status=model_status(),
    )


@app.get("/api/status")
def api_status() -> Any:
    return jsonify(as_jsonable(model_status()))


@app.post("/api/predict")
def api_predict() -> Any:
    status = model_status()
    if status["missing"]:
        return jsonify({"ok": False, "status": status}), 503

    try:
        frame, annual_km = parse_input(request.get_json(force=True) or {})
        models = load_models()

        category = str(models["vehicle_category"].predict(frame)[0])
        predicted_co2 = float(models["co2_regression"].predict(frame)[0])
        model_impact = str(models["environmental_impact"].predict(frame)[0])
        co2_impact = impact_level(predicted_co2)

        annual_miles = annual_km * 0.621371
        annual_kg_co2 = max(predicted_co2, 0.0) * annual_miles / 1000.0
        five_year_kg_co2 = annual_kg_co2 * 5.0

        payload = {
            "ok": True,
            "input": frame.iloc[0].to_dict(),
            "annual_km": annual_km,
            "vehicle_category": category,
            "category_probabilities": probability_table(models["vehicle_category"], frame),
            "predicted_co2_g_per_mile": predicted_co2,
            "co2_based_impact_level": co2_impact,
            "model_impact_level": model_impact,
            "impact_probabilities": probability_table(models["environmental_impact"], frame),
            "annual_kg_co2": annual_kg_co2,
            "five_year_kg_co2": five_year_kg_co2,
        }
        return jsonify(as_jsonable(payload))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True, "status": model_status()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
