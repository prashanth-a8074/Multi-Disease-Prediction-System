
import os
import pickle
import pandas as pd
import numpy as np

_MODEL_CACHE = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_prediction_artifacts(disease):
    """Load trained model and optional scaler."""

    disease = disease.lower().strip()

    if disease in _MODEL_CACHE:
        return _MODEL_CACHE[disease]

    model_path = os.path.join(MODEL_DIR, f"{disease}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{disease}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please run train_models.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = None

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Get feature names from the scaler or trained model.
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        features = list(scaler.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise ValueError(
            f"Cannot determine input features for {disease}."
        )

    artifacts = {
        "model": model,
        "scaler": scaler,
        "features": features
    }

    _MODEL_CACHE[disease] = artifacts
    return artifacts


def clean_key(key):
    """Normalize feature names for comparison."""

    return (
        str(key).lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "")
    )


def predict_disease(disease, raw_inputs):
    """
    Predict disease status and return risk percentage.

    Returns:
        prediction: 0 or 1
        risk_percentage: 0 to 100
    """

    artifacts = load_prediction_artifacts(disease)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["features"]

    # Normalize user input keys.
    clean_inputs = {
        clean_key(k): v
        for k, v in raw_inputs.items()
    }

    abbrev_map = {
        "chestpaintype": ["chest_pain", "cp"],
        "restingbp": ["trestbps", "bp", "blood_pressure"],
        "fastingbs": ["fbs"],
        "restingecg": ["restecg"],
        "maxhr": ["thalach", "max_heart_rate", "heart_rate"],
        "exerciseangina": ["exang"],
        "st_slope": ["slope"],
        "ca": ["ca_vessels"],
        "thal": ["thalassemia"],
        "alamine_aminotransferase": ["alt", "sgpt"],
        "aspartate_aminotransferase": ["ast", "sgot"],
        "total_protiens": ["total_proteins", "tp"],
        "albumin_and_globulin_ratio": ["ag_ratio", "ag"],
        "mdvpfohz": ["fo"],
        "mdvpfhihz": ["fhi"],
        "mdvp flohz": ["flo"],
        "jitter": ["jitter_percent"],
    }

    mapped_inputs = {}

    # Match the input fields with model features.
    for feature in features:

        feature_clean = clean_key(feature)

        if feature_clean in clean_inputs:
            mapped_inputs[feature] = clean_inputs[feature_clean]
            continue

        matched = False

        for target, aliases in abbrev_map.items():

            if feature_clean == target:
                for alias in aliases:
                    if alias in clean_inputs:
                        mapped_inputs[feature] = clean_inputs[alias]
                        matched = True
                        break

            if matched:
                break

        if matched:
            continue

        # Match similar feature names.
        for key, value in clean_inputs.items():
            if key in feature_clean or feature_clean in key:
                mapped_inputs[feature] = value
                matched = True
                break

        if matched:
            continue

        # Fallback when an input is missing.
        if "gender" in feature_clean or "sex" in feature_clean:
            mapped_inputs[feature] = 1.0
        else:
            mapped_inputs[feature] = 0.0

    # Convert categorical values into numeric values.
    for feature, value in mapped_inputs.items():

        if isinstance(value, str):

            value_clean = value.lower().strip()
            feature_clean = clean_key(feature)

            if feature_clean in ["sex", "gender"]:
                mapped_inputs[feature] = (
                    0.0 if "female" in value_clean else 1.0
                )

            elif feature_clean in ["cp", "chestpaintype"]:
                options = [
                    "typical angina",
                    "atypical angina",
                    "non-anginal pain",
                    "asymptomatic"
                ]

                index = next(
                    (i for i, item in enumerate(options)
                     if item in value_clean),
                    0
                )
                mapped_inputs[feature] = float(index)

            elif feature_clean in ["fbs", "fastingbs"]:
                mapped_inputs[feature] = (
                    1.0 if any(x in value_clean for x in
                    ["yes", "true", "> 120"]) else 0.0
                )

            elif feature_clean in ["restecg", "restingecg"]:
                options = [
                    "normal",
                    "st-t wave abnormality",
                    "lvh",
                    "left ventricular hypertrophy"
                ]

                index = next(
                    (i for i, item in enumerate(options)
                     if item in value_clean),
                    0
                )
                mapped_inputs[feature] = float(
                    2 if index == 3 else index
                )

            elif feature_clean in ["exang", "exerciseangina"]:
                mapped_inputs[feature] = (
                    1.0 if "yes" in value_clean or
                    "true" in value_clean else 0.0
                )

            elif feature_clean in ["slope", "st_slope"]:
                options = ["upsloping", "flat", "downsloping"]

                index = next(
                    (i for i, item in enumerate(options)
                     if item in value_clean),
                    1
                )
                mapped_inputs[feature] = float(index + 1)

            elif feature_clean in ["thal", "thalassemia"]:
                options = [
                    "normal",
                    "fixed defect",
                    "reversible defect"
                ]

                index = next(
                    (i for i, item in enumerate(options)
                     if item in value_clean),
                    1
                )
                mapped_inputs[feature] = float(index + 1)

            else:
                try:
                    mapped_inputs[feature] = float(value)
                except ValueError:
                    mapped_inputs[feature] = 0.0

    # Prepare DataFrame in the exact training feature order.
    input_df = pd.DataFrame(
        [[mapped_inputs[feature] for feature in features]],
        columns=features
    )

    # Apply scaling only when a scaler exists.
    if scaler is not None:
        model_inputs = scaler.transform(input_df)
    else:
        model_inputs = input_df

    # Generate prediction.
    prediction = int(model.predict(model_inputs)[0])

    # Calculate probability.
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(model_inputs)[0][1]

    elif hasattr(model, "decision_function"):
        decision = model.decision_function(model_inputs)[0]
        probability = 1 / (1 + np.exp(-decision))

    else:
        probability = float(prediction)

    risk_percentage = round(float(probability) * 100, 2)

    return prediction, risk_percentage
