# ============================================================
# PowerGuard AI - Backend
# Live Smart Grid Prediction Backend
# ============================================================

import os
import gzip
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# MODEL FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


FAULT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "transformer_fault_model.pkl.gz"
)

FAULT_SELECTOR_FILE = os.path.join(
    BASE_DIR,
    "feature_selector.pkl"
)

FAULT_FEATURES_FILE = os.path.join(
    BASE_DIR,
    "selected_features.pkl"
)


OVERLOAD_MODEL_FILE = os.path.join(
    BASE_DIR,
    "overload_model.pkl.gz"
)

OVERLOAD_SELECTOR_FILE = os.path.join(
    BASE_DIR,
    "overload_feature_selector.pkl"
)

OVERLOAD_FEATURES_FILE = os.path.join(
    BASE_DIR,
    "overload_selected_features.pkl"
)


# ============================================================
# FILE LOADER
# ============================================================

def load_file(path):
    """
    Loads .pkl and .pkl.gz files.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model file not found:\n{path}"
        )

    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)

        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise RuntimeError(
            f"Unable to load file:\n{path}\n\nError: {e}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    fault_model = load_file(FAULT_MODEL_FILE)
    fault_selector = load_file(FAULT_SELECTOR_FILE)
    fault_selected_features = load_file(FAULT_FEATURES_FILE)

    overload_model = load_file(OVERLOAD_MODEL_FILE)
    overload_selector = load_file(OVERLOAD_SELECTOR_FILE)
    overload_selected_features = load_file(OVERLOAD_FEATURES_FILE)

except Exception as e:

    print("MODEL LOADING ERROR")
    print(e)

    fault_model = None
    fault_selector = None
    fault_selected_features = None

    overload_model = None
    overload_selector = None
    overload_selected_features = None


# ============================================================
# REQUIRED FEATURES FROM YOUR TRAINED MODELS
# ============================================================

FAULT_REQUIRED_FEATURES = [
    "Voltage Fluctuation (%)",
    "Temperature (°C)",
    "Electricity Price (USD/kWh)",
    "Hour",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    "IsPeakHour",
    "Renewable_Ratio",
    "Absolute_Power_Imbalance"
]


OVERLOAD_REQUIRED_FEATURES = [
    "Voltage (V)",
    "Solar Power (kW)",
    "Grid Supply (kW)",
    "Predicted Load (kW)",
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "IsPeakHour",
    "Total_Renewable_Power",
    "Voltage_Deviation"
]


# ============================================================
# FEATURE NAME EXTRACTION
# ============================================================

def extract_feature_names(feature_object):

    if feature_object is None:
        return None

    if isinstance(feature_object, (list, tuple, np.ndarray, pd.Index)):
        return list(feature_object)

    if isinstance(feature_object, dict):

        for key in [
            "features",
            "selected_features",
            "feature_names",
            "columns"
        ]:
            if key in feature_object:
                value = feature_object[key]

                if isinstance(
                    value,
                    (list, tuple, np.ndarray, pd.Index)
                ):
                    return list(value)

    return None


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    input_data,
    required_features,
    selector=None
):
    """
    Converts live/user input into model-ready features.
    """

    if not isinstance(input_data, dict):
        raise ValueError(
            "Input data must be a dictionary."
        )

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Check required features
    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "MODEL FEATURE MISMATCH.\n\n"
            f"Missing features: {missing_features}\n\n"
            f"Required features: {required_features}\n\n"
            f"Supplied features: {list(df.columns)}"
        )

    # Keep exact feature order expected by model
    X = df[required_features].copy()

    # Convert values to numeric
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    if X.isnull().any().any():

        bad_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            f"Invalid numeric values in: {bad_columns}"
        )

    # --------------------------------------------------------
    # Apply saved feature selector if available
    # --------------------------------------------------------

    if selector is not None:

        try:

            X_selected = selector.transform(X)

            return X_selected

        except Exception:

            # Some saved selectors may expect a DataFrame
            try:

                X_selected = selector.transform(
                    pd.DataFrame(X)
                )

                return X_selected

            except Exception as e:

                raise RuntimeError(
                    "Feature selector could not transform "
                    "the prepared input.\n\n"
                    f"Selector error: {e}"
                )

    return X


# ============================================================
# GENERIC MODEL PREDICTION
# ============================================================

def run_model(model, X):

    if model is None:
        raise RuntimeError(
            "Model was not loaded."
        )

    prediction = model.predict(X)

    prediction_value = prediction[0]

    probability = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(X)

            probability = float(
                np.max(probabilities[0])
            )

        except Exception:
            probability = None

    return prediction_value, probability


# ============================================================
# FAULT PREDICTION
# ============================================================

def predict_fault(input_data):

    X = prepare_features(
        input_data,
        FAULT_REQUIRED_FEATURES,
        fault_selector
    )

    prediction, probability = run_model(
        fault_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "type": "Transformer Fault"
    }


# ============================================================
# OVERLOAD PREDICTION
# ============================================================

def predict_overload(input_data):

    X = prepare_features(
        input_data,
        OVERLOAD_REQUIRED_FEATURES,
        overload_selector
    )

    prediction, probability = run_model(
        overload_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "type": "Overload"
    }


# ============================================================
# PREDICTION LABEL
# ============================================================

def prediction_to_label(prediction):

    value = str(prediction).strip().lower()

    if value in [
        "1",
        "true",
        "yes",
        "fault",
        "faulty",
        "abnormal",
        "overload",
        "high"
    ]:
        return "HIGH RISK"

    if value in [
        "0",
        "false",
        "no",
        "normal",
        "healthy",
        "safe",
        "low"
    ]:
        return "NORMAL"

    return str(prediction)


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    fault_result,
    overload_result
):

    fault_prediction = prediction_to_label(
        fault_result["prediction"]
    )

    overload_prediction = prediction_to_label(
        overload_result["prediction"]
    )

    fault_high = (
        fault_prediction == "HIGH RISK"
    )

    overload_high = (
        overload_prediction == "HIGH RISK"
    )

    if fault_high and overload_high:

        risk_level = "CRITICAL"

    elif fault_high or overload_high:

        risk_level = "HIGH"

    else:

        risk_level = "NORMAL"

    return {
        "level": risk_level,
        "fault_status": fault_prediction,
        "overload_status": overload_prediction
    }


# ============================================================
# LIVE DATA GENERATOR
# ============================================================

def get_live_data():

    """
    Simulates real-time smart-grid sensor data.

    IMPORTANT:
    Replace this function with an actual IoT/API/MQTT
    data source when deploying with real sensors.
    """

    now = datetime.now()

    # Simulated electrical measurements
    voltage = random.uniform(
        215,
        245
    )

    temperature = random.uniform(
        25,
        80
    )

    solar_power = random.uniform(
        0,
        50
    )

    wind_power = random.uniform(
        0,
        30
    )

    grid_supply = random.uniform(
        40,
        150
    )

    predicted_load = random.uniform(
        30,
        160
    )

    power_imbalance = (
        grid_supply - predicted_load
    )

    # Voltage deviation from nominal 230 V
    voltage_deviation = abs(
        voltage - 230
    )

    # Voltage fluctuation percentage
    voltage_fluctuation = (
        abs(voltage - 230) / 230
    ) * 100

    # Renewable generation
    total_renewable_power = (
        solar_power + wind_power
    )

    total_generation = (
        grid_supply
        + total_renewable_power
    )

    if total_generation > 0:

        renewable_ratio = (
            total_renewable_power
            / total_generation
        )

    else:

        renewable_ratio = 0

    # Time features
    hour = now.hour
    month = now.month
    day_of_week = now.weekday()

    is_weekend = (
        1
        if day_of_week >= 5
        else 0
    )

    # Peak hours example
    is_peak_hour = (
        1
        if hour in [7, 8, 9, 18, 19, 20, 21]
        else 0
    )

    # Simulated electricity price
    electricity_price = (
        0.12
        + (0.08 if is_peak_hour else 0)
        + random.uniform(
            -0.02,
            0.02
        )
    )

    return {

        # Basic sensor values
        "Voltage": voltage,
        "Temperature": temperature,
        "Solar Power": solar_power,
        "Wind Power": wind_power,
        "Grid Supply": grid_supply,
        "Predicted Load": predicted_load,

        # Fault features
        "Voltage Fluctuation (%)":
            voltage_fluctuation,

        "Temperature (°C)":
            temperature,

        "Electricity Price (USD/kWh)":
            electricity_price,

        "Hour":
            hour,

        "Month":
            month,

        "DayOfWeek":
            day_of_week,

        "IsWeekend":
            is_weekend,

        "IsPeakHour":
            is_peak_hour,

        "Renewable_Ratio":
            renewable_ratio,

        "Absolute_Power_Imbalance":
            abs(power_imbalance),

        # Overload features
        "Voltage (V)":
            voltage,

        "Solar Power (kW)":
            solar_power,

        "Grid Supply (kW)":
            grid_supply,

        "Predicted Load (kW)":
            predicted_load,

        "Total_Renewable_Power":
            total_renewable_power,

        "Voltage_Deviation":
            voltage_deviation,

        # Additional values for dashboard
        "Wind Power (kW)":
            wind_power,

        "Power Imbalance (kW)":
            power_imbalance,

        "Timestamp":
            now
    }


# ============================================================
# COMPLETE LIVE ANALYSIS
# ============================================================

def analyze_live_data():

    input_data = get_live_data()

    fault_result = predict_fault(
        input_data
    )

    overload_result = predict_overload(
        input_data
    )

    risk = calculate_risk(
        fault_result,
        overload_result
    )

    return {
        "timestamp":
            input_data["Timestamp"],

        "data":
            input_data,

        "fault":
            fault_result,

        "overload":
            overload_result,

        "risk":
            risk
    }
