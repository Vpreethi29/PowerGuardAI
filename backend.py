import os
import gzip
import pickle
import tempfile
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL LOCATIONS
# ============================================================

FAULT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "transformer_fault_model.pkl.gz"
)

OVERLOAD_MODEL_FILE = os.path.join(
    BASE_DIR,
    "overload_model.pkl.gz"
)

FAULT_SELECTOR_FILE = os.path.join(
    BASE_DIR,
    "feature_selector.pkl"
)

FAULT_FEATURES_FILE = os.path.join(
    BASE_DIR,
    "selected_features.pkl"
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
# OPTIONAL MODEL URLS
#
# If models are hosted somewhere else, put the URLs in
# Streamlit Secrets:
#
# FAULT_MODEL_URL = "..."
# OVERLOAD_MODEL_URL = "..."
# ============================================================

FAULT_MODEL_URL = os.environ.get(
    "FAULT_MODEL_URL",
    ""
)

OVERLOAD_MODEL_URL = os.environ.get(
    "OVERLOAD_MODEL_URL",
    "")


# ============================================================
# DOWNLOAD MODEL IF NOT PRESENT
# ============================================================

def ensure_model(
    local_path,
    url,
    display_name
):

    if os.path.exists(local_path):
        return local_path

    if not url:
        raise FileNotFoundError(
            f"""
{display_name} was not found.

Expected file:
{local_path}

Because this model is larger than GitHub's normal
file limit, either:

1. Add the model using Git LFS, OR
2. Host the model externally and set:
   {display_name.upper()}_URL

The application cannot load a model that does not
exist locally or at a configured URL.
"""
        )

    try:

        urllib.request.urlretrieve(
            url,
            local_path
        )

        if not os.path.exists(local_path):

            raise RuntimeError(
                f"Download failed for {display_name}"
            )

        return local_path

    except Exception as error:

        raise RuntimeError(
            f"Could not download {display_name}: {error}"
        )


# ============================================================
# LOAD PICKLE / GZIP MODEL
# ============================================================

def load_model(
    path,
    url="",
    display_name="model"
):

    path = ensure_model(
        path,
        url,
        display_name
    )

    try:

        if path.endswith(".gz"):

            with gzip.open(
                path,
                "rb"
            ) as file:

                model = pickle.load(file)

        else:

            with open(
                path,
                "rb"
            ) as file:

                model = pickle.load(file)

        return model

    except Exception as error:

        raise RuntimeError(
            f"""
Could not load {display_name}.

File:
{path}

Error:
{type(error).__name__}: {error}

IMPORTANT:
The .pkl.gz file must be a gzip-compressed
pickle file created from the original model.
"""
        )


# ============================================================
# LOAD SMALL FILES
# ============================================================

def load_pickle(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(
        path,
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    fault_model = load_model(
        FAULT_MODEL_FILE,
        FAULT_MODEL_URL,
        "transformer_fault_model.pkl.gz"
    )

    overload_model = load_model(
        OVERLOAD_MODEL_FILE,
        OVERLOAD_MODEL_URL,
        "overload_model.pkl.gz"
    )

    fault_selector = load_pickle(
        FAULT_SELECTOR_FILE
    )

    fault_selected_features = load_pickle(
        FAULT_FEATURES_FILE
    )

    overload_selector = load_pickle(
        OVERLOAD_SELECTOR_FILE
    )

    overload_selected_features = load_pickle(
        OVERLOAD_FEATURES_FILE
    )

    MODELS_LOADED = True
    MODEL_ERROR = None

except Exception as error:

    MODELS_LOADED = False
    MODEL_ERROR = str(error)

    fault_model = None
    overload_model = None
    fault_selector = None
    fault_selected_features = None
    overload_selector = None
    overload_selected_features = None


# ============================================================
# REQUIRED FEATURES
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
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    data,
    required_features,
    selector
):

    df = pd.DataFrame(
        [data]
    )

    missing = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "MODEL FEATURE MISMATCH.\n\n"
            f"Missing: {missing}\n\n"
            f"Required: {required_features}\n\n"
            f"Supplied: {list(df.columns)}"
        )

    X = df[
        required_features
    ].copy()

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    if X.isnull().any().any():

        raise ValueError(
            "Invalid numeric value in model features."
        )

    if selector is not None:

        try:

            X = selector.transform(X)

        except Exception as error:

            raise RuntimeError(
                f"Feature selector error: {error}"
            )

    return X


# ============================================================
# GENERIC PREDICTION
# ============================================================

def run_model(
    model,
    X
):

    prediction = model.predict(X)

    value = prediction[0]

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(X)

            probability = float(
                np.max(
                    probabilities[0]
                )
            )

        except Exception:
            probability = None

    return value, probability


# ============================================================
# FAULT PREDICTION
# ============================================================

def predict_fault(data):

    if not MODELS_LOADED:

        raise RuntimeError(
            MODEL_ERROR
        )

    X = prepare_features(
        data,
        FAULT_REQUIRED_FEATURES,
        fault_selector
    )

    prediction, probability = run_model(
        fault_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability
    }


# ============================================================
# OVERLOAD PREDICTION
# ============================================================

def predict_overload(data):

    if not MODELS_LOADED:

        raise RuntimeError(
            MODEL_ERROR
        )

    X = prepare_features(
        data,
        OVERLOAD_REQUIRED_FEATURES,
        overload_selector
    )

    prediction, probability = run_model(
        overload_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability
    }


# ============================================================
# NORMALIZE PREDICTION
# ============================================================

def is_risk(value):

    value = str(
        value
    ).strip().lower()

    return value in [
        "1",
        "true",
        "yes",
        "fault",
        "faulty",
        "abnormal",
        "overload",
        "high",
        "critical"
    ]


# ============================================================
# LIVE DATA
# ============================================================

def get_live_data():

    import random

    now = datetime.now()

    voltage = random.uniform(
        220,
        240
    )

    temperature = random.uniform(
        30,
        75
    )

    solar = random.uniform(
        0,
        50
    )

    wind = random.uniform(
        0,
        30
    )

    grid_supply = random.uniform(
        50,
        150
    )

    predicted_load = random.uniform(
        40,
        160
    )

    voltage_fluctuation = (
        abs(voltage - 230)
        / 230
    ) * 100

    voltage_deviation = abs(
        voltage - 230
    )

    renewable_power = (
        solar + wind
    )

    total_available = (
        grid_supply
        + renewable_power
    )

    renewable_ratio = (
        renewable_power
        / total_available
        if total_available > 0
        else 0
    )

    power_imbalance = (
        grid_supply
        - predicted_load
    )

    hour = now.hour

    month = now.month

    day_of_week = now.weekday()

    is_weekend = int(
        day_of_week >= 5
    )

    is_peak_hour = int(
        hour in [
            7, 8, 9,
            18, 19, 20, 21
        ]
    )

    electricity_price = (
        0.12
        + (0.08 if is_peak_hour else 0)
        + random.uniform(
            -0.01,
            0.01
        )
    )

    return {

        # Fault model features

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

        # Overload model features

        "Voltage (V)":
            voltage,

        "Solar Power (kW)":
            solar,

        "Grid Supply (kW)":
            grid_supply,

        "Predicted Load (kW)":
            predicted_load,

        "Total_Renewable_Power":
            renewable_power,

        "Voltage_Deviation":
            voltage_deviation,

        # Dashboard values

        "Voltage":
            voltage,

        "Temperature":
            temperature,

        "Solar Power":
            solar,

        "Wind Power":
            wind,

        "Grid Supply":
            grid_supply,

        "Predicted Load":
            predicted_load,

        "Power Imbalance":
            power_imbalance,

        "Timestamp":
            now
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_live_data():

    data = get_live_data()

    fault = predict_fault(
        data
    )

    overload = predict_overload(
        data
    )

    fault_risk = is_risk(
        fault["prediction"]
    )

    overload_risk = is_risk(
        overload["prediction"]
    )

    if fault_risk and overload_risk:
        overall = "CRITICAL"

    elif fault_risk or overload_risk:
        overall = "HIGH"

    else:
        overall = "NORMAL"

    return {

        "data": data,

        "fault": fault,

        "overload": overload,

        "overall_risk": overall
    }
