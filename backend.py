import os
import gzip
import pickle
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL FILES
# ============================================================

FAULT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "transformer_fault_model.pkl.gz"
)

FAULT_SELECTOR_PATH = os.path.join(
    BASE_DIR,
    "feature_selector.pkl"
)

FAULT_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "selected_features.pkl"
)


OVERLOAD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "overload_model.pkl.gz"
)

OVERLOAD_SELECTOR_PATH = os.path.join(
    BASE_DIR,
    "overload_feature_selector.pkl"
)

OVERLOAD_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "overload_selected_features.pkl"
)


# ============================================================
# FILE LOADER
# ============================================================

def load_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {os.path.basename(path)}"
        )

    try:

        if path.endswith(".gz"):

            with gzip.open(path, "rb") as f:
                return pickle.load(f)

        else:

            with open(path, "rb") as f:
                return pickle.load(f)

    except Exception as e:

        raise RuntimeError(
            f"Could not load {os.path.basename(path)}: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    fault_model = load_file(
        FAULT_MODEL_PATH
    )

    fault_selector = load_file(
        FAULT_SELECTOR_PATH
    )

    fault_selected_features = load_file(
        FAULT_FEATURES_PATH
    )

    overload_model = load_file(
        OVERLOAD_MODEL_PATH
    )

    overload_selector = load_file(
        OVERLOAD_SELECTOR_PATH
    )

    overload_selected_features = load_file(
        OVERLOAD_FEATURES_PATH
    )

    MODELS_LOADED = True
    MODEL_ERROR = None

except Exception as e:

    MODELS_LOADED = False
    MODEL_ERROR = str(e)

    fault_model = None
    fault_selector = None
    fault_selected_features = None

    overload_model = None
    overload_selector = None
    overload_selected_features = None


# ============================================================
# ACTUAL FEATURES REQUIRED BY YOUR MODELS
# ============================================================

FAULT_FEATURES = [
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


OVERLOAD_FEATURES = [
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

    df = pd.DataFrame([data])

    missing = [
        f
        for f in required_features
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            "MODEL FEATURE MISMATCH.\n\n"
            f"Missing features: {missing}\n\n"
            f"Required features: {required_features}\n\n"
            f"Supplied features: {list(df.columns)}"
        )

    X = df[required_features].copy()

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    if X.isnull().any().any():

        raise ValueError(
            "One or more input values are invalid."
        )

    # Apply saved selector
    if selector is not None:

        X = selector.transform(X)

    return X


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(
    model,
    X
):

    prediction = model.predict(X)

    prediction_value = prediction[0]

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

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

def predict_fault(data):

    if not MODELS_LOADED:

        raise RuntimeError(
            MODEL_ERROR
        )

    X = prepare_features(
        data,
        FAULT_FEATURES,
        fault_selector
    )

    prediction, probability = run_prediction(
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
        OVERLOAD_FEATURES,
        overload_selector
    )

    prediction, probability = run_prediction(
        overload_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability
    }


# ============================================================
# RISK CONVERSION
# ============================================================

def is_high_risk(value):

    value = str(value).lower().strip()

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
# COMPLETE USER INPUT ANALYSIS
# ============================================================

def analyze_input(
    voltage,
    temperature,
    electricity_price,
    solar_power,
    wind_power,
    grid_supply,
    predicted_load,
    hour,
    month,
    day_of_week
):

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    is_weekend = int(
        day_of_week >= 5
    )

    is_peak_hour = int(
        hour in [
            7, 8, 9,
            18, 19, 20, 21
        ]
    )


    # --------------------------------------------------------
    # Derived features
    # --------------------------------------------------------

    voltage_fluctuation = (
        abs(voltage - 230)
        / 230
    ) * 100


    voltage_deviation = abs(
        voltage - 230
    )


    total_renewable_power = (
        solar_power
        + wind_power
    )


    total_available_power = (
        grid_supply
        + total_renewable_power
    )


    renewable_ratio = (

        total_renewable_power
        / total_available_power

        if total_available_power > 0
        else 0
    )


    power_imbalance = (
        grid_supply
        - predicted_load
    )


    # --------------------------------------------------------
    # Complete feature dictionary
    # --------------------------------------------------------

    data = {

        # Fault model
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


        # Overload model
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
            voltage_deviation
    }


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    fault_result = predict_fault(
        data
    )

    overload_result = predict_overload(
        data
    )


    fault_high = is_high_risk(
        fault_result["prediction"]
    )

    overload_high = is_high_risk(
        overload_result["prediction"]
    )


    if fault_high and overload_high:

        overall_risk = "CRITICAL"

    elif fault_high or overload_high:

        overall_risk = "HIGH"

    else:

        overall_risk = "NORMAL"


    return {

        "inputs": data,

        "fault": fault_result,

        "overload": overload_result,

        "overall_risk": overall_risk,

        "derived": {

            "Voltage Fluctuation":
                voltage_fluctuation,

            "Voltage Deviation":
                voltage_deviation,

            "Renewable Ratio":
                renewable_ratio,

            "Total Renewable Power":
                total_renewable_power,

            "Power Imbalance":
                power_imbalance,

            "Is Weekend":
                is_weekend,

            "Is Peak Hour":
                is_peak_hour
        }
    }
