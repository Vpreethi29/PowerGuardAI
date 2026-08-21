# ============================================================
# PowerGuard AI - Backend
# ============================================================

import os
import gzip
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL FILE PATHS
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
# SAFE FILE LOADER
# ============================================================

def load_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {os.path.basename(path)}"
        )

    try:

        if path.endswith(".gz"):

            with gzip.open(path, "rb") as file:

                return pickle.load(file)

        else:

            with open(path, "rb") as file:

                return pickle.load(file)

    except Exception as error:

        raise RuntimeError(
            f"Could not load {os.path.basename(path)}: "
            f"{error}"
        )


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    try:

        fault_model = load_file(
            FAULT_MODEL_PATH
        )

        fault_selector = load_file(
            FAULT_SELECTOR_PATH
        )

        fault_features = load_file(
            FAULT_FEATURES_PATH
        )

        overload_model = load_file(
            OVERLOAD_MODEL_PATH
        )

        overload_selector = load_file(
            OVERLOAD_SELECTOR_PATH
        )

        overload_features = load_file(
            OVERLOAD_FEATURES_PATH
        )

        return (
            fault_model,
            fault_selector,
            fault_features,
            overload_model,
            overload_selector,
            overload_features
        )

    except Exception as error:

        raise RuntimeError(
            "MODEL LOADING FAILED\n\n"
            f"{error}"
        )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        fault_model,
        fault_selector,
        fault_selected_features,
        overload_model,
        overload_selector,
        overload_selected_features

    ) = load_models()

    MODELS_LOADED = True
    MODEL_ERROR = None

except Exception as error:

    fault_model = None
    fault_selector = None
    fault_selected_features = None

    overload_model = None
    overload_selector = None
    overload_selected_features = None

    MODELS_LOADED = False
    MODEL_ERROR = str(error)


# ============================================================
# EXACT MODEL FEATURES
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
# GET FEATURE NAMES
# ============================================================

def get_saved_feature_names(feature_object):

    if feature_object is None:
        return None

    if isinstance(
        feature_object,
        (list, tuple, np.ndarray, pd.Index)
    ):

        return list(feature_object)

    if isinstance(feature_object, dict):

        possible_keys = [

            "features",
            "selected_features",
            "feature_names",
            "columns"

        ]

        for key in possible_keys:

            if key in feature_object:

                value = feature_object[key]

                if isinstance(
                    value,
                    (list, tuple, np.ndarray, pd.Index)
                ):

                    return list(value)

    return None


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    input_data,
    required_features,
    selector=None
):

    if not isinstance(input_data, dict):

        raise ValueError(
            "Input data must be a dictionary."
        )


    df = pd.DataFrame(
        [input_data]
    )


    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing = [

        feature
        for feature in required_features
        if feature not in df.columns

    ]


    if missing:

        raise ValueError(

            "MODEL FEATURE MISMATCH.\n\n"

            f"Missing features: {missing}\n\n"

            f"Required features: "
            f"{required_features}\n\n"

            f"Supplied features: "
            f"{list(df.columns)}"

        )


    # --------------------------------------------------------
    # Exact order
    # --------------------------------------------------------

    X = df[
        required_features
    ].copy()


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    if X.isnull().any().any():

        bad_columns = [

            column
            for column in X.columns
            if X[column].isnull().any()

        ]

        raise ValueError(

            "Invalid numeric values found in: "
            f"{bad_columns}"

        )


    # --------------------------------------------------------
    # Feature selector
    # --------------------------------------------------------

    if selector is not None:

        try:

            X = selector.transform(X)

        except Exception as error:

            raise RuntimeError(

                "Feature selector failed.\n\n"

                f"Selector error: {error}"

            )


    return X


# ============================================================
# MODEL PREDICTION
# ============================================================

def run_prediction(
    model,
    X
):

    if model is None:

        raise RuntimeError(
            "Model is not loaded."
        )


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
                np.max(
                    probabilities[0]
                )
            )

        except Exception:

            probability = None


    return (
        prediction_value,
        probability
    )


# ============================================================
# FAULT PREDICTION
# ============================================================

def predict_fault(
    input_data
):

    if not MODELS_LOADED:

        raise RuntimeError(
            MODEL_ERROR
        )


    X = prepare_features(

        input_data,

        FAULT_REQUIRED_FEATURES,

        fault_selector

    )


    prediction, probability = run_prediction(

        fault_model,

        X

    )


    return {

        "prediction":
            prediction,

        "probability":
            probability,

        "type":
            "Transformer Fault"

    }


# ============================================================
# OVERLOAD PREDICTION
# ============================================================

def predict_overload(
    input_data
):

    if not MODELS_LOADED:

        raise RuntimeError(
            MODEL_ERROR
        )


    X = prepare_features(

        input_data,

        OVERLOAD_REQUIRED_FEATURES,

        overload_selector

    )


    prediction, probability = run_prediction(

        overload_model,

        X

    )


    return {

        "prediction":
            prediction,

        "probability":
            probability,

        "type":
            "Overload"

    }


# ============================================================
# NORMALIZE MODEL OUTPUT
# ============================================================

def prediction_to_risk(
    prediction
):

    value = str(
        prediction
    ).strip().lower()


    high_values = [

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


    normal_values = [

        "0",
        "false",
        "no",
        "normal",
        "healthy",
        "safe",
        "low"

    ]


    if value in high_values:

        return "HIGH RISK"


    if value in normal_values:

        return "NORMAL"


    return str(
        prediction
    )


# ============================================================
# OVERALL RISK
# ============================================================

def calculate_risk(

    fault_result,

    overload_result

):

    fault_status = prediction_to_risk(

        fault_result[
            "prediction"
        ]

    )


    overload_status = prediction_to_risk(

        overload_result[
            "prediction"
        ]

    )


    fault_high = (
        fault_status == "HIGH RISK"
    )


    overload_high = (
        overload_status == "HIGH RISK"
    )


    if fault_high and overload_high:

        overall = "CRITICAL"

    elif fault_high or overload_high:

        overall = "HIGH"

    else:

        overall = "NORMAL"


    return {

        "level":
            overall,

        "fault_status":
            fault_status,

        "overload_status":
            overload_status

    }


# ============================================================
# LIVE DATA
# ============================================================

def get_live_data():

    """
    Demo live-data generator.

    This generates continuously changing
    smart-grid values.

    Replace this function later with:
    - IoT
    - MQTT
    - REST API
    - SCADA
    - Smart meter
    """

    import random


    now = datetime.now()


    # --------------------------------------------------------
    # Base sensor values
    # --------------------------------------------------------

    voltage = random.uniform(
        220,
        240
    )


    temperature = random.uniform(
        30,
        75
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
        50,
        150
    )


    predicted_load = random.uniform(
        40,
        160
    )


    # --------------------------------------------------------
    # Derived values
    # --------------------------------------------------------

    voltage_fluctuation = (

        abs(
            voltage - 230
        )

        / 230

    ) * 100


    voltage_deviation = abs(
        voltage - 230
    )


    total_renewable = (

        solar_power
        + wind_power

    )


    total_available_power = (

        grid_supply
        + total_renewable

    )


    if total_available_power > 0:

        renewable_ratio = (

            total_renewable
            / total_available_power

        )

    else:

        renewable_ratio = 0


    power_imbalance = (

        grid_supply
        - predicted_load

    )


    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    hour = now.hour

    month = now.month

    day_of_week = now.weekday()


    is_weekend = (

        1
        if day_of_week >= 5
        else 0

    )


    is_peak_hour = (

        1
        if hour in [
            7,
            8,
            9,
            18,
            19,
            20,
            21
        ]
        else 0

    )


    electricity_price = (

        0.12

        + (
            0.08
            if is_peak_hour
            else 0
        )

        + random.uniform(
            -0.01,
            0.01
        )

    )


    # --------------------------------------------------------
    # Return ALL required features
    # --------------------------------------------------------

    return {

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
            total_renewable,

        "Voltage_Deviation":
            voltage_deviation,


        # Display values

        "Voltage":
            voltage,

        "Temperature":
            temperature,

        "Solar Power":
            solar_power,

        "Wind Power":
            wind_power,

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
            input_data[
                "Timestamp"
            ],

        "data":
            input_data,

        "fault":
            fault_result,

        "overload":
            overload_result,

        "risk":
            risk

    }
