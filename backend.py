```python
import os
import gzip
import pickle
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD FILE
# Supports both .pkl and .pkl.gz
# ============================================================

def load_model(filename):

    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FILE NOT FOUND: {filename}\n"
            f"Expected at: {path}"
        )

    # --------------------------------------------------------
    # GZIP FILE
    # --------------------------------------------------------

    if filename.endswith(".gz"):

        with gzip.open(path, "rb") as f:

            try:
                return pickle.load(f)

            except Exception:

                f.seek(0)
                return joblib.load(f)

    # --------------------------------------------------------
    # NORMAL PKL
    # --------------------------------------------------------

    try:

        return joblib.load(path)

    except Exception:

        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================
# TRANSFORMER FAULT FILES
# ============================================================

fault_selector = load_model(
    "feature_selector.pkl"
)

fault_selected_features = load_model(
    "selected_features.pkl"
)

fault_model = load_model(
    "transformer_fault_model.pkl.gz"
)


# ============================================================
# OVERLOAD FILES
# ============================================================

overload_selector = load_model(
    "overload_feature_selector.pkl"
)

overload_selected_features = load_model(
    "overload_selected_features.pkl"
)

overload_model = load_model(
    "overload_model.pkl.gz"
)


# ============================================================
# FEATURE NAME EXTRACTION
# ============================================================

def extract_feature_names(data):

    if data is None:
        return None

    if isinstance(
        data,
        (list, tuple, np.ndarray)
    ):
        return list(data)

    if isinstance(data, pd.Index):
        return list(data)

    if hasattr(
        data,
        "get_feature_names_out"
    ):

        try:
            return list(
                data.get_feature_names_out()
            )

        except Exception:
            pass

    return None


# ============================================================
# CREATE DATAFRAME
# ============================================================

def create_dataframe(input_data):

    if isinstance(
        input_data,
        pd.DataFrame
    ):
        return input_data.copy()

    if isinstance(
        input_data,
        dict
    ):
        return pd.DataFrame(
            [input_data]
        )

    raise ValueError(
        "Input must be a dictionary or pandas DataFrame."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    input_data,
    selected_features,
    selector
):

    df = create_dataframe(
        input_data
    )

    # Clean column names
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    feature_names = extract_feature_names(
        selected_features
    )

    # --------------------------------------------------------
    # SELECT REQUIRED FEATURES
    # --------------------------------------------------------

    if feature_names is not None:

        # Exact match
        if all(
            feature in df.columns
            for feature in feature_names
        ):

            X = df[
                feature_names
            ].copy()

        else:

            # Case-insensitive matching
            column_map = {
                str(c).lower(): c
                for c in df.columns
            }

            missing = []

            matched_columns = []

            for feature in feature_names:

                key = str(
                    feature
                ).lower()

                if key in column_map:

                    matched_columns.append(
                        column_map[key]
                    )

                else:

                    missing.append(
                        feature
                    )

            if missing:

                raise ValueError(
                    "MODEL FEATURE MISMATCH.\n\n"
                    "Required features:\n"
                    + ", ".join(
                        map(
                            str,
                            feature_names
                        )
                    )
                    + "\n\nYour app supplied:\n"
                    + ", ".join(
                        map(
                            str,
                            df.columns
                        )
                    )
                )

            X = df[
                matched_columns
            ].copy()

            X.columns = feature_names

    else:

        X = df.copy()


    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)


    # --------------------------------------------------------
    # APPLY FEATURE SELECTOR
    # --------------------------------------------------------

    if selector is not None:

        try:

            transformed = selector.transform(
                X
            )

            # Preserve names where possible
            if hasattr(
                selector,
                "get_support"
            ):

                try:

                    support = (
                        selector.get_support()
                    )

                    original_columns = list(
                        X.columns
                    )

                    selected_columns = [
                        original_columns[i]
                        for i, value
                        in enumerate(support)
                        if value
                    ]

                    X = pd.DataFrame(
                        transformed,
                        columns=selected_columns
                    )

                except Exception:

                    X = pd.DataFrame(
                        transformed
                    )

            else:

                X = pd.DataFrame(
                    transformed
                )

        except Exception:

            # Some saved selectors may already
            # have been applied during training.
            pass


    return X


# ============================================================
# PROBABILITY
# ============================================================

def get_probability(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = (
                model.predict_proba(X)[0]
            )

            return float(
                np.max(probabilities)
            )

        except Exception:

            return None

    return None


# ============================================================
# TRANSFORMER FAULT PREDICTION
# ============================================================

def predict_fault(input_data):

    X = prepare_features(
        input_data,
        fault_selected_features,
        fault_selector
    )

    prediction = fault_model.predict(
        X
    )[0]

    probability = get_probability(
        fault_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "features": X
    }


# ============================================================
# OVERLOAD PREDICTION
# ============================================================

def predict_overload(input_data):

    X = prepare_features(
        input_data,
        overload_selected_features,
        overload_selector
    )

    prediction = overload_model.predict(
        X
    )[0]

    probability = get_probability(
        overload_model,
        X
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "features": X
    }


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(
    model,
    X
):

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X
        )

    except Exception:

        return pd.DataFrame({
            "Feature": list(X.columns),
            "Value": X.iloc[0].values,
            "SHAP Value": np.zeros(
                len(X.columns)
            ),
            "Impact": [
                "Unavailable"
                for _ in X.columns
            ]
        })


    # --------------------------------------------------------
    # SHAP LIST FORMAT
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        prediction = model.predict(
            X
        )[0]

        try:

            class_index = list(
                model.classes_
            ).index(
                prediction
            )

        except Exception:

            class_index = 1

        values = np.asarray(
            shap_values[class_index]
        )[0]


    # --------------------------------------------------------
    # SHAP ARRAY FORMAT
    # --------------------------------------------------------

    elif isinstance(
        shap_values,
        np.ndarray
    ):

        if shap_values.ndim == 3:

            prediction = model.predict(
                X
            )[0]

            try:

                class_index = list(
                    model.classes_
                ).index(
                    prediction
                )

            except Exception:

                class_index = 1

            values = shap_values[
                0,
                :,
                class_index
            ]

        elif shap_values.ndim == 2:

            values = shap_values[0]

        else:

            values = shap_values.flatten()

    else:

        values = np.asarray(
            shap_values
        ).flatten()


    # --------------------------------------------------------
    # BUILD TABLE
    # --------------------------------------------------------

    feature_names = list(
        X.columns
    )

    values = np.asarray(
        values
    ).flatten()

    count = min(
        len(feature_names),
        len(values)
    )

    explanation = pd.DataFrame({

        "Feature":
            feature_names[:count],

        "Value":
            X.iloc[0].values[:count],

        "SHAP Value":
            values[:count],

        "Impact": [
            "Increases Risk"
            if value > 0
            else "Decreases Risk"
            for value in values[:count]
        ]

    })

    explanation[
        "Absolute Impact"
    ] = explanation[
        "SHAP Value"
    ].abs()

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return explanation


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    input_data,
    fault_prediction,
    overload_prediction
):

    recommendations = []

    df = create_dataframe(
        input_data
    )

    columns = {
        str(c).lower(): c
        for c in df.columns
    }


    def get_value(names):

        for name in names:

            key = name.lower()

            if key in columns:

                try:

                    return float(
                        df.iloc[0][
                            columns[key]
                        ]
                    )

                except Exception:

                    return None

        return None


    # Current
    current = get_value([
        "current",
        "load_current",
        "transformer_current"
    ])

    if current is not None:

        if current > 80:

            recommendations.append(
                "🔴 High current detected. Reduce excessive electrical loading."
            )


    # Voltage
    voltage = get_value([
        "voltage",
        "grid_voltage",
        "line_voltage"
    ])

    if voltage is not None:

        if voltage < 210:

            recommendations.append(
                "🟠 Low voltage detected. Check voltage regulation."
            )

        elif voltage > 250:

            recommendations.append(
                "🟠 High voltage detected. Inspect voltage regulation equipment."
            )


    # Temperature
    temperature = get_value([
        "temperature",
        "transformer_temperature",
        "temp"
    ])

    if temperature is not None:

        if temperature > 80:

            recommendations.append(
                "🔴 High temperature detected. Inspect transformer cooling."
            )


    # Power factor
    power_factor = get_value([
        "power_factor",
        "pf"
    ])

    if power_factor is not None:

        if power_factor < 0.80:

            recommendations.append(
                "🟠 Low power factor detected. Consider power-factor correction."
            )


    # Imbalance
    imbalance = get_value([
        "power_imbalance",
        "imbalance",
        "phase_imbalance"
    ])

    if imbalance is not None:

        if imbalance > 10:

            recommendations.append(
                "🟠 High phase imbalance detected. Balance the electrical load."
            )


    # Fault
    fault_text = str(
        fault_prediction
    ).lower()

    if (
        "fault" in fault_text
        or fault_text in [
            "1",
            "true",
            "yes"
        ]
    ):

        recommendations.append(
            "🚨 Transformer fault risk detected. Schedule inspection."
        )


    # Overload
    overload_text = str(
        overload_prediction
    ).lower()

    if (
        "overload" in overload_text
        or overload_text in [
            "1",
            "true",
            "yes"
        ]
    ):

        recommendations.append(
            "🚨 Overload risk detected. Reduce non-critical loads."
        )


    if not recommendations:

        recommendations.append(
            "✅ System conditions appear normal. Continue regular monitoring."
        )


    return list(
        dict.fromkeys(
            recommendations
        )
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(
    fault_probability,
    overload_probability
):

    values = []

    if fault_probability is not None:

        values.append(
            fault_probability * 100
        )

    if overload_probability is not None:

        values.append(
            overload_probability * 100
        )

    if not values:

        return 0.0

    return round(
        max(values),
        2
    )
```
