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
# LOAD MODEL / PICKLE FILE
# Supports .pkl and .pkl.gz
# ============================================================

def load_model(filename):

    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {filename}\n"
            f"Expected location: {path}"
        )

    # --------------------------------------------------------
    # GZIP PICKLE
    # --------------------------------------------------------

    if filename.endswith(".gz"):

        try:
            with gzip.open(path, "rb") as file:
                return pickle.load(file)

        except Exception:

            # Some files may have been compressed from joblib
            with gzip.open(path, "rb") as file:
                return joblib.load(file)

    # --------------------------------------------------------
    # NORMAL PICKLE
    # --------------------------------------------------------

    try:
        return joblib.load(path)

    except Exception:

        with open(path, "rb") as file:
            return pickle.load(file)


# ============================================================
# LOAD TRANSFORMER FAULT FILES
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
# LOAD OVERLOAD FILES
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

def extract_feature_names(feature_data):

    if feature_data is None:
        return None

    # List / tuple / numpy array
    if isinstance(
        feature_data,
        (list, tuple, np.ndarray)
    ):
        return list(feature_data)

    # Pandas Index
    if isinstance(
        feature_data,
        pd.Index
    ):
        return list(feature_data)

    # Objects with feature names
    if hasattr(
        feature_data,
        "get_feature_names_out"
    ):

        try:
            return list(
                feature_data.get_feature_names_out()
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

    df = create_dataframe(input_data)

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    feature_names = extract_feature_names(
        selected_features
    )

    # --------------------------------------------------------
    # SELECT SAVED FEATURES
    # --------------------------------------------------------

    if feature_names is not None:

        missing = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing:

            # Try case-insensitive matching
            column_map = {
                str(column).lower(): column
                for column in df.columns
            }

            alternative_missing = []

            for feature in missing:

                if str(feature).lower() not in column_map:
                    alternative_missing.append(feature)

            if alternative_missing:

                raise ValueError(
                    "Missing model features:\n"
                    + ", ".join(
                        map(
                            str,
                            alternative_missing
                        )
                    )
                    + "\n\nAvailable input features:\n"
                    + ", ".join(
                        map(
                            str,
                            df.columns
                        )
                    )
                )

            # Case-insensitive reconstruction
            matched_columns = []

            for feature in feature_names:

                matched_columns.append(
                    column_map[
                        str(feature).lower()
                    ]
                )

            X = df[
                matched_columns
            ].copy()

            X.columns = feature_names

        else:

            X = df[
                feature_names
            ].copy()

    else:

        X = df.copy()

    # --------------------------------------------------------
    # Convert numeric values
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

            selected = selector.transform(X)

            # Preserve feature names if possible
            if hasattr(
                selector,
                "get_support"
            ):

                try:

                    support = selector.get_support()

                    original_columns = list(
                        X.columns
                    )

                    selected_columns = [
                        original_columns[index]
                        for index, is_selected
                        in enumerate(support)
                        if is_selected
                    ]

                    X = pd.DataFrame(
                        selected,
                        columns=selected_columns
                    )

                except Exception:

                    X = pd.DataFrame(
                        selected
                    )

            else:

                X = pd.DataFrame(
                    selected
                )

        except Exception as error:

            # If selector was already applied
            # during training, continue with current data.
            print(
                "Feature selector warning:",
                error
            )

    return X


# ============================================================
# GET MODEL PROBABILITY
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

    prediction = fault_model.predict(X)[0]

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

    prediction = overload_model.predict(X)[0]

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

    except Exception as error:

        # Return a fallback explanation
        # instead of crashing the entire dashboard.
        return pd.DataFrame({
            "Feature": list(X.columns),
            "Value": X.iloc[0].values,
            "SHAP Value": np.zeros(
                len(X.columns)
            ),
            "Impact": [
                "Unavailable"
                for _ in X.columns
            ],
            "Absolute Impact": np.zeros(
                len(X.columns)
            )
        })


    # ========================================================
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # ========================================================

    if isinstance(
        shap_values,
        list
    ):

        prediction = model.predict(X)[0]

        try:

            class_index = list(
                model.classes_
            ).index(prediction)

        except Exception:

            class_index = 1

        values = np.asarray(
            shap_values[class_index]
        )[0]

    elif isinstance(
        shap_values,
        np.ndarray
    ):

        if shap_values.ndim == 3:

            prediction = model.predict(X)[0]

            try:

                class_index = list(
                    model.classes_
                ).index(prediction)

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


    # ========================================================
    # CREATE EXPLANATION TABLE
    # ========================================================

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
# RECOMMENDATION ENGINE
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

    # --------------------------------------------------------
    # Case-insensitive column lookup
    # --------------------------------------------------------

    columns = {
        str(column).lower(): column
        for column in df.columns
    }


    def get_value(names):

        for name in names:

            if name.lower() in columns:

                try:

                    return float(
                        df.iloc[0][
                            columns[
                                name.lower()
                            ]
                        ]
                    )

                except Exception:

                    return None

        return None


    # ========================================================
    # CURRENT
    # ========================================================

    current = get_value([
        "current",
        "load_current",
        "transformer_current"
    ])

    if current is not None:

        if current > 80:

            recommendations.append(
                "🔴 Reduce excessive load and inspect transformer loading."
            )


    # ========================================================
    # VOLTAGE
    # ========================================================

    voltage = get_value([
        "voltage",
        "grid_voltage",
        "line_voltage"
    ])

    if voltage is not None:

        if voltage < 210:

            recommendations.append(
                "🟠 Low voltage detected. Check voltage regulation and grid stability."
            )

        elif voltage > 250:

            recommendations.append(
                "🟠 High voltage detected. Inspect voltage regulation equipment."
            )


    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature = get_value([
        "temperature",
        "transformer_temperature",
        "temp"
    ])

    if temperature is not None:

        if temperature > 80:

            recommendations.append(
                "🔴 High transformer temperature. Inspect cooling and reduce loading."
            )


    # ========================================================
    # POWER FACTOR
    # ========================================================

    power_factor = get_value([
        "power_factor",
        "pf"
    ])

    if power_factor is not None:

        if power_factor < 0.80:

            recommendations.append(
                "🟠 Low power factor detected. Consider power-factor correction."
            )


    # ========================================================
    # POWER IMBALANCE
    # ========================================================

    imbalance = get_value([
        "power_imbalance",
        "imbalance",
        "phase_imbalance"
    ])

    if imbalance is not None:

        if imbalance > 10:

            recommendations.append(
                "🟠 High power imbalance. Balance the load across phases."
            )


    # ========================================================
    # FAULT PREDICTION
    # ========================================================

    fault_text = str(
        fault_prediction
    ).lower().strip()

    if (
        "fault" in fault_text
        or fault_text == "yes"
        or fault_text == "true"
        or fault_text == "1"
    ):

        recommendations.append(
            "🚨 Transformer fault risk detected. Inspect transformer operating conditions."
        )


    # ========================================================
    # OVERLOAD PREDICTION
    # ========================================================

    overload_text = str(
        overload_prediction
    ).lower().strip()

    if (
        "overload" in overload_text
        or overload_text == "yes"
        or overload_text == "true"
        or overload_text == "1"
    ):

        recommendations.append(
            "🚨 Overload risk detected. Reduce non-critical loads and monitor current continuously."
        )


    # ========================================================
    # DEFAULT
    # ========================================================

    if len(recommendations) == 0:

        recommendations.append(
            "✅ Operating conditions appear normal. Continue regular monitoring."
        )


    # Remove duplicates
    recommendations = list(
        dict.fromkeys(
            recommendations
        )
    )

    return recommendations


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
