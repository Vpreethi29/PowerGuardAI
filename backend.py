import os
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# MODEL DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD PICKLE FILE
# ============================================================

def load_model(filename):

    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return joblib.load(path)


# ============================================================
# LOAD TRANSFORMER FAULT PIPELINE
# ============================================================

fault_selector = load_model(
    "feature_selector.pkl"
)

fault_selected_features = load_model(
    "selected_features.pkl"
)

fault_model = load_model(
    "transformer_fault_model.pkl"
)


# ============================================================
# LOAD OVERLOAD PIPELINE
# ============================================================

overload_selector = load_model(
    "overload_feature_selector.pkl"
)

overload_selected_features = load_model(
    "overload_selected_features.pkl"
)

overload_model = load_model(
    "overload_model.pkl"
)


# ============================================================
# FEATURE NAME HELPER
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
    if isinstance(feature_data, pd.Index):
        return list(feature_data)

    # sklearn object
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
# CREATE INPUT DATAFRAME
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
        "Input must be a dictionary or DataFrame."
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

    feature_names = extract_feature_names(
        selected_features
    )

    # --------------------------------------------------------
    # Use saved selected features
    # --------------------------------------------------------

    if feature_names is not None:

        missing = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    map(str, missing)
                )
            )

        X = df[feature_names].copy()

    else:

        X = df.copy()

    # --------------------------------------------------------
    # Apply feature selector
    # --------------------------------------------------------

    if selector is not None:

        try:

            X_selected = selector.transform(X)

            # Try to preserve feature names
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
                        original_columns[i]
                        for i, selected
                        in enumerate(support)
                        if selected
                    ]

                    X = pd.DataFrame(
                        X_selected,
                        columns=selected_columns
                    )

                except Exception:

                    X = pd.DataFrame(
                        X_selected
                    )

            else:

                X = pd.DataFrame(
                    X_selected
                )

        except Exception as error:

            print(
                "Feature selector was not applied:",
                error
            )

    return X


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

def get_probability(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)[0]
        )

        return float(
            np.max(probabilities)
        )

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

    # --------------------------------------------------------
    # TreeExplainer is ideal for Random Forest
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    # --------------------------------------------------------
    # SHAP versions differ for binary classification.
    # Handle both old and new formats.
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        # For binary classification,
        # use the predicted class
        prediction = model.predict(X)[0]

        try:

            class_index = list(
                model.classes_
            ).index(prediction)

        except Exception:

            class_index = 1

        values = shap_values[
            class_index
        ][0]

    elif isinstance(
        shap_values,
        np.ndarray
    ):

        # New SHAP format:
        # samples × features × classes

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

        else:

            values = shap_values[0]

    else:

        values = np.array(
            shap_values
        ).flatten()

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = list(
        X.columns
    )

    # Protect against mismatch
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

        "Impact":
            [
                "Increases Risk"
                if value > 0
                else "Decreases Risk"
                for value in values[:count]
            ]

    })

    # Sort by absolute contribution
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

    # Convert column names to lowercase
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
                            columns[name.lower()]
                        ]
                    )

                except Exception:

                    return None

        return None

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VOLTAGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # POWER FACTOR
    # --------------------------------------------------------

    power_factor = get_value([
        "power_factor",
        "pf"
    ])

    if power_factor is not None:

        if power_factor < 0.8:

            recommendations.append(
                "🟠 Low power factor detected. Consider power-factor correction."
            )

    # --------------------------------------------------------
    # POWER IMBALANCE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FAULT PREDICTION
    # --------------------------------------------------------

    fault_text = str(
        fault_prediction
    ).lower()

    if (
        "fault" in fault_text
        or "yes" in fault_text
        or "1" == fault_text
        or "true" in fault_text
    ):

        recommendations.append(
            "🚨 Transformer fault risk detected. Inspect transformer operating conditions."
        )

    # --------------------------------------------------------
    # OVERLOAD
    # --------------------------------------------------------

    overload_text = str(
        overload_prediction
    ).lower()

    if (
        "overload" in overload_text
        or "yes" in overload_text
        or "1" == overload_text
        or "true" in overload_text
    ):

        recommendations.append(
            "🚨 Overload risk detected. Reduce non-critical loads and monitor current continuously."
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.append(
            "✅ Operating conditions appear normal. Continue regular monitoring."
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

        return 0

    return round(
        max(values),
        2
    )
