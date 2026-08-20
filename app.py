import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend import (
    predict_fault,
    predict_overload,
    generate_shap_explanation,
    generate_recommendations,
    calculate_risk
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PowerGuard AI",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #cccccc;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ POWERGUARD AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Grid Intelligence • Fault Detection • Overload Prediction • Explainable AI'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Grid Parameters")

st.sidebar.write(
    "Enter the electrical parameters below."
)


voltage = st.sidebar.number_input(
    "Voltage",
    min_value=0.0,
    value=230.0
)

current = st.sidebar.number_input(
    "Current",
    min_value=0.0,
    value=50.0
)

power = st.sidebar.number_input(
    "Power Consumption",
    min_value=0.0,
    value=100.0
)

power_factor = st.sidebar.number_input(
    "Power Factor",
    min_value=0.0,
    max_value=1.0,
    value=0.95
)

temperature = st.sidebar.number_input(
    "Temperature",
    min_value=0.0,
    value=55.0
)

power_imbalance = st.sidebar.number_input(
    "Power Imbalance",
    min_value=0.0,
    value=5.0
)

solar_power = st.sidebar.number_input(
    "Solar Power",
    min_value=0.0,
    value=30.0
)

wind_power = st.sidebar.number_input(
    "Wind Power",
    min_value=0.0,
    value=20.0
)

predicted_load = st.sidebar.number_input(
    "Predicted Load",
    min_value=0.0,
    value=100.0
)


# ============================================================
# INPUT DICTIONARY
# ============================================================

input_data = {

    "Voltage": voltage,

    "Current": current,

    "Power Consumption": power,

    "Power Factor": power_factor,

    "Temperature": temperature,

    "Power Imbalance": power_imbalance,

    "Solar Power": solar_power,

    "Wind Power": wind_power,

    "Predicted Load": predicted_load
}


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 ANALYZE GRID",
    type="primary",
    use_container_width=True
)


if analyze:

    try:

        # ====================================================
        # PREDICTION
        # ====================================================

        with st.spinner(
            "Analyzing smart-grid conditions..."
        ):

            fault_result = predict_fault(
                input_data
            )

            overload_result = predict_overload(
                input_data
            )


        fault_prediction = (
            fault_result["prediction"]
        )

        overload_prediction = (
            overload_result["prediction"]
        )

        fault_probability = (
            fault_result["probability"]
        )

        overload_probability = (
            overload_result["probability"]
        )


        # ====================================================
        # RISK
        # ====================================================

        risk = calculate_risk(
            fault_probability,
            overload_probability
        )


        # ====================================================
        # SYSTEM STATUS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🚦 System Status'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Transformer Fault",
                str(fault_prediction)
            )

            if fault_probability is not None:

                st.progress(
                    min(
                        fault_probability,
                        1.0
                    )
                )

                st.caption(
                    f"Confidence: "
                    f"{fault_probability * 100:.2f}%"
                )


        with col2:

            st.metric(
                "Overload",
                str(overload_prediction)
            )

            if overload_probability is not None:

                st.progress(
                    min(
                        overload_probability,
                        1.0
                    )
                )

                st.caption(
                    f"Confidence: "
                    f"{overload_probability * 100:.2f}%"
                )


        with col3:

            st.metric(
                "Overall Risk",
                f"{risk:.2f}%"
            )

            st.progress(
                min(
                    risk / 100,
                    1.0
                )
            )


        # ====================================================
        # RISK MESSAGE
        # ====================================================

        if risk >= 75:

            st.error(
                "🚨 HIGH RISK — Immediate inspection recommended."
            )

        elif risk >= 50:

            st.warning(
                "⚠️ WARNING — Abnormal operating condition detected."
            )

        else:

            st.success(
                "✅ NORMAL — System is operating within monitored conditions."
            )


        # ====================================================
        # LIVE MONITORING
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Grid Monitoring'
            '</div>',
            unsafe_allow_html=True
        )


        monitor_col1, monitor_col2, monitor_col3 = st.columns(3)


        with monitor_col1:

            st.metric(
                "Voltage",
                f"{voltage:.2f} V"
            )

            st.metric(
                "Current",
                f"{current:.2f} A"
            )

            st.metric(
                "Power",
                f"{power:.2f}"
            )


        with monitor_col2:

            st.metric(
                "Power Factor",
                f"{power_factor:.2f}"
            )

            st.metric(
                "Temperature",
                f"{temperature:.2f} °C"
            )

            st.metric(
                "Power Imbalance",
                f"{power_imbalance:.2f}"
            )


        with monitor_col3:

            st.metric(
                "Solar Power",
                f"{solar_power:.2f}"
            )

            st.metric(
                "Wind Power",
                f"{wind_power:.2f}"
            )

            st.metric(
                "Predicted Load",
                f"{predicted_load:.2f}"
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🔍 Explainable AI — Why did this happen?'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            SHAP explains how each input feature contributed to
            the model's current prediction.
            """
        )


        # ====================================================
        # FAULT SHAP
        # ====================================================

        st.subheader(
            "⚡ Transformer Fault Explanation"
        )

        fault_shap = generate_shap_explanation(
            model=__import__(
                "backend"
            ).fault_model,
            X=fault_result["features"]
        )


        if not fault_shap.empty:

            display_fault = fault_shap[
                [
                    "Feature",
                    "Value",
                    "SHAP Value",
                    "Impact"
                ]
            ].head(10)

            st.dataframe(
                display_fault,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # SHAP BAR CHART
            # ------------------------------------------------

            chart_data = fault_shap.head(8).copy()

            chart_data = chart_data.sort_values(
                "SHAP Value"
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                chart_data["Feature"],
                chart_data["SHAP Value"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Transformer Fault — SHAP Contributions"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ====================================================
        # OVERLOAD SHAP
        # ====================================================

        st.subheader(
            "🔥 Overload Explanation"
        )

        overload_shap = generate_shap_explanation(
            model=__import__(
                "backend"
            ).overload_model,
            X=overload_result["features"]
        )


        if not overload_shap.empty:

            display_overload = overload_shap[
                [
                    "Feature",
                    "Value",
                    "SHAP Value",
                    "Impact"
                ]
            ].head(10)

            st.dataframe(
                display_overload,
                use_container_width=True,
                hide_index=True
            )


            chart_data = overload_shap.head(8).copy()

            chart_data = chart_data.sort_values(
                "SHAP Value"
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                chart_data["Feature"],
                chart_data["SHAP Value"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Overload — SHAP Contributions"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ====================================================
        # TOP RISK CONTRIBUTORS
        # ====================================================

        st.divider()

        st.subheader(
            "📌 Top Risk Contributors"
        )


        if not fault_shap.empty:

            top_features = fault_shap.head(5)

            for _, row in top_features.iterrows():

                value = row["SHAP Value"]

                if value > 0:

                    st.warning(
                        f"⚠️ **{row['Feature']}** "
                        f"is increasing the predicted risk "
                        f"(SHAP: {value:.4f})"
                    )

                else:

                    st.success(
                        f"✅ **{row['Feature']}** "
                        f"is reducing the predicted risk "
                        f"(SHAP: {value:.4f})"
                    )


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '💡 Recommended Actions'
            '</div>',
            unsafe_allow_html=True
        )


        recommendations = generate_recommendations(
            input_data,
            fault_prediction,
            overload_prediction
        )


        for recommendation in recommendations:

            st.info(
                recommendation
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Current Grid Parameters"
        )


        parameter_table = pd.DataFrame({

            "Parameter": [
                "Voltage",
                "Current",
                "Power Consumption",
                "Power Factor",
                "Temperature",
                "Power Imbalance",
                "Solar Power",
                "Wind Power",
                "Predicted Load"
            ],

            "Value": [
                voltage,
                current,
                power,
                power_factor,
                temperature,
                power_imbalance,
                solar_power,
                wind_power,
                predicted_load
            ]
        })


        st.dataframe(
            parameter_table,
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(error)
        )

        st.warning(
            """
            This usually means that the feature names or feature
            order entered in the dashboard do not exactly match
            the features used when the model was trained.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PowerGuard AI • Machine Learning + SHAP + Smart Recommendations"
)
