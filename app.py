import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import backend

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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 45px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
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
    'Smart Grid Intelligence • Transformer Fault Detection • '
    'Overload Prediction • Explainable AI'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Grid Parameters")

st.sidebar.write(
    "Enter the operating conditions of the electrical grid."
)


# ============================================================
# TRANSFORMER FAULT FEATURES
# ============================================================

st.sidebar.subheader("🔧 Transformer Fault Parameters")


voltage_fluctuation = st.sidebar.number_input(
    "Voltage Fluctuation (%)",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-50.0,
    max_value=150.0,
    value=55.0,
    step=0.5
)


electricity_price = st.sidebar.number_input(
    "Electricity Price (USD/kWh)",
    min_value=0.0,
    max_value=10.0,
    value=0.15,
    step=0.01
)


power_imbalance = st.sidebar.number_input(
    "Power Imbalance",
    min_value=-100000.0,
    max_value=100000.0,
    value=0.0,
    step=0.1
)


renewable_ratio = st.sidebar.number_input(
    "Renewable Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01
)


absolute_power_imbalance = abs(
    power_imbalance
)


# ============================================================
# OVERLOAD MODEL FEATURES
# ============================================================

st.sidebar.subheader("⚡ Overload Parameters")


voltage = st.sidebar.number_input(
    "Voltage (V)",
    min_value=0.0,
    max_value=1000.0,
    value=230.0,
    step=0.1
)


solar_power = st.sidebar.number_input(
    "Solar Power (kW)",
    min_value=0.0,
    max_value=100000.0,
    value=10.0,
    step=0.1
)


grid_supply = st.sidebar.number_input(
    "Grid Supply (kW)",
    min_value=0.0,
    max_value=100000.0,
    value=50.0,
    step=0.1
)


predicted_load = st.sidebar.number_input(
    "Predicted Load (kW)",
    min_value=0.0,
    max_value=100000.0,
    value=40.0,
    step=0.1
)


wind_power = st.sidebar.number_input(
    "Wind Power (kW)",
    min_value=0.0,
    max_value=100000.0,
    value=5.0,
    step=0.1
)


total_renewable_power = (
    solar_power + wind_power
)


voltage_deviation = st.sidebar.number_input(
    "Voltage Deviation",
    min_value=-1000.0,
    max_value=1000.0,
    value=0.0,
    step=0.1
)


# ============================================================
# TIME FEATURES
# ============================================================

st.sidebar.subheader("🕐 Time Information")


selected_datetime = st.sidebar.datetime_input(
    "Date and Time",
    value=datetime.now()
)


hour = selected_datetime.hour

month = selected_datetime.month

day_of_week = selected_datetime.weekday()

is_weekend = int(
    day_of_week >= 5
)


# Peak-hour definition
is_peak_hour = int(
    hour in [
        7,
        8,
        9,
        18,
        19,
        20,
        21
    ]
)


peak_override = st.sidebar.checkbox(
    "Peak Hour",
    value=bool(is_peak_hour)
)

is_peak_hour = int(
    peak_override
)


# ============================================================
# COMPLETE INPUT DICTIONARY
# ============================================================

input_data = {

    # --------------------------------------------------------
    # FAULT MODEL FEATURES
    # --------------------------------------------------------

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
        absolute_power_imbalance,


    # --------------------------------------------------------
    # OVERLOAD MODEL FEATURES
    # --------------------------------------------------------

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


# ============================================================
# CALCULATED VALUES
# ============================================================

with st.sidebar.expander("📋 Calculated Features"):

    st.write(
        f"**Hour:** {hour}"
    )

    st.write(
        f"**Month:** {month}"
    )

    st.write(
        f"**DayOfWeek:** {day_of_week}"
    )

    st.write(
        f"**IsWeekend:** {is_weekend}"
    )

    st.write(
        f"**IsPeakHour:** {is_peak_hour}"
    )

    st.write(
        f"**Absolute Power Imbalance:** "
        f"{absolute_power_imbalance:.2f}"
    )

    st.write(
        f"**Total Renewable Power:** "
        f"{total_renewable_power:.2f} kW"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 ANALYZE GRID",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    try:

        with st.spinner(
            "🤖 AI is analyzing smart-grid conditions..."
        ):

            # ------------------------------------------------
            # TRANSFORMER FAULT MODEL
            # ------------------------------------------------

            fault_result = predict_fault(
                input_data
            )


            # ------------------------------------------------
            # OVERLOAD MODEL
            # ------------------------------------------------

            overload_result = predict_overload(
                input_data
            )


        # ====================================================
        # RESULTS
        # ====================================================

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


        # ----------------------------------------------------
        # FAULT
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Transformer Fault",
                str(fault_prediction)
            )

            if fault_probability is not None:

                st.progress(
                    min(
                        max(
                            fault_probability,
                            0.0
                        ),
                        1.0
                    )
                )

                st.caption(
                    f"Confidence: "
                    f"{fault_probability * 100:.2f}%"
                )


        # ----------------------------------------------------
        # OVERLOAD
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Overload",
                str(overload_prediction)
            )

            if overload_probability is not None:

                st.progress(
                    min(
                        max(
                            overload_probability,
                            0.0
                        ),
                        1.0
                    )
                )

                st.caption(
                    f"Confidence: "
                    f"{overload_probability * 100:.2f}%"
                )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Overall Risk",
                f"{risk:.2f}%"
            )

            st.progress(
                min(
                    max(
                        risk / 100,
                        0.0
                    ),
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
        # GRID MONITORING
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Grid Monitoring'
            '</div>',
            unsafe_allow_html=True
        )


        monitor1, monitor2, monitor3 = st.columns(3)


        with monitor1:

            st.metric(
                "Voltage",
                f"{voltage:.2f} V"
            )

            st.metric(
                "Temperature",
                f"{temperature:.2f} °C"
            )

            st.metric(
                "Voltage Fluctuation",
                f"{voltage_fluctuation:.2f}%"
            )


        with monitor2:

            st.metric(
                "Solar Power",
                f"{solar_power:.2f} kW"
            )

            st.metric(
                "Grid Supply",
                f"{grid_supply:.2f} kW"
            )

            st.metric(
                "Predicted Load",
                f"{predicted_load:.2f} kW"
            )


        with monitor3:

            st.metric(
                "Total Renewable Power",
                f"{total_renewable_power:.2f} kW"
            )

            st.metric(
                "Voltage Deviation",
                f"{voltage_deviation:.2f}"
            )

            st.metric(
                "Electricity Price",
                f"${electricity_price:.3f}/kWh"
            )


        # ====================================================
        # TIME INFORMATION
        # ====================================================

        st.subheader("🕐 Time Features")


        time1, time2, time3, time4 = st.columns(4)


        with time1:

            st.metric(
                "Hour",
                str(hour)
            )


        with time2:

            st.metric(
                "Month",
                str(month)
            )


        with time3:

            st.metric(
                "Day of Week",
                str(day_of_week)
            )


        with time4:

            st.metric(
                "Peak Hour",
                "Yes" if is_peak_hour else "No"
            )


        # ====================================================
        # SHAP
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🔍 Explainable AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "SHAP explains which features influenced each prediction."
        )


        # ====================================================
        # FAULT SHAP
        # ====================================================

        st.subheader(
            "⚡ Transformer Fault Explanation"
        )


        fault_shap = generate_shap_explanation(
            model=backend.fault_model,
            X=fault_result["features"]
        )


        if not fault_shap.empty:

            st.dataframe(
                fault_shap[
                    [
                        "Feature",
                        "Value",
                        "SHAP Value",
                        "Impact"
                    ]
                ].head(10),
                use_container_width=True,
                hide_index=True
            )


            chart_data = (
                fault_shap
                .head(10)
                .sort_values("SHAP Value")
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                chart_data["Feature"].astype(str),
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
                "Transformer Fault Feature Contributions"
            )

            plt.tight_layout()

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
            model=backend.overload_model,
            X=overload_result["features"]
        )


        if not overload_shap.empty:

            st.dataframe(
                overload_shap[
                    [
                        "Feature",
                        "Value",
                        "SHAP Value",
                        "Impact"
                    ]
                ].head(10),
                use_container_width=True,
                hide_index=True
            )


            chart_data = (
                overload_shap
                .head(10)
                .sort_values("SHAP Value")
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.barh(
                chart_data["Feature"].astype(str),
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
                "Overload Feature Contributions"
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🛠️ AI Recommendations'
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
        # MODEL INPUT DATA
        # ====================================================

        st.divider()

        with st.expander(
            "📋 View Complete Model Input Data"
        ):

            input_df = pd.DataFrame(
                [input_data]
            )

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander(
            "🤖 Model Information"
        ):

            st.write(
                "PowerGuard AI uses two independent machine-learning "
                "pipelines."
            )

            st.write(
                "Transformer Fault Detection uses the saved "
                "feature selector, selected feature list, and "
                "transformer fault model."
            )

            st.write(
                "Overload Detection uses its own saved feature "
                "selector, selected feature list, and overload model."
            )

            st.write(
                "SHAP is used to provide explainable predictions."
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            "❌ Unable to analyze the supplied parameters."
        )

        st.error(
            str(error)
        )

        with st.expander(
            "🔧 Technical Error Details"
        ):

            st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ PowerGuard AI | Machine Learning + "
    "Explainable AI for Smart Grid Monitoring"
)
