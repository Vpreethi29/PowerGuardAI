import streamlit as st
import backend


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PowerGuard AI",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚡ PowerGuard AI"
)

st.subheader(
    "Smart Grid Transformer Fault & Overload Prediction"
)

st.write(
    "Enter the current grid parameters to analyze "
    "transformer fault and overload risk using machine learning."
)


# ============================================================
# MODEL CHECK
# ============================================================

if not backend.MODELS_LOADED:

    st.error(
        "❌ Model loading failed."
    )

    st.code(
        backend.MODEL_ERROR
    )

    st.stop()


st.success(
    "✅ ML Models Loaded"
)


# ============================================================
# USER INPUT
# ============================================================

st.header(
    "🔌 Grid Parameters"
)


col1, col2, col3 = st.columns(3)


with col1:

    voltage = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        max_value=500.0,
        value=230.0,
        step=0.1
    )


    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=150.0,
        value=35.0,
        step=0.5
    )


    electricity_price = st.number_input(
        "Electricity Price (USD/kWh)",
        min_value=0.0,
        max_value=10.0,
        value=0.15,
        step=0.01
    )


with col2:

    solar_power = st.number_input(
        "Solar Power (kW)",
        min_value=0.0,
        value=20.0,
        step=1.0
    )


    wind_power = st.number_input(
        "Wind Power (kW)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


    grid_supply = st.number_input(
        "Grid Supply (kW)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )


with col3:

    predicted_load = st.number_input(
        "Predicted Load (kW)",
        min_value=0.0,
        value=90.0,
        step=1.0
    )


    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )


    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )


day_of_week = st.selectbox(
    "Day of Week",
    options=list(range(7)),
    format_func=lambda x: [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ][x]
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()


analyze = st.button(
    "🔍 Analyze Grid Condition",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    try:

        result = backend.analyze_input(

            voltage=voltage,

            temperature=temperature,

            electricity_price=electricity_price,

            solar_power=solar_power,

            wind_power=wind_power,

            grid_supply=grid_supply,

            predicted_load=predicted_load,

            hour=hour,

            month=month,

            day_of_week=day_of_week
        )


        fault = result[
            "fault"
        ]

        overload = result[
            "overload"
        ]

        overall = result[
            "overall_risk"
        ]

        derived = result[
            "derived"
        ]


        # ====================================================
        # RESULTS
        # ====================================================

        st.header(
            "🤖 AI Prediction Results"
        )


        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # FAULT
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "🔧 Transformer Fault"
            )

            if backend.is_high_risk(
                fault["prediction"]
            ):

                st.error(
                    "⚠️ HIGH RISK"
                )

            else:

                st.success(
                    "✅ NORMAL"
                )


            st.write(
                "Model Prediction:",
                fault["prediction"]
            )


            if fault["probability"] is not None:

                st.metric(
                    "Confidence",
                    f"{fault['probability'] * 100:.1f}%"
                )


        # ----------------------------------------------------
        # OVERLOAD
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "⚡ Overload"
            )

            if backend.is_high_risk(
                overload["prediction"]
            ):

                st.error(
                    "⚠️ HIGH RISK"
                )

            else:

                st.success(
                    "✅ NORMAL"
                )


            st.write(
                "Model Prediction:",
                overload["prediction"]
            )


            if overload["probability"] is not None:

                st.metric(
                    "Confidence",
                    f"{overload['probability'] * 100:.1f}%"
                )


        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        with col3:

            st.subheader(
                "🚨 Overall Risk"
            )


            if overall == "CRITICAL":

                st.error(
                    "🔴 CRITICAL"
                )

            elif overall == "HIGH":

                st.warning(
                    "🟠 HIGH"
                )

            else:

                st.success(
                    "🟢 NORMAL"
                )


        # ====================================================
        # DERIVED FEATURES
        # ====================================================

        st.header(
            "📊 Calculated Grid Features"
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Voltage Fluctuation",
            f"{derived['Voltage Fluctuation']:.2f}%"
        )


        c2.metric(
            "Voltage Deviation",
            f"{derived['Voltage Deviation']:.2f} V"
        )


        c3.metric(
            "Renewable Ratio",
            f"{derived['Renewable Ratio'] * 100:.2f}%"
        )


        c4.metric(
            "Power Imbalance",
            f"{derived['Power Imbalance']:.2f} kW"
        )


        # ====================================================
        # OPERATING CONDITIONS
        # ====================================================

        st.header(
            "🕒 Operating Conditions"
        )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Weekend",
                "Yes"
                if derived["Is Weekend"]
                else "No"
            )


        with c2:

            st.metric(
                "Peak Hour",
                "Yes"
                if derived["Is Peak Hour"]
                else "No"
            )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.header(
            "💡 Recommendation"
        )


        if overall == "CRITICAL":

            st.error(
                "Critical condition detected. "
                "Both transformer fault and overload "
                "indicators require immediate attention."
            )

        elif overall == "HIGH":

            st.warning(
                "Elevated risk detected. "
                "Review the grid parameters and "
                "investigate the affected condition."
            )

        else:

            st.success(
                "The system is operating within the "
                "current model's normal prediction range."
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "🔍 Technical Details"
        ):

            st.write(
                "### Transformer Fault Features"
            )

            st.write(
                backend.FAULT_FEATURES
            )


            st.write(
                "### Overload Features"
            )

            st.write(
                backend.OVERLOAD_FEATURES
            )


            st.write(
                "### Input Values"
            )

            st.json(
                result["inputs"]
            )


    except Exception as error:

        st.error(
            "❌ Unable to analyze the supplied parameters."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ PowerGuard AI | Machine Learning + "
    "Explainable Smart Grid Monitoring"
)
