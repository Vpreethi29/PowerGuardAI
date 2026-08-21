# ============================================================
# PowerGuard AI
# Live Smart Grid Monitoring Dashboard
# ============================================================

import time

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
# TITLE
# ============================================================

st.title(
    "⚡ PowerGuard AI"
)

st.subheader(
    "Live Smart Grid Monitoring & Predictive Analytics"
)

st.write(
    "Continuous machine-learning based monitoring "
    "for transformer faults and overload conditions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Monitoring"
)


refresh_interval = st.sidebar.slider(

    "Refresh interval (seconds)",

    min_value=1,

    max_value=10,

    value=3

)


live_mode = st.sidebar.toggle(

    "Live Monitoring",

    value=True

)


st.sidebar.info(

    "Demo mode uses simulated live sensor data. "
    "The ML models are loaded from the saved "
    "model files in the repository."

)


# ============================================================
# MODEL STATUS
# ============================================================

if not backend.MODELS_LOADED:

    st.error(
        "❌ Model loading failed."
    )

    st.code(
        backend.MODEL_ERROR
    )

    st.stop()


st.sidebar.success(
    "✅ ML Models Loaded"
)


# ============================================================
# DASHBOARD FUNCTION
# ============================================================

def render_dashboard():

    try:

        result = backend.analyze_live_data()

    except Exception as error:

        st.error(
            "Unable to analyze the supplied live data."
        )

        st.exception(error)

        return


    data = result[
        "data"
    ]

    fault = result[
        "fault"
    ]

    overload = result[
        "overload"
    ]

    risk = result[
        "risk"
    ]


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = result[
        "timestamp"
    ]


    st.caption(

        "🟢 LIVE | Last update: "

        + timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    )


    # ========================================================
    # LIVE SENSOR DATA
    # ========================================================

    st.markdown(
        "## 📡 Live Grid Measurements"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "Voltage",

            f"{data['Voltage']:.2f} V"

        )


    with c2:

        st.metric(

            "Temperature",

            f"{data['Temperature']:.2f} °C"

        )


    with c3:

        st.metric(

            "Grid Supply",

            f"{data['Grid Supply']:.2f} kW"

        )


    with c4:

        st.metric(

            "Predicted Load",

            f"{data['Predicted Load']:.2f} kW"

        )


    # ========================================================
    # RENEWABLE DATA
    # ========================================================

    st.markdown(
        "## 🌱 Renewable Energy"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "Solar",

            f"{data['Solar Power']:.2f} kW"

        )


    with c2:

        st.metric(

            "Wind",

            f"{data['Wind Power']:.2f} kW"

        )


    with c3:

        st.metric(

            "Total Renewable",

            f"{data['Total_Renewable_Power']:.2f} kW"

        )


    with c4:

        st.metric(

            "Renewable Ratio",

            f"{data['Renewable_Ratio'] * 100:.1f}%"

        )


    # ========================================================
    # AI PREDICTIONS
    # ========================================================

    st.markdown(
        "## 🤖 AI Predictions"
    )


    c1, c2, c3 = st.columns(3)


    # --------------------------------------------------------
    # Fault
    # --------------------------------------------------------

    with c1:

        st.markdown(
            "### 🔧 Transformer Fault"
        )


        if (
            risk["fault_status"]
            == "HIGH RISK"
        ):

            st.error(
                "⚠️ HIGH RISK"
            )

        else:

            st.success(
                "✅ NORMAL"
            )


        if (
            fault["probability"]
            is not None
        ):

            st.metric(

                "Confidence",

                f"{fault['probability'] * 100:.1f}%"

            )


    # --------------------------------------------------------
    # Overload
    # --------------------------------------------------------

    with c2:

        st.markdown(
            "### ⚡ Overload"
        )


        if (
            risk["overload_status"]
            == "HIGH RISK"
        ):

            st.error(
                "⚠️ HIGH RISK"
            )

        else:

            st.success(
                "✅ NORMAL"
            )


        if (
            overload["probability"]
            is not None
        ):

            st.metric(

                "Confidence",

                f"{overload['probability'] * 100:.1f}%"

            )


    # --------------------------------------------------------
    # Overall Risk
    # --------------------------------------------------------

    with c3:

        st.markdown(
            "### 🚨 Overall Risk"
        )


        if (
            risk["level"]
            == "CRITICAL"
        ):

            st.error(
                "🔴 CRITICAL"
            )

        elif (
            risk["level"]
            == "HIGH"
        ):

            st.warning(
                "🟠 HIGH"
            )

        else:

            st.success(
                "🟢 NORMAL"
            )


    # ========================================================
    # GRID CONDITION
    # ========================================================

    st.markdown(
        "## 📊 Grid Condition"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "Voltage Deviation",

            f"{data['Voltage_Deviation']:.2f} V"

        )


    with c2:

        st.metric(

            "Voltage Fluctuation",

            f"{data['Voltage Fluctuation (%)']:.2f}%"

        )


    with c3:

        st.metric(

            "Power Imbalance",

            f"{data['Power Imbalance']:.2f} kW"

        )


    with c4:

        st.metric(

            "Electricity Price",

            f"${data['Electricity Price (USD/kWh)']:.3f}"

        )


    # ========================================================
    # TIME INFORMATION
    # ========================================================

    st.markdown(
        "## 🕒 Operating Conditions"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "Hour",

            str(data["Hour"])

        )


    with c2:

        st.metric(

            "Month",

            str(data["Month"])

        )


    with c3:

        st.metric(

            "Weekend",

            "Yes"
            if data["IsWeekend"]
            else "No"

        )


    with c4:

        st.metric(

            "Peak Hour",

            "Yes"
            if data["IsPeakHour"]
            else "No"

        )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    st.markdown(
        "## 💡 Recommended Action"
    )


    if risk["level"] == "CRITICAL":

        st.error(

            """
            **Critical condition detected.**

            Both fault and overload indicators are
            currently elevated.

            Verify sensor readings and investigate the
            affected equipment according to operational
            procedures.
            """

        )


    elif risk["level"] == "HIGH":

        st.warning(

            """
            **Elevated risk detected.**

            At least one ML pipeline has identified a
            potentially abnormal operating condition.

            Review the live parameters and investigate
            the affected condition.
            """

        )


    else:

        st.success(

            """
            **Normal operating condition.**

            No elevated risk was identified by the
            current prediction outputs.
            """

        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "🔍 Technical Details"
    ):

        st.write(
            "### Fault Model Features"
        )

        st.write(
            backend.FAULT_REQUIRED_FEATURES
        )


        st.write(
            "### Overload Model Features"
        )

        st.write(
            backend.OVERLOAD_REQUIRED_FEATURES
        )


        st.write(
            "### Architecture"
        )

        st.code(

            """
Live Sensor Data
       ↓
Feature Engineering
       ↓
Fault Pipeline ─────→ Fault Model
       │
       └─────────────→ Fault Prediction

Live Sensor Data
       ↓
Feature Engineering
       ↓
Overload Pipeline ──→ Overload Model
       │
       └─────────────→ Overload Prediction

Fault + Overload
       ↓
Risk Assessment
       ↓
Streamlit Dashboard
            """

        )


# ============================================================
# RUN DASHBOARD
# ============================================================

if live_mode:

    container = st.empty()


    while True:

        with container.container():

            render_dashboard()


        time.sleep(
            refresh_interval
        )


        st.rerun()


else:

    render_dashboard()


    st.info(
        "Live monitoring is currently paused."
    )
