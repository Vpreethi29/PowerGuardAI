import time

import streamlit as st

import backend


# ============================================================
# PAGE
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
    "Live Smart Grid Monitoring & Predictive Analytics"
)

st.write(
    "Continuous machine-learning based monitoring "
    "for transformer faults and overload conditions."
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

    st.info(
        """
Your large model files are not currently available
inside the Streamlit repository.

If they are hosted externally, configure these
Streamlit Secrets:

FAULT_MODEL_URL
OVERLOAD_MODEL_URL
"""
    )

    st.stop()


st.success(
    "✅ ML Models Loaded"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Live Monitoring"
)

refresh = st.sidebar.slider(
    "Refresh interval",
    1,
    10,
    3
)

live = st.sidebar.checkbox(
    "Enable Live Monitoring",
    True
)


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    try:

        result = backend.analyze_live_data()

    except Exception as error:

        st.error(
            "❌ Prediction failed"
        )

        st.exception(error)

        return

    data = result["data"]

    fault = result["fault"]

    overload = result["overload"]

    overall = result["overall_risk"]


    # ========================================================
    # TIME
    # ========================================================

    st.caption(
        "🟢 LIVE | "
        + data["Timestamp"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    # ========================================================
    # LIVE DATA
    # ========================================================

    st.header(
        "📡 Live Grid Data"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Voltage",
        f"{data['Voltage']:.2f} V"
    )

    c2.metric(
        "Temperature",
        f"{data['Temperature']:.2f} °C"
    )

    c3.metric(
        "Grid Supply",
        f"{data['Grid Supply']:.2f} kW"
    )

    c4.metric(
        "Predicted Load",
        f"{data['Predicted Load']:.2f} kW"
    )


    # ========================================================
    # RENEWABLE
    # ========================================================

    st.header(
        "🌱 Renewable Energy"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Solar",
        f"{data['Solar Power']:.2f} kW"
    )

    c2.metric(
        "Wind",
        f"{data['Wind Power']:.2f} kW"
    )

    c3.metric(
        "Total Renewable",
        f"{data['Total_Renewable_Power']:.2f} kW"
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.header(
        "🤖 AI Predictions"
    )

    c1, c2, c3 = st.columns(3)


    with c1:

        st.subheader(
            "🔧 Transformer Fault"
        )

        if backend.is_risk(
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
            "Prediction:",
            fault["prediction"]
        )

        if fault["probability"] is not None:

            st.metric(
                "Confidence",
                f"{fault['probability'] * 100:.1f}%"
            )


    with c2:

        st.subheader(
            "⚡ Overload"
        )

        if backend.is_risk(
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
            "Prediction:",
            overload["prediction"]
        )

        if overload["probability"] is not None:

            st.metric(
                "Confidence",
                f"{overload['probability'] * 100:.1f}%"
            )


    with c3:

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


    # ========================================================
    # GRID CONDITION
    # ========================================================

    st.header(
        "📊 Grid Condition"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Voltage Deviation",
        f"{data['Voltage_Deviation']:.2f} V"
    )

    c2.metric(
        "Voltage Fluctuation",
        f"{data['Voltage Fluctuation (%)']:.2f}%"
    )

    c3.metric(
        "Power Imbalance",
        f"{data['Power Imbalance']:.2f} kW"
    )

    c4.metric(
        "Electricity Price",
        f"${data['Electricity Price (USD/kWh)']:.3f}"
    )


    # ========================================================
    # OPERATING CONDITION
    # ========================================================

    st.header(
        "🕒 Operating Conditions"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Hour",
        data["Hour"]
    )

    c2.metric(
        "Month",
        data["Month"]
    )

    c3.metric(
        "Weekend",
        "Yes"
        if data["IsWeekend"]
        else "No"
    )

    c4.metric(
        "Peak Hour",
        "Yes"
        if data["IsPeakHour"]
        else "No"
    )


    # ========================================================
    # ACTION
    # ========================================================

    st.header(
        "💡 Recommended Action"
    )

    if overall == "CRITICAL":

        st.error(
            "Critical condition detected. "
            "Both transformer fault and overload "
            "indicators require immediate investigation."
        )

    elif overall == "HIGH":

        st.warning(
            "Elevated risk detected. "
            "Review the affected grid condition."
        )

    else:

        st.success(
            "Normal operating condition detected."
        )


# ============================================================
# RUN
# ============================================================

if live:

    placeholder = st.empty()

    while True:

        with placeholder.container():

            dashboard()

        time.sleep(
            refresh
        )

        st.rerun()

else:

    dashboard()
