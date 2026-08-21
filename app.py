# ============================================================
# PowerGuard AI
# Live Smart Grid Monitoring Dashboard
# ============================================================

import time

import streamlit as st

from backend import analyze_live_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PowerGuard AI",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚡ PowerGuard AI")

st.subheader(
    "Real-Time Smart Grid Monitoring & Predictive Analytics"
)

st.write(
    "PowerGuard AI continuously monitors simulated grid "
    "parameters and uses machine learning to predict "
    "transformer faults and overload conditions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Monitoring Settings"
)

refresh_seconds = st.sidebar.slider(
    "Refresh interval (seconds)",
    min_value=1,
    max_value=10,
    value=3
)

auto_refresh = st.sidebar.checkbox(
    "Enable Live Monitoring",
    value=True
)


st.sidebar.info(
    "Current version uses simulated live sensor data. "
    "The same backend can later be connected to IoT, "
    "MQTT, SCADA or API data."
)


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:

    st.session_state.running = True


# ============================================================
# LIVE ANALYSIS FUNCTION
# ============================================================

def display_dashboard():

    try:

        result = analyze_live_data()

        data = result["data"]

        fault = result["fault"]

        overload = result["overload"]

        risk = result["risk"]

    except Exception as e:

        st.error(
            "Unable to analyze live data."
        )

        st.exception(e)

        return


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = result["timestamp"]

    st.caption(
        f"🟢 Live data received at: "
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    st.markdown("## 📡 Live Grid Measurements")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Voltage",
            f"{data['Voltage']:.2f} V"
        )

    with col2:

        st.metric(
            "Temperature",
            f"{data['Temperature']:.2f} °C"
        )

    with col3:

        st.metric(
            "Grid Supply",
            f"{data['Grid Supply (kW)']:.2f} kW"
        )

    with col4:

        st.metric(
            "Predicted Load",
            f"{data['Predicted Load (kW)']:.2f} kW"
        )


    # ========================================================
    # RENEWABLE ENERGY
    # ========================================================

    st.markdown("## ☀️ Renewable Generation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Solar Power",
            f"{data['Solar Power (kW)']:.2f} kW"
        )

    with col2:

        st.metric(
            "Wind Power",
            f"{data['Wind Power (kW)']:.2f} kW"
        )

    with col3:

        st.metric(
            "Total Renewable",
            f"{data['Total_Renewable_Power']:.2f} kW"
        )


    # ========================================================
    # SYSTEM CONDITION
    # ========================================================

    st.markdown("## 🤖 AI Predictions")

    col1, col2, col3 = st.columns(3)


    # Fault
    with col1:

        st.markdown("### 🔧 Transformer Fault")

        fault_status = (
            risk["fault_status"]
        )

        if fault_status == "HIGH RISK":

            st.error(
                "⚠️ HIGH RISK"
            )

        else:

            st.success(
                "✅ NORMAL"
            )

        if fault["probability"] is not None:

            st.metric(
                "Model Confidence",
                f"{fault['probability'] * 100:.1f}%"
            )


    # Overload
    with col2:

        st.markdown("### ⚡ Overload")

        overload_status = (
            risk["overload_status"]
        )

        if overload_status == "HIGH RISK":

            st.error(
                "⚠️ HIGH RISK"
            )

        else:

            st.success(
                "✅ NORMAL"
            )

        if overload["probability"] is not None:

            st.metric(
                "Model Confidence",
                f"{overload['probability'] * 100:.1f}%"
            )


    # Overall Risk
    with col3:

        st.markdown("### 🚨 Overall Risk")

        risk_level = risk["level"]

        if risk_level == "CRITICAL":

            st.error(
                "🔴 CRITICAL"
            )

        elif risk_level == "HIGH":

            st.warning(
                "🟠 HIGH"
            )

        else:

            st.success(
                "🟢 NORMAL"
            )


    # ========================================================
    # GRID DETAILS
    # ========================================================

    st.markdown(
        "## 📊 Detailed Live Parameters"
    )

    detail_col1, detail_col2 = st.columns(2)


    with detail_col1:

        st.write(
            "### Electrical Parameters"
        )

        st.write(
            f"**Voltage:** "
            f"{data['Voltage']:.2f} V"
        )

        st.write(
            f"**Voltage Deviation:** "
            f"{data['Voltage_Deviation']:.2f} V"
        )

        st.write(
            f"**Voltage Fluctuation:** "
            f"{data['Voltage Fluctuation (%)']:.2f}%"
        )

        st.write(
            f"**Power Imbalance:** "
            f"{data['Power Imbalance (kW)']:.2f} kW"
        )

        st.write(
            f"**Temperature:** "
            f"{data['Temperature']:.2f} °C"
        )


    with detail_col2:

        st.write(
            "### Operational Parameters"
        )

        st.write(
            f"**Hour:** "
            f"{data['Hour']}"
        )

        st.write(
            f"**Day of Week:** "
            f"{data['DayOfWeek']}"
        )

        st.write(
            f"**Weekend:** "
            f"{'Yes' if data['IsWeekend'] else 'No'}"
        )

        st.write(
            f"**Peak Hour:** "
            f"{'Yes' if data['IsPeakHour'] else 'No'}"
        )

        st.write(
            f"**Electricity Price:** "
            f"${data['Electricity Price (USD/kWh)']:.3f}/kWh"
        )


    # ========================================================
    # RENEWABLE RATIO
    # ========================================================

    st.markdown(
        "## 🌱 Renewable Contribution"
    )

    renewable_ratio = (
        data["Renewable_Ratio"] * 100
    )

    st.progress(
        min(
            max(
                int(renewable_ratio),
                0
            ),
            100
        )
    )

    st.write(
        f"Renewable contribution: "
        f"**{renewable_ratio:.2f}%**"
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        "## 💡 Recommended Action"
    )

    if risk["level"] == "CRITICAL":

        st.error(
            """
            **Immediate attention recommended.**

            Both transformer fault and overload indicators
            are currently showing elevated risk.

            Verify the live sensor readings and inspect the
            affected equipment according to operational
            procedures.
            """
        )

    elif risk["level"] == "HIGH":

        st.warning(
            """
            **Elevated risk detected.**

            At least one predictive model indicates a
            potentially abnormal operating condition.

            Review the live electrical parameters and
            investigate the affected condition.
            """
        )

    else:

        st.success(
            """
            **System operating within the current
            prediction range.**

            Continue normal monitoring.
            """
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "🔍 Technical Model Information"
    ):

        st.write(
            "### Transformer Fault Model"
        )

        st.write(
            "The fault pipeline uses the trained "
            "transformer fault model and its saved "
            "feature-selection artifacts."
        )

        st.write(
            "### Overload Model"
        )

        st.write(
            "The overload pipeline uses the trained "
            "overload model and its separate "
            "feature-selection artifacts."
        )

        st.write(
            "### Explainability"
        )

        st.write(
            "SHAP explainability can be added to the "
            "live inference output using the appropriate "
            "explainer for the trained model."
        )


# ============================================================
# MAIN LIVE LOOP
# ============================================================

if auto_refresh:

    placeholder = st.empty()

    while True:

        with placeholder.container():

            display_dashboard()

        time.sleep(
            refresh_seconds
        )

        st.rerun()

else:

    display_dashboard()

    st.info(
        "Live monitoring is paused. "
        "Enable it from the sidebar."
    )
