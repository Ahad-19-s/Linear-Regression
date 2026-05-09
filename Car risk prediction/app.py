import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Car Risk System",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = pickle.load(open("car_risk_model.pkl", "rb"))

# ---------------------------
# HEADER
# ---------------------------
st.title("🚗 AI Car Risk Prediction System")
st.markdown("### Real-world Machine Learning Safety Intelligence Dashboard")
st.markdown("---")

# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.header("🧑 Driver Profile")

age = st.sidebar.slider("Driver Age", 18, 80, 30)
speed = st.sidebar.slider("Current Speed (km/h)", 0, 250, 80)
car_age = st.sidebar.slider("Car Age (years)", 0, 30, 5)
experience = st.sidebar.slider("Driving Experience (years)", 0, 50, 10)

driving_style = st.sidebar.selectbox(
    "Driving Style",
    ["Normal", "Aggressive", "Careful"]
)

# ---------------------------
# STYLE ADJUSTMENT (LIGHT FEATURE ENGINEERING)
# ---------------------------
adjusted_speed = speed

if driving_style == "Aggressive":
    adjusted_speed *= 1.15
elif driving_style == "Careful":
    adjusted_speed *= 0.90

# ---------------------------
# MODEL INPUT
# ---------------------------
input_data = pd.DataFrame(
    [[adjusted_speed, car_age, experience]],
    columns=["speed", "car_age", "experience"]
)

# ---------------------------
# PREDICTION
# ---------------------------
prediction = model.predict(input_data)[0]

risk_score = np.clip(prediction, 0, 100)

# ---------------------------
# RISK LEVEL LOGIC
# ---------------------------
if risk_score < 30:
    risk_level = "🟢 Low Risk"
    alert_type = "success"
elif risk_score < 60:
    risk_level = "🟡 Medium Risk"
    alert_type = "warning"
else:
    risk_level = "🔴 High Risk"
    alert_type = "error"

# ---------------------------
# MAIN DASHBOARD
# ---------------------------

col1, col2, col3 = st.columns(3)

col1.metric("🚗 Speed", f"{adjusted_speed:.0f} km/h")
col2.metric("🚙 Car Age", f"{car_age} years")
col3.metric("🧠 Experience", f"{experience} years")

st.markdown("---")

# ---------------------------
# RESULT SECTION
# ---------------------------
st.subheader("📊 Risk Analysis Result")

left, right = st.columns([1, 1])

with left:
    st.metric("Risk Score", f"{risk_score:.2f}")

with right:
    if alert_type == "success":
        st.success(risk_level)
    elif alert_type == "warning":
        st.warning(risk_level)
    else:
        st.error(risk_level)

# ---------------------------
# PROGRESS BAR (RISK METER)
# ---------------------------
st.progress(int(risk_score))

# ---------------------------
# FEATURE VISUALIZATION
# ---------------------------
st.subheader("📈 Feature Impact Overview")

chart_data = pd.DataFrame({
    "Features": ["Speed", "Car Age", "Experience"],
    "Values": [adjusted_speed, car_age, experience]
})

st.area_chart(chart_data.set_index("Features"))

# ---------------------------
# INSIGHT ENGINE
# ---------------------------
st.subheader("🧠 AI Insight Engine")

if risk_score > 70:
    st.error("⚠️ High risk detected: Reduce speed and improve driving behavior.")
elif risk_score > 40:
    st.warning("⚠️ Moderate risk: Drive carefully and maintain safe speed.")
else:
    st.success("✅ Safe driving behavior detected.")

st.info(
    "💡 Insight: Risk increases significantly with higher speed and lower experience. "
    "Aggressive driving style amplifies accident probability."
)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("⚡ Built with Streamlit | AI Safety Intelligence System v1.0")