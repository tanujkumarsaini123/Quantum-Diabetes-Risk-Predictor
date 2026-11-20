
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import db
from qml_core import Preprocessor, QuantumClassifier, FEATURES

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Quantum Diabetes AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM UI
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global Font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main Background */
.stApp {
    background: #ffffff !important;
    color: #1f2937 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc !important;
}

[data-testid="stSidebar"] * {
    color: #1f2937;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: #f8fafc !important;
    padding: 18px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

/* Metric Values */
[data-testid="stMetricValue"] {
    color: #2563eb !important;
}

/* Headings */
h1, h2, h3 {
    color: #1e293b !important;
}

/* Paragraphs and Labels */
p, label {
    color: #334155;
}

/* Input Fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    border-radius: 8px;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    border-radius: 9px;
    font-weight: 600;
}

/* Prediction Form */
div[data-testid="stForm"] {
    background: #ffffff !important;
    padding: 22px;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
}

/* Dividers */
hr {
    border-color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EXISTING QUANTUM MODEL
# --------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "qml_weights.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


@st.cache_resource
def load_models():
    db.init_db()

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            "qml_weights.json is missing from the model folder."
        )

    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            "metrics.json is missing from the model folder."
        )

    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        weights = json.load(f)

    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    pre = Preprocessor(**weights["pre"])

    quantum = QuantumClassifier(
        weights["w"],
        weights["a"],
        weights["b"],
        weights["layers"],
        weights.get("s")
    )

    dataset = db.load_dataset()

    train_data, _ = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset.Outcome,
        random_state=42
    )

    classical = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=500
        )
    )

    classical.fit(
        train_data[FEATURES].values,
        train_data.Outcome.values
    )

    reference = np.median(
        pre.transform(dataset),
        axis=0
    )

    with db.conn() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM training_runs"
        ).fetchone()[0]

        if count == 0:
            for name, result in metrics["results"].items():
                db.save_run(
                    name,
                    result,
                    f'{metrics["epochs"]} epochs'
                    if name.startswith("Quantum") else ""
                )

    return pre, quantum, classical, metrics, reference


# --------------------------------------------------
# LOAD APPLICATION
# --------------------------------------------------

try:
    PRE, QC, CLASSICAL, METRICS, REF = load_models()
except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def risk_level(probability):
    if probability < 0.35:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    return "High"


def make_prediction(values):
    row = pd.DataFrame([{
        "Glucose": values["glucose"],
        "BMI": values["bmi"],
        "Age": values["age"],
        "BloodPressure": values["blood_pressure"],
        "DiabetesPedigreeFunction": values["family_index"],
        "Pregnancies": values["pregnancies"]
    }])

    transformed = PRE.transform(row)[0]

    variants = np.vstack([
        transformed
    ] + [
        np.where(
            np.arange(len(transformed)) == i,
            REF[i],
            transformed
        )
        for i in range(len(transformed))
    ])

    probabilities = QC.proba(variants)

    quantum_probability = float(probabilities[0])

    contributions = []

    labels = {
        "Glucose": "Blood glucose",
        "BMI": "Body mass index",
        "Age": "Age",
        "BloodPressure": "Blood pressure",
        "DiabetesPedigreeFunction": "Family history",
        "Pregnancies": "Pregnancies"
    }

    for i, feature in enumerate(FEATURES):
        contributions.append({
            "Feature": labels[feature],
            "Impact": float(
                quantum_probability - probabilities[i + 1]
            )
        })

    contributions.sort(
        key=lambda item: abs(item["Impact"]),
        reverse=True
    )

    classical_probability = float(
        CLASSICAL.predict_proba(row[FEATURES].values)[0, 1]
    )

    return (
        quantum_probability,
        classical_probability,
        contributions
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🧬 Quantum Diabetes AI")
st.sidebar.caption("Intelligent Healthcare Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Risk Prediction",
        "Patient History",
        "Model Analytics",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Quantum ML Engine | PennyLane")
st.sidebar.caption("Classical ML | Scikit-learn")
st.sidebar.caption("Storage | SQLite")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🧬 Quantum Intelligence Dashboard")
    st.write(
        "Early Diabetes Risk Prediction using "
        "Quantum Machine Learning."
    )

    history = db.list_assessments()

    total = len(history)
    high = sum(
        item["risk_level"] == "High"
        for item in history
    )

    average = (
        np.mean([item["quantum_prob"] for item in history])
        if total else 0
    )

    dataset = db.load_dataset()

    a, b, c, d = st.columns(4)

    a.metric("Patient Assessments", total)
    b.metric("High Risk Cases", high)
    c.metric("Average Risk", f"{average * 100:.1f}%")
    d.metric("Training Records", len(dataset))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Assessment Risk Distribution")

        if history:
            distribution = pd.DataFrame(history)

            fig = px.pie(
                distribution,
                names="risk_level",
                hole=0.55,
                title="Patient Risk Categories"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No patient assessments available yet.")

    with right:
        st.subheader("Dataset Outcome Distribution")

        outcome = dataset["Outcome"].value_counts().reset_index()
        outcome.columns = ["Outcome", "Count"]

        fig = px.bar(
            outcome,
            x="Outcome",
            y="Count",
            color="Outcome",
            title="Diabetes Dataset"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Patient Assessments")

    if history:
        st.dataframe(
            pd.DataFrame(history).head(10),
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# RISK PREDICTION
# --------------------------------------------------

elif page == "Risk Prediction":

    st.title("Patient Risk Assessment")
    st.write("Enter patient information to generate predictions.")

    with st.form("prediction_form"):

        name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                18, 90, 35
            )

            glucose = st.number_input(
                "Glucose Level",
                40, 300, 120
            )

        with col2:
            pregnancies = st.number_input(
                "Pregnancies",
                0, 17, 0
            )

            bmi = st.number_input(
                "BMI",
                12.0, 65.0, 25.0
            )

        with col3:
            blood_pressure = st.number_input(
                "Blood Pressure",
                30, 140, 75
            )

            family_index = st.number_input(
                "Diabetes Pedigree Function",
                0.05, 2.5, 0.5
            )

        submitted = st.form_submit_button(
            "🧬 Analyze Diabetes Risk",
            use_container_width=True
        )

    if submitted:

        values = {
            "age": age,
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "bmi": bmi,
            "family_index": family_index
        }

        with st.spinner("Analyzing patient data..."):

            quantum, classical, contributions = make_prediction(values)

            level = risk_level(quantum)

            record = {
                "patient_name": name.strip() or "Anonymous",
                **values,
                "quantum_prob": quantum,
                "classical_prob": classical,
                "risk_level": level,
                "notes": ""
            }

            assessment_id = db.add_assessment(record)

        st.divider()
        st.subheader("Prediction Results")

        x, y, z = st.columns(3)

        x.metric(
            "Quantum Model Risk",
            f"{quantum * 100:.2f}%"
        )

        y.metric(
            "Classical Model Risk",
            f"{classical * 100:.2f}%"
        )

        z.metric(
            "Risk Classification",
            level
        )

        if level == "High":
            st.error("High predicted risk. Consider professional medical evaluation.")
        elif level == "Moderate":
            st.warning("Moderate predicted risk. Consider discussing screening with a healthcare professional.")
        else:
            st.success("Low predicted risk according to this model.")

        st.caption(
            f"Assessment ID: {assessment_id}"
        )

        st.subheader("Quantum Model Feature Contributions")

        contribution_df = pd.DataFrame(contributions)

        fig = px.bar(
            contribution_df,
            x="Impact",
            y="Feature",
            orientation="h",
            title="Feature Impact on Model Prediction"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "This is an experimental machine learning prediction, "
            "not a medical diagnosis. A low score does not rule out diabetes."
        )


# --------------------------------------------------
# PATIENT HISTORY
# --------------------------------------------------

elif page == "Patient History":

    st.title("Patient Assessment History")

    history = db.list_assessments()

    if history:

        dataframe = pd.DataFrame(history)

        st.metric(
            "Total Assessments",
            len(dataframe)
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        csv_data = dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Assessment CSV",
            csv_data,
            file_name="diabetes_assessments.csv",
            mime="text/csv"
        )

        st.divider()

        selected_id = st.selectbox(
            "Select Assessment ID to Delete",
            dataframe["id"].tolist()
        )

        if st.button("Delete Selected Assessment"):
            db.delete_assessment(int(selected_id))
            st.success("Assessment deleted.")
            st.rerun()

    else:
        st.info("No patient records found.")


# --------------------------------------------------
# MODEL ANALYTICS
# --------------------------------------------------

elif page == "Model Analytics":

    st.title("Machine Learning Model Analytics")

    st.subheader("Model Performance")

    results = METRICS["results"]

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "Model"

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")

    accuracy_df = metrics_df.reset_index()

    if "accuracy" in accuracy_df.columns:

        fig = px.bar(
            accuracy_df,
            x="Model",
            y="accuracy",
            color="Model",
            title="Model Accuracy Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Training Configuration")

    st.write(
        f"Quantum Training Epochs: {METRICS.get('epochs', 'N/A')}"
    )

    st.write(
        f"Quantum Circuit Layers: {QC.n_layers}"
    )

    st.write(
        f"Number of Qubits: {len(FEATURES)}"
    )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

elif page == "About Project":

    st.title("About Quantum Diabetes AI")

    st.write("""
    Quantum Diabetes AI is an experimental healthcare
    analytics application combining classical machine
    learning and quantum machine learning.

    **Core Technologies**

    - Python
    - PennyLane
    - Variational Quantum Classifier
    - Scikit-learn
    - Logistic Regression
    - Streamlit
    - SQLite
    - Plotly

    **Project Objectives**

    - Explore quantum machine learning for diabetes risk prediction.
    - Compare quantum and classical model predictions.
    - Analyze patient health parameters.
    - Store and visualize patient assessments.
    - Explore feature contributions.

    The model is intended for educational and research
    purposes and is not a substitute for clinical testing
    or professional medical advice.
    """)
