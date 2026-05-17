import os
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Risk Assessment Tool",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants & Data Loading
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"

_UI_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_UI_DIR)
_DATA_CSV = os.path.join(_PROJ_DIR, "data.csv")

@st.cache_data
def load_dataset():
    """Load data.csv once and cache it."""
    path = os.path.normpath(_DATA_CSV)
    df   = pd.read_csv(path)
    df   = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

df            = load_dataset()
FEATURE_NAMES = [c for c in df.columns if c != "diagnosis"]
_mins         = df[FEATURE_NAMES].min().values
_maxs         = df[FEATURE_NAMES].max().values
_means        = df[FEATURE_NAMES].mean().values

# Sample rows
_malignant_row = df[df["diagnosis"] == 1].iloc[0][FEATURE_NAMES].tolist()
_benign_row    = df[df["diagnosis"] == 0].iloc[0][FEATURE_NAMES].tolist()

# ─────────────────────────────────────────────
# Professional Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Typography and Spacing */
        .main-header {
            font-size: 2.2rem;
            font-weight: 300;
            color: #1E3A8A; /* Clinical Blue */
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        .sub-header {
            font-size: 1.1rem;
            font-weight: 400;
            color: #64748B; /* Slate Gray */
            margin-bottom: 2rem;
            margin-top: 5px;
        }
        
        /* Clean Alert Boxes */
        .alert-card {
            padding: 1.5rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .alert-malignant {
            background-color: #FEF2F2;
            border-left: 6px solid #DC2626;
            color: #991B1B;
        }
        .alert-benign {
            background-color: #F0FDF4;
            border-left: 6px solid #16A34A;
            color: #166534;
        }
        
        /* Subtle Disclaimer */
        .disclaimer-text {
            font-size: 0.85rem;
            color: #94A3B8;
            text-align: center;
            margin-top: 3rem;
            border-top: 1px solid #E2E8F0;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(
        """
        This assessment tool utilizes a **Logistic Regression** model trained on the 
        **Breast Cancer Wisconsin (Diagnostic)** dataset to evaluate tumor characteristics.

        **Specifications:**
        * **Algorithm:** Logistic Regression
        * **Feature Count:** 30 Clinical Metrics
        * **Accuracy:** ~97%
        * **Infrastructure:** scikit-learn + MLflow
        """
    )
    st.divider()
    
    with st.expander("View Feature Glossary"):
        for i, name in enumerate(FEATURE_NAMES, 1):
            st.markdown(f"`{i:02d}.` {name}")

    st.divider()
    st.caption("Group 2 — Machine Learning Project")

# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">Breast Cancer Risk Assessment Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Diagnostic prediction via logistic regression modeling</div>', unsafe_allow_html=True)

# Input mode selection (Placed outside the form so it updates defaults instantly)
input_col1, input_col2 = st.columns([1, 2])
with input_col1:
    input_mode = st.radio(
        "Data Entry Method",
        options=["Manual Entry", "Load Sample Data"],
        horizontal=True,
        label_visibility="collapsed"
    )

if input_mode == "Load Sample Data":
    with input_col2:
        sample_type = st.selectbox(
            "Select a clinical profile:",
            options=["Malignant Profile (Sample)", "Benign Profile (Sample)"],
            label_visibility="collapsed"
        )
    default_values = _malignant_row if "Malignant" in sample_type else _benign_row
    st.info(f"Loaded a known **{'Malignant' if 'Malignant' in sample_type else 'Benign'}** sample from the dataset. Review the parameters below and run the assessment.")
else:
    default_values = _means.tolist()

st.write("---")

# ─────────────────────────────────────────────
# Form-based Feature Input Grid
# ─────────────────────────────────────────────
st.markdown("#### Clinical Parameters")

# We wrap the inputs in a form. This prevents Streamlit from reloading the whole page 
# every time the user types a single number into one of the 30 inputs.
with st.form("clinical_data_form"):
    
    # Break 30 features into 3 tabs to reduce cognitive overload
    tab1, tab2, tab3 = st.tabs(["Parameters 1-10", "Parameters 11-20", "Parameters 21-30"])
    tabs = [tab1, tab2, tab3]
    
    feature_values = []
    
    # Distribute features across tabs
    for i, fname in enumerate(FEATURE_NAMES):
        tab_idx = i // 10
        current_tab = tabs[min(tab_idx, 2)]
        
        min_val = float(_mins[i])
        max_val = float(_maxs[i])
        default = float(default_values[i])
        default = max(min_val, min(max_val * 2, default))
        
        with current_tab:
            # Group into 2 columns within each tab for a cleaner layout
            if i % 2 == 0:
                colA, colB = st.columns(2)
                container = colA
            else:
                container = colB
                
            val = container.number_input(
                label=fname.replace("_", " ").title(), # Cleans up "radius_mean" to "Radius Mean"
                min_value=min_val,
                max_value=max_val * 2,
                value=default,
                format="%.5f",
                key=f"feat_{i}"
            )
            feature_values.append(val)
            
    st.write("") # Spacer
    submit_clicked = st.form_submit_button("Run Diagnostic Assessment", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# API Call & Results Processing
# ─────────────────────────────────────────────
if submit_clicked:
    with st.spinner("Analyzing clinical parameters..."):
        try:
            response = requests.post(API_URL, json={"features": feature_values}, timeout=10)

            if response.status_code == 200:
                result     = response.json()
                prediction = result.get("prediction")
                label      = result.get("label", "Unknown")
                prob       = result.get("probability", {})

                st.write("---")
                st.markdown("### Assessment Results")

                if prediction == 1:   # Malignant
                    st.markdown(
                        f'<div class="alert-card alert-malignant">Diagnosis Indication: {label.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    st.warning("The model indicates a high probability of malignancy. Immediate review by an oncologist is recommended.")
                else:                 # Benign
                    st.markdown(
                        f'<div class="alert-card alert-benign">Diagnosis Indication: {label.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    st.success("The model indicates a high probability of a benign tumor. Routine observation is advised.")

                # Clean Probability Metrics
                st.markdown("#### Confidence Metrics")
                
                # Using standard columns and progress bars instead of raw charts for a clinical feel
                m_prob = prob.get('malignant', 0)
                b_prob = prob.get('benign', 0)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Malignant Probability", f"{m_prob*100:.1f}%")
                    st.progress(m_prob)
                with metric_col2:
                    st.metric("Benign Probability", f"{b_prob*100:.1f}%")
                    st.progress(b_prob)

            else:
                error_msg = response.json().get("error", "Unknown error from API.")
                st.error(f"System Error ({response.status_code}): {error_msg}")

        except requests.exceptions.ConnectionError:
            st.error(
                "**Connection Refused:** Unable to communicate with the prediction engine.\n\n"
                "Ensure the backend service is running (`python api/app.py`)."
            )
        except requests.exceptions.Timeout:
            st.error("**Timeout:** The prediction engine took too long to respond.")
        except Exception as exc:
            st.error(f"**Unexpected Error:** {str(exc)}")

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer-text">'
    '<strong>Notice:</strong> This tool is a machine learning prototype designed for educational purposes. '
    'It does not constitute medical advice and should not be used as a primary diagnostic tool.'
    '</div>',
    unsafe_allow_html=True,
)