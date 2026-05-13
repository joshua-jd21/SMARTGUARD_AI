# ============================================================
# SmartGuard AI — Streamlit Dashboard
# File: dashboard/app.py
#
# FEATURES:
# ✅ Upload patient prediction JSON
# ✅ Run full multi-agent pipeline
# ✅ @st.cache_resource — pipeline instance survives reruns
# ✅ Session state — results persist across reruns
# ✅ Cooldown timer — prevents accidental repeated API calls
# ✅ Visual risk indicators
# ✅ Insurance recommendation display
# ✅ Evidence display
# ✅ Pipeline metrics
# ✅ Error handling
# ✅ Clean production UI
# ============================================================

import json
import sys
import time
from pathlib import Path

import streamlit as st

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# PIPELINE
# ============================================================

from agents.agent_pipeline import SmartGuardPipeline

# ============================================================
# CONSTANTS
# ============================================================

COOLDOWN_SECONDS = 30  # minimum seconds between pipeline runs

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartGuard AI",
    page_icon="🩺",
    layout="wide",
)

# ============================================================
# SESSION STATE INITIALISATION
# ============================================================

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

# ============================================================
# STYLES
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1rem;
}

.risk-low {
    color: #00C853;
    font-weight: bold;
}

.risk-medium {
    color: #FFD600;
    font-weight: bold;
}

.risk-high {
    color: #FF6D00;
    font-weight: bold;
}

.risk-critical {
    color: #D50000;
    font-weight: bold;
}

.metric-box {
    padding: 1rem;
    border-radius: 10px;
    background-color: #111111;
    border: 1px solid #333333;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("🩺 SmartGuard AI")
st.caption(
    "Multi-Agent Health Insurance Risk Assessment System"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("System")

st.sidebar.info("""
SmartGuard AI Pipeline

1. Risk Reader Agent
2. Web Researcher Agent
3. Policy Advisor Agent
""")

# ============================================================
# SAMPLE INPUT
# ============================================================

SAMPLE_INPUT = {
    "patient": {
        "name": "Rajan",
        "age": 45,
    },

    "risk_prediction": {
        "risk_label": "CRITICAL",
        "risk_class": 3,
        "confidence": 0.9875,
        "failsafe_triggered": False,
        "all_probabilities": {
            "LOW": 0.001,
            "MEDIUM": 0.004,
            "HIGH": 0.007,
            "CRITICAL": 0.988,
        }
    },

    "anomaly_detection": {
        "is_anomaly": True,
        "anomaly_score": 0.8732,
        "reconstruction_error": 0.09214,
        "threshold_used": 0.05120,
    }
}

# ============================================================
# HELPERS
# ============================================================

def get_risk_color(risk: str):
    mapping = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "orange",
        "CRITICAL": "red",
    }
    return mapping.get(risk, "white")


def render_risk_badge(risk: str):
    color = get_risk_color(risk)
    st.markdown(
        f"""<h2 style='color:{color};'>{risk}</h2>""",
        unsafe_allow_html=True
    )


# ============================================================
# CACHED PIPELINE FACTORY
# Single instance — survives Streamlit reruns
# ============================================================

@st.cache_resource
def get_pipeline():
    return SmartGuardPipeline()


# ============================================================
# RESULT RENDERER — extracted to avoid duplication
# ============================================================

def render_result(result):
    """Renders a PipelineResult to the Streamlit UI."""

    if not result.success:
        st.error("Pipeline execution failed.")
        st.code(result.error)
        return

    rc = result.risk_context
    rr = result.research_result
    ir = result.insurance_report

    # =======================================================
    # TOP METRICS
    # =======================================================

    st.success("Pipeline executed successfully.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patient", rc.patient_name)

    with col2:
        st.metric("Age", rc.patient_age)

    with col3:
        st.metric("Urgency", f"{rc.urgency_level}/5")

    with col4:
        st.metric("Execution Time", f"{result.execution_time_sec}s")

    st.divider()

    # =======================================================
    # RISK SECTION
    # =======================================================

    st.subheader("Risk Assessment")

    render_risk_badge(rc.risk_label)

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.metric("CNN Confidence", f"{rc.cnn_confidence:.2%}")
        st.metric("Anomaly Score", f"{rc.anomaly_score:.4f}")
        st.metric("Claim Probability", f"{ir.claim_probability}%")

    with risk_col2:
        st.metric("Premium Adjustment", f"+{ir.premium_adjustment}%")
        st.metric("Evidence Strength", rr.evidence_strength)
        st.metric(
            "Clinical Referral",
            "YES" if ir.clinical_referral else "NO"
        )

    st.divider()

    # =======================================================
    # MEDICAL SUMMARY
    # =======================================================

    st.subheader("Medical Summary")
    st.info(rc.medical_summary)

    # =======================================================
    # SYNTHESIS
    # =======================================================

    st.subheader("Research Synthesis")
    st.write(rr.synthesised_finding)

    # =======================================================
    # GUIDELINES + CLINICAL ACTIONS
    # =======================================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Guidelines")
        for item in rr.key_guidelines:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Clinical Actions")
        for item in rr.clinical_actions:
            st.markdown(f"- {item}")

    st.divider()

    # =======================================================
    # INSURANCE
    # =======================================================

    st.subheader("Insurance Recommendations")
    st.write(ir.agent_finding)

    st.subheader("Immediate Actions")
    for action in ir.immediate_actions:
        st.markdown(f"- {action}")

    st.subheader("Coverage Changes")
    for item in ir.coverage_changes:
        st.markdown(f"- {item}")

    st.subheader("Wellness Program")
    st.success(ir.wellness_program)

    st.subheader("Policy Rationale")
    st.write(ir.policy_rationale)

    st.divider()

    # =======================================================
    # SOURCES
    # =======================================================

    st.subheader("Sources")
    for src in rr.sources_cited:
        st.markdown(f"- {src}")

    # =======================================================
    # RAW REPORT
    # =======================================================

    with st.expander("Full Insurance Report"):
        st.code(ir.report_text, language="text")

    # =======================================================
    # EXPORT
    # =======================================================

    st.download_button(
        label="Download Full Result JSON",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="smartguard_report.json",
        mime="application/json",
        use_container_width=True,
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Patient Prediction Input")

uploaded_file = st.file_uploader(
    "Upload prediction JSON",
    type=["json"]
)

use_sample = st.checkbox(
    "Use sample prediction",
    value=True
)

prediction = None

if uploaded_file:
    try:
        prediction = json.load(uploaded_file)
        st.success("Prediction JSON loaded.")
    except Exception as e:
        st.error(f"Invalid JSON file: {e}")

elif use_sample:
    prediction = SAMPLE_INPUT

if prediction:
    with st.expander("View Prediction JSON"):
        st.json(prediction)

# ============================================================
# RUN BUTTON
# ============================================================

run_pipeline = st.button(
    "Run SmartGuard AI Pipeline",
    use_container_width=True
)

# ============================================================
# COOLDOWN CHECK
# ============================================================

can_run = True

if run_pipeline and st.session_state.last_run_time is not None:
    elapsed = time.time() - st.session_state.last_run_time
    if elapsed < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - elapsed)
        st.warning(
            f"Cooldown active — prevents accidental repeated API calls. "
            f"Try again in {remaining}s."
        )
        can_run = False

# ============================================================
# PIPELINE EXECUTION
# ============================================================

if run_pipeline and prediction and can_run:

    with st.spinner("Running SmartGuard AI Pipeline..."):

        try:
            pipeline = get_pipeline()
            result = pipeline.run(prediction)

            # Store in session state so result survives reruns
            st.session_state.pipeline_result = result
            st.session_state.last_run_time = time.time()

        except Exception as e:
            st.exception(e)

# ============================================================
# DISPLAY RESULT (from session state)
# ============================================================

if st.session_state.pipeline_result is not None:
    render_result(st.session_state.pipeline_result)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "SmartGuard AI — Multi-Agent Deep Learning "
    "Insurance Intelligence System"
)
