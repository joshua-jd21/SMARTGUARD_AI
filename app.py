"""
SmartGuard AI — Cloud Demo Entrypoint
=====================================

This file is the lightweight, zero-cost cloud presentation layer of
SmartGuard AI. It is intentionally decoupled from the heavy local
compute stack (TensorFlow, MQTT, LangChain, Gemini, SerpAPI) so that it
can run on free hosting tiers such as Hugging Face Spaces and
Streamlit Community Cloud.

Architecture intent
-------------------
- Local edge layer  -> deep_learning/, agents/, iot/  (preserved, unchanged)
- Cloud demo layer  -> this file + demo/              (this branch)

The cloud layer reads pre-computed sample outputs from ``demo/`` and
renders a SaaS-style dashboard. It never imports TensorFlow, never opens
an MQTT connection, never calls an LLM API, and never loads a model.

If the user wants the full pipeline they must run it locally, see
``README.md``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants and demo asset paths
# ---------------------------------------------------------------------------

APP_VERSION = "cloud-demo-1.0.0"
BUILD_CHANNEL = "Hugging Face Spaces / Streamlit Cloud"
DEMO_DIR = Path(__file__).resolve().parent / "demo"

PREDICTIONS_PATH = DEMO_DIR / "sample_prediction.json"
REPORTS_PATH = DEMO_DIR / "sample_report.json"
VITALS_PATH = DEMO_DIR / "sample_vitals.csv"

RISK_COLOR = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartGuard AI — Cloud Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling — enterprise / SaaS look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      .sg-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b6f 100%);
        border: 1px solid #1f2a44;
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        color: #e5edff;
      }
      .sg-hero h1 { margin: 0; font-size: 1.9rem; }
      .sg-hero p  { margin: 0.35rem 0 0 0; color: #b8c4dd; }

      .sg-pill {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-right: 0.4rem;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
        color: #e5edff;
      }
      .sg-pill.green { background: rgba(34,197,94,0.18); border-color: rgba(34,197,94,0.5); color: #c6f6d5; }
      .sg-pill.blue  { background: rgba(59,130,246,0.18); border-color: rgba(59,130,246,0.5); color: #cfe1ff; }
      .sg-pill.amber { background: rgba(234,179,8,0.18);  border-color: rgba(234,179,8,0.5);  color: #fde68a; }

      .sg-card {
        background: #0e1525;
        border: 1px solid #1f2a44;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #e5edff;
      }
      .sg-card h4 { margin: 0 0 0.4rem 0; }
      .sg-card p  { margin: 0; color: #94a3b8; font-size: 0.9rem; }

      .sg-risk {
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.85rem;
      }

      .sg-footer {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 1.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached demo loaders (read once, reuse across reruns)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_predictions() -> dict[str, Any]:
    with PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_reports() -> dict[str, Any]:
    with REPORTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_vitals() -> pd.DataFrame:
    df = pd.read_csv(VITALS_PATH, parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------


def risk_pill(label: str) -> str:
    color = RISK_COLOR.get(label, "#64748b")
    return (
        f"<span class='sg-risk' style='background:{color}22;border:1px solid {color};"
        f"color:{color}'>{label}</span>"
    )


def kpi_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class='sg-card'>
          <h4>{value}</h4>
          <p>{label}{f' &middot; <span style="color:#475569">{help_text}</span>' if help_text else ''}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — system identity + deployment status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### SmartGuard AI")
    st.caption("Centralized insurance intelligence dashboard")

    st.markdown(
        f"""
        <div class='sg-card' style='margin-top:0.5rem'>
          <p><b style='color:#e5edff'>Deployment</b><br/>
          <span class='sg-pill green'>CLOUD&nbsp;ACTIVE</span>
          <span class='sg-pill blue'>EDGE&nbsp;INFERENCE</span><br/>
          <span style='color:#94a3b8'>Build {APP_VERSION}</span><br/>
          <span style='color:#475569;font-size:0.8rem'>{BUILD_CHANNEL}</span>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Hybrid architecture")
    st.markdown(
        "- Local edge layer runs **TensorFlow + LSTM/CNN + LangChain agents**.\n"
        "- Cloud layer streams **decision-support outputs** to this dashboard.\n"
        "- All recommendations are **assistive** and require human review."
    )

    st.markdown("#### Demo mode")
    st.success("Loaded from `demo/` — no live model, no API calls.")

    st.markdown("#### Status")
    st.markdown(
        "<span class='sg-pill green'>API&nbsp;OK</span>"
        "<span class='sg-pill green'>STREAM&nbsp;OK</span>"
        "<span class='sg-pill blue'>SOC2-ready</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class='sg-hero'>
      <h1>🩺 SmartGuard AI — Cloud Monitoring Layer</h1>
      <p>AI-assisted health-risk telemetry and underwriting intelligence, served from a hybrid edge + cloud architecture.</p>
      <div style='margin-top:0.7rem'>
        <span class='sg-pill green'>EDGE INFERENCE ENABLED</span>
        <span class='sg-pill blue'>CLOUD MONITORING ACTIVE</span>
        <span class='sg-pill amber'>HYBRID DEPLOYMENT</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------------------------------------------------------
# Load demo data once
# ---------------------------------------------------------------------------

try:
    predictions = load_predictions()
    reports = load_reports()
    vitals = load_vitals()
except FileNotFoundError as exc:
    st.error(f"Demo assets missing: {exc}. Make sure the `demo/` directory was deployed.")
    st.stop()


patient_index: dict[str, dict[str, Any]] = {
    p["patient"]["patient_id"]: p for p in predictions["patients"]
}
report_index: dict[str, dict[str, Any]] = {r["patient_id"]: r for r in reports["reports"]}


# ---------------------------------------------------------------------------
# Top KPI strip — fleet-wide rollups
# ---------------------------------------------------------------------------

risk_counts = (
    pd.Series([p["risk_prediction"]["risk_label"] for p in predictions["patients"]])
    .value_counts()
    .reindex(RISK_ORDER, fill_value=0)
)

avg_claim_prob = float(
    np.mean([r["insurance"]["claim_probability_pct"] for r in reports["reports"]])
)
avg_premium_adj = float(
    np.mean([r["insurance"]["premium_adjustment_pct"] for r in reports["reports"]])
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Monitored members", f"{len(predictions['patients']):,}", "active policies")
with k2:
    kpi_card("CRITICAL alerts", f"{int(risk_counts.get('CRITICAL', 0))}", "last 24h")
with k3:
    kpi_card("HIGH alerts", f"{int(risk_counts.get('HIGH', 0))}", "last 24h")
with k4:
    kpi_card("Avg claim probability", f"{avg_claim_prob:.1f}%", "portfolio")
with k5:
    kpi_card("Avg premium adj.", f"{avg_premium_adj:+.1f}%", "actuarial corridor")

st.write("")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_monitor, tab_reports, tab_arch, tab_workflow, tab_deploy = st.tabs(
    [
        "Overview",
        "Live Monitoring",
        "Insurance Reports",
        "Architecture",
        "AI Workflow",
        "Deployment",
    ]
)


# ---------- OVERVIEW ----------
with tab_overview:
    st.subheader("Portfolio risk distribution")

    left, right = st.columns([1.1, 1])

    with left:
        rc_df = risk_counts.reset_index()
        rc_df.columns = ["risk", "count"]
        fig = px.bar(
            rc_df,
            x="risk",
            y="count",
            color="risk",
            color_discrete_map=RISK_COLOR,
            text="count",
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            height=340,
            plot_bgcolor="#0e1525",
            paper_bgcolor="#0e1525",
            font_color="#e5edff",
            xaxis_title="",
            yaxis_title="members",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### What this view shows")
        st.markdown(
            "- Real-time roll-up of CNN risk classification across all monitored members.\n"
            "- Backed by **LSTM anomaly detection** + **CNN classifier** running on the edge.\n"
            "- Cloud layer ingests **decision-support summaries**, not raw biometric streams."
        )
        st.info(
            "All recommendations are decision-support. SmartGuard AI does not auto-approve "
            "or auto-reject any claim or policy."
        )

    st.divider()
    st.subheader("Active fleet snapshot")

    table_rows = []
    for r in reports["reports"]:
        table_rows.append(
            dict(
                patient_id=r["patient_id"],
                patient=r["patient_name"],
                age=r["patient_age"],
                risk=r["risk_label"],
                cnn_conf=f"{r['cnn_confidence']*100:.1f}%",
                anomaly=f"{r['anomaly_score']:.2f}",
                claim_prob=f"{r['insurance']['claim_probability_pct']:.1f}%",
                premium=f"{r['insurance']['premium_adjustment_pct']:+.1f}%",
                referral="YES" if r["insurance"]["clinical_referral"] else "NO",
            )
        )
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )


# ---------- LIVE MONITORING ----------
with tab_monitor:
    st.subheader("Telemetry stream")
    st.caption(
        "Simulated 2-hour vitals window per patient — sourced from `demo/sample_vitals.csv`. "
        "In production this stream is provided by the MQTT edge layer."
    )

    patient_ids = sorted(vitals["patient_id"].unique().tolist())
    default_idx = patient_ids.index("SG-1042") if "SG-1042" in patient_ids else 0

    cols = st.columns([2, 1, 1])
    with cols[0]:
        selected_pid = st.selectbox(
            "Patient",
            patient_ids,
            index=default_idx,
            format_func=lambda pid: f"{pid} — {patient_index[pid]['patient']['name']}",
        )
    with cols[1]:
        metric = st.selectbox(
            "Metric",
            [
                ("heart_rate_bpm", "Heart rate (bpm)"),
                ("spo2_pct", "SpO₂ (%)"),
                ("systolic_bp", "Systolic BP (mmHg)"),
                ("diastolic_bp", "Diastolic BP (mmHg)"),
                ("temperature_c", "Temperature (°C)"),
                ("respiration_rate", "Respiration rate"),
            ],
            format_func=lambda x: x[1],
        )
    with cols[2]:
        show_anomalies = st.toggle("Highlight anomalies", value=True)

    pdf = vitals[vitals["patient_id"] == selected_pid].sort_values("timestamp")
    risk_label = patient_index[selected_pid]["risk_prediction"]["risk_label"]
    color = RISK_COLOR.get(risk_label, "#3b82f6")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pdf["timestamp"],
            y=pdf[metric[0]],
            mode="lines",
            line=dict(color=color, width=2),
            name=metric[1],
        )
    )
    if show_anomalies:
        anomalies = pdf[pdf["anomaly_flag"] == 1]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["timestamp"],
                    y=anomalies[metric[0]],
                    mode="markers",
                    marker=dict(size=9, color="#ef4444", symbol="x"),
                    name="anomaly",
                )
            )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#0e1525",
        paper_bgcolor="#0e1525",
        font_color="#e5edff",
        xaxis_title="",
        yaxis_title=metric[1],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    snap = patient_index[selected_pid]["vitals_snapshot"]
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("HR (bpm)", snap["heart_rate_bpm"])
    s2.metric("SpO₂ (%)", snap["spo2_pct"])
    s3.metric("Systolic", snap["systolic_bp"])
    s4.metric("Diastolic", snap["diastolic_bp"])
    s5.metric("Temp (°C)", snap["temperature_c"])
    s6.metric("Resp. rate", snap["respiration_rate"])

    st.markdown(
        f"Current risk classification: {risk_pill(risk_label)}",
        unsafe_allow_html=True,
    )


# ---------- INSURANCE REPORTS ----------
with tab_reports:
    st.subheader("AI-generated decision-support reports")
    st.caption(
        "Generated locally by the multi-agent pipeline (Risk Reader → Web Researcher → Policy Advisor). "
        "Cached snapshots are displayed here — no live LLM call is made from the cloud layer."
    )

    pid_options = [r["patient_id"] for r in reports["reports"]]
    selected_report_pid = st.selectbox(
        "Select patient report",
        pid_options,
        format_func=lambda pid: f"{pid} — {report_index[pid]['patient_name']} "
                               f"({report_index[pid]['risk_label']})",
    )

    r = report_index[selected_report_pid]

    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown(
            f"### {r['patient_name']}  \n"
            f"Patient ID `{r['patient_id']}` · Age {r['patient_age']} · "
            f"Urgency {r['urgency_level']}/5"
        )
        st.markdown(risk_pill(r["risk_label"]), unsafe_allow_html=True)
    with head_r:
        st.metric("Claim probability", f"{r['insurance']['claim_probability_pct']:.1f}%")
        st.metric("Premium adjustment", f"{r['insurance']['premium_adjustment_pct']:+.1f}%")

    st.markdown("#### Medical summary")
    st.info(r["medical_summary"])

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Research synthesis")
        st.write(r["research"]["synthesised_finding"])
        st.caption(f"Evidence strength: **{r['research']['evidence_strength']}**")
        st.markdown("**Key guidelines**")
        for g in r["research"]["key_guidelines"]:
            st.markdown(f"- {g}")
    with cc2:
        st.markdown("#### Recommended clinical actions")
        for a in r["research"]["clinical_actions"]:
            st.markdown(f"- {a}")
        st.markdown("**Sources cited**")
        for s in r["research"]["sources_cited"]:
            st.markdown(f"- {s}")

    st.divider()
    st.markdown("#### Insurance recommendation")
    st.write(r["insurance"]["agent_finding"])

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Immediate actions**")
        for a in r["insurance"]["immediate_actions"]:
            st.markdown(f"- {a}")
    with ic2:
        st.markdown("**Coverage changes**")
        for c in r["insurance"]["coverage_changes"]:
            st.markdown(f"- {c}")

    st.markdown("**Wellness program**")
    st.success(r["insurance"]["wellness_program"])

    st.markdown("**Policy rationale**")
    st.write(r["insurance"]["policy_rationale"])

    with st.expander("Raw report text"):
        st.code(r["report_text"], language="text")

    st.download_button(
        "Download report JSON",
        data=json.dumps(r, indent=2),
        file_name=f"smartguard_report_{r['patient_id']}.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------- ARCHITECTURE ----------
with tab_arch:
    st.subheader("Hybrid edge + cloud architecture")

    st.markdown(
        """
**Local edge layer** (compute-heavy, runs on customer-owned infra)
- IoT sensors → MQTT broker (HiveMQ)
- LSTM autoencoder for anomaly detection
- CNN classifier for 4-class risk labelling (LOW / MEDIUM / HIGH / CRITICAL)
- LangChain multi-agent pipeline (Risk Reader → Web Researcher → Policy Advisor)
- Gemini + SerpAPI for grounded medical research

**Cloud presentation layer** (this app — lightweight, free-tier hostable)
- Monitoring dashboard, KPI rollups, charts
- AI-generated report visualization
- SaaS-style access layer for underwriters, case managers, ops
- No model loading, no MQTT loops, no API calls
"""
    )

    st.markdown("#### Data flow")
    st.code(
        """
IoT sensors
   │ MQTT
   ▼
Edge ingestion ──► Preprocessing ──► LSTM (anomaly) ──► CNN (risk class)
                                                     │
                                                     ▼
                                         LangChain multi-agent pipeline
                                                     │
                                                     ▼
                                      Decision-support report (JSON)
                                                     │
                                                     ▼   sync / replicate
                                            ┌────────────────┐
                                            │  CLOUD  LAYER  │  ← this app
                                            │  Streamlit UI  │
                                            └────────────────┘
        """.strip(),
        language="text",
    )

    st.markdown("#### Why separate layers?")
    st.markdown(
        "- **Compliance**: PHI stays on the edge; only de-identified decision-support payloads leave the boundary.\n"
        "- **Cost**: GPU/inference cost is bounded to edge nodes; cloud layer scales horizontally and cheaply.\n"
        "- **Reliability**: Dashboard remains available even when the edge pipeline is offline.\n"
        "- **Auditability**: Every report is versioned and traceable to the originating model lineage."
    )


# ---------- AI WORKFLOW ----------
with tab_workflow:
    st.subheader("Multi-agent AI workflow (local pipeline)")
    st.caption(
        "These agents execute on the edge. The cloud layer only renders their cached outputs."
    )

    steps = [
        ("1 · Risk Reader Agent",
         "Parses the deep-learning prediction payload, extracts CNN risk class, "
         "LSTM anomaly score, urgency tier, and produces a structured medical summary."),
        ("2 · Web Researcher Agent",
         "Performs grounded medical research via SerpAPI + Gemini. Synthesises "
         "guideline-backed clinical context, with explicit evidence-strength tag and sources."),
        ("3 · Policy Advisor Agent",
         "Translates clinical + research output into an underwriting decision-support "
         "report: claim probability, premium adjustment, coverage changes, wellness program."),
    ]

    for title, body in steps:
        st.markdown(
            f"""
            <div class='sg-card' style='margin-bottom:0.6rem'>
              <h4 style='color:#e5edff'>{title}</h4>
              <p style='color:#cbd5e1'>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "SmartGuard AI never auto-approves or auto-rejects insurance claims. "
        "Every recommendation is grounded in model outputs, retrieved evidence, "
        "and configurable business rules — and requires a human reviewer."
    )


# ---------- DEPLOYMENT ----------
with tab_deploy:
    st.subheader("Deployment status")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            "<div class='sg-card'><h4>Cloud layer</h4>"
            "<p><span class='sg-pill green'>HEALTHY</span> Streamlit / HF Spaces</p></div>",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            "<div class='sg-card'><h4>Edge layer</h4>"
            "<p><span class='sg-pill blue'>EXTERNAL</span> Runs on customer infra</p></div>",
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            "<div class='sg-card'><h4>Compliance</h4>"
            "<p><span class='sg-pill blue'>SOC2-ready</span> Audit trail enabled</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Build metadata")
    meta = {
        "app_version": APP_VERSION,
        "channel": BUILD_CHANNEL,
        "served_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "demo_assets": {
            "predictions": str(PREDICTIONS_PATH.relative_to(DEMO_DIR.parent)),
            "reports": str(REPORTS_PATH.relative_to(DEMO_DIR.parent)),
            "vitals": str(VITALS_PATH.relative_to(DEMO_DIR.parent)),
        },
        "live_compute": False,
        "external_api_calls": False,
        "tensorflow_loaded": False,
        "mqtt_loop_running": False,
    }
    st.json(meta)

    st.markdown("#### Hosting recipes")
    st.markdown(
        "- **Hugging Face Spaces** — set SDK to `streamlit`, point `app_file` to `app.py`. "
        "Already pre-configured via this repo's README front-matter.\n"
        "- **Streamlit Community Cloud** — connect this branch, main file = `app.py`. "
        "No secrets required.\n"
        "- **Local sanity check** — `pip install -r requirements.txt && streamlit run app.py`."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class='sg-footer'>
      SmartGuard AI · Cloud presentation layer · Decision-support only —
      not an automated medical or underwriting authority.
    </div>
    """,
    unsafe_allow_html=True,
)
