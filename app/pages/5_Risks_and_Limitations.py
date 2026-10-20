"""Streamlit Module 5: Traffic Flow Physics, Boundaries & System Guardrails.

Rigorous documentation of macroscopic traffic flow theory (q = k * v), forensic target horizon verification,
zero-lookahead leakage prevention, and real-world Traffic Management Center (TMC) deployment roadmaps.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path so modules resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.styles import get_shared_css
from app.utils.plots import plot_fundamental_diagram_curve

st.set_page_config(
    page_title="Traffic Physics & System Guardrails | Smart Traffic TOC",
    page_icon="⚠️",
    layout="wide"
)

# Apply shared design tokens and enterprise CSS
st.markdown(get_shared_css(), unsafe_allow_html=True)

st.markdown("""
<div class="toc-header-bar">
    <div class="toc-title">⚠️ Module 5: Traffic Flow Physics, Boundaries & System Guardrails</div>
    <div class="toc-subtitle">Macroscopic traffic flow theory, forensic horizon verification, leakage prevention, and real-world deployment roadmaps</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <strong>Engineering Ethics & Scientific Integrity:</strong> An authentic AI/ML internship portfolio must honestly state physical limitations,
    theoretical boundaries, and empirical discoveries rather than overpromising capabilities.
</div>
""", unsafe_allow_html=True)

# Section 1: Traffic Flow Physics
st.subheader("1. Traffic Flow Theory: Why Traffic Volume ≠ Physical Congestion")

col_phys_txt, col_phys_plot = st.columns([1.1, 1.3])

with col_phys_txt:
    st.markdown(r"""
    A frequent mistake in student data science projects is assuming that **high traffic volume directly equals traffic congestion**.
    In macroscopic transportation engineering (Lighthill-Whitham-Richards and Greenshields models), traffic flow is governed by:

    $$q = k \cdot v$$

    Where:
    - **$q$**: Traffic Flow / Volume (vehicles / hour)
    - **$k$**: Traffic Density (vehicles / kilometer)
    - **$v$**: Space Mean Speed (kilometers / hour)

    **The Flow-Density Parabolic Duality:**
    Because the curve is parabolic, **the exact same flow rate $q$ occurs in two completely opposite physical traffic states**:
    1. **Free-Flow Regime (Green Curve):** High vehicle speed ($v \approx 100 \text{ km/h}$), low density ($k < k_{crit}$). The road is moving freely at the posted speed limit.
    2. **Congested Regime (Red Curve):** Severe stop-and-go gridlock ($v \approx 10 \text{ km/h}$), high density ($k > k_{crit}$). Vehicles are bumper-to-bumper.

    > **Scientific Conclusion:** Without direct measurements of vehicle **spot speed** ($v$) or detector **occupancy percentage**, volume alone cannot mathematically differentiate free-flow traffic from gridlock. Our system strictly forecasts *volume and relative flow*, avoiding unsubstantiated congestion claims.
    """)

with col_phys_plot:
    fig_fund = plot_fundamental_diagram_curve()
    st.plotly_chart(fig_fund, use_container_width=True)

st.markdown("---")

# Section 2: Forensic Target Horizon Verification
st.subheader("2. Forensic Target Horizon Verification (Dataset A Discrepancy)")

col_forensic1, col_forensic2 = st.columns([1.2, 1.0])

with col_forensic1:
    st.markdown(r"""
    During onboarding and audit in Phase 1 and 2, our empirical mathematical audit discovered an important discrepancy in Dataset A (UCI 734):
    - **Nominal Author Claim:** The source dataset description states a *"15-minute-ahead traffic flow prediction"*.
    - **Empirical Synchronization Reality:** Each input matrix $X$ contains 10 historical lag steps ($[t-9, \dots, t]$) at 15-minute intervals. However, numerical synchronization revealed that target $Y[:, t]$ corresponds to **offset $+2$ steps** relative to the latest input observation (`lag[9]`).
    - **Physical Reality:** Offset $+2$ steps at 15 minutes per step equates to **30 minutes ahead in real time** ($2 \times 15 = 30\text{ min}$).

    **Our Defensible Engineering Stance:**
    Rather than silently altering the target or mislabeling the horizon to match the author's docx text, our pipeline preserves the official benchmark target for literature comparability while **explicitly documenting the verified 30-minute horizon across all specifications, code, and dashboards.**
    """)

with col_forensic2:
    horizon_df = pd.DataFrame({
        "Specification Dimension": [
            "Nominal Author Description",
            "Verified Mathematical Target",
            "Input Feature Window",
            "Target Vector Offset",
            "Literature Comparability"
        ],
        "Engineering Value": [
            "15 minutes ahead",
            "30 minutes ahead (t+2 steps)",
            "10 lags (150 minutes of history)",
            "+2 steps beyond lag[9]",
            "100% Preserved (Unmodified Y)"
        ]
    })
    st.dataframe(horizon_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 3: Lookahead Leakage Prevention
st.subheader("3. Zero-Lookahead Bias & Data Leakage Prevention")
st.markdown("""
Time-series forecasting models in published literature frequently report artificially inflated accuracy due to subtle data leakage bugs.
Our pipeline enforces four mathematical barriers against lookahead leakage:
""")

leak_col1, leak_col2, leak_col3 = st.columns(3)

with leak_col1:
    st.markdown("""
    #### **🚫 Causal Cleaning**
    - **No Lookahead Interpolation:** Zero bidirectional interpolation (`.interpolate()`). Missing values and sensor dropouts are filled strictly using causal forward-fill (`.ffill()`).
    - **Outlier Guardrails:** Validated rainfall outliers capped to the 6-year non-anomalous maximum ($55.63\text{ mm}$), preventing training distortion.
    """)

with leak_col2:
    st.markdown("""
    #### **🛡️ Strict Chronological Splits**
    - **No Random K-Fold Shuffling:** Shuffling time series data leaks future information into past training samples.
    - **Causal Split Boundaries:** Train, validation, and official test sets are partitioned by strict chronological timestamps, preserving the true arrow of time.
    """)

with leak_col3:
    st.markdown(r"""
    #### **🔍 Telemetry Gap Filter**
    - **Non-Consecutive Target Dropping:** 2,589 target pairs where sensor outages caused $\Delta t > 1\text{ hour}$ were filtered out.
    - **No Cross-Gap Bridging:** Prevents a 10:00 AM reading from predicting a 4:00 PM observation across a 6-hour power outage.
    """)

st.markdown("---")

# Section 4: Production Deployment Roadmap
st.subheader("4. Production Traffic Management Center (TMC) Roadmap")
st.markdown("""
To transition this historical benchmark system into a real-time production deployment at a municipal agency (e.g. Virginia DOT (VDOT) or MnDOT RTMC), the following engineering infrastructure would be required:
""")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.markdown("""
    **1. Streaming Telemetry:**
    - Deploy Apache Kafka or AWS Kinesis to ingest high-frequency 30-second loop detector feeds.
    - Windowing aggregations computed in real time using Apache Flink.
    """)

with r2:
    st.markdown("""
    **2. Live Weather API:**
    - Replace historical weather observations with point-in-time NOAA / OpenWeatherMap numerical forecasts for upcoming hours.
    """)

with r3:
    st.markdown("""
    **3. Graph Deep Learning:**
    - Scale from tabular GBDTs to Spatio-Temporal Graph Neural Networks (STGCN, DCRNN) to model city-wide 5,000+ sensor freeway topologies.
    """)

with r4:
    st.markdown("""
    **4. Failover & MLOps:**
    - Automated fallback to persistence baseline whenever detector communications drop out.
    - Continuous drift detection (Evidently AI) for seasonal recalibration.
    """)
