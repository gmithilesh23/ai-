"""
Telemetry & Pitch Map Analytics Components

Features:
- 2D Pitch Map showing Yorker / Full / Good Length / Short zones
- Speed profile & aerodynamic deceleration curve
- 3-axis kinematic velocity breakdown (vx, vy, vz)
- Quantitative delivery movement telemetry table
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.hawk_eye import DRSReviewResult


def create_pitch_map_figure(
    bounce_x: float,
    bounce_y: float,
    length_category: str,
    line_category: str
) -> go.Figure:
    """
    Renders top-down 2D pitch map with ICC delivery length and line zones.
    """
    pitch = CricketPitch()
    fig = go.Figure()

    hw = pitch.dims.pitch_width / 2.0
    stump_y = pitch.batsman_stumps_y
    pop_y = pitch.batsman_popping_crease_y

    # Length Zones (measured back from popping crease pop_y ~18.9m)
    zones = [
        ("Yorker", pop_y - 1.5, pop_y, "rgba(239, 68, 68, 0.18)"),         # Red
        ("Full", pop_y - 3.5, pop_y - 1.5, "rgba(249, 115, 22, 0.18)"),    # Orange
        ("Good Length", pop_y - 6.0, pop_y - 3.5, "rgba(16, 185, 129, 0.18)"), # Green
        ("Short of Length", pop_y - 8.5, pop_y - 6.0, "rgba(59, 130, 246, 0.18)"), # Blue
        ("Bouncer", pop_y - 13.0, pop_y - 8.5, "rgba(168, 85, 247, 0.18)")  # Purple
    ]

    for zname, y0, y1, col in zones:
        fig.add_shape(
            type="rect",
            x0=-hw, x1=hw,
            y0=y0, y1=y1,
            fillcolor=col,
            line=dict(color="#475569", width=1)
        )
        # Zone label
        fig.add_annotation(
            x=-hw + 0.35, y=(y0 + y1) / 2.0,
            text=zname, showarrow=False,
            font=dict(color="#cbd5e1", size=10)
        )

    # Crease Lines
    fig.add_shape(type="line", x0=-hw, x1=hw, y0=pop_y, y1=pop_y, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=-hw, x1=hw, y0=stump_y, y1=stump_y, line=dict(color="white", width=2))

    # Stumps Corridor (-half_w to +half_w)
    half_w = pitch.dims.wicket_width / 2.0
    fig.add_shape(
        type="rect",
        x0=-half_w, x1=half_w,
        y0=0.0, y1=stump_y,
        fillcolor="rgba(255, 255, 255, 0.04)",
        line=dict(color="#38bdf8", width=1, dash="dot")
    )

    # Plot Bounce Point
    fig.add_trace(go.Scatter(
        x=[bounce_x],
        y=[bounce_y],
        mode="markers+text",
        marker=dict(size=14, color="#ef4444", symbol="circle", line=dict(color="white", width=2)),
        text=[f"Bounce ({length_category})"],
        textposition="top right",
        name="Bounce Spot",
        hoverinfo="x+y+text"
    ))

    fig.update_layout(
        title=dict(text=f"Pitch Map: {length_category} | {line_category}", font=dict(color="#f8fafc", size=13)),
        xaxis=dict(title="Lateral Line (m)", range=[-hw - 0.2, hw + 0.2], gridcolor="#334155", zerolinecolor="#64748b"),
        yaxis=dict(title="Pitch Distance Y (m)", range=[pop_y - 12.0, stump_y + 0.5], gridcolor="#334155"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e2430",
        margin=dict(l=30, r=30, b=30, t=50),
        height=320,
        showlegend=False
    )
    return fig


def create_kinematics_figure(telemetry_df: pd.DataFrame) -> go.Figure:
    """
    Renders delivery speed profile and 3D velocity components over flight time.
    """
    fig = go.Figure()

    if telemetry_df.empty:
        return fig

    # Total Speed curve (km/h)
    fig.add_trace(go.Scatter(
        x=telemetry_df["time_s"],
        y=telemetry_df["speed_kmh"],
        mode="lines",
        name="Total Speed (km/h)",
        line=dict(color="#38bdf8", width=3)
    ))

    # Forward velocity (m/s)
    fig.add_trace(go.Scatter(
        x=telemetry_df["time_s"],
        y=telemetry_df["vy_mps"] * 3.6,
        mode="lines",
        name="Forward Speed vy (km/h)",
        line=dict(color="#34d399", width=2, dash="dash")
    ))

    # Vertical velocity (m/s)
    fig.add_trace(go.Scatter(
        x=telemetry_df["time_s"],
        y=telemetry_df["vz_mps"] * 3.6,
        mode="lines",
        name="Vertical Speed vz (km/h)",
        line=dict(color="#fbbf24", width=2)
    ))

    # Lateral swing velocity (m/s)
    fig.add_trace(go.Scatter(
        x=telemetry_df["time_s"],
        y=telemetry_df["vx_mps"] * 3.6,
        mode="lines",
        name="Lateral Speed vx (km/h)",
        line=dict(color="#f43f5e", width=2)
    ))

    fig.update_layout(
        title=dict(text="Kinematic Velocity Profile Across Delivery Epochs", font=dict(color="#f8fafc", size=13)),
        xaxis=dict(title="Flight Time (seconds)", gridcolor="#334155"),
        yaxis=dict(title="Speed (km/h)", gridcolor="#334155", zerolinecolor="#64748b"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e2430",
        legend=dict(font=dict(color="#cbd5e1"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, b=30, t=50),
        height=320
    )
    return fig


def render_movement_telemetry_table(drs_result: DRSReviewResult):
    """
    Renders structured quantitative delivery telemetry metrics.
    """
    st.markdown("### 📊 Delivery Kinematics & Aerodynamic Movement")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Release Speed", value=f"{drs_result.release_speed_kmh:.1f} km/h", delta=f"{drs_result.release_speed_kmh * 0.621371:.1f} mph")
    with c2:
        st.metric(label="Bounce Speed", value=f"{drs_result.bounce_speed_kmh:.1f} km/h", delta=f"-{drs_result.speed_drop_pct:.1f}% drop", delta_color="inverse")
    with c3:
        st.metric(label="Swing (In-Air)", value=f"{drs_result.pre_bounce_swing_cm:.1f} cm", delta=f"{drs_result.pre_bounce_swing_deg:.1f}° drift")
    with c4:
        st.metric(label="Seam/Spin (Off-Pitch)", value=f"{drs_result.post_bounce_seam_cm:.1f} cm", delta=f"{drs_result.post_bounce_seam_deg:.1f}° cut")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(label="Length Category", value=drs_result.length_category)
    with c6:
        st.metric(label="Line Category", value=drs_result.line_category)
    with c7:
        st.metric(label="Pad Impact Height", value=f"{drs_result.impact_height_m*100:.1f} cm")
    with c8:
        st.metric(label="Stumps Height", value=f"{drs_result.stumps_height_m*100:.1f} cm")
