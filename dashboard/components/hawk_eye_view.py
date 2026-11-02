"""
Hawk-Eye 3D Interactive Pitch & DRS Review Visualization Component

Uses Plotly 3D to render:
- Cricket pitch surface, creases, and 3D wickets
- Ball flight path (Pre-bounce, Post-bounce, Projected)
- Front-on 2D Stumps Impact Graphic
- Broadcast-grade DRS LBW Verdict Cards
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.hawk_eye import DRSReviewResult
from cricket_vision.tracking.tracker import TrackedBallState


def create_3d_pitch_figure(
    trajectory: List[TrackedBallState],
    projected_pts: Optional[List[np.ndarray]] = None,
    bounce_pt: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Constructs an interactive 3D stadium pitch and ball flight visualization.
    """
    pitch = CricketPitch()
    fig = go.Figure()

    # 1. Pitch Surface (Clay Strip)
    # X: [-1.52, 1.52], Y: [-1.0, 21.12], Z: 0.0
    hw = pitch.dims.pitch_width / 2.0
    stump_y = pitch.batsman_stumps_y

    fig.add_trace(go.Mesh3d(
        x=[-hw, hw, hw, -hw],
        y=[-1.0, -1.0, stump_y + 1.0, stump_y + 1.0],
        z=[0.0, 0.0, 0.0, 0.0],
        color="#d2b48c",  # Tan clay
        opacity=0.9,
        name="Pitch Turf",
        hoverinfo="skip"
    ))

    # 2. Crease Lines
    creases = pitch.get_crease_lines()
    for name, line in creases.items():
        fig.add_trace(go.Scatter3d(
            x=[line[0, 0], line[1, 0]],
            y=[line[0, 1], line[1, 1]],
            z=[0.002, 0.002],  # Slightly above pitch to prevent z-fighting
            mode="lines",
            line=dict(color="white", width=4),
            name=name.replace("_", " ").title(),
            showlegend=False,
            hoverinfo="skip"
        ))

    # 3. Batsman Wickets (3 Stumps and Bails)
    wbox = pitch.get_wicket_bounding_box(is_batsman_end=True)
    stump_xs = [wbox["off_stump_x"], wbox["middle_stump_x"], wbox["leg_stump_x"]]
    stump_names = ["Off Stump", "Middle Stump", "Leg Stump"]

    for sx, sname in zip(stump_xs, stump_names):
        fig.add_trace(go.Scatter3d(
            x=[sx, sx],
            y=[stump_y, stump_y],
            z=[0.0, pitch.dims.stump_height],
            mode="lines",
            line=dict(color="#f0f0f5", width=7),
            name=sname,
            showlegend=False,
            hoverinfo="name"
        ))

    # Bails
    fig.add_trace(go.Scatter3d(
        x=[wbox["off_stump_x"], wbox["leg_stump_x"]],
        y=[stump_y, stump_y],
        z=[pitch.dims.total_wicket_height, pitch.dims.total_wicket_height],
        mode="lines",
        line=dict(color="#ffffff", width=5),
        name="Bails",
        showlegend=False,
        hoverinfo="skip"
    ))

    # 4. Trajectory Segments
    if trajectory:
        # Pre-bounce
        pre_pts = [s for s in trajectory if not s.is_bounced]
        if pre_pts:
            px = [s.pos_3d[0] for s in pre_pts]
            py = [s.pos_3d[1] for s in pre_pts]
            pz = [s.pos_3d[2] for s in pre_pts]
            speeds = [f"{s.speed_kmh} km/h" for s in pre_pts]

            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=pz,
                mode="lines+markers",
                line=dict(color="#00e5ff", width=6),
                marker=dict(size=3, color="#ffffff"),
                name="Flight (Pre-Bounce)",
                text=speeds,
                hoverinfo="text+x+y+z"
            ))

        # Post-bounce
        post_pts = [s for s in trajectory if s.is_bounced]
        if post_pts:
            qx = [s.pos_3d[0] for s in post_pts]
            qy = [s.pos_3d[1] for s in post_pts]
            qz = [s.pos_3d[2] for s in post_pts]
            speeds_post = [f"{s.speed_kmh} km/h" for s in post_pts]

            fig.add_trace(go.Scatter3d(
                x=qx, y=qy, z=qz,
                mode="lines+markers",
                line=dict(color="#ff6d00", width=6),
                marker=dict(size=3, color="#ff9100"),
                name="Flight (Post-Bounce)",
                text=speeds_post,
                hoverinfo="text+x+y+z"
            ))

    # 5. Bounce Bullseye Spot
    if bounce_pt is not None:
        fig.add_trace(go.Scatter3d(
            x=[bounce_pt[0]],
            y=[bounce_pt[1]],
            z=[0.01],
            mode="markers+text",
            marker=dict(size=9, color="#ffd600", symbol="diamond"),
            text=["BOUNCE"],
            textposition="top center",
            name="Pitch Bounce Spot",
            hoverinfo="name+x+y"
        ))

    # 6. Projected Virtual Path into Stumps
    if projected_pts:
        fx = [p[0] for p in projected_pts]
        fy = [p[1] for p in projected_pts]
        fz = [p[2] for p in projected_pts]

        fig.add_trace(go.Scatter3d(
            x=fx, y=fy, z=fz,
            mode="lines",
            line=dict(color="#76ff03", width=6, dash="dash"),
            name="Virtual Path (To Stumps)",
            hoverinfo="name+x+y+z"
        ))

    # Layout styling (Dark broadcast theme)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Lateral X (m)", range=[-2.2, 2.2], backgroundcolor="#1e2430", gridcolor="#334155"),
            yaxis=dict(title="Pitch Length Y (m)", range=[-1.0, 22.0], backgroundcolor="#1e2430", gridcolor="#334155"),
            zaxis=dict(title="Height Z (m)", range=[0.0, 3.0], backgroundcolor="#1e2430", gridcolor="#334155"),
            aspectmode="manual",
            aspectratio=dict(x=0.6, y=2.2, z=0.6),
            camera=dict(
                eye=dict(x=1.2, y=-1.5, z=0.8),
                center=dict(x=0, y=0.5, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor="#0f172a",
        legend=dict(
            font=dict(color="#f8fafc"),
            bgcolor="rgba(15, 23, 42, 0.7)",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02
        )
    )

    return fig


def create_stumps_impact_graphic(drs_result: DRSReviewResult) -> go.Figure:
    """
    Renders front-on 2D diagram of the 3 stumps, bails, and ball arrival position.
    """
    fig = go.Figure()

    pitch = CricketPitch()
    wbox = pitch.get_wicket_bounding_box(is_batsman_end=True)
    half_w = pitch.dims.wicket_width / 2.0
    stump_h = pitch.dims.stump_height
    tot_h = pitch.dims.total_wicket_height
    ball_r = pitch.dims.ball_radius

    # Draw 3 Stumps as vertical bars
    stump_xs = [wbox["off_stump_x"], wbox["middle_stump_x"], wbox["leg_stump_x"]]
    stump_labels = ["Off", "Middle", "Leg"]

    for sx, label in zip(stump_xs, stump_labels):
        fig.add_shape(
            type="rect",
            x0=sx - 0.018, x1=sx + 0.018,
            y0=0.0, y1=stump_h,
            fillcolor="#e2e8f0",
            line=dict(color="#94a3b8", width=1.5)
        )
        # Label underneath
        fig.add_annotation(
            x=sx, y=-0.04, text=label, showarrow=False,
            font=dict(color="#94a3b8", size=11)
        )

    # Draw Bails
    fig.add_shape(
        type="rect",
        x0=wbox["off_stump_x"] - 0.02, x1=wbox["leg_stump_x"] + 0.02,
        y0=stump_h, y1=tot_h,
        fillcolor="#cbd5e1",
        line=dict(color="#64748b", width=1.5)
    )

    # Stumps Zone Bounding Box (In-line corridor)
    fig.add_shape(
        type="rect",
        x0=-half_w, x1=half_w,
        y0=0.0, y1=tot_h,
        line=dict(color="#3b82f6", width=1.5, dash="dot"),
        fillcolor="rgba(59, 130, 246, 0.05)"
    )

    # Draw Ball at arrival point
    bx, by, bz = drs_result.stumps_arrival_coords
    is_hitting = drs_result.wickets_verdict in ["HITTING", "UMPIRE'S CALL"]
    ball_color = "#ef4444" if is_hitting else "#10b981"

    fig.add_shape(
        type="circle",
        x0=bx - ball_r, x1=bx + ball_r,
        y0=bz - ball_r, y1=bz + ball_r,
        fillcolor=ball_color,
        line=dict(color="#ffffff", width=2)
    )

    # Target Crosshair
    fig.add_annotation(
        x=bx, y=bz,
        text=f"{bz*100:.1f} cm",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f59e0b",
        ax=40, ay=-30,
        font=dict(color="#f8fafc", size=12)
    )

    fig.update_layout(
        title=dict(text="Stump-Line Arrival Visualizer (Front View)", font=dict(color="#f8fafc", size=14)),
        xaxis=dict(title="Lateral (m)", range=[-0.35, 0.35], zeroline=True, zerolinecolor="#475569", gridcolor="#334155"),
        yaxis=dict(title="Height (m)", range=[-0.08, 0.95], zeroline=True, zerolinecolor="#475569", gridcolor="#334155"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e2430",
        margin=dict(l=30, r=30, b=30, t=50),
        height=320
    )
    return fig


def render_drs_broadcast_card(drs_result: DRSReviewResult):
    """
    Renders the official Hawk-Eye DRS Review Decision Broadcast Card in Streamlit.
    """
    col1, col2, col3, col4 = st.columns(4)

    def get_badge(verdict: str):
        if verdict in ["IN LINE", "HITTING"]:
            return "background-color: #059669; color: white;" # Green
        elif verdict in ["UMPIRE'S CALL"]:
            return "background-color: #d97706; color: white;" # Amber
        else:
            return "background-color: #dc2626; color: white;" # Red

    with col1:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 12px; background: #1e2430; text-align: center;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: 600;">1. PITCHING</div>
            <div style="font-size: 16px; font-weight: 800; margin-top: 6px; padding: 4px; border-radius: 4px; {get_badge(drs_result.pitching_verdict)}">
                {drs_result.pitching_verdict}
            </div>
            <div style="color: #64748b; font-size: 11px; margin-top: 6px;">
                X: {drs_result.pitching_coords[0]:+.2f}m | Y: {drs_result.pitching_coords[1]:.1f}m
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 12px; background: #1e2430; text-align: center;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: 600;">2. IMPACT</div>
            <div style="font-size: 16px; font-weight: 800; margin-top: 6px; padding: 4px; border-radius: 4px; {get_badge(drs_result.impact_verdict)}">
                {drs_result.impact_verdict}
            </div>
            <div style="color: #64748b; font-size: 11px; margin-top: 6px;">
                Height: {drs_result.impact_height_m*100:.1f} cm
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 12px; background: #1e2430; text-align: center;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: 600;">3. WICKETS</div>
            <div style="font-size: 16px; font-weight: 800; margin-top: 6px; padding: 4px; border-radius: 4px; {get_badge(drs_result.wickets_verdict)}">
                {drs_result.wickets_verdict}
            </div>
            <div style="color: #64748b; font-size: 11px; margin-top: 6px;">
                {drs_result.stump_targeted}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        decision_color = "#dc2626" if "OUT" in drs_result.overall_lbw_decision and "NOT" not in drs_result.overall_lbw_decision else "#059669"
        st.markdown(f"""
        <div style="border: 2px solid {decision_color}; border-radius: 8px; padding: 12px; background: #1e2430; text-align: center;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: 600;">DRS DECISION</div>
            <div style="font-size: 18px; font-weight: 900; margin-top: 6px; color: {decision_color};">
                {drs_result.overall_lbw_decision}
            </div>
            <div style="color: #cbd5e1; font-size: 11px; margin-top: 6px;">
                Arrival: {drs_result.stumps_height_m*100:.1f} cm
            </div>
        </div>
        """, unsafe_allow_html=True)
