"""
Match-Adaptive Pitch Learning Visualization Component

Renders the Core Research Innovation:
1. Spatial 2D Pitch Restitution Heatmap e_z(X, Y) updated online from match deliveries
2. Empirical Prediction Error Convergence Curves (MAE / RMSE dropping over deliveries)
3. Bayesian credible interval analysis for post-bounce kinematics and wicket intersection
4. Delivery-by-delivery match learning log table
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cricket_vision.physics.pitch_learning import MatchAdaptivePitchModel


def create_restitution_heatmap_figure(pitch_model: MatchAdaptivePitchModel) -> go.Figure:
    """
    Renders 2D spatial contour heatmap of learned pitch restitution e_z(X, Y).
    """
    X_grid, Y_grid, Z_restitution = pitch_model.generate_pitch_restitution_heatmap()

    fig = go.Figure()

    # 2D Heatmap of Restitution
    fig.add_trace(go.Contour(
        x=X_grid[0, :],
        y=Y_grid[:, 0],
        z=Z_restitution,
        colorscale="Viridis",
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=10, color="white")
        ),
        colorbar=dict(
            title="Restitution (e_z)",
            titleside="right",
            titlefont=dict(color="#f8fafc", size=12),
            tickfont=dict(color="#cbd5e1", size=10)
        ),
        hoverinfo="x+y+z"
    ))

    # Scatter plot of all previously observed bounce spots in the match
    if pitch_model.history:
        obs_x = [o.bounce_x for o in pitch_model.history]
        obs_y = [o.bounce_y for o in pitch_model.history]
        obs_text = [
            f"Del #{o.delivery_idx}: e={o.measured_restitution:.2f}, Err={o.prediction_error_norm:.2f} m/s"
            for o in pitch_model.history
        ]

        fig.add_trace(go.Scatter(
            x=obs_x,
            y=obs_y,
            mode="markers+text",
            marker=dict(size=12, color="#ef4444", symbol="circle-dot", line=dict(color="white", width=2)),
            text=[f"#{o.delivery_idx}" for o in pitch_model.history],
            textposition="top center",
            hovertext=obs_text,
            hoverinfo="text",
            name="Learned Delivery Bounces"
        ))

    fig.update_layout(
        title=dict(
            text="Learned 2D Spatial Pitch Restitution Surface e_z(X, Y)",
            font=dict(color="#f8fafc", size=14)
        ),
        xaxis=dict(title="Lateral Pitch Corridor X (meters)", gridcolor="#334155", zerolinecolor="#64748b"),
        yaxis=dict(title="Distance Down Pitch Y (meters)", gridcolor="#334155"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e2430",
        margin=dict(l=40, r=40, b=40, t=50),
        height=360
    )
    return fig


def create_error_convergence_figure(pitch_model: MatchAdaptivePitchModel) -> go.Figure:
    """
    Renders delivery-by-delivery prediction error convergence curve.
    Demonstrates model improvement as more deliveries are observed in the match.
    """
    metrics = pitch_model.get_convergence_metrics()
    fig = go.Figure()

    if not metrics["delivery"]:
        return fig

    # Individual delivery residual errors
    fig.add_trace(go.Scatter(
        x=metrics["delivery"],
        y=metrics["error"],
        mode="markers+lines",
        name="Instantaneous Error (m/s)",
        marker=dict(size=8, color="#f59e0b"),
        line=dict(color="#f59e0b", width=2, dash="dot")
    ))

    # Cumulative MAE
    fig.add_trace(go.Scatter(
        x=metrics["delivery"],
        y=metrics["cumulative_mae"],
        mode="lines",
        name="Cumulative MAE (m/s)",
        line=dict(color="#10b981", width=3)
    ))

    fig.update_layout(
        title=dict(
            text="Online Model Learning Convergence Across Match Deliveries",
            font=dict(color="#f8fafc", size=14)
        ),
        xaxis=dict(title="Delivery Index in Match", dtick=1, gridcolor="#334155"),
        yaxis=dict(title="Post-Bounce Velocity Error (m/s)", gridcolor="#334155", zerolinecolor="#64748b"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e2430",
        legend=dict(font=dict(color="#cbd5e1"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, b=40, t=50),
        height=360
    )
    return fig


def render_match_learning_panel(pitch_model: MatchAdaptivePitchModel):
    """
    Renders full Streamlit panel for the Match-Adaptive Pitch Learning system.
    """
    st.markdown("### 🧠 Match-Adaptive Pitch Learning (Research Innovation)")
    st.markdown(
        "Standard ballistic trackers assume constant, static pitch restitution. In real cricket, turf hardness and seam grip vary across the pitch and degrade over time. "
        "Our **Physics-Informed Match-Adaptive Machine Learning Model (PIMA-Pitch)** learns from each delivery to progressively refine post-pitch bounce predictions."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_heat = create_restitution_heatmap_figure(pitch_model)
        st.plotly_chart(fig_heat, use_container_width=True)

    with c2:
        fig_conv = create_error_convergence_figure(pitch_model)
        st.plotly_chart(fig_conv, use_container_width=True)

    # Delivery log table
    if pitch_model.history:
        st.markdown("##### 📋 Sequential Match Delivery Learning Log")
        log_records = []
        for o in pitch_model.history:
            log_records.append({
                "Delivery #": o.delivery_idx,
                "Pitching Spot (X, Y)": f"({o.bounce_x:+.2f}m, {o.bounce_y:.1f}m)",
                "Incoming vz": f"{o.v_in[2]:.2f} m/s",
                "Actual Outgoing vz": f"{o.v_out_actual[2]:.2f} m/s",
                "Measured e_z": f"{o.measured_restitution:.3f}",
                "Seam Deflection": f"{o.seam_deflection_deg:+.2f}°",
                "Prediction Error": f"{o.prediction_error_norm:.3f} m/s",
                "Accuracy Status": "🎯 Calibrated" if o.prediction_error_norm < 0.8 else "🔄 Learning"
            })
        st.dataframe(pd.DataFrame(log_records), use_container_width=True)
    else:
        st.info("No deliveries logged yet in current match session. Run deliveries to train the online model!")

