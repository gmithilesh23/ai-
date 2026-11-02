"""
Streamlit Web Dashboard for AI Cricket Ball Tracking & Hawk-Eye DRS Adjudication

Features:
- Video Ingestion & Frame-by-frame Tracking Scrubber
- Official Hawk-Eye DRS Review Decision Cards
- 3D Interactive Stadium & Trajectory Visualization
- 2D Pitch Map & Stumps Impact Radar
- Physics-Informed Match-Adaptive Pitch Learning (Research Innovation)

Launch with:
  streamlit run dashboard/app.py
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.homography import CameraModel
from cricket_vision.physics.pitch_learning import MatchAdaptivePitchModel
from cricket_vision.video.generator import CricketDeliveryGenerator
from cricket_vision.video.video_processor import VideoTrackingPipeline
from dashboard.components.hawk_eye_view import (
    create_3d_pitch_figure,
    create_stumps_impact_graphic,
    render_drs_broadcast_card
)
from dashboard.components.telemetry_view import (
    create_pitch_map_figure,
    create_kinematics_figure,
    render_movement_telemetry_table
)
from dashboard.components.match_adaptation_view import render_match_learning_panel

# Page configuration
st.set_page_config(
    page_title="AI Cricket Ball Tracking & Hawk-Eye DRS",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark sports broadcast styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .metric-card {
        background: #1e2430;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🏏 AI Cricket Ball-Tracking & Hawk-Eye DRS System")
    st.markdown(
        "**Portfolio-Grade Computer Vision & Physics-Informed Match-Adaptive Machine Learning Engine**"
    )

    # Initialize Match Pitch Model in Session State
    if "match_pitch_model" not in st.session_state:
        st.session_state["match_pitch_model"] = MatchAdaptivePitchModel()

    match_pitch_model: MatchAdaptivePitchModel = st.session_state["match_pitch_model"]

    # Sidebar Controls
    st.sidebar.header("⚙️ Delivery Selection & Ingestion")

    data_source = st.sidebar.radio(
        "Source Input Mode",
        ["Preset Realistic Match Delivery", "Upload Match Video (.mp4)"]
    )

    video_path = None
    delivery_name = "Custom Upload"

    if data_source == "Preset Realistic Match Delivery":
        delivery_choice = st.sidebar.selectbox(
            "Select Delivery Scenario",
            [
                "Inswinging Yorker (144 km/h - Hitting)",
                "Outswinger Good Length (138 km/h - Missing Off)",
                "Leg-Break Turning Spinner (88 km/h - Sharp Turn)",
                "Fast Bouncer (148 km/h - Over Stumps)",
                "Close LBW Candidate (141 km/h - Umpire's Call)",
                "Late Reverse-Swing Yorker (146 km/h - Hitting Middle)"
            ]
        )

        preset_mapping = {
            "Inswinging Yorker (144 km/h - Hitting)": "inswinging_yorker",
            "Outswinger Good Length (138 km/h - Missing Off)": "outswinger_good_length",
            "Leg-Break Turning Spinner (88 km/h - Sharp Turn)": "leg_break_spinner",
            "Fast Bouncer (148 km/h - Over Stumps)": "bouncer",
            "Close LBW Candidate (141 km/h - Umpire's Call)": "lbw_candidate",
            "Late Reverse-Swing Yorker (146 km/h - Hitting Middle)": "reverse_swing_yorker"
        }
        delivery_name = preset_mapping[delivery_choice]

        # Generate or check existing sample
        sample_dir = "data/samples"
        os.makedirs(sample_dir, exist_ok=True)
        video_path = os.path.join(sample_dir, f"{delivery_name}.mp4")

        if not os.path.exists(video_path):
            with st.spinner(f"Synthesizing delivery video '{delivery_name}'..."):
                gen = CricketDeliveryGenerator(resolution=(1280, 720), fps=60.0)
                gen.generate_delivery(delivery_type=delivery_name, output_path=video_path)

    else:
        uploaded_file = st.sidebar.file_uploader("Upload Delivery Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            delivery_name = uploaded_file.name

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Tracking Parameters")
    conf_thresh = st.sidebar.slider("Detector Confidence Threshold", 0.15, 0.85, 0.35, 0.05)
    weights_path = "cricket_vision/models/weights/tracknet_cricket.pth"

    run_pipeline = st.sidebar.button("🚀 Track Ball & Run Hawk-Eye DRS", use_container_width=True)

    # Multi-Delivery Over Simulation Button
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Research Experiment")
    st.sidebar.caption("Simulate 6 match deliveries in sequence to demonstrate online pitch learning:")
    simulate_over = st.sidebar.button("⚡ Simulate Full 6-Ball Over", use_container_width=True)
    if st.sidebar.button("🧹 Reset Match Pitch Memory", use_container_width=True):
        st.session_state["match_pitch_model"] = MatchAdaptivePitchModel()
        st.session_state["pipeline_results"] = None
        st.rerun()

    # Session State Caching for Processed Results
    if "pipeline_results" not in st.session_state:
        st.session_state["pipeline_results"] = None

    if simulate_over:
        delivery_types = [
            "outswinger_good_length",
            "lbw_candidate",
            "inswinging_yorker",
            "bouncer",
            "leg_break_spinner",
            "reverse_swing_yorker"
        ]
        sample_dir = "data/samples"
        os.makedirs(sample_dir, exist_ok=True)

        over_progress = st.progress(0.0)
        status_text = st.empty()

        for idx, dtype in enumerate(delivery_types):
            d_path = os.path.join(sample_dir, f"{dtype}.mp4")
            if not os.path.exists(d_path):
                gen = CricketDeliveryGenerator(resolution=(1280, 720), fps=60.0)
                gen.generate_delivery(delivery_type=dtype, output_path=d_path)

            status_text.text(f"Processing Match Delivery #{idx+1}/6: {dtype}...")
            pipeline = VideoTrackingPipeline(
                weights_path=weights_path if os.path.exists(weights_path) else None,
                confidence_thresh=conf_thresh,
                match_pitch_model=match_pitch_model
            )
            res = pipeline.process_video(video_path=d_path)
            st.session_state["pipeline_results"] = res
            over_progress.progress((idx + 1) / 6.0)

        status_text.empty()
        over_progress.empty()
        st.success(f"Successfully simulated full 6-ball match over! Model observed {len(match_pitch_model.history)} bounces.")

    elif run_pipeline and video_path:
        with st.spinner("Processing video frames through Deep Learning Detector & Kalman Trajectory Pipeline..."):
            progress_bar = st.progress(0.0)

            pipeline = VideoTrackingPipeline(
                weights_path=weights_path if os.path.exists(weights_path) else None,
                confidence_thresh=conf_thresh,
                match_pitch_model=match_pitch_model
            )

            out_annotated_dir = "data/generated"
            os.makedirs(out_annotated_dir, exist_ok=True)
            out_annotated_video = os.path.join(out_annotated_dir, f"annotated_{os.path.basename(video_path)}")

            results = pipeline.process_video(
                video_path=video_path,
                output_video_path=out_annotated_video,
                progress_callback=lambda p: progress_bar.progress(p)
            )
            progress_bar.progress(1.0)
            st.session_state["pipeline_results"] = results
            st.success(f"Processing Complete! ({results['total_frames']} frames tracked at {results['fps']:.1f} FPS)")

    results = st.session_state.get("pipeline_results")

    if results is not None:
        drs_result = results.get("drs_result")
        trajectory = results.get("trajectory", [])
        projected = results.get("projected_path", [])
        telemetry_df = results.get("telemetry_df", pd.DataFrame())

        # 1. Official Hawk-Eye DRS Verdict Banner
        if drs_result:
            render_drs_broadcast_card(drs_result)

        st.markdown("---")

        # 2. Main Visual Panel (Video Player Scrubber + 3D Interactive Pitch)
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎥 Video Tracking HUD",
            "🌐 3D Interactive Pitch Trajectory",
            "📈 Full Kinematics & Data",
            "🧠 Match-Adaptive Pitch Learning"
        ])

        with tab1:
            col_vid, col_info = st.columns([2, 1])
            with col_vid:
                annotated_frames = results.get("annotated_frames", [])
                if annotated_frames:
                    st.markdown("##### 🎬 Frame-by-Frame Tracking Scrubber")
                    frame_idx = st.slider("Scrub Delivery Frame", 0, len(annotated_frames) - 1, 0)
                    # Convert BGR to RGB for Streamlit image display
                    f_rgb = annotated_frames[frame_idx][..., ::-1]
                    st.image(f_rgb, caption=f"Frame #{frame_idx} / {len(annotated_frames)}", use_container_width=True)

            with col_info:
                st.markdown("##### 📡 Tracking Telemetry")
                if not telemetry_df.empty and frame_idx < len(telemetry_df):
                    row = telemetry_df.iloc[frame_idx]
                    st.metric("Instantaneous Speed", f"{row['speed_kmh']:.1f} km/h")
                    st.metric("3D Position [X, Y, Z]", f"[{row['x_m']:+.2f}, {row['y_m']:.2f}, {row['z_m']:.2f}] m")
                    st.metric("Trajectory Phase", "Post-Bounce" if row['is_bounced'] else "In-Flight")
                    st.metric("Detector Confidence", f"{row['confidence']*100:.1f}%")
                else:
                    st.info("Scrub frames to inspect telemetry.")

        with tab2:
            st.markdown("##### 🌐 Interactive 3D Stadium Trajectory (Rotate, Pan & Zoom)")
            bounce_coord = np.array(drs_result.pitching_coords + (0.036,)) if drs_result else None
            fig_3d = create_3d_pitch_figure(trajectory, projected, bounce_coord)
            st.plotly_chart(fig_3d, use_container_width=True)

            # Two Sub-Charts: Pitch Map and Stumps Impact
            c_left, c_right = st.columns(2)
            with c_left:
                if drs_result:
                    fig_map = create_pitch_map_figure(
                        drs_result.pitching_coords[0],
                        drs_result.pitching_coords[1],
                        drs_result.length_category,
                        drs_result.line_category
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
            with c_right:
                if drs_result:
                    fig_stumps = create_stumps_impact_graphic(drs_result)
                    st.plotly_chart(fig_stumps, use_container_width=True)

        with tab3:
            if drs_result:
                render_movement_telemetry_table(drs_result)

            st.markdown("##### 📉 Kinematic Velocity Breakdown")
            fig_kin = create_kinematics_figure(telemetry_df)
            st.plotly_chart(fig_kin, use_container_width=True)

            st.markdown("##### 📋 Frame-by-Frame Kinematic State Table")
            st.dataframe(telemetry_df, use_container_width=True, height=260)
            csv_data = telemetry_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Telemetry CSV",
                data=csv_data,
                file_name=f"{delivery_name}_telemetry.csv",
                mime="text/csv"
            )

        with tab4:
            render_match_learning_panel(match_pitch_model)

    else:
        st.info("👈 Select a delivery preset from the sidebar and click **'Track Ball & Run Hawk-Eye DRS'** to start analysis.")


if __name__ == "__main__":
    main()
