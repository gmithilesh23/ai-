"""
Command Line Pipeline Runner for AI Cricket Ball Tracking & Hawk-Eye DRS

Usage Examples:
  # Process a single delivery video:
  python run_pipeline.py --delivery inswinging_yorker --render-video

  # Run full match-adaptation over experiment:
  python run_pipeline.py --simulate-over
"""

import os
import argparse
import time
import numpy as np
import pandas as pd

from cricket_vision.physics.pitch_learning import MatchAdaptivePitchModel
from cricket_vision.video.generator import CricketDeliveryGenerator
from cricket_vision.video.video_processor import VideoTrackingPipeline


def main():
    parser = argparse.ArgumentParser(description="AI Cricket Ball Tracking & Hawk-Eye DRS Pipeline")
    parser.add_argument("--delivery", type=str, default="inswinging_yorker",
                        choices=["inswinging_yorker", "outswinger_good_length", "leg_break_spinner",
                                 "bouncer", "lbw_candidate", "reverse_swing_yorker"],
                        help="Preset delivery type to generate and track")
    parser.add_argument("--video-input", type=str, default=None, help="Custom input video path")
    parser.add_argument("--render-video", action="store_true", help="Render high-definition annotated video")
    parser.add_argument("--simulate-over", action="store_true", help="Run 6-delivery match adaptation experiment")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    args = parser.parse_args()

    sample_dir = "data/samples"
    output_dir = "data/generated"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    weights_path = "cricket_vision/models/weights/tracknet_cricket.pth"

    if args.simulate_over:
        print("=" * 75)
        print("  SIMULATING 6-BALL MATCH OVER: ONLINE PITCH LEARNING BENCHMARK")
        print("=" * 75)

        over_deliveries = [
            "outswinger_good_length",
            "lbw_candidate",
            "inswinging_yorker",
            "bouncer",
            "leg_break_spinner",
            "reverse_swing_yorker"
        ]

        pitch_model = MatchAdaptivePitchModel()
        pipeline = VideoTrackingPipeline(
            weights_path=weights_path if os.path.exists(weights_path) else None,
            confidence_thresh=args.confidence,
            match_pitch_model=pitch_model
        )

        for ball_num, dtype in enumerate(over_deliveries, 1):
            vid_path = os.path.join(sample_dir, f"{dtype}.mp4")
            if not os.path.exists(vid_path):
                print(f"[*] Synthesizing video for delivery #{ball_num}: {dtype}...")
                gen = CricketDeliveryGenerator(resolution=(1280, 720), fps=60.0)
                gen.generate_delivery(delivery_type=dtype, output_path=vid_path)

            print(f"\n--- [Delivery {ball_num}/6: {dtype.upper()}] ---")
            t0 = time.time()
            res = pipeline.process_video(vid_path)
            el = time.time() - t0

            drs = res["drs_result"]
            obs = res.get("bounce_obs")

            if drs:
                print(f"  Release Speed:  {drs.release_speed_kmh:.1f} km/h | Bounce Speed: {drs.bounce_speed_kmh:.1f} km/h")
                print(f"  Pitching Zone:  {drs.pitching_verdict} at (X={drs.pitching_coords[0]:+.2f}m, Y={drs.pitching_coords[1]:.1f}m)")
                print(f"  Impact Zone:    {drs.impact_verdict} (Height: {drs.impact_height_m*100:.1f} cm)")
                print(f"  Wickets:        {drs.wickets_verdict} -> {drs.stump_targeted}")
                print(f"  DRS Verdict:    >>> {drs.overall_lbw_decision} <<<")

            if obs:
                print(f"  [Online Learner] Prior Error: {obs.prediction_error_norm:.3f} m/s | Measured e_z: {obs.measured_restitution:.3f}")

        print("\n" + "=" * 75)
        print("  MATCH LEARNING CONVERGENCE SUMMARY")
        print("=" * 75)
        conv = pitch_model.get_convergence_metrics()
        for d, err, mae in zip(conv["delivery"], conv["error"], conv["cumulative_mae"]):
            bar = "█" * int(min(30, max(1, 30 - err * 8)))
            print(f"  Delivery #{d}: Instantaneous Error = {err:.3f} m/s | Cumulative MAE = {mae:.3f} m/s  {bar}")
        print("=" * 75)

    else:
        video_path = args.video_input
        if video_path is None:
            video_path = os.path.join(sample_dir, f"{args.delivery}.mp4")
            if not os.path.exists(video_path):
                print(f"[*] Generating delivery video '{args.delivery}'...")
                gen = CricketDeliveryGenerator(resolution=(1280, 720), fps=60.0)
                gen.generate_delivery(delivery_type=args.delivery, output_path=video_path)

        out_video = os.path.join(output_dir, f"annotated_{os.path.basename(video_path)}") if args.render_video else None

        print("=" * 75)
        print(f"  PROCESSING DELIVERY: {video_path}")
        print("=" * 75)

        pipeline = VideoTrackingPipeline(
            weights_path=weights_path if os.path.exists(weights_path) else None,
            confidence_thresh=args.confidence
        )
        res = pipeline.process_video(video_path, output_video_path=out_video)

        drs = res["drs_result"]
        if drs:
            print(f"[OK] Release Speed:      {drs.release_speed_kmh:.1f} km/h")
            print(f"[OK] Bounce Speed:       {drs.bounce_speed_kmh:.1f} km/h ({drs.speed_drop_pct:.1f}% deceleration)")
            print(f"[OK] Pitching:           {drs.pitching_verdict} (X={drs.pitching_coords[0]:+.2f}m, Y={drs.pitching_coords[1]:.1f}m)")
            print(f"[OK] Impact:             {drs.impact_verdict} ({drs.line_category}, Height: {drs.impact_height_m*100:.1f} cm)")
            print(f"[OK] Wickets:            {drs.wickets_verdict} -> {drs.stump_targeted}")
            print(f"[OK] Final DRS Decision: {drs.overall_lbw_decision}")
            print(f"[OK] Lateral Swing:      {drs.pre_bounce_swing_cm:.1f} cm ({drs.pre_bounce_swing_deg:.1f} deg)")
            print(f"[OK] Seam/Cut Off Pitch: {drs.post_bounce_seam_cm:.1f} cm ({drs.post_bounce_seam_deg:.1f} deg)")

        if out_video:
            print(f"[OK] Rendered Annotated Video: {out_video}")

        # Save telemetry CSV
        telemetry_csv = os.path.join(output_dir, f"{args.delivery}_telemetry.csv")
        res["telemetry_df"].to_csv(telemetry_csv, index=False)
        print(f"[OK] Exported Kinematic Telemetry: {telemetry_csv}")
        print("=" * 75)


if __name__ == "__main__":
    main()
