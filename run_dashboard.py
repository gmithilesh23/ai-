"""
Convenience launcher for the Streamlit Cricket Ball Tracking Dashboard.

Run with:
  python run_dashboard.py
"""

import sys
import os
import subprocess


def launch():
    app_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"[*] Launching AI Cricket Ball Tracking Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    launch()

