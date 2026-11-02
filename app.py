"""
Root Application Shim for Cloud Deployment (Streamlit Community Cloud & Hugging Face).
Forwards execution to dashboard.app.
"""

import os
import sys
from pathlib import Path

# Ensure root directory is in Python path for absolute imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main dashboard app
import dashboard.app
