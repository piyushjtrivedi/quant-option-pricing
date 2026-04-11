import sys
from pathlib import Path
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.dashboard import render_dashboard

st.set_page_config(layout="wide")
render_dashboard()