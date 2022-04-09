import streamlit as st
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    # initial_sidebar_state="expanded"
)

# Custom imports 
from multipage import MultiPage
from PIL import  Image
import numpy as np
from src import model, eda, inference # import your pages here