import streamlit as st
st.set_page_config(
    page_title="Medical Insurance Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    # initial_sidebar_state="expanded"
)

# Custom imports 
from multipage import MultiPage
from PIL import  Image
import numpy as np
from src import train, eda, inference # import your pages here


# Create an instance of the app 
app = MultiPage()

display = Image.open('data/Logo.png')
display = np.array(display)

# Title of the main page
col1, col2 = st.columns(2)
col1.title("Medical Insurance Cost Analytics")
col1.write(
    """Health insurance costs are on the rise every year. Our objective is to determine the predictors which 
affect health insurance cost of an individual the most, by studying the relationship between the health insurance 
cost and various predictors as well as by investigating the correlation among the predictors."""
)  # description and instructions

col2.image(display, width = 300)

# Add all your applications (pages) here
app.add_page("Data Visualization", eda.app)
app.add_page("Model Creation", train.app)
app.add_page("Cost Prediction", inference.app)

# The main app
app.run()