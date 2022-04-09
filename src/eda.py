from unicodedata import category
import streamlit as st
from src.data import *
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json


def app():
    st.markdown("## Exploratory Data Analysis")
    st.write("\n")
    df = fetch_data()
    # print(df.head())
    config, chart = st.columns((4, 6))

    config_expander = config.expander('Configuration')

    hist_plot = config_expander.checkbox("Histogram Plot")

    interact_plot = config_expander.checkbox("Interaction Plot")

    corr_plot = config_expander.checkbox("Correlation Plot", )

    cat_var = ["sex", "region", "smoker", "bmi_cat", "children"]

    cont_var = [x for x in df.columns if x not in cat_var]
    # print("Continous Variable:", cont_var)
    # print("Categorical Variable:", cat_var)

    var1, var2, var3 = st.columns(3)

    x_axis = var1.selectbox("Select the first predictor variable", cont_var, index=0)
    y_axis = var2.selectbox("Select the second predictor vairable", cont_var, index=2)
    cat_var = var3.selectbox("Select the Categorical Variable", cat_var, index=2)

    if interact_plot:
        
        fig1 = px.scatter(df, x=x_axis, y=y_axis, color=cat_var)

        # Add figure title
        fig1.update_layout(
            title_text="{} v/s {}".format(y_axis, x_axis)
        )

        # Set x-axis title
        fig1.update_xaxes(title_text=x_axis)

        # Set y-axes titles
        fig1.update_yaxes(title_text="{}".format(y_axis))

        chart.plotly_chart(fig1)
    
    if hist_plot:
        fig2 = px.histogram(df, x=x_axis, color=cat_var)
        chart.plotly_chart(fig2)

    if corr_plot:
        df_corr = df.corr()
        x = list(df_corr.columns)
        y = list(df_corr.index)
        z = np.array(df_corr)

        fig3 = ff.create_annotated_heatmap(
            z,
            x = x,
            y = y ,
            annotation_text = np.around(z, decimals=2),
            hoverinfo='z',
            colorscale='Viridis'
            )

        chart.plotly_chart(fig3)
