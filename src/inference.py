import streamlit as st
import pickle
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def app():
    st.markdown("## Medical Cost Predictor Page!")
    st.write("\n")
    model, sc = None, None

    with open('model/model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with open('model/standardScaler.pkl', 'rb') as handle:
        sc = pickle.load(handle)

    # s = np.load('model/std.npy')
    # m = np.load('model/mean.npy')

    # print(s, m)

    col1, col2, col3 = st.columns(3)

    sex = col1.selectbox("Select Sex", ["male", "female"])
    if sex == "female":
        sex = 1
    else:
        sex = 2

    age = col2.slider('How old are you?', value=25, min_value=1, max_value=130, step=1)
    age = int(age)

    bmi = col2.slider('What is your body mass index? [Mass/Height]', value=25.0, min_value=22.0, max_value=54.0, step=0.01)
    bmi = float(bmi)

    region = col1.selectbox("Select your region", ["southwest", "northwest", "southeast", "northeast"])
    if region == "southwest":
        region = 1
    elif region == "southeast":
        region = 2
    elif region == "northwest":
        region = 3
    else:
        region = 4

    children = col1.selectbox("Number of Children", list(np.arange(10)))
    children = int(children)

    smoker = col2.selectbox("Do you smoke?", ["yes", "no"])
    if smoker == "yes":
        smoker = 1
    else:
        smoker = 2

    def bmi_cat(row):
        if row <= 25:
            return 3
        elif row <= 30 and row > 25:
            return 1
        else:
            return 2

    bmi_cat_ = bmi_cat(bmi)

    data = {"age":age, "sex":sex, "bmi":bmi,
                            "children":children, "smoker":smoker, "bmi_cat":bmi_cat_,
                            "region": region}

    df_test = pd.Series(data)

    X_test_scaled = sc.transform(df_test.values.reshape(1, -1))

    y_pred = model.predict(X_test_scaled)

    st.subheader("Medical Charges : {}".format(round(y_pred[0], 2)))

    