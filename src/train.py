from matplotlib import figure
import streamlit as st
from src.data import *
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

def main():
    st.markdown("## Welcome to model creation page!")
    st.write("\n")
    df = fetch_data()
    st.write("Brief overview of data!")
    st.dataframe(df.describe())

    df['sex'] = pd.factorize(df['sex'])[0] + 1
    df['region'] = pd.factorize(df['region'])[0] + 1
    df['smoker'] = pd.factorize(df['smoker'])[0] + 1
    df['bmi_cat'] = pd.factorize(df['bmi_cat'])[0] + 1

    print(df)

    X = df.drop('charges', axis = 1)
    y = df['charges']

    train_test_split_ratio = st.text_input("Enter the train test split ratio", value=0.3)
    train_test_split_ratio = float(train_test_split_ratio)

    k_fold = st.text_input("Enter the K Value for cross validation", value=5)
    k_fold = int(k_fold)

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=train_test_split_ratio, random_state=7406)

    scaler= StandardScaler()
    scaler.fit(X_train)
    X_train_scaled= scaler.transform(X_train)
    X_test_scaled= scaler.transform(X_test)

    algorithm_list = ["Linear Regression", "Gradient Boosting",  "XG Boost", "Decision Tree", "Random Forest"]

    select_algo = st.selectbox("Select the Algorithm", algorithm_list, index=2)

    if select_algo == "Linear Regression":
        st.subheader("Training model using {}".format(select_algo))
        linear_reg_model= LinearRegression()
        linear_reg_model.fit(X_train_scaled, y_train)
        y_pred = linear_reg_model.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        MAE_li_reg= metrics.mean_absolute_error(y_test, y_pred)
        MSE_li_reg = metrics.mean_squared_error(y_test, y_pred)
        RMSE_li_reg =np.sqrt(MSE_li_reg)
        st.dataframe(pd.DataFrame([MAE_li_reg, MSE_li_reg, RMSE_li_reg], index=['MAE_li_reg', 'MSE_li_reg', 'RMSE_li_reg'], columns=['Metrics']))

        scores = cross_val_score(linear_reg_model, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, linear_reg_model.predict(X_test_scaled))))

    elif select_algo == "Gradient Boosting":
        st.subheader("Training model using {}".format(select_algo))
        Gradient_model = GradientBoostingRegressor()
        Gradient_model.fit(X_train_scaled, y_train)

        y_pred = Gradient_model.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        MAE_gradient= metrics.mean_absolute_error(y_test, y_pred)
        MSE_gradient = metrics.mean_squared_error(y_test, y_pred)
        RMSE_gradient =np.sqrt(MSE_gradient)
        st.dataframe(pd.DataFrame([MAE_gradient, MSE_gradient, RMSE_gradient], index=['MAE_gradient', 'MSE_gradient', 'RMSE_gradient'], columns=['Metrics']))

        scores = cross_val_score(Gradient_model, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, Gradient_model.predict(X_test_scaled))))

    elif select_algo == "Decision Tree":
        st.subheader("Training model using {}".format(select_algo))
        tree_reg_model =DecisionTreeRegressor()
        tree_reg_model.fit(X_train_scaled, y_train)

        y_pred = tree_reg_model.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        MAE_tree_reg= metrics.mean_absolute_error(y_test, y_pred)
        MSE_tree_reg = metrics.mean_squared_error(y_test, y_pred)
        RMSE_tree_reg =np.sqrt(MSE_tree_reg)
        st.dataframe(pd.DataFrame([MAE_tree_reg, MSE_tree_reg, RMSE_tree_reg], index=['MAE_tree_reg', 'MSE_tree_reg', 'RMSE_tree_reg'], columns=['Metrics']))

        scores = cross_val_score(tree_reg_model, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, tree_reg_model.predict(X_test_scaled))))

    elif select_algo == "Random Forest":
        st.subheader("Training model using {}".format(select_algo))
        forest_reg_model =RandomForestRegressor()
        forest_reg_model.fit(X_train_scaled, y_train)

        y_pred = forest_reg_model.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        MAE_forest_reg= metrics.mean_absolute_error(y_test, y_pred)
        MSE_forest_reg = metrics.mean_squared_error(y_test, y_pred)
        RMSE_forest_reg =np.sqrt(MSE_forest_reg)
        st.dataframe(pd.DataFrame([MAE_forest_reg, MSE_forest_reg, RMSE_forest_reg], index=['MAE_forest_reg', 'MSE_forest_reg', 'RMSE_forest_reg'], columns=['Metrics']))

        scores = cross_val_score(forest_reg_model, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, forest_reg_model.predict(X_test_scaled))))

    else:
        st.subheader("Training model using {}".format(select_algo))
        XGB_model =XGBRegressor()
        XGB_model.fit(X_train_scaled, y_train)

        y_pred = XGB_model.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        MAE_XGB= metrics.mean_absolute_error(y_test, y_pred)
        MSE_XGB = metrics.mean_squared_error(y_test, y_pred)
        RMSE_XGB =np.sqrt(MSE_XGB)
        st.dataframe(pd.DataFrame([MAE_XGB, MSE_XGB, RMSE_XGB], index=['MAE_XGB', 'MSE_XGB', 'RMSE_XGB'], columns=['Metrics']))

        scores = cross_val_score(XGB_model, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, XGB_model.predict(X_test_scaled))))

        # with open('model/standardScaler.pkl', 'wb') as handle:
        #     pickle.dump(scaler, handle)
        std  = np.sqrt(scaler.var_)
        np.save('model/std.npy',std )
        np.save('model/mean.npy',scaler.mean_)

        with open('model/model.pkl', 'wb') as handle:
            pickle.dump(XGB_model, handle)

def app():
    def is_authenticated(password):
        return password == "admin"

    def generate_login_block():
        block1 = st.empty()
        block2 = st.empty()

        return block1, block2

    def clean_blocks(blocks):
        for block in blocks:
            block.empty()

    def login(blocks):
        blocks[0].markdown("""
                <style>
                    input {
                        -webkit-text-security: disc;
                    }
                </style>
            """, unsafe_allow_html=True)

        return blocks[1].text_input('Password')

    login_blocks = generate_login_block()
    password = login(login_blocks)

    if is_authenticated(password):
        clean_blocks(login_blocks)
        main()
    elif password:
        st.info("Please enter a valid password")
