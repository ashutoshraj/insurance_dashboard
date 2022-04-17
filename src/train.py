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
        linear_regression= LinearRegression()
        linear_regression.fit(X_train_scaled, y_train)
        y_pred = linear_regression.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        linear_regression_mae= metrics.mean_absolute_error(y_test, y_pred)
        linear_regression_mse = metrics.mean_squared_error(y_test, y_pred)
        linear_regression_rmse =np.sqrt(linear_regression_mse)
        st.dataframe(pd.DataFrame([linear_regression_mae, linear_regression_mse, linear_regression_rmse], index=['linear_regression_mae', 'linear_regression_mse', 'linear_regression_rmse'], columns=['Metrics']))

        scores = cross_val_score(linear_regression, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, linear_regression.predict(X_test_scaled))))

    elif select_algo == "Gradient Boosting":
        st.subheader("Training model using {}".format(select_algo))
        gradient_boosting = GradientBoostingRegressor()
        gradient_boosting.fit(X_train_scaled, y_train)

        y_pred = gradient_boosting.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        gradient_boosting_mae= metrics.mean_absolute_error(y_test, y_pred)
        gradient_boosting_mse = metrics.mean_squared_error(y_test, y_pred)
        gradient_boosting_rmse =np.sqrt(gradient_boosting_mse)
        st.dataframe(pd.DataFrame([gradient_boosting_mae, gradient_boosting_mse, gradient_boosting_rmse], index=['gradient_boosting_mae', 'gradient_boosting_mse', 'gradient_boosting_rmse'], columns=['Metrics']))

        scores = cross_val_score(gradient_boosting, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, gradient_boosting.predict(X_test_scaled))))

    elif select_algo == "Decision Tree":
        st.subheader("Training model using {}".format(select_algo))
        decision_tree =DecisionTreeRegressor()
        decision_tree.fit(X_train_scaled, y_train)

        y_pred = decision_tree.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        decision_tree_mae= metrics.mean_absolute_error(y_test, y_pred)
        decision_tree_mse = metrics.mean_squared_error(y_test, y_pred)
        decision_tree_rmse =np.sqrt(decision_tree_mse)
        st.dataframe(pd.DataFrame([decision_tree_mae, decision_tree_mse, decision_tree_rmse], index=['decision_tree_mae', 'decision_tree_mse', 'decision_tree_rmse'], columns=['Metrics']))

        scores = cross_val_score(decision_tree, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, decision_tree.predict(X_test_scaled))))

    elif select_algo == "Random Forest":
        st.subheader("Training model using {}".format(select_algo))
        random_forest =RandomForestRegressor()
        random_forest.fit(X_train_scaled, y_train)

        y_pred = random_forest.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        random_forest_mae= metrics.mean_absolute_error(y_test, y_pred)
        random_forest_mse = metrics.mean_squared_error(y_test, y_pred)
        random_forest_rmse =np.sqrt(random_forest_mse)
        st.dataframe(pd.DataFrame([random_forest_mae, random_forest_mse, random_forest_rmse], index=['random_forest_mae', 'random_forest_mse', 'random_forest_rmse'], columns=['Metrics']))

        scores = cross_val_score(random_forest, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, random_forest.predict(X_test_scaled))))

    else:
        st.subheader("Training model using {}".format(select_algo))
        xg_boost =XGBRegressor()
        xg_boost.fit(X_train_scaled, y_train)

        y_pred = xg_boost.predict(X_test_scaled)
        y_pred = pd.DataFrame(y_pred)
        xg_boost_mae= metrics.mean_absolute_error(y_test, y_pred)
        xg_boost_mse = metrics.mean_squared_error(y_test, y_pred)
        xg_boost_rmse =np.sqrt(xg_boost_mse)
        st.dataframe(pd.DataFrame([xg_boost_mae, xg_boost_mse, xg_boost_rmse], index=['xg_boost_mae', 'xg_boost_mse', 'xg_boost_rmse'], columns=['Metrics']))

        scores = cross_val_score(xg_boost, X_train_scaled, y_train, cv=k_fold)
        st.subheader("{} Cross Validation Score: {}".format(k_fold, np.sqrt(scores)))

        st.subheader("R2 Score: {}".format(r2_score(y_test, xg_boost.predict(X_test_scaled))))

        # with open('model/standardScaler.pkl', 'wb') as handle:
        #     pickle.dump(scaler, handle)
        std  = np.sqrt(scaler.var_)
        np.save('model/std.npy',std )
        np.save('model/mean.npy',scaler.mean_)

        with open('model/model.pkl', 'wb') as handle:
            pickle.dump(xg_boost, handle)

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
