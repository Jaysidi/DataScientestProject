import streamlit as st

st.set_page_config(
    page_title="Conception d'un modèle de Machine Learning des données",
    layout="wide",

    menu_items={})

import numpy as np
import statsmodels.api as sm
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, StandardScaler
from xgboost import XGBRegressor

from matplotlib import pyplot as plt

from Libraries.Data import vgsales_metacritic_scores_df
from Libraries.Models import run_models, models_tried

st.image("Images/ML.png")
tab1, tab2, tab3 = st.tabs(["Recherche d'un modèle", "XGBoost Regressor", "Analyse de sentiments"], )
with tab1:
    st.header("Recherche d'un modèle")
    st.write("Nous avons entrainé plusieurs modèles afin de déterminer celui qui donne les meilleurs résultats")

    drop_columns = ['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    target_columns = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

    # target_col = st.selectbox('Choix de la cible', target_columns)
    target_col = 'Global_Sales'
    drop_columns.remove(target_col)

    st.sidebar.write("***Options***")
    test_size = st.sidebar.select_slider("Test size en %", np.arange(10, 35, 5), value=20)
    verbosity = st.sidebar.checkbox("Verbose output", value=True)
    plot_pred = st.sidebar.checkbox("Plot prediction", value=True)

    all_data_preprocessed = (
        vgsales_metacritic_scores_df.copy().drop(columns=drop_columns))
    feats = all_data_preprocessed.drop(columns=target_col)
    target = all_data_preprocessed[target_col]

    with st.expander("Prévisualisation des attributs et de la cible (Valeurs brutes)", expanded=False, icon=None):
        nb_lignes = st.select_slider("Nombre de lignes à afficher", range(1, 51), value=5, key='s1')
        st.write("Attributs:")
        st.dataframe(feats.head(nb_lignes))
        st.write("Cible")
        st.dataframe(target.head(nb_lignes))
        st.write(target.min(), target.max())
        fig, ax = plt.subplots(figsize=(3, 3))
        sm.qqplot(target, fit=True, line='s', ax=ax)
        st.pyplot(fig, use_container_width=False)

    cat_col = ['Platform', 'Genre', 'Rate', 'Year', 'Publisher', 'Developer']
    b_encoder = ce.BinaryEncoder(cols=cat_col, return_df=True)
    feats = b_encoder.fit_transform(feats)

    feats['Type'] = feats['Type'].map({'Salon': 1, 'Portable': 0})

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=test_size / 100)

    num_col = ['Critic_score', 'Critic_positive_reviews', 'Critic_mixed_reviews',
               'Critic_negative_reviews', 'User_score', 'User_positive_reviews',
               'User_mixed_reviews', 'User_negative_reviews']
    x_train_scaled = X_train.copy()
    x_test_scaled = X_test.copy()
    x_encoders = ["StandardScaler", "RobustScaler", "MinMaxScaler"]
    scaler_ = st.sidebar.radio(
        "Encodage des attributs numériques",
        x_encoders,
        captions=[],
    )
    if scaler_ == "StandardScaler":
        x_scaler = StandardScaler()
        x_train_scaled[num_col] = x_scaler.fit_transform(x_train_scaled[num_col])
        x_test_scaled[num_col] = x_scaler.transform(x_test_scaled[num_col])
    elif scaler_ == "RobustScaler":
        x_scaler = RobustScaler()
        x_train_scaled[num_col] = x_scaler.fit_transform(x_train_scaled[num_col])
        x_test_scaled[num_col] = x_scaler.transform(x_test_scaled[num_col])
    elif scaler_ == "MinMaxScaler":
        x_scaler = MinMaxScaler()
        x_train_scaled[num_col] = x_scaler.fit_transform(x_train_scaled[num_col])
        x_test_scaled[num_col] = x_scaler.transform(x_test_scaled[num_col])

    # y_encoders = ["RobustScaler", "Box-Cox", "Yéo-Johnson", "QuantileTransformer"]
    y_encoders = ["RobustScaler", "Box-Cox", "QuantileTransformer"]
    if target_col != 'Global_Sales':
        y_encoders.remove("Box-Cox")

    scaler_ = st.sidebar.radio(
        "Encodage de la cible",
        y_encoders,
        captions=[],
    )
    if scaler_ == "RobustScaler":
        y_scaler = RobustScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    elif scaler_ == "Box-Cox":
        y_scaler = PowerTransformer(method='box-cox', standardize=True)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    elif scaler_ == "Yéo-Johnson":
        y_scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    elif scaler_ == "QuantileTransformer":
        y_scaler = QuantileTransformer()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    st.sidebar.button("Rafraichir")

    with st.expander("Prévisualisation des attributs et de la cible (Valeurs d'entrainement encodées)", expanded=False,
                     icon=None):
        nb_lignes = st.select_slider("Nombre de lignes à afficher", range(1, 51), value=5, key='s2')
        st.write("Attributs:")
        st.dataframe(x_train_scaled.head(nb_lignes))
        st.write("Cible")
        st.dataframe(y_train_scaled[:nb_lignes])
        st.write(y_train_scaled.min(), y_train_scaled.max())
        fig, ax = plt.subplots(figsize=(3, 3))
        sm.qqplot(target, fit=True, line='s', ax=ax)
        st.pyplot(fig, use_container_width=False)

    model_selection = st.multiselect(
        "Quels modèles voulez vous évaluer ?",
        [model for model in models_tried.keys()],
        [model for model in models_tried.keys()]
    )

    models_to_run = {key: models_tried[key] for key in model_selection}
    _ = run_models(models_to_run,
                   x_train_scaled,
                   x_test_scaled,
                   y_train_scaled,
                   y_test_scaled,
                   y_scaler=y_scaler,
                   test_size=test_size,
                   verbose=verbosity, graph=plot_pred)

with tab2:
    tab2_col1, tab2_col2, tab2_col3, tab2_col4 = st.columns(4)
    with tab2_col1:
        eta = st.select_slider('eta', np.arange(0.12, 0.15, 0.01), value=0.12)
    with tab2_col2:
        max_depth = st.select_slider('max_depth', np.arange(5, 9, 1), value=7)
    with tab2_col3:
        subsample = st.select_slider('subsample', np.arange(0.8, 1.1, 0.1), value=0.9)
    with tab2_col4:
        n_estimators = st.select_slider('n_estimators', np.arange(100, 1600, 100), value=1000)
    hyperparameters = {
        'eta': eta,
        'max_depth': max_depth,
        'subsample': subsample,
        'n_estimators': n_estimators
    }

    model = {'XGBRegressor': XGBRegressor(**hyperparameters)}

    _ = run_models(model,
                   x_train_scaled,
                   x_test_scaled,
                   y_train_scaled,
                   y_test_scaled,
                   y_scaler=y_scaler,
                   test_size=test_size,
                   verbose=verbosity, graph=plot_pred)
