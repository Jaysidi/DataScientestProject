import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, StandardScaler
from xgboost import XGBRegressor

import plotly.express as px
from Libraries.Data import vgsales_metacritic_scores_df
from Libraries.Models import run_models, models_tried, qq_plot_plotly

import streamlit as st
st.set_page_config(
    page_title="Conception d'un modèle de Machine Learning des données",
    layout="wide",
    menu_items={})

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.image("Images/ML.png")

drop_columns = ['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
target_columns = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# target_col = st.selectbox('Choix de la cible', target_columns)
target_col = 'Global_Sales'
drop_columns.remove(target_col)
all_data_preprocessed = (vgsales_metacritic_scores_df.copy().drop(columns=drop_columns))

st.sidebar.write("***Options***")
t_r = (all_data_preprocessed[target_col].min(), all_data_preprocessed[target_col].max())
target_range = st.sidebar.slider("Sélectionnez les ventes min et max",
                   t_r[0],
                   t_r[1],
                   t_r)

test_size = st.sidebar.select_slider("Test size en %", np.arange(10, 35, 5), value=20)
c1, c2, c3 = st.sidebar.columns(3)
with c1:
    verbosity = st.checkbox("Verb.", value=True)
with c2:
    plot_pred = st.checkbox("Plot", value=True)
with c3:
    plot_shap = st.checkbox("Shap", value=False)
all_data_preprocessed = all_data_preprocessed[(all_data_preprocessed[target_col] >= target_range[0])
                                              & (all_data_preprocessed[target_col] <= target_range[1])]
feats = all_data_preprocessed.drop(columns=target_col)
target = all_data_preprocessed[target_col]


tab0, tab1, tab2, tab3, tab4 = st.tabs(["Présentation des données", "Encodage", "Recherche d'un modèle", "XGBoost Regressor", "Analyse de sentiments"], )
with tab0:
    st.header("Jeu de données utilisé")
    st.markdown("""Pour la modélisation finale, nous avons travaillé sur le jeu de données enrichi par celles récupérées sur le site Metacritic.  """)
    st.dataframe(feats.head())
    st.markdown(
        """Pour la cible, nous nous sommes concentrés sur les ventes globales (*Global_Sales*) des jeux""")

with tab1:
    st.header("Préparation des données <-> Encodage des variables")
    nunique = pd.DataFrame(vgsales_metacritic_scores_df[['Platform', 'Year', 'Genre', 'Publisher', 'Rate', 'Developer', 'Type']].nunique())
    nunique.columns = ['Valeur unique']
    st.markdown(f"""Après avoir testé le ***One Hot Encoder*** et rapidement constaté le trop grand nombre de colonnes générées ~= {nunique['Valeur unique'].sum()-4}""")
    st.dataframe(nunique)
    st.markdown("""Nous sommes passés au ***Binary Encoder*** """)
    st.code("""import category_encoders as ce  
b_encoder = ce.BinaryEncoder(cols=cat_col, return_df=True)  
ce.BinaryEncoder(cols=['Genre'], return_df=True).fit_transform(vgsales_metacritic_scores_df).sample(10)[['Genre_0', 'Genre_1', 'Genre_2', 'Genre_3']])""")
    st.markdown(f"""Le ***Binary Encoder*** encode sur n bits chaque catégorie de telle sorte que $2^n < nb.values$.  
    Pour la variable '*Genre*' cela donne $n=4$ soit $2^4=16$ valeurs possible, pour 11 valeurs uniques à encoder. Nous avons donc 4 colonnes au lieur de 10.""")
    st.dataframe(ce.BinaryEncoder(cols=['Genre'], return_df=True).fit_transform(vgsales_metacritic_scores_df).sample(10)[['Genre_0', 'Genre_1', 'Genre_2', 'Genre_3']])
    st.caption("Sur un échantillon de 10 lignes au hasard")
    st.markdown("""La différence est encore plus notable avec la variable *Developer* et ses 1006 valeurs uniques -> $n=10$,  soit $2^{10}=1024 > 1006$.  
    10 colonnes au lieu de 1005 !
    """)
with tab2:
    st.header("Recherche d'un modèle")
    st.write("Nous avons entrainé plusieurs modèles afin de déterminer celui qui donne les meilleurs résultats")

    with st.expander("Prévisualisation des attributs et de la cible (Valeurs brutes)", expanded=False, icon=None):
        nb_lignes = st.select_slider("Nombre de lignes à afficher", range(1, 51), value=5, key='s1')
        st.write("Attributs:")
        st.dataframe(feats.head(nb_lignes))
        st.write("Cible")
        st.dataframe(target.head(nb_lignes))
        st.write(f"""Valeur minimum = {target.min()}    
        Valeur minimum = {target.max()}""")
        # fig, ax = plt.subplots(figsize=(3, 3))
        # sm.qqplot(target, fit=True, line='s', ax=ax)

        fig = qq_plot_plotly(target)
        st.plotly_chart(fig, use_container_width=False)

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
    if scaler_ == "Box-Cox":
        standard = st.sidebar.checkbox("Standard BC", value=False)
    if scaler_ == "RobustScaler":
        y_scaler = RobustScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    elif scaler_ == "Box-Cox":
        y_scaler = PowerTransformer(method='box-cox', standardize=standard)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    elif scaler_ == "Yéo-Johnson":
        y_scaler = PowerTransformer(method='yeo-johnson', standardize=False)
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
        st.write(f"""Valeur minimum (train) = {y_train_scaled.min()}    
                Valeur minimum (train) = {y_train_scaled.max()}""")
        fig1 = qq_plot_plotly(y_train_scaled)
        st.plotly_chart(fig1, use_container_width=False)

    model_selection = st.multiselect(
        "Quels modèles voulez vous évaluer ?",
        [model for model in models_tried.keys()],
        []
        # [model for model in models_tried.keys()]
    )

    models_to_run = {key: models_tried[key] for key in model_selection}
    with st.spinner(f"Running model(s): "):
        _ = run_models(models_to_run,
                       x_train_scaled,
                       x_test_scaled,
                       y_train_scaled,
                       y_test_scaled,
                       y_scaler=y_scaler,
                       test_size=test_size,
                       verbose=verbosity,
                       graph=plot_pred,
                       plot_shap=plot_shap)

with tab3:
    tab2_col1, tab2_col2, tab2_col3, tab2_col4 = st.columns(4)
    with tab2_col1:
        eta = st.select_slider('eta', np.arange(0.12, 0.15, 0.01), value=0.12)
    with tab2_col2:
        max_depth = st.select_slider('max_depth', np.arange(5, 9, 1), value=7)
    with tab2_col3:
        subsample = st.select_slider('subsample', np.arange(0.8, 1.01, 0.1), value=0.9)
    with tab2_col4:
        n_estimators = st.select_slider('n_estimators', np.arange(100, 1600, 100), value=1000)

    hyperparameters = {
        'eta': eta,
        'max_depth': max_depth,
        'subsample': subsample,
        'n_estimators': n_estimators
    }

    model = {'XGBRegressor': XGBRegressor(**hyperparameters)}
    with st.spinner(f"Running model(s): "):
        _ = run_models(model,
                       x_train_scaled,
                       x_test_scaled,
                       y_train_scaled,
                       y_test_scaled,
                       y_scaler=y_scaler,
                       test_size=test_size,
                       verbose=verbosity, graph=plot_pred, plot_shap=plot_shap)

with tab4:
    st.title("Analyse de sentiment")
    st.markdown("Avec les données recueillies par Web Scraping des commentaires utilisateurs du site Metacritic, "
                "nous avons commencé une analyse de sentiment")
    st.page_link("pages/4_🌐_Web_scraping.py", label="Cliquer ici pour accéder à la page WebScraping")

    st.markdown("""Pour entrainé un modèle, nous avons filtré le jeu de données pour ne garder que les jeux récoltés 
    avec 500 commentaires.  
    Ensuite, à l'aide de la fonction *detect* de la librairie *langdetect* nous n'avons gardé que les commentaires en 
    anglais (majoritaires).""")
    st.code("""from langdetect import detect  
    data['Is_English'] = data['Quote'].apply(detect_english)
    data = data.loc[data.Is_English]""")

    st.markdown("""Ensuite, nous avons créer la variable cible, sur base de la métrique utilisé par Metacritic:""")
    st.image("Images/metacritic_metric.png")
    st.code("""# Encode the target column Sentiment based on Metacritic ranges: 0-4 = Negative, 5-7 = Mixed, 8-10 = Positive  
data['Sentiment'] = data['Score'].apply(lambda x: -1 if x < 5 else 0 if x >= 5  and x < 8 else 1)""")
    st.markdown("""Ensuite, comment le nettoyage du texte des caractères inutiles ainsi que des stop words""")
    st.code("""# Set quotes to lower case and remove all non alpha numeric
# or not white space characters with an empty string
data['Quote'] = data['Quote'].str.lower().replace('[^\w\s]','', regex=True)
data["Quote"] = data["Quote"].apply(word_tokenize)
data["Quote"] = data["Quote"].apply(stop_words_filtering)
data["Quote"] = data["Quote"].apply(" ".join)""")
    st.markdown("""Au passage nous avons réalisé deux *word clouds*:""")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Positif")
        st.image("Images/word_cloud_positive.png")
    with col2:
        st.write("#### Négatif")
        st.image("Images/word_cloud_negative.png")
    st.markdown("""Séparation des variables, application d'une *lematization*, 'split' et application d'une vectorisation 
    TF-IDF:""")
    st.code("""X = data.Quote
y = data.Sentiment  
X = X.apply(lambda x: " ". join(lemmatizing(x.split())))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 27)
vec = TfidfVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)""")
    st.markdown("""Étant donnée le fort déséquilibre entre les labels, nous avons tenté d'effectuer une méthode de 
    over/under sampling assez élaborée: ***Over-sampling using SMOTE and cleaning using Tomek links***.""")
    st.page_link(page="https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html",
                 label="Link vers la page de SMOTETomek")
    st.code("""smt = SMOTETomek(sampling_strategy='all')
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
""")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Données brutes")
        st.image("Images/count_unbalanced.png")
    with col2:
        st.write("#### Données *re-sampled*")
        st.image("Images/count_balanced.png")

    st.markdown(f"""Nous avons testé un modèle GradientBoostingClassifier avec les paramètres par défaut avec
{round(155392*0.85)} commentaires""")
    conf_mat = [[4619, 132, 2471], [610, 274, 1628], [471, 76, 13028]]
    conf_mat_resampled = [[5632, 847, 743], [725, 1096, 691], [1885, 1408, 10282]]

    labels = ['Class -1', 'Class 0', 'Class 1']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""#### Résultat du modèle entrainé avec les données brutes:  
##### Classification report""")
        st.image("Images/class_report.png")
        st.write("""##### Confusion Matrix""")
        fig1 = px.imshow(conf_mat,
                         labels=dict(x="Predicted", y="Expected", color="count"),
                         x=labels,
                         y=labels,
                         text_auto=True, color_continuous_scale = "GnBu")
        st.plotly_chart(fig1)
    with col2:
        st.markdown("""#### Résultat du modèle entrainé avec les données rééquilibrées par SMOTETomek:    
##### Classification report""")
        st.image("Images/class_report_resampled.png")
        st.write("""##### Confusion Matrix""")
        fig2 = px.imshow(conf_mat_resampled,
                         labels=dict(x="Predicted", y="Expected", color="count"),
                         x=labels,
                         y=labels,
                         text_auto=True, color_continuous_scale = "GnBu")
        st.plotly_chart(fig2)
