import streamlit as st

st.set_page_config(
    page_title="Données brutes: Conception d'un modèle de Machine Learning",
    layout="wide",
    menu_items={})

import numpy as np
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import TargetEncoder
from category_encoders import CountEncoder
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

from Libraries.Data import vgsales_cleaned_df

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
tab1, tab2 = st.tabs(['Préparation', 'Modélisation'])
with tab1:
    df = vgsales_cleaned_df.copy()
    # on passe les Name en minuscules dans df_uvlist et df_no_year
    df.loc[:, 'Name'] = df['Name'].str.lower()
    df.loc[:, 'Publisher'] = df['Publisher'].str.lower()
    # on retire toutes les informations inutiles dans le nom de df_no_year, elles sont entre parenthèses (JP sales), etc.
    df.loc[:, 'Name'] = df['Name'].str.split('(').str[0]
    df.loc[:, 'Publisher'] = df['Publisher'].str.split('(').str[0]

    # On ne conserve que les mots et les espaces dans les Names
    df.loc[:, 'Name'] = df['Name'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df.loc[:, 'Publisher'] = df['Publisher'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # on remplace les espaces doubles par des simples
    df.loc[:, 'Name'] = df['Name'].str.replace("  ", " ")
    df.loc[:, 'Publisher'] = df['Publisher'].str.replace("  ", " ")

    # on retire tous les espaces en début et fin de Name
    df.loc[:, 'Name'] = df['Name'].str.strip()
    df.loc[:, 'Publisher'] = df['Publisher'].str.strip()

    li_salon = ['Wii', 'NES', 'X360', 'PS3', 'PS2', 'SNES', 'PS4', 'N64', 'PS', 'XB', 'PC', '2600', 'XOne', 'GC', 'GEN',
                'DC', 'SAT', 'SCD', 'NG', 'TG16', '3DO', 'PCFX']
    li_portable = ['GB', 'DS', 'GBA', '3DS', 'PSP', 'WiiU', 'PSV', 'WS', 'GG']
    df['Type'] = np.where(df['Platform'].isin(li_salon), 'Salon', 'Portable')
    df['Year'] = df['Year'].astype(int)

    st.write("### Préparation des données")
    st.write(
        "Suite à l'exploration initiale des données et nos premières constatations, nous ajoutons le type de plateforme Salon/Portable.")

  #   code = '''
  # li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
  # li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']
  # df['Type'] = np.where(df['Platform'].isin(li_salon), 'Salon', 'Portable')'''
  #
  #   st.code(code)

    # st.write("Nous passons aussi l'année en entier.")
    # code = '''df['Year'] = df['Year'].astype(int)'''
    # st.code(code)
    ### FIN PARTIE 1
    ### Affichage des premières lignes du df
    st.dataframe(df.head())

    ### PARTIE 2 - étude la répartition de Global_Sales

    st.write("### Répartition de la variable Global_Sales")
    st.write(
        "Les différents modèles que nous avons essayés, lors de notre première tentative, renvoyaient des résultats nuls ou négatifs, quelques fussent les variations !")
    st.write(
        "Il est apparu clair que la distribution de la variable cible empêchait toute modélisation, à notre niveau, comme on peut le voir ci-dessous.")

    # Créer les sous-graphiques
    fig = make_subplots(rows=1, cols=2)

    # Ajouter un Scatter plot
    i = 1
    for colonne in ['Global_Sales']:
        fig.add_trace(
            go.Scatter(x=df[colonne], name=colonne),
            row=1, col=i
        )
        i += 1

    i = 2
    for colonne in ['Global_Sales']:
        fig.add_trace(
            go.Histogram(x=df[colonne], name=colonne),
            row=1, col=i
        )
        i += 1

    fig.update_layout(width=800, height=400)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    ### FIN DE LA PARTIE 2

    ### PARTIE 3
    ### APPLICATION DE LA METHODE BOX-COX ET VISUALISATION DE LA TRANSFORMATION
    st.write("### Application de la méthode Box-Cox sur la variable cible")
    st.write(
        """Cette méthode est employée car nous n'avons des valeurs strictement positives, autrement il eut fallu utiliser 
        la méthode Yeo-Johnson qui supporte de telles valeurs.""")

    pt = PowerTransformer(method='box-cox', standardize=False)

    pt.fit(df[['Global_Sales']])

    df['Global_Sales_boxed'] = pt.transform(df[['Global_Sales']])

    only_those = ['Name', 'Global_Sales', 'Global_Sales_boxed']
    df_only_those = df[only_those]
    st.dataframe(df_only_those.head(5))

    fig = make_subplots(
        rows=1, cols=2
    )

    i = 1
    for colonne in ['Global_Sales_boxed']:
        fig.add_trace(
            go.Scatter(x=df[colonne], name=colonne),
            row=1, col=i
        )
        i += 1

    i = 2
    for colonne in ['Global_Sales_boxed']:
        fig.add_trace(
            go.Histogram(x=df[colonne], name=colonne),
            row=1, col=2
        )
        i += 1

    fig.update_layout(width=800, height=400)
    st.write(
        "Nous voyons l'effet de la 'normalisation' de la variable cible plus clairement sur les graphiques ci-dessous.")
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    ### FIN DE LA PARTIE 3

    ### PARTIE 4
    ### FEATURE ENGINEERING, CREATION DE NOUVELLES VARIABLES
    st.write("### Feature Engineering à partir des données de base")
    st.write(
        "Compte tenu du nombre limité de variables à disposition, nous avons essayé d'en ajouter de nouvelles à partir des existantes.")
    st.write(
        "Période d'existence au sein du jeu de données des éditeurs et des plateformes ainsi que des associations potentiellement utiles.")

    ### checkbox pour afficher le code
    code = '''
  def assign_longevite(group):
    plat_long = group.max() - group.min()
    return plat_long

  df['Game_Sales_Period'] = df.groupby('Platform')['Year'].transform(assign_longevite)
  df['Publisher_Sales_Period'] = df.groupby('Publisher')['Year'].transform(assign_longevite)

  df['Pub_Plat'] = df['Publisher'] + '_' + df['Platform']
  df['Pub_Genre'] = df['Publisher'] + '_' + df['Genre']
  df['Plat_Year'] = df['Platform'] + '_' + df['Year'].astype(str)
  df['Plat_Genre'] = df['Platform'] + '_' + df['Genre']
  df['Genre_Year'] = df['Genre'] + '_' + df['Year'].astype(str)
  '''
    with st.expander('Afficher le code'):
        st.code(code, language="python")


    ### Définition de 'durée de vie' pour les Plateformes et les éditeurs
    def assign_longevite(group):
        plat_long = group.max() - group.min()
        return plat_long


    df['Game_Sales_Period'] = df.groupby('Platform')['Year'].transform(assign_longevite)


    def assign_longevite(group):
        plat_long = group.max() - group.min()
        return plat_long


    df['Publisher_Sales_Period'] = df.groupby('Publisher')['Year'].transform(assign_longevite)

    # Création de combinaisons de variables
    df['Pub_Plat'] = df['Publisher'] + '_' + df['Platform']
    df['Pub_Genre'] = df['Publisher'] + '_' + df['Genre']
    df['Plat_Year'] = df['Platform'] + '_' + df['Year'].astype(str)
    df['Plat_Genre'] = df['Platform'] + '_' + df['Genre']
    df['Genre_Year'] = df['Genre'] + '_' + df['Year'].astype(str)

    df['PSP_x_GSP'] = df['Publisher_Sales_Period'] * df['Game_Sales_Period']

    st.dataframe(df.head())

with tab2:

    ########################################################## CODE POUR LA PAGE 3 ##########
    df = vgsales_cleaned_df.copy()

    # on passe les Name en minuscules dans df_uvlist et df_no_year
    df.loc[:, 'Name'] = df['Name'].str.lower()
    df.loc[:, 'Publisher'] = df['Publisher'].str.lower()
    # on retire toutes les informations inutiles dans le nom de df_no_year, elles sont entre parenthèses (JP sales), etc.
    df.loc[:, 'Name'] = df['Name'].str.split('(').str[0]
    df.loc[:, 'Publisher'] = df['Publisher'].str.split('(').str[0]

    # On ne conserve que les mots et les espaces dans les Names
    df.loc[:, 'Name'] = df['Name'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df.loc[:, 'Publisher'] = df['Publisher'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # on remplace les espaces doubles par des simples
    df.loc[:, 'Name'] = df['Name'].str.replace("  ", " ")
    df.loc[:, 'Publisher'] = df['Publisher'].str.replace("  ", " ")

    # on retire tous les espaces en début et fin de Name
    df.loc[:, 'Name'] = df['Name'].str.strip()
    df.loc[:, 'Publisher'] = df['Publisher'].str.strip()

    li_salon = ['Wii', 'NES', 'X360', 'PS3', 'PS2', 'SNES', 'PS4', 'N64', 'PS', 'XB', 'PC', '2600', 'XOne', 'GC',
                'GEN', 'DC', 'SAT', 'SCD', 'NG', 'TG16', '3DO', 'PCFX']
    # li_portable = ['GB', 'DS', 'GBA', '3DS', 'PSP', 'WiiU', 'PSV', 'WS', 'GG']
    df['Type'] = np.where(df['Platform'].isin(li_salon), 'Salon', 'Portable')
    df['Year'] = df['Year'].astype(int)


    ### Définition de 'durée de vie' pour les Plateformes et les éditeurs
    def assign_longevite(group):
        plat_long = group.max() - group.min()
        return plat_long


    df['Game_Sales_Period'] = df.groupby('Platform')['Year'].transform(assign_longevite)

    df['Publisher_Sales_Period'] = df.groupby('Publisher')['Year'].transform(assign_longevite)

    # Création de combinaisons de variables
    df['Pub_Plat'] = df['Publisher'] + '_' + df['Platform']
    df['Pub_Genre'] = df['Publisher'] + '_' + df['Genre']
    df['Plat_Year'] = df['Platform'] + '_' + df['Year'].astype(str)
    df['Plat_Genre'] = df['Platform'] + '_' + df['Genre']
    df['Genre_Year'] = df['Genre'] + '_' + df['Year'].astype(str)

    df['PSP_x_GSP'] = df['Publisher_Sales_Period'] * df['Game_Sales_Period']

    ############################################################################# FIN DU CODE POUR LA PAGE 3 ########################################
    st.write("### Machine Learning - Données de base")
    st.write(
        "Nous allons procéder sur deux jeux de test et d'entrainement, un normalisé par Box-Cox et l'autre non.")

    #### Séparation du jeu de données
    X_scaled = df.drop(['Rank', 'NA_Sales',
                        'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Year'
                        ], axis=1)
    y_scaled = df['Global_Sales']

    X_non_scaled = df.drop(['Rank', 'NA_Sales',
                            'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Year'
                            ], axis=1)
    y_non_scaled = df['Global_Sales']

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled,
                                                                                    test_size=0.3, random_state=43)

    X_train_non_scaled, X_test_non_scaled, y_train_non_scaled, y_test_non_scaled = train_test_split(X_non_scaled,
                                                                                                    y_non_scaled,
                                                                                                    test_size=0.3,
                                                                                                    random_state=43)

    pt = PowerTransformer(method='box-cox', standardize=False).set_output(transform="pandas")

    y_train_scaled_trans = pt.fit_transform(y_train_scaled.values.reshape(-1, 1))
    y_test_scaled_trans = pt.transform(y_test_scaled.values.reshape(-1, 1))

    global_sales_lambda = pt.lambdas_[0]
    with  st.expander('Afficher X_train et y_train scaled'):
        st.write("X_train_scaled")
        st.dataframe(X_train_scaled.head(2))
        st.write("y_train_scaled")
        st.write(y_train_scaled_trans.head(2))

    with st.expander('Afficher X_train et y_train non_scaled'):
        st.write("X_train_non_scaled")
        st.dataframe(X_train_scaled.head(2))
        st.write("y_train_non_scaled")
        st.write(y_train_non_scaled.head(2))

    ### ENCODAGE DES VARIABLES
    # Target_Encoder - SCALED
    te_cat = ['Name']

    te = TargetEncoder(categories='auto', target_type='continuous', smooth='auto', cv=5, shuffle=False).set_output(
        transform="pandas")

    X_train_scaled[te_cat] = te.fit_transform(X_train_scaled[te_cat], y_train_scaled)
    X_test_scaled[te_cat] = te.transform(X_test_scaled[te_cat])

    ### FREQUENCY ENCODER - SCALED
    freq_cat = ['Publisher', 'Platform', 'Genre', 'Pub_Plat', 'Plat_Year', 'Plat_Genre', 'Type', 'Pub_Genre',
                'Genre_Year']

    fr = CountEncoder(normalize=True).set_output(transform="pandas")
    X_train_scaled_encoded = fr.fit_transform(X_train_scaled[freq_cat])
    X_test_scaled_encoded = fr.transform(X_test_scaled[freq_cat])

    X_train_scaled = pd.concat([X_train_scaled.drop(freq_cat, axis=1), X_train_scaled_encoded], axis=1)
    X_test_scaled = pd.concat([X_test_scaled.drop(freq_cat, axis=1), X_test_scaled_encoded], axis=1)

    # Target_Encoder - NON SCALED
    te_ns_cat = ['Name']

    te_ns = TargetEncoder(categories='auto', target_type='continuous', smooth='auto', cv=5,
                          shuffle=False).set_output(transform="pandas")

    X_train_non_scaled[te_ns_cat] = te_ns.fit_transform(X_train_non_scaled[te_ns_cat], y_train_non_scaled)
    X_test_non_scaled[te_ns_cat] = te_ns.transform(X_test_non_scaled[te_ns_cat])

    ### FREQUENCY ENCODER - NON SCALED
    freq_cat_ns = ['Publisher', 'Platform', 'Genre', 'Pub_Plat', 'Plat_Year', 'Plat_Genre', 'Type', 'Pub_Genre',
                   'Genre_Year']

    fr_ns = CountEncoder(normalize=True).set_output(transform="pandas")
    X_train_non_scaled_encoded = fr_ns.fit_transform(X_train_non_scaled[freq_cat_ns])
    X_test_non_scaled_encoded = fr_ns.transform(X_test_non_scaled[freq_cat_ns])

    X_train_non_scaled = pd.concat([X_train_non_scaled.drop(freq_cat_ns, axis=1), X_train_non_scaled_encoded],
                                   axis=1)
    X_test_non_scaled = pd.concat([X_test_non_scaled.drop(freq_cat_ns, axis=1), X_test_non_scaled_encoded], axis=1)

    X_train_scaled['Pub_x_PSP'] = X_train_scaled['Publisher'] * X_train_scaled['Publisher_Sales_Period']
    X_test_scaled['Pub_x_PSP'] = X_test_scaled['Publisher'] * X_test_scaled['Publisher_Sales_Period']

    X_train_scaled['Plat_x_GSP'] = X_train_scaled['Platform'] * X_train_scaled['Game_Sales_Period']
    X_test_scaled['Plat_x_GSP'] = X_test_scaled['Platform'] * X_test_scaled['Game_Sales_Period']

    X_train_non_scaled['Pub_x_PSP'] = X_train_non_scaled['Publisher'] * X_train_non_scaled['Publisher_Sales_Period']
    X_test_non_scaled['Pub_x_PSP'] = X_test_non_scaled['Publisher'] * X_test_non_scaled['Publisher_Sales_Period']

    X_train_non_scaled['Plat_x_GSP'] = X_train_non_scaled['Platform'] * X_train_non_scaled['Game_Sales_Period']
    X_test_non_scaled['Plat_x_GSP'] = X_test_non_scaled['Platform'] * X_test_non_scaled['Game_Sales_Period']

    ### FEATURE ENGINEERING, CREATION DE NOUVELLES VARIABLES
    st.write("### Feature Engineering à partir des données de base")
    st.write(
        "Compte tenu du nombre limité de variables à disposition, nous avons essayé d'en ajouter de nouvelles à partir des existantes.")
    st.write("Une fois l'encodage réalisé")

    code = '''
  X_train_scaled['Pub_x_PSP'] = X_train_scaled['Publisher'] * X_train_scaled['Publisher_Sales_Period']
  X_test_scaled['Pub_x_PSP'] = X_test_scaled['Publisher'] * X_test_scaled['Publisher_Sales_Period']

  X_train_scaled['Plat_x_GSP'] = X_train_scaled['Platform'] * X_train_scaled['Game_Sales_Period']
  X_test_scaled['Plat_x_GSP'] = X_test_scaled['Platform'] * X_test_scaled['Game_Sales_Period']


  X_train_non_scaled['Pub_x_PSP'] = X_train_non_scaled['Publisher'] * X_train_non_scaled['Publisher_Sales_Period']
  X_test_non_scaled['Pub_x_PSP'] = X_test_non_scaled['Publisher'] * X_test_non_scaled['Publisher_Sales_Period']

  X_train_non_scaled['Plat_x_GSP'] = X_train_non_scaled['Platform'] * X_train_non_scaled['Game_Sales_Period']
  X_test_non_scaled['Plat_x_GSP'] = X_test_non_scaled['Platform'] * X_test_non_scaled['Game_Sales_Period']
  '''
    with  st.expander('Afficher le code'):
        st.code(code, language="python")

    with st.expander('Afficher X_train_scaled et X_train_non_scaled encodés'):
        st.dataframe(X_train_scaled.head(2))
        st.dataframe(X_train_non_scaled.head(2))

    scaler = StandardScaler().set_output(transform="pandas")
    minmaxscaler = MinMaxScaler().set_output(transform="pandas")

    # ### X_scaled
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test_scaled)

    ### X_non_scaled
    X_train_non_scaled = scaler.fit_transform(X_train_non_scaled)
    X_test_non_scaled = scaler.transform(X_test_non_scaled)

    # Titre de l'application
    st.title('Modèles Pré-Entraînés')
    # Dictionnaire des modèles disponibles
    model_dict = {
        'RandomForrest scaled': 'Datasets/rf_scaled.joblib',
        'XGBregressor scaled': 'Datasets/xg_scaled.joblib',
        'RandomForrest non scaled': 'Datasets/rf_non_scaled.joblib',
        'XGBregressor non scaled': 'Datasets/xg_non_scaled.joblib',
    }

    # Créez une selectbox pour choisir le modèle
    selected_model_name = st.selectbox('Sélectionnez un modèle', list(model_dict.keys()))

    # Chargez le modèle sélectionné
    selected_model_path = model_dict[selected_model_name]
    model = joblib.load(selected_model_path)


    # Fonction de prédiction utilisant le modèle sélectionné
    def predict(in_data):
        return model.predict(in_data)


    # Entrée des utilisateurs pour la prédiction
    if model_dict[selected_model_name] == "Datasets/rf_scaled.joblib" or model_dict[
        selected_model_name] == "Datasets/xg_scaled.joblib":
        input_data = X_test_scaled
        input_data_train = X_train_scaled

    if model_dict[selected_model_name] == "Datasets/rf_non_scaled.joblib" or model_dict[
        selected_model_name] == "Datasets/xg_non_scaled.joblib":
        input_data = X_test_non_scaled
        input_data_train = X_train_non_scaled

    if st.button('Prédire'):
        result = predict(input_data)
        result_train = predict(input_data_train)

        ######################################################## Prédictions et résultats sur le jeu scaled avec RANDOM FORREST
        if model_dict[selected_model_name] == "Datasets/rf_scaled.joblib":
            mse = mean_squared_error(y_test_scaled_trans, result)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test_scaled_trans, result)
            r2 = r2_score(y_test_scaled_trans, result)
            medae = median_absolute_error(y_test_scaled_trans, result)

            mse_t = mean_squared_error(y_train_scaled_trans, result_train)
            rmse_t = mse_t ** 0.5
            mae_t = mean_absolute_error(y_train_scaled_trans, result_train)
            r2_t = r2_score(y_train_scaled_trans, result_train)
            medae_t = median_absolute_error(y_train_scaled_trans, result_train)

            st.write('Résultat de la prédiction sur test et train:\n\n')
            st.write('R2 (test):', r2, 'R2 (train):', r2_t, '\n')
            st.write('MSE (test):', mse, 'MSE (train):', mse_t, '\n')
            st.write('MAE (test):', mae, 'MAE (train):', mae_t, '\n')
            st.write('RMSE (test):', rmse, 'RMSE (train):', rmse_t, '\n')
            st.write('MedAE (test):', medae, 'MedAE (train):', medae_t, '\n')

            st.write('# Valeurs réelles - Valeurs résiduelles:\n\n')

            le_dict = {
                'Global_Sales': global_sales_lambda
            }
            y_pred = inv_boxcox(result, [global_sales_lambda])
            y_test = inv_boxcox(y_test_scaled_trans, [global_sales_lambda])

            residuals = y_test['x0'] - y_pred

            comparison_df = pd.DataFrame(
                {'Valeurs Réelles': y_test['x0'], 'Valeurs Prédites': y_pred, 'Residuals': residuals})
            comparison_df.sort_values(by='Valeurs Réelles', ascending=True, inplace=True)

            # Création de 2 colonnes dans streamlit
            col1, col2 = st.columns(2)
            with col1:
                st.write("10 plus petites valeurs réelles")
                st.dataframe(comparison_df.head(10))
            with col2:
                st.write("10 plus grandes valeurs réelles")
                st.dataframe(comparison_df.tail(10))

            st.dataframe(comparison_df.describe())

            fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# Valeurs réelles - Valeurs prédites:\n\n')
            fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# SHAP values:\n\n')
            shap_values_test = shap.TreeExplainer(model).shap_values(X_test_scaled)

            # X_test_scaled_array = X_test_scaled
            X_test_scaled_array = X_test_scaled.values
            plt.figure()
            shap.summary_plot(shap_values_test, X_test_scaled_array, feature_names=X_test_scaled.columns)
            st.pyplot(plt)

            st.write('# Matrice de corrélations:\n\n')
            # y_pred = inv_boxcox(result_train, [global_sales_lambda])
            y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

            X_all = pd.concat([X_train_scaled, y_pred], axis=1)

            corr_matrix = X_all.corr()
            plt.figure(figsize=(12, 12))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Features')
            st.pyplot(plt)

            ################################################################## Prédictions et résultats sur le jeu scaled avec XGB
        if model_dict[selected_model_name] == "Datasets/xg_scaled.joblib":
            mse = mean_squared_error(y_test_scaled_trans, result)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test_scaled_trans, result)
            r2 = r2_score(y_test_scaled_trans, result)
            medae = median_absolute_error(y_test_scaled_trans, result)

            mse_t = mean_squared_error(y_train_scaled_trans, result_train)
            rmse_t = mse_t ** 0.5
            mae_t = mean_absolute_error(y_train_scaled_trans, result_train)
            r2_t = r2_score(y_train_scaled_trans, result_train)
            medae_t = median_absolute_error(y_train_scaled_trans, result_train)

            st.write('Résultat de la prédiction sur test et train:\n\n')
            st.write('R2 (test):', r2, 'R2 (train):', r2_t, '\n')
            st.write('MSE (test):', mse, 'MSE (train):', mse_t, '\n')
            st.write('MAE (test):', mae, 'MAE (train):', mae_t, '\n')
            st.write('RMSE (test):', rmse, 'RMSE (train):', rmse_t, '\n')
            st.write('MedAE (test):', medae, 'MedAE (train):', medae_t, '\n')
            st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

            le_dict = {
                'Global_Sales': global_sales_lambda
            }
            y_pred = inv_boxcox(result, [global_sales_lambda])
            y_test = inv_boxcox(y_test_scaled_trans, [global_sales_lambda])

            residuals = y_test['x0'] - y_pred

            comparison_df = pd.DataFrame(
                {'Valeurs Réelles': y_test['x0'], 'Valeurs Prédites': y_pred, 'Residuals': residuals})
            comparison_df.sort_values(by='Valeurs Réelles', ascending=True, inplace=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("10 plus petites valeurs")
                st.dataframe(comparison_df.head(10))
            with col2:
                st.write("10 plus grandes valeurs")
                st.dataframe(comparison_df.tail(10))

            st.dataframe(comparison_df.describe())

            fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# Valeurs réelles - Valeurs prédites:\n\n')
            fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# SHAP values:\n\n')
            shap_values_test = shap.TreeExplainer(model).shap_values(X_test_scaled)

            # X_test_scaled_array = X_test_scaled
            X_test_scaled_array = X_test_scaled.values
            plt.figure(figsize=(6, 12))
            shap.summary_plot(shap_values_test, X_test_scaled_array, feature_names=X_test_scaled.columns,
                              show=False)
            st.pyplot(plt)

            st.write('# Matrice de corrélations:\n\n')
            # y_pred = inv_boxcox(result_train, [global_sales_lambda])
            y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

            X_all = pd.concat([X_train_scaled, y_pred], axis=1)

            corr_matrix = X_all.corr()
            plt.figure(figsize=(12, 12))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Features')
            st.pyplot(plt)

            ################################################################ Prédictions et résultats sur le jeu non scaled avec RANDOM FORREST
        if model_dict[selected_model_name] == "Datasets/rf_non_scaled.joblib":
            mse = mean_squared_error(y_test_non_scaled, result)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test_non_scaled, result)
            r2 = r2_score(y_test_non_scaled, result)
            medae = median_absolute_error(y_test_non_scaled, result)

            mse_t = mean_squared_error(y_train_non_scaled, result_train)
            rmse_t = mse_t ** 0.5
            mae_t = mean_absolute_error(y_train_non_scaled, result_train)
            r2_t = r2_score(y_train_non_scaled, result_train)
            medae_t = median_absolute_error(y_train_non_scaled, result_train)

            st.write('Résultat de la prédiction sur test et train:\n\n')
            st.write('R2 (test):', r2, 'R2 (train):', r2_t, '\n')
            st.write('MSE (test):', mse, 'MSE (train):', mse_t, '\n')
            st.write('MAE (test):', mae, 'MAE (train):', mae_t, '\n')
            st.write('RMSE (test):', rmse, 'RMSE (train):', rmse_t, '\n')
            st.write('MedAE (test):', medae, 'MedAE (train):', medae_t, '\n')

            st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

            y_pred = result
            y_test = y_test_non_scaled

            residuals = y_test - y_pred

            comparison_df = pd.DataFrame(
                {'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred, 'Residuals': residuals})
            comparison_df.sort_values(by='Valeurs Réelles', ascending=True, inplace=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("10 plus petites valeurs")
                st.dataframe(comparison_df.head(10))
            with col2:
                st.write("10 plus grandes valeurs")
                st.dataframe(comparison_df.tail(10))

            st.dataframe(comparison_df.describe())

            fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# Valeurs réelles - Valeurs prédites:\n\n')
            fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# SHAP values:\n\n')
            shap_values_test = shap.TreeExplainer(model).shap_values(X_test_non_scaled)

            # X_test_non_scaled_array = X_test_non_scaled
            X_test_non_scaled_array = X_test_non_scaled.values
            plt.figure()
            shap.summary_plot(shap_values_test, X_test_non_scaled_array, feature_names=X_test_non_scaled.columns)
            st.pyplot(plt)

            st.write('# Matrice de corrélations:\n\n')
            # y_pred = result_train
            y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

            X_all = pd.concat([X_train_scaled, y_pred], axis=1)

            corr_matrix = X_all.corr()
            plt.figure(figsize=(12, 12))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Features')
            st.pyplot(plt)

            #################################################################### Prédictions et résultats sur le jeu non scaled avec XGB
        if model_dict[selected_model_name] == "Datasets/xg_non_scaled.joblib":
            mse = mean_squared_error(y_test_non_scaled, result)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test_non_scaled, result)
            r2 = r2_score(y_test_non_scaled, result)
            medae = median_absolute_error(y_test_non_scaled, result)

            mse_t = mean_squared_error(y_train_non_scaled, result_train)
            rmse_t = mse_t ** 0.5
            mae_t = mean_absolute_error(y_train_non_scaled, result_train)
            r2_t = r2_score(y_train_non_scaled, result_train)
            medae_t = median_absolute_error(y_train_non_scaled, result_train)

            st.write('Résultat de la prédiction sur test et train:\n\n')
            st.write('R2 (test):', r2, 'R2 (train):', r2_t, '\n')
            st.write('MSE (test):', mse, 'MSE (train):', mse_t, '\n')
            st.write('MAE (test):', mae, 'MAE (train):', mae_t, '\n')
            st.write('RMSE (test):', rmse, 'RMSE (train):', rmse_t, '\n')
            st.write('MedAE (test):', medae, 'MedAE (train):', medae_t, '\n')

            st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

            y_pred = result
            y_test = y_test_non_scaled

            residuals = y_test - y_pred

            comparison_df = pd.DataFrame(
                {'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred, 'Residuals': residuals})
            comparison_df.sort_values(by='Valeurs Réelles', ascending=True, inplace=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("10 plus petites valeurs")
                st.dataframe(comparison_df.head(10))
            with col2:
                st.write("10 plus grandes valeurs")
                st.dataframe(comparison_df.tail(10))

            st.dataframe(comparison_df.describe())

            fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# Valeurs réelles - Valeurs prédites:\n\n')
            fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")

            fig.update_layout(width=800, height=400)

            st.plotly_chart(fig)

            st.write('# SHAP values:\n\n')
            shap_values_test = shap.TreeExplainer(model).shap_values(X_test_non_scaled)

            # X_test_non_scaled_array = X_test_non_scaled
            X_test_non_scaled_array = X_test_non_scaled.values
            plt.figure()
            shap.summary_plot(shap_values_test, X_test_non_scaled_array, feature_names=X_test_non_scaled.columns)
            st.pyplot(plt)

            st.write('# Matrice de corrélations:\n\n')
            # y_pred = result_train
            y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

            X_all = pd.concat([X_train_scaled, y_pred], axis=1)

            corr_matrix = X_all.corr()
            plt.figure(figsize=(12, 12))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Features')
            # plt.show()
            st.pyplot(plt)

    else:
        st.write('Veuillez choisir un modèle.')