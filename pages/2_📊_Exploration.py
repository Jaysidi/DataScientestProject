import streamlit as st

st.set_page_config(
    page_title="Exploration des données",
    layout="wide",
    menu_items={}
)

import uuid
import plotly.express as px
from Libraries.Data import uvl_df, vgsales_original_df, metacritic_scores_df, vgsales_cleaned_df, vgsales_new_df

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 2rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
st.image("Images/exploration.png")
st.markdown("""### Exploration  
Le dataset contient 16598 lignes indexées de 0 à 16597.  
Il est composé de 3 types de données:

* 6 colonnes de type float (Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales et Global_Sales)
* 1 colonne de type int (Rank)
* 4 colonnes de type object (Name, Platform, Genre et Publisher)""")

with st.expander('Afficher les premières lignes du jeu de données original', expanded=True):
    st.dataframe(vgsales_original_df.head(10))

na_summary = vgsales_original_df.isna().sum()[vgsales_original_df.isna().sum() != 0].reset_index()
na_summary['%'] = round(na_summary[0] / len(vgsales_original_df) * 100, 2)
na_summary.columns = ['Variable', '# NaN', '% total']

st.markdown("""### Constatations

* Valeurs manquantes pour la variable Year (271) et Publisher (58)

""")
st.dataframe(na_summary)
st.markdown("""

* Valeurs = 'Unknown' pour Publisher (203)""")


with st.expander('Afficher les lignes en question'):
    st.dataframe(vgsales_original_df.loc[
                     (vgsales_original_df['Publisher'] == 'Unknown') | (vgsales_original_df['Publisher'].isna()) |
                     vgsales_original_df['Year'].isna()])


st.write("### Remplacement des données manquantes ou inconnues des variables Year et Publisher")
st.page_link("pages/4_🌐_Web_scraping.py", label="Cliquer ici pour accéder à la page WebScraping")
st.markdown("""Nous avons utilisé les données récupérées par Web Scrapping sur différents sites web.
Les données ont été récupérées et exportées dans des fichiers *.csv. Afin de pouvoir compléter les 
informations du jeu de données original, nous avons liés les différents jeux de données sur les couple Name/Platform.
Les noms ont été préalablement nettoyés et standardisés afin d'optimiser le nombre de correspondances.""")

with st.expander("Code source de la fonction faisant la mise à jour des données"):
    st.code("""def update_na_values(target_df, target_filter_column, na_column, source_df,
                         source_filter_column, source_column, format):
        \"""
        Helps recover some NA data from another DataFrame source.
    
        Args:
          target_df: DataFrame to be updated
          target_filter_column: source_df columns to filter on (matching rows) (2 columns)
          na_column: column with missing values to update
          source_df: source DataFrame to find missing values
          source_filter_column: source_df columns names to filter to filter on (matching rows) (2 columns)
          source_column: column name of the source DataFrame where to find the missing values
        \"""
        counter = 0
        # Save the target and source columns names to variables
        t_col1, t_col2 = target_filter_column
        s_col1, s_col2 = source_filter_column
        # Create df copies
        target_df_formated = target_df.copy()
        source_df_formated = source_df.copy()
    
        # Format the values of the columns used for matching the lines (lower case,
        # removes non alphabetic characters) after removal of data between parenthesis
        # (JP Sales), (2008)
        if format:
            target_df_formated[t_col1] = target_df_formated[t_col1].str.lower().apply(lambda x: re.sub(r'\W+', '', re.sub(r'\(.+\)', '', x)))
            target_df_formated[t_col2] = target_df_formated[t_col2].str.lower().apply(lambda x: re.sub(r'\W+', '', re.sub(r'\(.+\)', '', x)))
            source_df_formated[s_col1] = source_df_formated[s_col1].str.lower().apply(lambda x: re.sub(r'\W+', '', re.sub(r'\(.+\)', '', x)))
            source_df_formated[s_col2] = source_df_formated[s_col2].str.lower().apply(lambda x: re.sub(r'\W+', '', re.sub(r'\(.+\)', '', x)))
    
        # Filter rows where the target df has NA values in the na_column (values to
        # retreive and replace)
        na_rows = target_df_formated.loc[target_df_formated[na_column].isna(), target_filter_column]
        print(f"Nombre de lignes à mettre à jour = {color.BOLD}{len(na_rows)}{color.END}")
    
        # Iterate over rows
        for row in na_rows.iterrows():
            # Get the data to match on
            col1_t_v = row[1][t_col1]
            col2_t_v = row[1][t_col2]
    
            # Get the matching value if any
            match_data = source_df_formated.loc[(source_df_formated[s_col1] == col1_t_v) & (source_df_formated[s_col2] == col2_t_v), source_column]
    
            # initialyse some matching data conditions
            is_not_empty = not match_data.empty  # is match data exist ?
            is_not_unknown = True  # is match data not 'Unknown' ?
            is_type_str = False  # is match data str type ?
            is_not_nan = True  # is match data NaN ?
    
            # Check the conditions on the match data
            if is_not_empty:
                is_not_unknown = match_data.values[0] != 'Unknown'
                is_type_str = type(match_data.values[0]) == str
                if not is_type_str:
                    is_not_nan = not np.isnan(match_data.values[0])
            # If there is a match and match data is not equal to 'Unknown' or np.nan
            if is_not_empty and ((is_type_str and is_not_unknown) or (is_not_unknown and is_not_nan)):
                # Update the target DataFrame directly with the matchin value
                target_df.loc[(target_df_formated[t_col1] == col1_t_v) & (target_df_formated[t_col2] == col2_t_v), na_column] = match_data.values[0]
                # Print out some informations
                print(f"{col1_t_v} - {col2_t_v}: {color.BOLD}{match_data.values[0]}{color.END}")
                # Increase the matching counter
                counter += 1
        # Print out the results
        print(f"{color.UNDERLINE}Nombre de lignes modifiées{color.END} = {counter}")
        print(f"{color.UNDERLINE}Nombre de lignes {color.BOLD}NON{color.END}{color.UNDERLINE} modifiées{color.END} = {target_df[na_column].isna().sum()}")""", language='python')


st.write("* VGChartz - MetaCritic - UVLIST (scripts et csv en annexes du rendu final)")

with st.expander('Afficher les données VGChartz'):
    st.text(f"VGChartz : {vgsales_new_df.shape[0]} lignes")
    st.dataframe(vgsales_new_df.head(50))
    df_plat_count = vgsales_original_df["Platform"].value_counts().reset_index()
    df_plat_count.columns = ["Platform", "nb_games_origine"]
    vgsales_new_df_plat_count = vgsales_new_df["Platform"].value_counts().reset_index()
    vgsales_new_df_plat_count.columns = ["Platform", "nb_games_scrap"]

    df_both = df_plat_count.merge(vgsales_new_df_plat_count, on="Platform", how="outer")
    df_both = df_both.loc[df_both["nb_games_origine"].notna()]
    st.dataframe(df_both)
    df_both['Platform'] = df_both['Platform'].astype(str).astype('category')

    fig = px.bar(df_both, y='Platform', x='nb_games_scrap', text_auto='.2s',
                 hover_data=['nb_games_origine', 'nb_games_scrap'], color='nb_games_origine',
                 labels={'Nb jeux': 'Jeux par plateforme'}, height=800)
    fig.update_layout(barmode='group')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Nombre de jeux")
    st.plotly_chart(fig, key = uuid.uuid4())

with st.expander('Afficher les données MetaCritic'):
    update_platform_name = {'Dreamcast': 'DC',
                            'Game Boy Advance': 'GBA',
                            'GameCube': 'GC',
                            'Meta Quest': 'MQ',
                            'Nintendo 64': 'N64',
                            'Nintendo Switch': 'SWITCH',
                            'PlayStation': 'PS',
                            'PlayStation 2': 'PS2',
                            'PlayStation 3': 'PS3',
                            'PlayStation 4': 'PS4',
                            'PlayStation 5': 'PS5',
                            'PlayStation Vita': 'PSV',
                            'Wii': 'Wii',
                            'Wii U': 'WiiU',
                            'Xbox': 'XB',
                            'Xbox 360': 'X360',
                            'Xbox One': 'XOne',
                            'Xbox Series X': 'XBSX',
                            'iOS (iPhone/iPad)': 'IOS'}
    metacritic_scores_df.Platform = metacritic_scores_df.Platform.replace(update_platform_name)
    # np.sort(df_meta.Platform.unique()), np.sort(df_meta.Platform.unique())
    st.text(f"MetaCritic : {metacritic_scores_df.shape[0]} lignes")
    st.dataframe(metacritic_scores_df.head(50))
    df_plat_count = vgsales_original_df["Platform"].value_counts().reset_index()
    df_plat_count.columns = ["Platform", "nb_games_origine"]
    metacritic_scores_df_plat_count = metacritic_scores_df["Platform"].value_counts().reset_index()
    metacritic_scores_df_plat_count.columns = ["Platform", "nb_games_scrap"]

    df_both = df_plat_count.merge(metacritic_scores_df_plat_count, on="Platform", how="outer")
    df_both = df_both.loc[df_both["nb_games_origine"].notna()]
    st.dataframe(df_both)
    df_both['Platform'] = df_both['Platform'].astype(str).astype('category')

    fig = px.bar(df_both, y='Platform', x='nb_games_scrap', text_auto='.2s',
                 hover_data=['nb_games_origine', 'nb_games_scrap'], color='nb_games_origine',
                 labels={'Nb jeux': 'Jeux par plateforme'}, height=800)
    fig.update_layout(barmode='group')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Nombre de jeux")
    st.plotly_chart(fig, key = uuid.uuid4())

with st.expander('Afficher les données UVLIST'):
    st.text(f"UVLIST : {uvl_df.shape[0]} lignes")
    st.dataframe(uvl_df.head(50))
    df_plat_count = vgsales_original_df["Platform"].value_counts().reset_index()
    df_plat_count.columns = ["Platform", "nb_games_origine"]
    uvl_df_plat_count = uvl_df["Platform"].value_counts().reset_index()
    uvl_df_plat_count.columns = ["Platform", "nb_games_scrap"]

    df_both = df_plat_count.merge(uvl_df_plat_count, on="Platform", how="outer")
    df_both = df_both.loc[df_both["nb_games_origine"].notna()]
    st.dataframe(df_both)
    df_both['Platform'] = df_both['Platform'].astype(str).astype('category')

    fig = px.bar(df_both, y='Platform', x='nb_games_scrap', text_auto='.2s',
                 hover_data=['nb_games_origine', 'nb_games_scrap'], color='nb_games_origine',
                 labels={'Nb jeux': 'Jeux par plateforme'}, height=800)
    fig.update_layout(barmode='group')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Nombre de jeux")
    st.plotly_chart(fig, key = uuid.uuid4())

st.write("* Quatre lignes ont été mises à jour manuellement, leurs dates étant supérieur à 2016, année de collecte des données.")
df_last_miss = vgsales_original_df[vgsales_original_df.Year > 2016]
st.dataframe(df_last_miss)

st.write("Enfin, certains doublons sont écartés en ne conservait que ceux présentant le plus de ventes globales")
st.image("Images/last_duplicates.png")
st.write("* Suppression des 60 lignes dont la correspondance n'a pu être faîte malgré la somme d'informations récupérées.")

if st.checkbox('Afficher les données nettoyées', value=True):
    st.text(f"Fichier complété : {vgsales_cleaned_df.shape[0]} lignes conservées sur les {vgsales_original_df.shape[0]} lignes de départ.")
    st.dataframe(vgsales_cleaned_df.head(50))
