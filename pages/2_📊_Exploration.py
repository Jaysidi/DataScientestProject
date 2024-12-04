import streamlit as st

st.set_page_config(
    page_title="Exploration des données",
    layout="wide",
    menu_items={}
)
import plotly.express as px

from Libraries.Data import uvl_df, vgsales_original_df, metacritic_scores_df, vgsales_cleaned_df, vgsales_new_df

st.markdown("""### Exploration  
Le dataset contient 16598 lignes de 0 à 16597.  
Il est composé de 3 types de données:

* 6 colonnes de type float (Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales et Global_Sales)
* 1 colonne de type int (Rank)
* 4 colonnes de type object (Name, Platform, Genre et Publisher)""")
with st.expander('Afficher les premières lignes du jeu de données original'):
    st.dataframe(vgsales_original_df.head(10))

st.markdown("""### Constations  
* Valeurs manquantes pour la variable Year (271) et Publisher (58)
* Valeurs Unknown (203) pour Publisher""")

with st.expander('Afficher les lignes en question'):
    st.dataframe(vgsales_original_df.loc[
                     (vgsales_original_df['Publisher'] == 'Unknown') | (vgsales_original_df['Publisher'].isna()) |
                     vgsales_original_df['Year'].isna()])

st.markdown("""### WebScrapping  
Les données ont été récupérées et exportées dans des fichiers csv afin de pouvoir compléter les informations du jeu de données original.  

* VGChartz - MetaCritic - UVLIST (scripts et csv en annexes du rendu final)""")

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
    st.plotly_chart(fig)

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
    st.plotly_chart(fig)

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
    st.plotly_chart(fig)

st.write("<br><ul><li>Quatre lignes ont été mises à jour manuellement</li></ul>", unsafe_allow_html=True)
df_last_miss = vgsales_original_df[vgsales_original_df.Year > 2016]
st.dataframe(df_last_miss)

st.write(
    "<br><ul><li>Suppression des 60 lignes dont la correspondance n'a pu être faîte malgré la somme d'informations récupérée.</li></ul>",
    unsafe_allow_html=True)
afficher_clean = st.checkbox('Afficher les données nettoyées')
if afficher_clean:
    st.text(f"Fichier complété : {vgsales_cleaned_df.shape[0]} lignes")
    st.dataframe(vgsales_cleaned_df.head(50))
