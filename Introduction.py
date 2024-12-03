import streamlit as st
st.set_page_config(
    page_title="Application Streamlit - Projet jeux vidéo",
    layout="wide",
    menu_items={}
)
from Libraries.Data import vgsales_df


st.image("pixel_art.jpg")
st.markdown("""
# Analyse des ventes de jeux vidéo

**Description du projet:**

Estimer les ventes totales d’un jeu vidéo à l’aide d’informations descriptives comme:
* Le pays d’origine
* Le studio l’ayant développé
* L’éditeur l’ayant publié
* La description du jeu
* La plateforme sur laquelle le jeu est sortie
* Le genre

Les données de base proviennent du site https://www.kaggle.com/datasets/gregorut/videogamesales

Les autres descripteurs devront être récupérées via du web scraping à l’aide de la libraire BeautifulSoup (par exemple).
""")

with st.expander("Metadata de la page Kaggle", expanded=False, icon=None):
    st.markdown("""
This dataset contains a list of video games with sales greater than 100,000 copies. It was generated by a scrape of vgchartz.com.

Fields include:

* Rank - Ranking of overall sales
* Name - The games name
* Platform - Platform of the games release (i.e. PC,PS4, etc.)
* Year - Year of the game's release
* Genre - Genre of the game
* Publisher - Publisher of the game
* NA_Sales - Sales in North America (in millions)
* EU_Sales - Sales in Europe (in millions)
* JP_Sales - Sales in Japan (in millions)
* Other_Sales - Sales in the rest of the world (in millions)
* Global_Sales - Total worldwide sales.

The script to scrape the data is available at https://github.com/GregorUT/vgchartzScrape.
It is based on BeautifulSoup using Python.
There are 16,598 records. 2 records were dropped due to incomplete information.
""")



st.title("Introduction")
st.dataframe(vgsales_df.head(10))
st.write(vgsales_df.shape)
st.dataframe(vgsales_df.describe())
if st.checkbox("Afficher les NA"):
    st.dataframe(vgsales_df.isna().sum())




