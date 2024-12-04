import streamlit as st
st.set_page_config(
    page_title="WebScraping de données complémentaires",
    layout="wide",
    menu_items={}
)

from Libraries.Data import vgsales_cleaned_df, vgsales_original_df, uvl_df, metacritic_scores_df, metacritic_scores_md, \
    metacritic_reviews_md, metacritic_user_reviews_df

st.image("WS.png")

tab1, tab2, tab3 = st.tabs(['Remplacement des NaN', 'Enrichissement', 'Analyse de sentiment'])
with tab1:
    st.title("Remplacement des valeurs manquantes du jeu de données initial")
    st.markdown(f"""
    Le jeu de données initial contient 329 valeurs manquantes (NaN).""")
    na_summary = vgsales_original_df.isna().sum()[vgsales_original_df.isna().sum() != 0].reset_index()
    na_summary['%'] = round(na_summary[0] / len(vgsales_cleaned_df) * 100, 2)
    na_summary.columns = ['Variable', '# NaN', '% total']
    st.dataframe(na_summary)
    st.markdown("""Afin de réduire au maximum la suppression de ligne, nous avons tenté de récupérer ces informations manquantes en les recherchant par la technique du Web Scraping.  
    Quatre sites différents ont été scrappés: 
    
* Le site Universal Videogame List (https://www.uvlist.net/gamesearch/) répertoriant 160,912 références
* Le site d'origine du jeu de données VGChartz (https://www.vgchartz.com/gamedb/) répertoriant 65,865 références
* Le site Metacritic (https://www.metacritic.com/browse/game/) répertoriant 13,503 références
* Le site Mobigames (https://www.mobygames.com/) répertoriant 289,909 références""")
    st.markdown("## 1. UVList - https://www.uvlist.net/gamesearch/")
    with st.popover("Aperçu du site"):
        st.image("uvlist_game_grid_preview.png")
    st.markdown("""
    Les données ont été collectées à l'aide de la librairie *lxml* et par le biais des *XPath* des éléments du code html.  
    Le scrap a permis ici de récupérer le nom des jeux, l'URL du détail des jeux, leurs plateformes, l'année de sortie 
    ainsi que l'éditeur des jeux:""")
    st.dataframe(uvl_df.head())
    st.write("## 2. VGChartz - https://www.vgchartz.com/gamedb/")
    with st.popover("Aperçu du site"):
        st.image("vgchartz_game_grid_preview.png")
    st.markdown("""
    Les données ont été collectées à l'aide de la librairie BeautifulSoup.  
    Afin de voir si nous ouvions avoir des information plus à jour, notamment concernant les volumes de ventes des jeux,
    nous avons reparcouru l'ensemble des jeux résents sur le site.  
    
    Il s'est avéré que les informations actuellement présentes sont malheureusement moins complète que lors du scrape 
    de 2016 (source Kaggle).""")

    st.write("## 3. Metacritic - https://www.metacritic.com/browse/game/")
    with st.popover("Aperçu du site"):
        st.image("meta_game_grid_preview.png")
    st.markdown("""
        Les données ont été collectées à l'aide de la librairie BeautifulSoup.
        
        En plus du nom, année de sortie, éditeur, plateforme, le script a permis de collecter le développeur, le rating,
        la description du jeu, la note des 'critics', la note utilisateur, ainsi que les nombres d'avis positifs/négatifs/mitigés
        par type de votant (critic/user).""")
    st.dataframe(metacritic_scores_df.head())
    with st.expander("Metadata de l'ensemble des données recueillies", expanded=False, icon='📝'):
        st.markdown(metacritic_scores_md)

    st.write("## 4. MobyGames - https://www.mobygames.com/")
    with st.popover("Aperçu du site"):
        st.image("mobigames_game_grid_preview.png")
    st.markdown("""
        Les données ont été collectées à l'aide de la librairie BeautifulSoup.  
        
        Pour chaque valeur manquante listée, la fonction effectuait une recherche sur le site pour le jeu concerné, 
        et tentait ensuite de trouver l'information manquante.""")

with tab2:
    st.title("Enrichissement du jeu de données par WebScraping")
    st.markdown(f"""
    Les données provenant du site Kaggle (Recueillies sur le site VGChartz.com) ne contiennent que des variables 
    catégorielles et, dans le cadre de notre projet, les seules variables quantitatives sont les variables cibles 
    (*Nombre de jeux vendus par région*, en million):""")
    with st.popover("Aperçu du site"):
        st.image("vgchartz_game_grid_preview.png")
    st.dataframe(vgsales_cleaned_df.head(5))
    st.markdown(f"""Dans le but d'enrichir notre jeu de données et permettre d'entrainer plus finement nos modèles, 
    nous avons fusionné les données de base (Kaggle) et celles récupérées par sur le site Metacritic.  
    Les informations complémentaires recueillies sont, pour chacun des couple jeu/plateforme:""")
    with st.popover("Aperçu du site"):
        st.image("meta_score_details_preview.png")
    st.markdown(f"""
* Le développeur 
* La date complète de sortie
* Le 'Rating' représentant le type de publique concerné par le jeu (https://en.wikipedia.org/wiki/Entertainment_Software_Rating_Board#Ratings)
* La description du jeux
* La note moyenne des **Critiques** (note sur 100) - https://www.metacritic.com/about-us/
* Le nombre de votes '*Positifs*' par les Critiques
* Le nombre de votes '*Mitigés*' par les Critiques
* Le nombre de votes '*Négatifs*' par les Critiques
* La note moyenne des **Utilisateurs** (note sur 10)
* Le nombre de votes '*Positifs*' par les Utilisateurs
* Le nombre de votes '*Mitigés*' par les Utilisateurs
* Le nombre de votes '*Négatifs*' par les Utilisateurs
* L'URL de la fiche du jeu
""")
    st.dataframe(metacritic_scores_df.head())
    with st.expander("Metadata de l'ensemble des données recueillies", expanded=False, icon='📝'):
        st.markdown(metacritic_scores_md)

    st.markdown("""Nous avons également récupéré le nombre de vue de vidéo 'trailer' sur Youtube à l'aide de 
    la librairie Python Selenium""")

with tab3:
    st.title("Récupération des commentaires des utilisateurs laissés sur le site Metacritic")
    st.markdown(f"""Dans le but de faire une analyse de sentiment, nous avons recueilli les plus vieux commentaires
des jeux ayant un minimum de 50 commentaires, en nous limitant à 500 commentaires maximum.  
La librairie BeautifulSoup a été utilisée pour analyser les données recueillies au format JSON via l'API du site, 
sans laquelle l'opération aurait été impossible dans des délais raisonnable (page dynamique). """)
    with st.popover("Aperçu du site"):
        st.image("meta_user_review_preview.png")
    st.markdown(f"""
Nous avons recueilli les informations suivantes:  
* Le nom du jeu
* La plateforme du jeu
* Le nombres **total** d'avis présents sur le site
* Le nombre d'avis **recueillis**
* L'avis
* La note attribuée par l'utilisateur ayant laissé l'avis (Note sur 10)
* La date du dépot de l'avis sur le site
* Le pseudo de l'auteur de l'avis
* Un flag si l'avis est jugé être un "spoiler"
""")
    st.write("#### Aperçu des données")
    st.caption("Le fichier csv généré faisant 381 Mo, seul un échantillon de 1000 lignes prises aléatoirement ont été chargées")
    st.dataframe(metacritic_user_reviews_df.head())
    with st.expander("Metadata de l'ensemble des données recueillies", expanded=False, icon='📝'):
        st.markdown(metacritic_reviews_md)