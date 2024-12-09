import streamlit as st
st.set_page_config(
    page_title="Conclusions",
    layout="wide",
    menu_items={}
)

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

st.image("Images/conclusion.png")
st.markdown("""
Après avoir analysé et mis à jour ce jeu de données, nous pensons qu'il est possible de réaliser un modèle de prédiction 
des ventes d'un jeu vidéo, sur base des variables qualitatives (Publisher, Developer, Genre, Platform, Type, Rate, Year) 
et les variables quantitatives annexes (notations, et nombre de sentiments) issues du site MetaCritic, qui permettent 
d'expliquer une part assez significative de la variance.  

Cependant, le jeu de données, comme nous l'avons montré précédemment, englobe plusieurs périodes du marché en question 
plus ou moins bien renseignées. Il conviendrait pour établir un modèle pertinent, d'avoir des données en quantité 
suffisante pour chacune d'entre elles.
La réduction du jeu de données lors de la fusion des différents jeux de données, démontre bien que sur le jeu 
d'entrainement la variance est très bien expliquée mais que le modèle peine à généraliser sur le jeu de test.
Cela dit, les résultats restent tout de même satisfaisants.  
 
La difficulté principale réside dans l'obtention d'un volume de données conséquent en liaison avec les données d'un site 
tel que MetaCritic et de ***chiffres de ventes fiables et idéalement marqués dans le temps***.

Afin d'étudier l'impact de l'engouement pré-commercialisation, il apparait évident que ce marquage temporel est 
une donnée primordiale, et permettrait d'étudier les variations sur la durée de commercialisation du jeu, cette 
donnée ne nous est pas accessible ici, dans un volume suffisant pour l'analyser et la modéliser, de même que les données sur 
l'engouement avant commercialisation. Dans notre cas, l'accès aux sentiments avant ventes est difficile étant donnée la 
période couverte.

Le manque de temps alloué à ce projet ne nous a pas permis d'aller plus loin dans la modélisation '*simple*', tout comme 
celle permettant de faire le lien entre les sentiments après ventes et les volumes de ventes. Ce dernier nécessiterait 
deux modèles en cascade, ou bien un modèle unique très sophistiqué.""")