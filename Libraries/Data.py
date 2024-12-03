import os.path
import pandas as pd
import streamlit as st

datasets_path = "Datasets"

vgsales_csv_name = "cleaned_by_script_vgsales.csv"
metacritic_scores_csv_name = "Scores_Metacritic_V2.csv"
metacritic_user_reviews = "UserReviews_Metacritic_Max500.csv"
vg_merged_meta = "VGSales_Metacritic_Scores.csv"

@st.cache_data
def load_csv_to_df(path_name, file_name):
    return pd.read_csv(os.path.join(path_name, file_name))


vgsales_df = load_csv_to_df(datasets_path, vgsales_csv_name)
metacritic_scores_df = load_csv_to_df(datasets_path, metacritic_scores_csv_name)
#metacritic_user_reviews_df = load_csv_to_df(datasets_path, metacritic_user_reviews)
vgsales_metacritic_scores_df = load_csv_to_df(datasets_path, vg_merged_meta)

