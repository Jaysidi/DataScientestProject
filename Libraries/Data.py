import os.path
import joblib
import pandas as pd
import pickle
import streamlit as st

datasets_path = "Datasets"

vgsales_original = "vgsales-original.csv"
cleaned_vgsales = "cleaned_by_script_vgsales.csv"
new_vgsales = "vgsales_new.csv"
uvl = "uvlist.csv"
metacritic_scores = "Scores_Metacritic_V2.csv"
metacritic_user_reviews = "UserReviews_Metacritic_Max500_Sample.csv"
vg_merged_meta = "VGSales_Metacritic_Scores.csv"
meta_metacritic_scores = "METADATA_Scores_Metacritic_V2.md"
meta_metacritic_reviews = "METADATA_UserReviews_Metacritic_Max500.md"
data_sentiment = 'data_sentiment_predicted.joblib'
data_sentiment_100_500 = 'data_sentiment_predicted_100_500.joblib'

@st.cache_data
def load_csv_to_df(path_name, file_name):
    return pd.read_csv(os.path.join(path_name, file_name))


@st.cache_data
def read_file(path_name, file_name):
    with open(os.path.join(path_name, file_name), 'r') as f:
        return f.read()

@st.cache_data
def read_pickle(path_name, file_name):
    return pickle.load(open(os.path.join(path_name, file_name), 'rb'))

@st.cache_data
def read_joblib(path_name, file_name):
    return joblib.load(os.path.join(path_name, file_name))

with st.spinner("Loading data..."):
    vgsales_original_df = load_csv_to_df(datasets_path, vgsales_original)
    vgsales_cleaned_df = load_csv_to_df(datasets_path, cleaned_vgsales)
    vgsales_new_df = load_csv_to_df(datasets_path, new_vgsales)
    uvl_df = load_csv_to_df(datasets_path, uvl).drop(columns='Unnamed: 0')
    metacritic_scores_df = load_csv_to_df(datasets_path, metacritic_scores)
    metacritic_user_reviews_df = load_csv_to_df(datasets_path, metacritic_user_reviews)
    vgsales_metacritic_scores_df = load_csv_to_df(datasets_path, vg_merged_meta)
    metacritic_scores_md = read_file(datasets_path, meta_metacritic_scores)
    metacritic_reviews_md = read_file(datasets_path, meta_metacritic_reviews)
    data_sentiment_df = read_joblib(datasets_path, data_sentiment)
    data_sentiment_100_500_df = read_joblib(datasets_path, data_sentiment_100_500)
