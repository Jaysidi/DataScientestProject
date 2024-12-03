import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer

import category_encoders as ce

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from xgboost import XGBRegressor, plot_importance, plot_tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

import shap

models_tried = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'XGBRegressor': XGBRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor()
}


def run_models(models, x_train, x_test, y_train, y_test, y_scaler, test_size, verbose=True, graph=False):
    """
    Function to help test and tune different models.
    Train models and compute different metrics to help compare them.

    Args:
      models: a dict with model names as keys and model instances as values.
      x_train:  X train data
      x_test: X test data
      y_train: y train data
      y_test: y test data
      verbose: boolean to print metrics
      graph: boolean to plot graph of predictions vs real values
      y_scaler: fitted scaler instance used for target
      test_size: test size used for train test split (in %)

    Returns:
      results: a dict with model names as keys and metrics as values.

    """
    results = {}
    y_train_unscaled = np.round(y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel(), 2)
    y_test_unscaled = np.round(y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), 2)

    y_range = np.min([y_train.min(), y_test.min()]), np.max([y_train.max(), y_test.max()])

    y_range_reverted = \
        np.min([y_train_unscaled.min(), y_test_unscaled.min()]), \
            np.max([y_train_unscaled.max(), y_test_unscaled.max()])

    results['y_range'] = y_range
    if verbose:
        st.markdown(f"#### ⚠️ **Les métriques sont calculées sur les valeurs brutes <-> {y_range_reverted}**")

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_train_unscaled = np.round(y_scaler.inverse_transform(y_pred_train.reshape(-1, 1), ).ravel(), 2)

        y_pred_test = model.predict(x_test)
        y_pred_test_unscaled = np.round(y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel(), 2)

        r2_train = round(model.score(x_train, y_train), 3)
        r2_test = round(model.score(x_test, y_test), 3)

        # f1_score_train = round(f1_score(y_train_unscaled, y_pred_train_unscaled), 3)
        # f1_score_test = round(f1_score(y_test_unscaled, y_pred_test_unscaled), 3)

        mae_train = round(mean_absolute_error(y_train_unscaled, y_pred_train_unscaled), 3)
        mae_test = round(mean_absolute_error(y_test_unscaled, y_pred_test_unscaled), 3)

        median_ae_train = round(median_absolute_error(y_train_unscaled, y_pred_train_unscaled), 3)
        median_ae_test = round(median_absolute_error(y_test_unscaled, y_pred_test_unscaled), 3)

        rmse_train = round(np.sqrt(mean_squared_error(y_train_unscaled, y_pred_train_unscaled)), 3)
        rmse_test = round(np.sqrt(mean_squared_error(y_test_unscaled, y_pred_test_unscaled)), 3)

        mse_train = round(mean_squared_error(y_train_unscaled, y_pred_train_unscaled), 3)
        mse_test = round(mean_squared_error(y_test_unscaled, y_pred_test_unscaled), 3)

        results[model_name] = {'Metrics':
                                   {'Train':
                                        {'R²': r2_train,
                                         #  'F1_Score': f1_score_train,
                                         'MAE': mae_train,
                                         'MSE': mse_train,
                                         'MedAE': median_ae_train,
                                         'RMSE': rmse_train},
                                    'Test':
                                        {'R²': r2_test,
                                         # 'F1_Score': f1_score_test,
                                         'MAE': mae_test,
                                         'MSE': mse_test,
                                         'MedAE': median_ae_test,
                                         'RMSE': rmse_test}},
                               'Model_instance':
                                   model
                               }
        if graph:
            fig, ax = plt.subplots(figsize=(4, 3))
            plt.plot(y_test_unscaled, y_test_unscaled, 'r--', lw=0.5)
            sns.scatterplot(x=y_test_unscaled.ravel(), y=y_pred_test_unscaled, color='orange', alpha=0.5)
            plt.xlabel('Valeurs de test (Brutes)', fontsize=8)
            plt.ylabel('Valeurs prédites (Brutes)', fontsize=9)
            plt.title(f'{model_name}\n'
                      f'Valeurs de test vs Valeurs prédites.\nTest split = {round(test_size)}%',
                      fontsize=8)
            if not verbose:
                st.subheader(f"Modèle {model_name}")
                st.pyplot(fig, use_container_width=True)
        if verbose:
            st.subheader(f"Modèle {model_name}")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### Métriques")
                st.dataframe(results[model_name]['Metrics'])

            with col2:
                if graph:
                    st.pyplot(fig, use_container_width=True)




            # st.write(f"{model_name}: \n "
            #          f"Metrics on TRAIN values:\n\n"
            #          f"R2= {r2_train}\t\t"
            #          # f"F1_Score= {f1_score_train}\t"
            #          f"MAE= {mae_train}\t "
            #          f"MSE= {mse_train}\t "
            #          f"RMSE= {rmse_train}\t "
            #          f"MedAE= {median_ae_train}"
            #          f"\n\n"
            #          f"Metrics on TEST values:\n\n"
            #          f"R2= {r2_test}\t "
            #          # f"F1_Score= {f1_score_test}\t"
            #          f"MAE= {mae_test}\t "
            #          f"MSE= {mse_test}\t "
            #          f"RMSE= {rmse_test}\t "
            #          f"MedAE= {median_ae_test}"
            #          f"\n\n")

    return results


def plot_xgb_feature_importances(model, title_prefix=''):
    """
    Function to plot XGBoost feature importances.
    3 types are drawn: gain, weight, cover.
    Args:
      model: XGBoost model instance.
      title_prefix: string to add to the title.
    Returns:
      None
    """
    plt.figure(figsize=(5, 10))
    ax1 = plt.subplot(311)
    plot_importance(model, importance_type='gain', show_values=False,
                    max_num_features=10, xlabel='Gain',
                    title=title_prefix + "\nFeature importance - Top 10", ax=ax1)
    ax2 = plt.subplot(312)
    plot_importance(model, importance_type='weight', show_values=False,
                    max_num_features=10, xlabel='Weight', title='', ax=ax2)
    ax3 = plt.subplot(313)
    plot_importance(model, importance_type='cover', show_values=False,
                    max_num_features=10, xlabel='Cover', title='', ax=ax3)
    # plt.tight_layout()
    plt.show()
