import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import uuid
import os
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

import shap
import streamlit as st
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

XGBRegressor_hyperparameters = {
    'eta': np.arange(0.12, 0.15, 0.01),
    'max_depth': np.arange(5, 8, 1),
    'subsample': np.arange(0.8, 1, 0.1)
}

model_path = "Models"

def get_xgbregressor_features_importances(model, i_types, ):
    """
        Function to return XGBoost feature f_importances for each type in i_types.
        i_types can be : gain, weight, cover, total_weight, total_cover.
        Args:
          model: XGBoost model instance.
          i_types: list of importance_type to return.
        Returns:
          dict of DataFrame {type: DataFrame([features, f_importances])}.
        """
    f_importances = {}
    for i_type in i_types:
        fi = model.get_booster().get_score(importance_type=i_type)
        fi = pd.DataFrame.from_dict(fi, orient='index').reset_index()
        fi.columns = ['Feature', 'ImportanceValue']
        f_importances[i_type] = fi
    return f_importances


def fit_model(model, x, y):
    model.fit(x, y)


def predict_model(model, x):
    return model.predict(x)


def plot_xgb_feature_importances(f_importances, max_features=np.inf, title_prefix=''):
    """
    Function to plot top 'max_features' XGBoost feature importances in f_importances
    Args:
      f_importances: dict of DataFrame {type: DataFrame([features, f_importances])}.
      max_features: max number of feature importances to plot.
      title_prefix: string to add to the title.
    Returns:
      None
    """
    if max_features != np.inf:
        st.write(f"{title_prefix} Top {max_features} feature importances")
    else:
        st.write(f"{title_prefix} Feature importances")

    cols = st.columns(len(f_importances))
    for i, col_i in enumerate(cols):
        with col_i:
            key = list(f_importances.keys())[i]
            fi_sorted = f_importances[key].sort_values(by='ImportanceValue', ascending=True).tail(max_features)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=fi_sorted['ImportanceValue'],
                                 y=fi_sorted['Feature'],
                                 orientation='h',
                                 text=round(fi_sorted['ImportanceValue'],2),
                                 marker=dict(color='LightSkyBlue'),
                                 marker_line=dict(width=1, color='gray'),
                                 opacity=0.9
                                 ))
            fig.update_layout(xaxis_title=key.capitalize(),
                              margin=dict(t=20, b=50)
                              )
            st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())


def qq_plot_plotly(data):
    qq_data = sm.qqplot(data, fit=True, line='s').gca().lines
    fig = go.Figure()
    fig.add_trace({
        'type': 'scatter',
        'x': qq_data[0].get_xdata(),
        'y': qq_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qq_data[1].get_xdata(),
        'y': qq_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }})
    fig.update_layout(
        title="Quartile to quartile plot of the target 'Global_Sales' as normality test",
        showlegend=False,
        width=600
    )
    return fig


def run_models(models, x_train, x_test, y_train, y_test, y_scaler, test_size, verbose=True, graph=False, plot_shap=False,
               param_name=''):
    """
    Function to help test and tune different models.
    Train models and compute different metrics to help compare them.

    Args:
      models: a dict with model names as keys and model instances as values.
      x_train:  X train data
      x_test: X test data
      y_train: y train data
      y_test: y test data
      y_scaler: fitted scaler instance used for target
      test_size: test size used for train test split (in %)
      verbose: boolean to print metrics
      graph: boolean to plot graph of predictions vs real values
      plot_shap: boolean to plot shap values
      param_name: str to use in the file name for saving/loading the fitted model
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
        st.caption(f"###### ⚠️ **Les métriques sont calculées sur les valeurs brutes: "
                    f"{str(y_range_reverted[0])} -> {str(y_range_reverted[1])} million(s) de ventes**")
    for model_name, model in models.items():
        # model.fit(x_train, y_train)
        with st.spinner(f"Fitting {model_name}..."):
            if not os.path.exists(os.path.join(model_path, model_name+param_name+'.joblib')):
                st.caption(f"{os.path.join(model_path, model_name+param_name+'.joblib')} not found")
                fit_model(model, x_train, y_train)
                # joblib.dump(model,os.path.join(model_path, model_name+param_name+'.joblib'))
            else:
                model = joblib.load(os.path.join(model_path, model_name+param_name+'.joblib'))
        # y_pred_train = model.predict(x_train)
        with st.spinner(f"Predict values from {model_name}..."):
            y_pred_train = predict_model(model, x_train)
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
                                         'Mean Absolute Error': mae_train,
                                         'Median Absolute Error': median_ae_train,
                                         'Mean Squared Error': mse_train,
                                         'Root Mean Squared Error': rmse_train},
                                    'Test':
                                        {'R²': r2_test,
                                         # 'F1_Score': f1_score_test,
                                         'Mean Absolute Error': mae_test,
                                         'Median Absolute Error': median_ae_test,
                                         'Mean Squared Error': mse_test,
                                         'Root Mean Squared Error': rmse_test}},
                               'Model_instance':
                                   model
                               }
        fig = go.Figure()
        if graph:
            # fig = go.Figure()
            fig.add_traces([go.Scatter(x=y_test_unscaled,
                                    y=y_test_unscaled,
                                    line=dict(
                                        color='Orange',
                                        width=1,
                                    dash='dot')),
                            go.Scatter(x=y_test_unscaled.ravel(),
                                       y=y_pred_test_unscaled,
                                       mode='markers',
                                       marker=dict(
                                           color='LightSkyBlue',
                                           opacity = 0.3,
                                           size=8,
                                           line=dict(
                                               color='Black',
                                               width=1)
                                           )
                                       ) ])
            fig.update_layout(xaxis_title="Valeurs de test (Brutes)",
                              yaxis_title="Valeurs prédites (Brutes)",
                              showlegend=False,
                              margin=dict(l=20, r=20, t=20, b=20),
                              title=f'{model_name}\n'
                                    f'Valeurs de test vs Valeurs prédites.\nTest split = {round(test_size)}%')
            if not verbose:
                st.subheader(f"Modèle {model_name}")
                st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())
                if model_name == "XGBRegressor":
                    plot_xgb_feature_importances(results[model_name]['Model_instance'], model_name)
        if verbose:
            st.subheader(f"Modèle {model_name}")
            # col1, col2, col3 = st.columns(3)
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Métriques")
                st.dataframe(results[model_name]['Metrics'])

            with col2:
                if graph:
                    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

        if graph:
            if model_name == "XGBRegressor":
                f_i = get_xgbregressor_features_importances(results[model_name]['Model_instance'],
                                                      ['gain', 'weight', 'cover'])
                plot_xgb_feature_importances(f_i, 10, model_name)
                if plot_shap:
                    st.write("### SHAP values")
                    with st.spinner(f"Computing SHAP interpretation..."):
                        shap_values_test = shap.TreeExplainer(results[model_name]['Model_instance']).shap_values(x_test)

                        x_test_array = x_test.values
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = plt.figure()
                            shap.summary_plot(shap_values_test, x_test_array, plot_type="bar",
                                              feature_names=x_test.columns)
                            st.pyplot(fig1, use_container_width=False)

                        with col2:
                            fig2 = plt.figure()
                            shap.summary_plot(shap_values_test, x_test_array, feature_names=x_test.columns)
                            st.pyplot(fig2, use_container_width=False)
    return results




def prepare_xgbregressor_model():
    eta = st.sidebar.select_slider('eta', np.arange(0.12, 0.15, 0.01), value=0.13)
    max_depth = st.sidebar.select_slider('max_depth', np.arange(5, 8, 1), value=7)
    subsample = st.sidebar.select_slider('subsample', np.arange(0.8, 1, 0.1), value=0.9)
    hyperparameters = {
        'eta': eta,
        'max_depth': max_depth,
        'subsample': subsample
    }

    return {'XGBRegressor': XGBRegressor(hyperparameters)}