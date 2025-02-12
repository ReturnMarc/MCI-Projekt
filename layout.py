"""
File for the general app layout for the dash app. All components, widgets etc. are created here
"""

from dash import html, dcc
import pandas as pd

# Load the dataset
housing_df = pd.read_csv('housing.csv')
housing_df = housing_df.dropna()
initial_data = housing_df.to_json(date_format='iso', orient='split')

def create_layout():
    return html.Div(id='main_container',
                    className='main-container',
                    children=[
        dcc.Store(id='housing-dataframe-store', data=initial_data),
        html.Div([
            dcc.Dropdown(
                id='plot-type',
                options=[
                    {'label': 'Feature Importance', 'value': 'feature_importance'},
                    {'label': 'SHAP Values', 'value': 'shap'},
                    {'label': 'Partial Dependence', 'value': 'partial_dependence'},
                    {'label': 'LIME Explanation', 'value': 'lime'}
                ],
                value='feature_importance'
            ),
            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': col, 'value': col} for col in housing_df.columns],
                value='median_income'
            ),
            dcc.Input(
                id='instance-selector',
                type='number',
                value=0,
                min=0,
                max=len(housing_df)-1
            ),
            dcc.Graph(id='xai-plot')
        ])
    ])