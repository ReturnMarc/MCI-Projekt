"""
File for the general app layout for the dash app. All components, widgets etc. are created here
"""

from dash import html, dcc
import pandas as pd

# Load the dataset
housing_df = pd.read_csv('housing.csv')

# Reformat it to json -> needed for dcc.Store component
initial_data = housing_df.to_json(date_format='iso', orient='split')

def create_layout():
    return html.Div(id='main_container',
                    className='main-container',
                    children=[dcc.Store(id='housing-dataframe-store', data=initial_data)])