"""
File for the general app layout for the dash app. All components, widgets etc. are created here
"""

from dash import html, dcc
import os

def create_layout():
    return html.Div(id='main_container',
                    className='main-container',
                    children=[
        html.H1('Model Analysis Dashboard', className='dashboard-title'),
        
        # Dataset and target selection container
        html.Div([
            html.Div([
                html.Label('Select Dataset:', className='dropdown-label'),
                dcc.Dropdown(
                    id='dataset-selector',
                    options=[
                        {'label': f.replace('.csv', ''), 'value': f.replace('.csv', '')} 
                        for f in os.listdir('datasets') if f.endswith('.csv')
                    ],
                    value=None,
                    className='dropdown'
                )
            ], className='selector-container'),
            
            html.Div([
                html.Label('Select Target Variable:', className='dropdown-label'),
                dcc.Dropdown(
                    id='target-selector',
                    options=[],  # Will be populated based on dataset selection
                    value=None,
                    className='dropdown'
                )
            ], className='selector-container')
        ], className='selection-row'),
        
        # Analysis controls container
        html.Div([
            html.Div([
                html.Label('Select Plot Type:', className='dropdown-label'),
                dcc.Dropdown(
                    id='plot-type',
                    options=[
                        {'label': 'Feature Importance', 'value': 'feature_importance'},
                        {'label': 'SHAP Values', 'value': 'shap'},
                        {'label': 'Partial Dependence', 'value': 'partial_dependence'},
                        {'label': 'LIME Explanation', 'value': 'lime'}
                    ],
                    value='feature_importance',
                    className='dropdown'
                )
            ], className='control-container'),
            
            html.Div([
                html.Label('Select Feature:', className='dropdown-label'),
                dcc.Dropdown(
                    id='feature-selector',
                    options=[],  # Will be populated based on dataset selection
                    value=None,
                    className='dropdown'
                )
            ], className='control-container'),
            
            html.Div([
                html.Label('Select Instance Index:', className='input-label'),
                dcc.Input(
                    id='instance-selector',
                    type='number',
                    value=0,
                    min=0,
                    className='number-input'
                )
            ], className='control-container')
        ], className='controls-row'),
        
        # Visualization container
        html.Div([
            dcc.Graph(
                id='xai-plot',
                className='plot-container'
            )
        ], className='visualization-row')
    ])

# CSS styles can be added to assets/style.css