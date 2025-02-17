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

        # Visualization
        # Most outer containers, contain one selection div and one graph div, one left and one right
        html.Div([
            html.Div([
                html.Div([
                    html.H3('FEATURE IMPORTANCE', className='graph-label'),
                    html.Div([
                        dcc.Dropdown(id='feature-importance-dropdown',
                                     options=[],
                                     placeholder='Select Target Feature',
                                     value=None,
                                     className='dropdown-graphs'),
                        html.Button('?', n_clicks=0, className='button'),
                    ], className='graph-selection-row'),
                    html.Div([dcc.Graph(id='feature-importance-plot',
                                       className='plot-container')])
                ])
            ]),
            html.Div([
                html.Div([
                    html.H3('PARTIAL DEPENDENCE', className='graph-label'),
                    html.Div([
                        dcc.Dropdown(id='partial-dependence-variable-dropdown',
                                     options=[],
                                     value=None,
                                     placeholder='Select Target Variable',
                                     className='dropdown-partial-dependence'),
                        dcc.Dropdown(id='partial-dependence-feature-dropdown',
                                     options=[],
                                     value=None,
                                     placeholder='Select Target Feature',
                                     className='dropdown-partial-dependence'),
                        html.Button('?', n_clicks=0, className='button',
                                    id='partial-dependence-button'),
                    ], className='graph-selection-row'),
                    html.Div([dcc.Graph(id='partial-dependence-plot',
                                        className='plot-container')])
                ])
            ])
        ], className='visualization-row'),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('LIME', className='graph-label'),
                    html.Div([
                        dcc.Dropdown(id='lime-dropdown',
                                     options=[],
                                     value=None,
                                     className='dropdown-graphs'),
                        html.Button('?', n_clicks=0, className='button',
                                    id='lime-button'),
                    ], className='graph-selection-row'),
                    html.Div([dcc.Graph(id='lime-plot',
                                        className='plot-container')])
                ])
            ]),
            html.Div([
                html.Div([
                    html.H3('SHAP VALUES', className='graph-label'),
                    html.Div([
                        dcc.Dropdown(id='shap-dropdown',
                                     options=[],
                                     value=None,
                                     className='dropdown-graphs'),
                        html.Button('?', n_clicks=0, className='button',
                                    id='shap-button'),
                    ], className='graph-selection-row'),
                    html.Div([dcc.Graph(id='shap-plot',
                                        className='plot-container')])
                ])
            ])
        ], className='visualization-row')
    ])

    #
    #     html.Div(children=[
    #         # left selection div
    #         html.Div(children=[
    #             html.Label('FEATURE IMPORTANCE', className='graph-label'),
    #             html.Button('Platzhalter', n_clicks=0, id='button'),
    #             html.Div(children=[
    #                 dcc.Graph(id='feature-importance-plot',
    #                           className='plot-container'
    #                           )])],
    #             className='control-container'),
    #         html.Div(children=[
    #             html.Label('PARTIAL DEPENDENCE', className='graph-label'),
    #             html.Div(children=[
    #                 dcc.Graph(id='partial-dependence-plot',
    #                           className='plot-container')
    #             ])],
    #             className='control-container')
    #         ],
    #         className='visualization-row'),
    #     html.Div(children=[
    #         # left selection div
    #         html.Div(children=[
    #             html.Label('LIME', className='graph-label'),
    #             html.Div(children=[
    #                 dcc.Graph(id='lime-plot',
    #                           className='plot-container'
    #                           )])],
    #             className='control-container'),
    #         html.Div(children=[
    #             html.Label('SHAP VALUES', className='graph-label'),
    #             html.Div(children=[
    #                 dcc.Graph(id='shap-values-plot',
    #                           className='plot-container')
    #             ])],
    #             className='control-container')
    #         ],
    #         className='visualization-row')
    # ])

# CSS styles can be added to assets/style.css