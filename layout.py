"""
File for the general app layout for the dash app. All components, widgets etc. are created here
"""

from dash import html, dcc
import os
import dash_bootstrap_components as dbc


def create_layout():
    return html.Div(id='main_container',
                    className='main-container',
                    children=[

                        html.H1('MODEL ANALYSIS DASHBOARD', className='dashboard-title'),

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
                        # Visualization
                        # Most outer containers, contain one selection div and one graph div, one left and one right
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.H3('Feature Importance', className='graph-label'),
                                    html.Div([
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='feature-importance-help'),
                                    ], className='graph-selection-row-feature-importance'),
                                    dbc.Modal([
                                        dbc.ModalHeader(dbc.ModalTitle('Feature Importance Explanation')),
                                        dbc.ModalBody('Feature importance tells you which input variables (features) have'
                                                      ' the biggest impact on the model`s predictions. Think of it as a '
                                                      'ranking: The higher the importance, the more influence that feature'
                                                      ' has on the model´s decision-making. This is useful for understanding'
                                                      ' which factors drive predictions in general. '),
                                        dbc.ModalFooter(html.Button('Close', id='close-modal-feature-importance',
                                                                    n_clicks=0, className='button'))
                                    ],
                                        id='modal-feature-importance',
                                        is_open=False),
                                    html.Div([dcc.Graph(id='feature-importance-plot',
                                                        className='plot-container')])
                                ])
                            ], className='feature-importance-partial-dependence-container'),
                            html.Div([
                                html.Div([
                                    html.H3('Partial Dependence', className='graph-label'),
                                    html.Div([
                                        dcc.Dropdown(id='partial-dependence-feature-dropdown',
                                                     options=[],
                                                     value=None,
                                                     placeholder='Select Feature',
                                                     className='dropdown-partial-dependence'),
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='partial-dependence-help'),
                                    ], className='graph-selection-row'),
                                    dbc.Modal([
                                        dbc.ModalHeader(dbc.ModalTitle('Partial Dependence Explanation')),
                                        dbc.ModalBody('Partial Dependence shows the marginal effect one or two '
                                                      'features have on the predicted outcome of a machine '
                                                      'learning model. A partial dependence plot can show whether the '
                                                      'relationship between the target and the feature is linear, '
                                                      'monotonic or more complex. This helps you see trends, like whether '
                                                      'increasing a feature makes predictions go up or down. It works well for '
                                                      'understanding the overall relationship between a feature '
                                                      'and the predicted outcome. '),
                                        dbc.ModalFooter(html.Button('Close', id='close-modal-partial-dependence',
                                                                    n_clicks=0, className='button'))
                                    ],
                                        id='modal-partial-dependence',
                                        is_open=False),
                                    html.Div([dcc.Graph(id='partial-dependence-plot',
                                                        className='plot-container')])
                                ])
                            ], className = 'feature-importance-partial-dependence-container'),
                        ], className='visualization-row'),
                        html.Hr(),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.H3('LIME', className='graph-label'),
                                    html.Div([
                                        dcc.Dropdown(id='lime-dropdown',
                                                     options=[],
                                                     value=None,
                                                     placeholder='Select Features',
                                                     multi=True,
                                                     className='dropdown-shap-lime'),
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='lime-help'),
                                    ], className='graph-selection-row'),
                                    dbc.Modal([
                                        dbc.ModalHeader(dbc.ModalTitle('LIME Explanation')),
                                        dbc.ModalBody('LIME (Local Interpretable Model-agnostic Explanations) explains'
                                                      ' a single prediction, it does so by creating a simplified, '
                                                      'interpretable model around that specific data point. It slightly'
                                                      ' changes the input and observes how the prediction shifts,'
                                                      ' helping to approximte the model´s reasoning for that one instance.'
                                                      '\n\n'
                                                      'IMPORTANT: SHAP and LIME only explain one row at a time - they '
                                                      'don´t give insights into the model´s overall behaviour, just why'
                                                      ' it made a specific decision for a given instance (in this case, '
                                                      'the first row of the remaining dataset after applying filters).'),
                                        dbc.ModalFooter(html.Button('Close', id='close-modal-lime',
                                                                    n_clicks=0, className='button'))
                                    ],
                                        id='modal-lime',
                                        is_open=False),
                                    html.Div(children=[],
                                             id='dynamic-filters-lime',
                                             className='filter-options'),
                                    html.Div(html.Label('Selected Instance: ', className='dropdown-label')),
                                    html.Div(children=[],
                                             id='show-lime-instance',
                                             style={'margin-bottom': '10px'}),
                                    html.Div([dcc.Graph(id='lime-plot',
                                                        className='plot-container-big')]),
                                ])
                            ], className='shap-lime-container'),
                            html.Hr(),
                            html.Div([
                                html.Div([
                                    html.H3('SHAP Values', className='graph-label'),
                                    html.Div([
                                        dcc.Dropdown(id='shap-dropdown',
                                                     options=[],
                                                     value=None,
                                                     placeholder='Select Features',
                                                     multi=True,
                                                     className='dropdown-shap-lime'),
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='shap-help'),
                                    ], className='graph-selection-row'),
                                    dbc.Modal([
                                        dbc.ModalHeader(dbc.ModalTitle('SHAP Explanation')),
                                        dbc.ModalBody('SHAP (SHapley Additive exPlanations) explain a single prediction'
                                                      ' by showing how much each feature contributed to that specific '
                                                      'outcome. Think of it as breaking down a decision for one '
                                                      'particular row of data. Some features push the prediction higher,'
                                                      ' others lower. SHAP is made for detailed, instance-level '
                                                      'explanations.\n\n'
                                                      'IMPORTANT: SHAP and LIME only explain one row at a time - they '
                                                      'don´t give insights into the model´s overall behaviour, just why'
                                                      ' it made a specific decision for a given instance (in this case, '
                                                      'the first row of the remaining dataset after applying filters).'),
                                        dbc.ModalFooter(html.Button('Close', id='close-modal-shap',
                                                                    n_clicks=0, className='button'))
                                    ],
                                        id='modal-shap',
                                        is_open=False),
                                    html.Div(children=[],
                                             id='dynamic-filters',
                                             className='filter-options'),
                                    html.Div(html.Label('Selected Instance: ', className='dropdown-label')),
                                    html.Div(children=[],
                                             id='show-shap-instance'),
                                    dcc.Store(id='stored-data'),
                                    html.Div([dcc.Graph(id='shap-plot',
                                                        className='plot-container-big')])
                                ])
                            ], className='shap-lime-container')
                        ])
                    ])

# CSS styles can be added to assets/style.css
