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
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='feature-importance-help'),
                                    ], className='graph-selection-row'),
                                    html.Div([dcc.Graph(id='feature-importance-plot',
                                                        className='plot-container')])
                                ])
                            ]),
                            html.Div([
                                html.Div([
                                    html.H3('PARTIAL DEPENDENCE', className='graph-label'),
                                    html.Div([
                                        dcc.Dropdown(id='partial-dependence-feature-dropdown',
                                                     options=[],
                                                     value=None,
                                                     placeholder='Select Feature',
                                                     className='dropdown-partial-dependence'),
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='partial-dependence-help'),
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
                                                    id='lime-help'),
                                    ], className='graph-selection-row'),
                                    html.Div([dcc.Graph(id='lime-plot',
                                                        className='plot-container-big')])
                                ])
                            ]),
                            html.Div([
                                html.Div([
                                    html.H3('SHAP VALUES', className='graph-label'),
                                    html.Div([
                                        dcc.Dropdown(id='shap-dropdown',
                                                     options=[],
                                                     value=None,
                                                     placeholder='Select Attributes',
                                                     multi=True,
                                                     className='dropdown-shap-lime'),
                                        html.Button('?', n_clicks=0, className='button',
                                                    id='shap-help'),
                                    ], className='graph-selection-row'),
                                    html.Div(children=[],
                                             id='dynamic-filters',
                                             className='filter-options'),
                                    dcc.Store(id='stored-data'),
                                        # html.Button('Filter Data', n_clicks=0, className='button',
                                        #             id='shap-additional-filtering'),
                                        # dbc.Modal([
                                        #     dbc.ModalHeader(dbc.ModalTitle('Filter Data')),
                                        #     dbc.ModalBody([html.Div(children=[],
                                        #                             style={'display': 'flex',
                                        #                                    'flexWrap': 'wrap',
                                        #                                    'gap': '10px',
                                        #                                    'justifyContent': 'space-between'},
                                        #                             id='dynamic-filters',
                                        #                             className='modal-widgets-row'),
                                        #                    html.Div(children=[],
                                        #                             id='data-table')]),
                                        #     dbc.ModalFooter(
                                        #         [html.Button("Use Filter", id='apply-filters', n_clicks=0,
                                        #                      className='button',
                                        #                      style={'width': 200}),
                                        #          html.Button('Close', id='close-modal', n_clicks=0,
                                        #                      className='button')]),
                                        #
                                        # ],
                                        #     id='modal-filter',
                                        #     size='xl',
                                        #     backdrop=True,
                                        #     centered=True,
                                        #     is_open=False,
                                        # ),
                                    html.Div([dcc.Graph(id='shap-plot',
                                                        className='plot-container-big')])
                                ])
                            ])
                        ])
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
