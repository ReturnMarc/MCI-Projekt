from dash import Input, Output, State, dcc, html, no_update, ALL
from plotly import graph_objs as go
import io
from model_utils import (
    plot_feature_importance, 
    plot_shap_values, 
    plot_partial_dependence,
    plot_lime_explanation,
)
from model_utils import load_dataset_models, read_dataset
import pandas as pd
def register_callbacks(app):
    """
    This function initializes various callbacks for the main app. Each callback consists of an @app.callback part and
    the function for the callback itself. Each interaction with the app triggers a callback with one ore more Outputs or
    Inputs and States.
    General callback overview:
    OUTPUTS are the component ids and their properties which change based on the callback.
    Example: An image gets shown after the upload

    INPUTS are the component ids and their properties which are used to create the outputs.
    Example: The contents (e.g. a list) with the uploaded files

    STATES are dynamic properties which need to be considered in the callback.
    Example: The children, e.g. a list with image names, already exist in the app. Without the state property,
    the old image data would be deleted and replaced with the new uploads.
    The function itself has to consider their parameters based on the order of the inputs and the states. When the
    callback has 2 Inputs and one State, the function must have 3 parameters AND RETURN the number of outputs in the
    order in which the Output properties are sorted.

    :param app: The main Dash Application where the callbacks will be registered
    :return: None
    """
    @app.callback(
        [Output('target-selector', 'options'),
         Output('shap-dropdown', 'options'),
         Output('stored-data', 'data')],
        Input('dataset-selector', 'value')
    )
    def update_target_options(dataset_name):
        if not dataset_name:
            return no_update, no_update,  no_update
        
        models = load_dataset_models(dataset_name)
        df = read_dataset(dataset_name)
        if models:
            return ([{'label': target, 'value': target} for target in models.keys()],
                    [{'label': target, 'value': target} for target in models.keys()],
                    df.to_dict())
        return no_update, no_update, no_update

    @app.callback(
        Output('partial-dependence-feature-dropdown', 'options'),
        [Input('dataset-selector', 'value'),
        Input('target-selector', 'value')]
    )
    def update_feature_options(dataset_name, target_variable):
        if not dataset_name or not target_variable:
            return []
            
        # Only show feature selector for partial dependence plots
        # if plot_type != 'partial_dependence':
        #     return []
            
        models = load_dataset_models(dataset_name)
        if models and target_variable in models:
            model_data = models[target_variable]
            features = model_data['X_train'].columns.tolist()
            # Remove target variable from feature list
            if target_variable in features:
                features.remove(target_variable)
            return [{'label': feature, 'value': feature} for feature in features]
        return []
        
    @app.callback(
        Output('feature-selector', 'value'),
        [Input('feature-selector', 'options')]
    )
    def reset_feature_selection(available_options):
        # Reset selection when options change
        return available_options[0]['value'] if available_options else None

    @app.callback(
         Output('feature-importance-plot', 'figure'),
         Input('target-selector', 'value'),
         State('dataset-selector', 'value'),
    )
    def update_feature_importance_plot(target_variable, dataset_name):
        if not dataset_name or not target_variable:
            return no_update # -> does not update any layout or components

        # Load models for selected dataset
        models = load_dataset_models(dataset_name)
        if not models:
            return no_update

        # Get model and data for selected target - check back if selection is correct!
        model_data = models[target_variable]
        model = model_data['model']
        X_test = model_data['X_test']
        return plot_feature_importance(model, X_test.columns)

    @app.callback(
        Output('partial-dependence-plot', 'figure'),
        [Input('target-selector', 'value'),
         Input('partial-dependence-feature-dropdown', 'value')],
        State('dataset-selector', 'value')
    )

    def update_partial_dependence_plot(target_variable, selected_feature, dataset_name):
        if not dataset_name or not target_variable or not selected_feature:
            return no_update

        models = load_dataset_models(dataset_name)
        if not models or target_variable not in models:
            return no_update

        model_data = models[target_variable]
        model = model_data['model']
        X_train = model_data['X_train']
        return plot_partial_dependence(model, X_train, selected_feature, target_variable)

    @app.callback(
        Output('xai-plot', 'figure'),
        [Input('dataset-selector', 'value'),
        Input('target-selector', 'value'),
        Input('feature-selector', 'value'),
        Input('plot-type', 'value'),
        Input('instance-selector', 'value')]
    )
    def update_xai_plot(dataset_name, target_variable, selected_feature, plot_type, instance_idx):
        # Return empty figure with fixed height if no data selected
        if not dataset_name or not target_variable:
            return go.Figure().update_layout(height=600)
        
        # Load models for selected dataset
        models = load_dataset_models(dataset_name)
        if not models or target_variable not in models:
            return go.Figure().update_layout(height=600)
        
        # Get model and data for selected target
        model_data = models[target_variable]
        model = model_data['model']
        X_train = model_data['X_train']
        X_test = model_data['X_test']
        
        # Create new figure for each plot type
        if plot_type == 'feature_importance':
            return plot_feature_importance(model, X_test.columns)
        
        elif plot_type == 'shap':
            X_sample = X_test.iloc[instance_idx:instance_idx+1]
            return plot_shap_values(model, X_sample, X_test.columns)
            
        elif plot_type == 'partial_dependence':
            return plot_partial_dependence(model, X_train, selected_feature, target_variable)
            
        elif plot_type == 'lime':
            X_sample = X_test.iloc[instance_idx:instance_idx+1]
            return plot_lime_explanation(model, X_sample, X_train, target_variable)

    @app.callback(
        Output('dynamic-filters', 'children'),
        [Input('dataset-selector', 'value'),
        Input('shap-dropdown', 'value')]
    )

    def update_filters(selected_dataset, shap_dropdown_values):
        if not selected_dataset or not shap_dropdown_values:
            return no_update
        df = read_dataset(selected_dataset)
        df_filtered = df[shap_dropdown_values]
        filter_elements = []
        for column in df_filtered.columns:
            if column == 'ID':
                continue
            if df[column].dtype in ['int64', 'float64']:
                filter_elements.append(html.Div([html.Label(f'{column}:'),
                                                 dcc.RangeSlider(
                                                     id={'type': 'rangeslider', 'index': f'{column}'},
                                                     min=df[column].min(),
                                                     max=df[column].max(),
                                                     value=[df[column].min(), df[column].max()],
                                                     marks={df[column].min(): f'{df[column].min()}',
                                                            df[column].max(): f'{df[column].max()}'},
                                                     tooltip={'always_visible': False, 'placement': 'bottom'},
                                                     className='slider'
                                                 )],
                                                style={'width': '200px'}))
            else:
                unique_values = df[column].unique()
                filter_elements.append(html.Div([html.Label(f'{column}'),
                                                 dcc.Checklist(
                                                     id={'type': 'checklist', 'index': f'{column}'},
                                                     options=[{'label': str(val), 'value': str(val)} for val in unique_values if val is not None],
                                                     value=list(unique_values),
                                                     className='checklist'
                                                 )]))
        return filter_elements

    @app.callback(
        Output('shap-plot', 'figure'),
        [Input({'type': 'rangeslider', 'index': ALL}, 'value'),
         Input({'type': 'checklist', 'index': ALL}, 'value'),
         Input('target-selector', 'value')],
        [State('stored-data', 'data'),
        State('shap-dropdown', 'value'),
         State('dataset-selector', 'value')]
    )

    def update_shap_plot(slider_values, checklist_values, target_variable, stored_data, shapely_features, dataset_name):
        if not dataset_name or not stored_data:
            return no_update
        try:
            df = pd.DataFrame.from_dict(stored_data)
        except KeyError:
            return no_update

        filtered_df = df.copy()
        models = load_dataset_models(dataset_name)
        if not models or target_variable not in models:
            return go.Figure().update_layout(height=600)

        for i, col in enumerate(shapely_features[:len(slider_values)]):
            min_val, max_val = slider_values[i]
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
        for i, col in enumerate(shapely_features[:len(checklist_values)]):
            selected_values = checklist_values[i]
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
        # Get model and data for selected target
        model_data = models[target_variable]
        model = model_data['model']
        X_test = model_data['X_test']
        X_sample = filtered_df.iloc[0:1]
        return plot_shap_values(model, X_sample, X_test.columns)