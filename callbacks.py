from dash import Input, Output, State, dcc
import pandas as pd
from model_utils import (
    plot_feature_importance, 
    plot_shap_values, 
    plot_partial_dependence,
    plot_lime_explanation
)
from model_store import store

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
        Output('xai-plot', 'figure'),
        [Input('feature-selector', 'value'),
         Input('plot-type', 'value'),
         Input('instance-selector', 'value')]
    )
    def update_xai_plot(selected_feature, plot_type, instance_idx):
        # Use stored model and data instead of training again
        model = store.model
        X_train = store.X_train
        X_test = store.X_test
        
        if plot_type == 'feature_importance':
            return plot_feature_importance(model, X_test.columns)
        
        elif plot_type == 'shap':
            X_sample = X_test.iloc[instance_idx:instance_idx+1]
            return plot_shap_values(model, X_sample, X_test.columns)
        
        elif plot_type == 'partial_dependence':
            return plot_partial_dependence(model, X_train, selected_feature)
        
        elif plot_type == 'lime':
            X_sample = X_test.iloc[instance_idx:instance_idx+1]
            return plot_lime_explanation(model, X_sample, X_train)


