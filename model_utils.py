import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import shap

def train_model(df):
    # Separate features
    numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']
    
    categorical_features = ['ocean_proximity'] 
    
    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create a pipeline with preprocessor and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Prepare data
    X = df[numeric_features + categorical_features]
    y = df['median_house_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit pipeline
    model.fit(X_train, y_train)
    
    return model, X_train, X_test

def plot_feature_importance(model, feature_names):
    """Generate feature importance plot"""
    # Extract the actual model from the pipeline
    if hasattr(model, 'named_steps'):
        # Get the final estimator
        rf_model = model.named_steps['regressor']
    else:
        rf_model = model
    
    # Get feature names after preprocessing
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        # Get transformed feature names
        numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out() if len(preprocessor.named_transformers_['cat'].get_feature_names_out()) > 0 else []
        transformed_feature_names = list(numeric_features) + list(categorical_features)
    else:
        transformed_feature_names = feature_names
    
    importances = rf_model.feature_importances_
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=transformed_feature_names,
        y=importances,
        name='Feature Importance'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        xaxis_tickangle=-45  # Angle feature names for better readability
    )
    return fig

def plot_shap_values(model, X_sample, feature_names):
    """Generate SHAP values plot for a specific instance"""
    # Extract the actual model from the pipeline
    if hasattr(model, 'named_steps'):
        # Get the final estimator
        rf_model = model.named_steps['regressor']
    else:
        rf_model = model
    
    # Transform the data using the pipeline's preprocessor
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        X_transformed = model.named_steps['preprocessor'].transform(X_sample)
        # Get transformed feature names
        numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out() if len(preprocessor.named_transformers_['cat'].get_feature_names_out()) > 0 else []
        transformed_feature_names = list(numeric_features) + list(categorical_features)
    else:
        transformed_feature_names = feature_names
        X_transformed = X_sample

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_transformed)
    
    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        name='SHAP values',
        orientation='h',
        y=feature_names,
        x=shap_values[0],
        connector={'mode': 'spanning'}
    ))
    
    fig.update_layout(
        title='SHAP Values for Selected Instance',
        xaxis_title='Impact on prediction',
        showlegend=False
    )
    return fig

def plot_partial_dependence(model, X, feature_name, num_points=50):
    """Generate partial dependence plot for a specific feature"""
    # Get preprocessor and final estimator from pipeline
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        rf_model = model.named_steps['regressor']
        
        # Check if feature is numeric or categorical
        numeric_features = preprocessor.transformers_[0][2]  # Get numeric feature names
        categorical_features = preprocessor.transformers_[1][2]  # Get categorical feature names
        
        if feature_name in numeric_features:
            # Handle numeric feature
            feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), num_points)
            predictions = []
            
            for value in feature_values:
                X_modified = X.copy()
                X_modified[feature_name] = value
                pred = model.predict(X_modified)
                predictions.append(pred.mean())
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=predictions,
                mode='lines',
                name='Partial Dependence'
            ))
            
            title = f'Partial Dependence Plot for {feature_name}'
            x_label = feature_name
            
        else:
            # Handle categorical feature
            feature_values = X[feature_name].unique()
            predictions = []
            
            for value in feature_values:
                X_modified = X.copy()
                X_modified[feature_name] = value
                pred = model.predict(X_modified)
                predictions.append(pred.mean())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_values,
                y=predictions,
                name='Partial Dependence'
            ))
            
            title = f'Partial Dependence Plot for {feature_name}'
            x_label = feature_name
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title='Predicted House Value',
            xaxis_tickangle=-45 if feature_name in categorical_features else 0
        )
        
        return fig
    else:
        raise ValueError("Model must be a scikit-learn pipeline with preprocessor and regressor steps")