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
import lime
import lime.lime_tabular
import joblib
import os

def save_model(model, X_train, X_test, filename='trained_model.joblib'):
    """Save the trained model and data to disk"""
    try:
        model_data = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test
        }
        joblib.dump(model_data, filename)
        if not os.path.exists(filename):
            raise Exception("File was not created")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(filename='trained_model.joblib'):
    """Load the trained model and data from disk"""
    model_data = joblib.load(filename)
    return model_data['model'], model_data['X_train'], model_data['X_test']

def train_model(df):
    # Remove target variable from feature detection
    features_df = df.drop('median_house_value', axis=1)
    
    # Automatically detect numeric and categorical features
    numeric_features = features_df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    categorical_features = features_df.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist() 
    
    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
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
        y=transformed_feature_names,
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
    
def plot_lime_explanation(model, X_sample, X_train, num_features=10):
    """Generate LIME explanation plot for a specific instance"""
    # Get preprocessor and final estimator from pipeline
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        rf_model = model.named_steps['regressor']
        
        # Get transformed feature names
        numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        transformed_feature_names = list(numeric_features) + list(categorical_features)
        
        # Transform training data for LIME explainer
        train_data = preprocessor.transform(X_train)
        
        # Create LIME explainer with training data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_data,
            feature_names=transformed_feature_names,
            class_names=['median_house_value'],
            mode='regression',
            training_labels=None,
            categorical_features=list(range(len(categorical_features))) if len(categorical_features) > 0 else None
        )
        
        # Transform and explain the sample instance
        X_transformed = preprocessor.transform(X_sample)
        exp = explainer.explain_instance(
            X_transformed[0], 
            rf_model.predict,
            num_features=num_features
        )
        
        # Extract feature contributions
        feature_importance = exp.as_list()
        features, values = zip(*feature_importance)
        
        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values,
            y=features,
            orientation='h'
        ))
        
        fig.update_layout(
            title='LIME Feature Contributions',
            xaxis_title='Impact on Prediction',
            yaxis_title='Features',
            height=max(400, 30 * len(features))  # Dynamic height based on feature count
        )
        
        return fig
    else:
        raise ValueError("Model must be a scikit-learn pipeline with preprocessor and regressor steps")