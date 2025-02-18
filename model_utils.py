import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from dash import no_update, dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import shap
import lime
import lime.lime_tabular
import joblib
import os
import warnings

def clean_data(df):
    """Clean and prepare the DataFrame for modeling"""
    df = df.copy()
    
    # Drop columns with more than 10% missing data
    df = df.replace(r'^\s*$', np.nan, regex=True) # empty Strings
    columns_to_drop = []
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio > 0.1:
            columns_to_drop.append(col)
        
    if columns_to_drop:
        print(f"Dropping columns due to more than 10% missing data: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    df = df.dropna()
    
    # Calculate ratio of unique values to total rows for categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns
    rows_count = len(df)
    
    columns_to_drop = []
    for col in categorical_columns:
        unique_ratio = df[col].nunique() / rows_count
        if unique_ratio > 0.2:
            columns_to_drop.append(col)
    
    # Drop categorical columns with too many unique values
    if columns_to_drop:
        print(f"Dropping columns due to high cardinality: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    if constant_columns:
        print(f"Dropping constant columns: {constant_columns}")
        df = df.drop(columns=constant_columns)
    
    return df

def train_models_for_dataset(dataset_path):
    """Train models for all numeric features in a dataset"""
    # Load dataset
    df = pd.read_csv(dataset_path)
    df = clean_data(df)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Get all numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    models_dict = {}
    
    # Train a model for each numeric feature as target
    for target in numeric_features:
        try:
            # Create feature set excluding current target
            features_df = df.drop(target, axis=1)
            
            # Split features into numeric and categorical
            remaining_numeric = features_df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            categorical_features = features_df.select_dtypes(
                include=['object', 'category', 'bool']
            ).columns.tolist()
            
            # Create transformers based on available feature types
            transformers = []
            if remaining_numeric:
                numeric_transformer = Pipeline(steps=[
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', numeric_transformer, remaining_numeric))
                
            if categorical_features:
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
                ])
                transformers.append(('cat', categorical_transformer, categorical_features))
            
            preprocessor = ColumnTransformer(transformers=transformers)
            
            # Create pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Prepare data
            X = df[remaining_numeric + categorical_features]
            y = df[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit pipeline
            model.fit(X_train, y_train)
            
            # Store model and data
            models_dict[target] = {
                'model': model,
                'X_train': X_train,
                'X_test': X_test
            }
            
        except Exception as e:
            print(f"Error training model for {target}: {str(e)}")
            continue
    
    # Save all models for this dataset
    if models_dict:
        os.makedirs('stored_models', exist_ok=True)
        joblib.dump(models_dict, f'stored_models/{dataset_name}_models.joblib')
    
    return models_dict

def train_all_datasets():
    """Train models for all untrained datasets in the datasets folder"""
    # Get all dataset files
    dataset_files = [f for f in os.listdir('datasets') if f.endswith('.csv')]
    trained_datasets = {}
    
    # Get list of already trained datasets
    existing_models = [f.replace('_models.joblib', '.csv') 
                      for f in os.listdir('stored_models') 
                      if f.endswith('_models.joblib')]
    
    # Filter for only untrained datasets
    untrained_datasets = [f for f in dataset_files if f not in existing_models]
    
    # Train models only for new datasets
    for dataset_file in untrained_datasets:
        print(f"Training models for new dataset: {dataset_file}")
        dataset_path = os.path.join('datasets', dataset_file)
        trained_datasets[dataset_file] = train_models_for_dataset(dataset_path)
    
    if not untrained_datasets:
        print("No new datasets to train - all datasets have existing models")
        
    return trained_datasets

def load_dataset_models(dataset_name):
    """Load models for a specific dataset"""
    filepath = f'stored_models/{dataset_name}_models.joblib'
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None

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
        numeric_features = []
        categorical_features = []
        
        for name, _, _ in preprocessor.transformers_:
            if name == 'num':
                numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            elif name == 'cat':
                categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                
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
        xaxis_tickangle=-30,  # Angle feature names for better readability
        height=600,
        margin=dict(l=40, r=10, t=30, b=80)
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
        numeric_features = []
        categorical_features = []
        
        for name, _, _ in preprocessor.transformers_:
            if name == 'num':
                numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            elif name == 'cat':
                categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                
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
        showlegend=False,
        height=max(600, 30 * len(transformed_feature_names))
    )
    return fig

def plot_partial_dependence(model, X, feature_name, target_feature, num_points=50):
    """Generate partial dependence plot for a specific feature"""
    if target_feature == feature_name:
        warnings.warn("Target feature cannot be the same as the feature for partial dependence")
        return go.Figure().update_layout(
            title='Cannot create partial dependence plot',
            annotations=[{
                'text': 'Target feature cannot be the same as the feature for partial dependence',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }],
            height=400
        )
        
    # Get preprocessor and final estimator from pipeline
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        rf_model = model.named_steps['regressor']
        
        # Check if feature is numeric or categorical
        numeric_features = []
        categorical_features = []
        
        # Safely get feature names from transformers
        for name, _, cols in preprocessor.transformers_:
            if name == 'num':
                numeric_features = cols
            elif name == 'cat':
                categorical_features = cols
        
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
            is_categorical = False
            
        elif feature_name in categorical_features:
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
            is_categorical = True
        else:
            raise ValueError(f"Feature {feature_name} not found in dataset")
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=f'Predicted {target_feature}',
            xaxis_tickangle=-45 if is_categorical else 0,
            margin = dict(l=40, r=10, t=30, b=60)
            #height=600
        )
        
        return fig
    else:
        raise ValueError("Model must be a scikit-learn pipeline with preprocessor and regressor steps")
    
def plot_lime_explanation(model, X_sample, X_train, target_name, num_features=10):
    """Generate LIME explanation plot for a specific instance"""
    # Get preprocessor and final estimator from pipeline
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        rf_model = model.named_steps['regressor']
        numeric_features = []
        categorical_features = []
        categorical_indices = []
        
        for i, (name, _, _) in enumerate(preprocessor.transformers_):
            if name == 'num':
                numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            elif name == 'cat':
                categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                categorical_indices = list(range(len(numeric_features), 
                                              len(numeric_features) + len(categorical_features)))
                
        transformed_feature_names = list(numeric_features) + list(categorical_features)
         
        # Transform training data for LIME explainer
        train_data = preprocessor.transform(X_train)
        
        # Create LIME explainer with training data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_data,
            feature_names=transformed_feature_names,
            class_names=[target_name],
            mode='regression',
            training_labels=None,
            categorical_features=categorical_indices if categorical_indices else None
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
            height=max(600, 30 * len(features))  # Dynamic height based on feature count
        )
        
        return fig
    else:
        raise ValueError("Model must be a scikit-learn pipeline with preprocessor and regressor steps")

# def extract_counterfactuals(dice_exp):
#     cf_examples = dice_exp.cf_examples_list[0]
#     cf_df = pd.DataFrame(cf_examples.final_cfs_df, columns=cf_examples.feature_names_including_target)
#
#     return cf_df
#
# def plot_dice_table(X_sample, dice_exp):
#     cf_df = extract_counterfactuals(dice_exp)
#     original = X_sample.to_frame().T
#
#     combined_df = pd.concat([original, cf_df], ignore_index=True)
#     combined_df['Type'] = pd.Series(['Original'] + [f"CF {i+1}" for i in range(len(cf_df))])
#
#     return dash_table.DataTable(
#         columns=[{'name': i, 'id': i} for i in combined_df.columns],
#         data=combined_df.to_dict("records"),
#         style_data_conditional=[
#             {'if': {'row_index': 0}, 'backgroundColor': 'lightblue'}
#         ]
#     )
def read_dataset(dataset_name):
    filepath = f'datasets/{dataset_name}.csv'
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def get_filter_elements(dataset, features, type):
    if not dataset or not features:
        return no_update
    df = read_dataset(dataset)

    df_filtered = df[features]
    filter_elements = []

    for column in df_filtered.columns:
        if column == 'ID':
            continue
        if df[column].dtype in ['int64', 'float64']:
            filter_elements.append(html.Div([html.Label(f'{column}:'),
                                             dcc.RangeSlider(
                                                 id={'type': f'rangeslider-{type}', 'index': f'{column}'},
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
                                                 id={'type': f'checklist-{type}', 'index': f'{column}'},
                                                 options=[{'label': str(val), 'value': str(val)} for val in
                                                          unique_values if val is not None],
                                                 value=list(unique_values),
                                                 className='checklist'
                                             )]))
    return filter_elements

def get_shap_lime_items(sliders, checklists, target_var, data, features, dataset_name, plot_type):
    if not dataset_name or not data:
        return no_update
    try:
        df = pd.DataFrame.from_dict(data)
    except KeyError:
        return no_update

    filtered_df = df.copy()

    models = load_dataset_models(dataset_name)
    if not models or target_var not in models:
        return go.Figure().update_layout(height=600)

    for i, col in enumerate(features[:len(sliders)]):
        min_val, max_val = sliders[i]
        filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
    for i, col in enumerate(features[:len(checklists)]):
        selected_values = checklists[i]
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    numeric_features = [feature for feature in features if feature in df.columns and df[feature].dtype in ['int64', 'float64']]
    if numeric_features:
        filtered_df = filtered_df.sort_values(by=numeric_features, ascending=True)
    # Get model and data for selected target
    model_data = models[target_var]
    model = model_data['model']
    X_train = model_data['X_train']
    X_test = model_data['X_test']
    X_sample = filtered_df.iloc[0:1]
    if plot_type=='shap':
        return model, X_sample, X_test
    else:
        return model, X_sample, X_train

