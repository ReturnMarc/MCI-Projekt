"""
Main File to start the Dash App. To keep it small,
the callbacks and the app layout are in different modules.
"""
import dash_bootstrap_components.themes
from dash import Dash
from layout import create_layout
from callbacks import register_callbacks
from model_utils import train_all_datasets
import os


# Initialize Dash Server
app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

# Create stored_models directory if it doesn't exist
os.makedirs('stored_models', exist_ok=True)

# Train models for any new datasets
trained_datasets = train_all_datasets()

# Set the layout of the app (see layout.py)
app.layout = create_layout()

# Register the callbacks for the app (see callbacks.py)
register_callbacks(app)

# Set App title
app.title = "XAI in MCI"

if __name__ == '__main__':
    app.run(debug=False)