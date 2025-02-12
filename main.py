"""
Main File to start the Dash App. To keep it small,
the callbacks and the app layout are in different modules.
"""

from dash import Dash
from layout import create_layout
from callbacks import register_callbacks
from model_utils import train_model
from model_store import store
import pandas as pd


# Initialize Dash Server
app = Dash(__name__, suppress_callback_exceptions=True)

# Load data and train model once
df = pd.read_csv('housing.csv')
model, X_train, X_test = train_model(df)

# Store model and data in global store
store.model = model
store.X_train = X_train
store.X_test = X_test


# Set the layout of the app (see layout.py)
app.layout = create_layout()

# Register the callbacks for the app (see callbacks.py)
register_callbacks(app)

# Set App title
app.title = "XAI in MCI"

if __name__ == '__main__':
    app.run_server(debug=True)
