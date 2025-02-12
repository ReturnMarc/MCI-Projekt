"""Global storage for trained model and data"""
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
import pandas as pd

@dataclass
class ModelStore:
    model: Pipeline = None
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None

# Global instance
store = ModelStore()