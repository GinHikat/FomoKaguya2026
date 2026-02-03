import pandas as pd 
import numpy as np
import os, sys
import pickle
from pathlib import Path

class ML():
    def __init__(self, model_name):
        self.model_name = model_name

    def load_archive(self):
        base_dir = Path(__file__).resolve().parent
        artifact_dir = base_dir / "artifact"

        with open(artifact_dir/f'{self.model_name}.pkl', 'rb') as f:
            model = pickle.load(f)

        return model

    def predict(self, X):

        model = self.load_archive()
        
        y_pred = model.predict(X)

        return y_pred