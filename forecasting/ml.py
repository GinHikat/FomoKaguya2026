import pandas as pd 
import numpy as np
import os, sys
import pickle
from pathlib import Path

class ML():
    def __init__(self, model_name):
        self.model_name = model_name

    def load_archive(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        artifact_path = os.path.join(current_dir, 'artifact', f'{self.model_name}.pkl')
        with open(artifact_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def predict(self, X):

        model = self.load_archive()
        
        y_pred = model.predict(X)

        return y_pred