import pandas as pd 
import numpy as np
import os, sys
import pickle

class ML():
    def __init__(self, model_name):
        self.model_name = model_name

    def load_archive(self):
        with open(f'artifact/{self.model_name}.pkl', 'rb') as f:
            model = pickle.load(f)

        return model

    def predict(self, X):

        model = self.load_archive()
        
        y_pred = model.predict(X)

        return y_pred