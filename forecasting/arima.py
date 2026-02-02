import pandas as pd 
import numpy as np
import os, sys
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.arima.model import ARIMAResults

exog = ['Error', 'No Change', 'Not Found', 'Redirected', 'Success']

class ARIMAS():
    def __init__(self, model_name, output_size):
        self.model_name = model_name
        self.output_size = output_size

    def load_archive(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        artifact_path = os.path.join(current_dir, 'artifact', 'sarimax.pkl')
        model = SARIMAXResults.load(artifact_path)

        return model

    def predict(self, X):

        model = self.load_archive()
        
        y_pred = model.forecast(steps = self.output_size, exog = X[exog])

        y = np.array(y_pred)
        y[y < 0] = 0
        y = np.log1p(y)

        return y