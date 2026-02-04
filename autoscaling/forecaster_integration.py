import os
import sys
import pandas as pd
import numpy as np

# Add project root and forecasting dir to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
forecasting_dir = os.path.join(project_root, "forecasting")

for path in [project_root, forecasting_dir]:
    if path not in sys.path:
        sys.path.append(path)

try:
    from forecasting.get_forecasting_model import Predictor
except ImportError:
    print("Warning: Could not import Predictor from forecasting.get_forecasting_model")
    Predictor = None

class Forecaster:
    def __init__(self, model_name="bilstm_attention", interval="1min"):
        self.model_name = model_name
        self.interval = interval
        self.predictor = None
        if Predictor:
            self.predictor = Predictor(model_name, interval)

    def generate_forecasts(self, df_data):
        """
        Generate forecasts for the given data DataFrame.
        
        Args:
            df_data (pd.DataFrame): Dataframe with 'time', 'size', 'status_label' etc.
                                    matches what Forecasting Predictor expects.
        
        Returns:
            np.array: Forecast values aligned with the input data length (padded with NaNs or real values)
        """
        if not self.predictor:
            print("Predictor not available, returning dummy zeros.")
            return np.zeros(len(df_data)), np.zeros(len(df_data))
        
        try:
            X, y, y_pred = self.predictor.get_prediction(df_data)
            
            # Inverse transform: log_time -> size
            pred_size = np.exp(y_pred).flatten()
            pred_size = np.nan_to_num(pred_size)
            
            # Align predictions with aggregated data length
            df_agg = self.predictor.agg_df(df_data)
            total_len = len(df_agg)
            forecasts = np.zeros(total_len)
            valid_len = len(pred_size)
            
            if valid_len > 0:
                forecasts[-valid_len:] = pred_size
                
            return forecasts, df_agg["size"].values
            
        except Exception as e:
            print(f"Error generating forecasts: {e}")
            return np.zeros(len(df_data)), np.zeros(len(df_data))

def get_forecasts(data_path, config):
    """
    Helper to load data and get forecasts.
    """
    if not os.path.exists(data_path):
        return np.array([]), np.array([])
        
    df = pd.read_csv(data_path)
    
    # Config parameters
    model_name = config.get("forecasting", {}).get("model", "bilstm_attention")
    interval = config.get("simulation", {}).get("interval", "1min")
    
    forecaster = Forecaster(model_name=model_name, interval=interval)
    predictions, actuals = forecaster.generate_forecasts(df)
    
    return predictions, actuals
