import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.01):
        """
        Anomaly Detector using Isolation Forest.
        
        Args:
            contamination (float): Expected proportion of outliers in the data.
        """
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False
        
    def fit(self, data):
        """
        Train the model on historical load data.
        
        Args:
            data (np.array): 1D array of load values.
        """
        # IsolationForest expects 2D array (n_samples, n_features)
        X = data.reshape(-1, 1)
        self.model.fit(X)
        self.is_fitted = True
        print("Anomaly Detector fitted.")
        
    def detect(self, value):
        """
        Check if a single load value is an anomaly.
        
        Args:
            value (float): The current load value.
            
        Returns:
            bool: True if anomaly, False otherwise.
        """
        if not self.is_fitted:
            # Default to False if not fitted to avoid crashing
            return False
            
        X = np.array([[value]])
        pred = self.model.predict(X)
        # IsolationForest returns -1 for anomaly, 1 for normal
        return pred[0] == -1
        
    def detect_batch(self, data):
        """
        Detect anomalies in a batch of data (vectorized).
        
        Args:
            data (np.array): 1D array of load values.
            
        Returns:
            np.array: Boolean array (True if anomaly).
        """
        if not self.is_fitted:
            return np.zeros(len(data), dtype=bool)
            
        X = data.reshape(-1, 1)
        pred = self.model.predict(X)
        return pred == -1
        
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Anomaly Detector saved to {path}")
        
    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            print(f"Anomaly Detector loaded from {path}")
        else:
            print(f"Model file not found at {path}")

if __name__ == "__main__":
    import pandas as pd
    import yaml
    
    # helper for loading data identical to run_simulation
    def load_data_simple(path, interval="1min"):
        if not os.path.exists(path):
            print(f"Data not found at {path}")
            return None
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df_agg = df.resample(interval)["size"].sum().fillna(0)
        return df_agg.values

    # Determine paths relative to this file
    # This file is in autoscaling/bonus/anomaly_detector.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # project root is 2 levels up: autoscaling/bonus -> autoscaling -> root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    config_path = os.path.join(project_root, "autoscaling", "config.yaml")
    train_path = os.path.join(project_root, "data", "processed", "train.csv")
    model_path = os.path.join(current_dir, "anomaly_model.joblib")
    
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loading training data from {train_path}")
    train_data = load_data_simple(train_path, interval=config["simulation"]["interval"])
    
    if train_data is not None:
        print("Training Anomaly Detector...")
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(train_data)
        detector.save(model_path)
        print("Done.")
    else:
        print("Training failed due to missing data.")
