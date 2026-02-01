import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dl import *
from rvfl import *
from ml import *

class Predictor():
    def __init__(self, model_name, interval, window_size):
        self.model_name = model_name
        self.interval = interval
        self.window_size = window_size

    def agg_df(self, df): 

        '''
        Create an aggregate dataframe based on the original input dataframe
        The resulting dataframe contains the resampled time (1min, 5min, 15min,....), the sum of file size and the number of Request status in that interval

        Input: 
            df: original df (from read_csv)

        Output: 
            df_agg: the resulting df
        '''

        interval = self.interval

        df['time'] = pd.to_datetime(df['time'])
        df['status'] = df['status'].astype(int)
        df['size'] = df['size'].astype(float)

        status_counts = (
            df
            .set_index("time")
            .groupby([pd.Grouper(freq=interval), "status_label"])
            .size()
            .unstack(fill_value=0)
        )

        size_agg = (
            df
            .set_index("time")
            .groupby(pd.Grouper(freq=interval))["size"]
            .sum()
        )

        mean_size = size_agg.median()

        df_agg = (
            size_agg
            .to_frame("size")
            .join(status_counts, how="left")
            .fillna(mean_size)
            .reset_index()
        )

        df_agg['anomaly'] = df_agg['size'] == 0

        df_agg.loc[df_agg['anomaly'], 'size'] = mean_size

        df_agg['log_time'] = np.log(df_agg['size'])

        df_agg['log_time'] = (
            df_agg['log_time']
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        return df_agg

    def moving_window(self, 
        df,
        target_col = 'log_time'):
        """
        create moving window for x and y; 
        
        Imagine with window = 60 then [0->60] predict 61, [1->61] predict 62, [2->62] predict 63

        Input:
            df: original df (from read_csv)
            target_col: the target for prediction (default is log_time, don't choose 'size')
        
        Output:

            X: input array (batch_size, window_size, num_features = 6 based on input_cols below)
            y: output array (if window size is 12 then y starts at 13 -> all) (batch_size,)
        """

        df = self.agg_df(df)

        input_cols = [
            "Error",
            "No Change",
            "Not Found",
            "Redirected",
            "Success",
            "log_time"
        ]

        X, y = [], []

        data_X = df[input_cols].values.astype(np.float32)
        data_y = df[target_col].values.astype(np.float32)


        for i in range(len(df) - self.window_size):
            X.append(data_X[i : i + self.window_size])
            y.append(data_y[i + self.window_size])

        X = np.array(X)  # (batch_size, window_size, num_features)
        y = np.array(y)  # (batch_size,)

        return X, y

    # Process data for Input of each type

    def ml_input(self, df, target = 'log_time'):
        '''
        Return df input for ML models

        Input: 
            df: original df (from read_csv)
            target: target col (default is "log_time")
        Output:
            X: df
            y: df
        '''
        df_agg = self.agg_df(df)

        lags = [1,2,3]
        windows = [4,12,30]

        for l in lags:
            df_agg[f'{target}_lag_{l}'] = df_agg[target].shift(l)

        for w in windows:
            df_agg[f'{target}_roll_mean_{w}'] = df_agg[target].shift(1).rolling(w).mean()
            df_agg[f'{target}_roll_std_{w}'] = df_agg[target].shift(1).rolling(w).std()

        df_agg["hour"] = df_agg['time'].dt.hour
        df_agg["minute"] = df_agg['time'].dt.minute
        df_agg["second"] = df_agg['time'].dt.second
        
        # Cyclical encoding
        df_agg["hour_sin"] = np.sin(2 * np.pi * df_agg["hour"] / 24)
        df_agg["hour_cos"] = np.cos(2 * np.pi * df_agg["hour"] / 24)
        
        df_agg["minute_sin"] = np.sin(2 * np.pi * df_agg["minute"] / 60)
        df_agg["minute_cos"] = np.cos(2 * np.pi * df_agg["minute"] / 60)
        
        df_agg["second_sin"] = np.sin(2 * np.pi * df_agg["second"] / 60)
        df_agg["second_cos"] = np.cos(2 * np.pi * df_agg["second"] / 60)

        df_agg.index = df_agg['time']
        
        df_agg['dayofweek'] = df_agg.index.dayofweek
        df_agg['weekofyear'] = df_agg.index.isocalendar().week.astype(int)
        df_agg['month'] = df_agg.index.month
        df_agg['is_weekend'] = (df_agg['dayofweek'] >= 5).astype(int)
        
        df_agg = df_agg.dropna()
        
        y = df_agg[[target]]
        
        X = df_agg.drop(['time', 'size', 'log_time'], axis = 1)

        return X, y

    def rvfl_input(self, df):
        '''
        Process from original df to 2-dim array to match with the model (will be changed later to fit with further models)

        Input:
            df: original df (from read_csv)

        Output:
            X: np.array(batch_size, num_features * window_size)
            y: (batch_size,)
        '''   
        
        X, y = self.moving_window(df)

        X = np.asarray(X).reshape(X.shape[0], -1)

        return X,y

    def dl_input(self, df):
        '''
        
        '''
        X, y = self.moving_window(df)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        return X_tensor, y_tensor

    def get_prediction(self, df):
        '''
        List of available models:

        1. ml-base
            - xgboost
            - lgbm

        2. rvfl-base
            - rvfl
            - d-rvfl
            - de-rvfl

        3. dl-base
            - lstm (x2)
            - bilstm (x2)
            - transformer 
            - bilstm_attention
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using:", device)

        model_type = {
            'ml': ['xgboost', 'lgbm'],
            'rvfl': ['rvfl', 'd-rvfl', 'de-rvfl'],
            'dl': ['lstm', 'bilstm', 'transformer', 'bilstm_attention']
        }

        for key, value in model_type.items():
            if self.model_name in value:
                model = key
                break
        
        # get processing function based on model type
        input_fn = getattr(self, f"{model}_input")

        X, y = input_fn(df)

        dl = DL(model_name = self.model_name, input_dim = X.shape[-1])

        y_pred = dl.predict(X)

        return X,y,y_pred

        



