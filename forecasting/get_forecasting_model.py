import pandas as pd 
import numpy as np

# don't care about these 2 yet
class DeepRVFL:
    def __init__(
        self,
        input_dim,
        hidden_dims=(100, 100),
        activation=np.tanh,
        reg=1e-3,
        seed=42
    ):
        rng = np.random.default_rng(seed)

        self.activation = activation
        self.reg = reg
        self.hidden_dims = hidden_dims

        self.W = []
        self.b = []

        prev_dim = input_dim
        for hdim in hidden_dims:
            self.W.append(rng.normal(0, 1, (prev_dim, hdim)))
            self.b.append(rng.normal(0, 1, hdim))
            prev_dim = hdim

    def _forward_hidden(self, X):
        H_list = []
        H = X
        for W, b in zip(self.W, self.b):
            H = self.activation(H @ W + b)
            H_list.append(H)
        return H_list

    def fit(self, X, y):
        H_list = self._forward_hidden(X)

        Z = np.hstack([X] + H_list)

        I = np.eye(Z.shape[1])
        self.beta = np.linalg.solve(
            Z.T @ Z + self.reg * I,
            Z.T @ y
        )

    def predict(self, X):
        H_list = self._forward_hidden(X)
        Z = np.hstack([X] + H_list)
        return Z @ self.beta

class DeepEnsembleRVFL:
    def __init__(
        self,
        input_dim,
        hidden_dims=(100, 100),
        n_estimators=10,
        activation=np.tanh,
        reg=1e-3,
        seed=42
    ):
        self.models = []
        for i in range(n_estimators):
            self.models.append(
                DeepRVFL(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    activation=activation,
                    reg=reg,
                    seed=seed + i
                )
            )

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        preds = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return preds.mean(axis=1)

    def predict_interval(self, X, alpha=0.05):
        preds = np.column_stack([
            model.predict(X) for model in self.models
        ])
        lower = np.quantile(preds, alpha / 2, axis=1)
        upper = np.quantile(preds, 1 - alpha / 2, axis=1)
        return lower, upper

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

    def for_dl_input(self, 
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

        return np.array(X), y

    def process_input(self, df):
        '''
        Process from original df to 2-dim array to match with the model (will be changed later to fit with further models)

        Input:
            df: original df (from read_csv)

        Output:
            X: np.array(batch_size, num_features * window_size)
            y: (batch_size,)
        '''   
        
        X, y = self.for_dl_input(df, window_size=self.window_size)

        X = np.asarray(X).reshape(X.shape[0], -1)

        return X,y
