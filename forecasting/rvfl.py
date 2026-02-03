import pandas as pd 
import numpy as np
import os, sys
import pickle
from pathlib import Path

class RVFL:
    def __init__(self, input_dim, hidden_dim=100, 
                 activation=np.tanh, reg=1e-3, seed=42):
        rng = np.random.default_rng(seed)

        self.W = rng.normal(0, 1, (input_dim, hidden_dim))
        self.b = rng.normal(0, 1, hidden_dim)
        self.activation = activation
        self.reg = reg

    def _hidden(self, X):
        return self.activation(X @ self.W + self.b)

    def fit(self, X, y):
        H = self._hidden(X)
        Z = np.hstack([X, H])  # direct link

        I = np.eye(Z.shape[1])
        self.beta = np.linalg.solve(
            Z.T @ Z + self.reg * I,
            Z.T @ y
        )

    def predict(self, X):
        H = self._hidden(X)
        Z = np.hstack([X, H])
        return Z @ self.beta

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

class rvfl():
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


    