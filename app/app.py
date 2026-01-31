import streamlit as st
import pickle
import sys
import matplotlib.pyplot as plt 
from forecasting.get_forecasting_model import Predictor, DeepRVFL
import pandas as pd 
import numpy as np 

# model selector sẽ có 1 hàm riêng 
# Expect that you run app.py in main dir
sys.modules['__main__'].DeepRVFL = DeepRVFL

st.title("Forecasting")

model_path= "forecasting/artifact/de-rvfl.pkl"
processed_test= "data/processed/test.csv"

def load_info(model_path, data_path, model_name, interval, window_size):
    info= {}
    # load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    info["model"]=model
    df= pd.read_csv(data_path)
    predictor= Predictor(model_name, interval, window_size)
    X,y= predictor.process_input(df)
    info["X"]= X
    info["y"]= y
    return info

def predict_visualize(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.exp(y_pred)
    y_true = np.exp(y)
    t = range(len(y))
    # figure

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y_true, label="Real", alpha=0.7)
    ax.plot(t, y_pred, label="Predicted", alpha=0.7)

    ax.set_title("Prediction vs Real (Time Series)")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig

info= load_info(model_path, processed_test, "de-rvfl", '5min', 12)
fig= predict_visualize(info["model"], info["X"], info["y"])

st.pyplot(fig)

