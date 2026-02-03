import os 
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd 
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from forecasting.rvfl import DeepRVFL, RVFL

sys.modules['__main__'].DeepRVFL = DeepRVFL
sys.modules['__main__'].RVFL = RVFL

from forecasting.get_forecasting_model import Predictor

processed_test= "data/processed/test.csv"

# Expect that you run app.py in main dir

st.title("Forecasting")

def load_info(data_path, model_name, interval):
    info= {}
    df= pd.read_csv(data_path)
    predictor= Predictor(model_name, interval)
    X, y_true, y_pred = predictor.get_prediction(df)

    info["X"]= X
    info["y_true"]= y_true
    info["y_pred"]= y_pred
    return info

# xem lại phần scale (log, không scale) và sửa
# sarimax không scale 
def predict_visualize(info, model_name):

    y_pred = np.exp(info["y_pred"])
    y = info["y_true"]
    
    # add this to suppress warning when working with DL models
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y_true = np.exp(y)
    
    t = range(len(y_pred))
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

model_name= st.selectbox("Select the model", ("sarimax", "bilstm", "bilstm_attention", "rvfl", "d-rvfl", "de-rvfl"))
interval= st.selectbox("Select interval", ("1min", "5min", "15min"))

if st.button("Submit"):
    info= load_info(processed_test, model_name, interval)

    y_pred = np.exp(info["y_pred"])
    y = info["y_true"]

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y_true = np.exp(y)


    fig= predict_visualize(info, model_name)
    st.pyplot(fig)

    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    df= pd.DataFrame({
        'R^2 score': [r2],
        'MAPE': [mape]
    })
    st.title("Evaluation results")
    st.dataframe(df)