import streamlit as st
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.get_forecasting_model import Predictor
from forecasting.rvfl import DeepRVFL
sys.modules['__main__'].DeepRVFL = DeepRVFL

processed_test= "data/processed/test.csv"

# model selector sẽ có 1 hàm riêng 
# Expect that you run app.py in main dir

st.title("Forecasting")

def load_info(data_path, model_name, interval, window_size):
    info= {}
    df= pd.read_csv(data_path)
    predictor= Predictor(model_name, interval, window_size)
    X, y_true, y_pred = predictor.get_prediction(df)
    info["X"]= X
    info["y_true"]= y_true
    info["y_pred"]= y_pred
    return info

def predict_visualize(info):
    y_true = info["y_true"]
    y_pred= info["y_pred"]
    
    t = range(len(y_true))
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

model_name= st.selectbox("Select the model", ("bilstm_attention", "de-rvfl"))
if st.button("Submit"):
    info= load_info(processed_test, model_name, '5min', 12)
    fig= predict_visualize(info)
    st.pyplot(fig)

