import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import sys

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoscaling.run_simulation import load_data
from autoscaling.cost_model import CostModel
from autoscaling.simulator import Simulator
from autoscaling.policies import ReactivePolicy, PredictivePolicy, HybridPolicy, OptimalFixedPolicy
from autoscaling.visualization.charts import plot_simulation_timeline_plotly, plot_cost_comparison_plotly
from autoscaling.forecaster_integration import get_forecasts

st.set_page_config(page_title="Autoscaling Demo", layout="wide")

st.title("Autoscaling Policy Simulation")

# Config Loader
config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Sidebar Controls
st.sidebar.header("Simulation Settings")
capacity = st.sidebar.number_input("Capacity Per Server", value=config["cost_parameters"]["capacity_per_server"], step=100000)
min_servers = st.sidebar.number_input("Min Servers", value=config["simulation"]["min_servers"], min_value=1)
max_servers = st.sidebar.number_input("Max Servers", value=config["simulation"]["max_servers"], min_value=1)

# Update config object locally
config["cost_parameters"]["capacity_per_server"] = capacity
config["simulation"]["min_servers"] = min_servers
config["simulation"]["max_servers"] = max_servers

forecast_mode = st.sidebar.radio("Prediction Source", ["Oracle (Actuals)", "Real Forecast (Bi-LSTM)"])

# Run Button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running Simulation..."):
        # Load Data
        data_path = os.path.join(os.path.dirname(__file__), "../../data/processed/test.csv")
        load_series = load_data(data_path, interval="1min")
        
        # Determine predictions based on selection
        st.info(f" **Prediction Mode:** {forecast_mode}")
        
        if forecast_mode == "Oracle (Actuals)":
             predictions = load_series
             st.success("Using Oracle (perfect foresight)")
        else:
             # Real Forecast
             st.info("Loading Bi-LSTM model...")
             # ensure forecasting config
             if "forecasting" not in config:
                  config["forecasting"] = {"model": "bilstm_attention", "window_size": 60}
             
             full_data_path = os.path.abspath(data_path)
             forecasts, aligned_actuals = get_forecasts(full_data_path, config)
             
             if len(forecasts) > 0:
                 predictions = forecasts
                 min_len = min(len(predictions), len(load_series))
                 pred_diff = np.abs(predictions[:min_len] - load_series[:min_len]).mean()
                 st.success(f"Bi-LSTM predictions loaded. Mean absolute error: {pred_diff:,.0f} Bytes")   
             else:
                 st.warning("Forecasting failed. Using Oracle.")
                 predictions = load_series

        cost_model = CostModel(config)
        
        policies = {
            "Reactive": ReactivePolicy(config),
            "Predictive": PredictivePolicy(config),
            "Hybrid": HybridPolicy(config),
            "Fixed": OptimalFixedPolicy(config, fixed_servers=max_servers)
        }
        
        sim = Simulator(config, cost_model, policies)
        results = {}
        
        for name in policies:
            results[name] = sim.run_scenario(name, load_series, predictions)
            
        # Analysis
        summary_rows = []
        for name, res in results.items():
            metrics = cost_model.calculate_total_cost(res)
            # Rename for display compatibility with charts.py
            display_metrics = {
                "Policy": name,
                "Total Cost": metrics["total_cost"],
                "Server Cost": metrics["server_cost"],
                "Scale Cost": metrics["scale_cost"],
                "Violation Cost": metrics["violation_cost"],
                "Dropped Reqs": metrics["total_dropped_requests"]
            }
            summary_rows.append(display_metrics)
            
        summary_df = pd.DataFrame(summary_rows)
        
        # Display Metrics
        st.subheader("Performance Metrics")
        st.dataframe(summary_df.style.format({
            "Total Cost": "${:.2f}",
            "Server Cost": "${:.2f}",
            "Violation Cost": "${:.2f}",
            "Scale Cost": "${:.2f}"
        }))
        
        # Visualizations
        st.subheader("Cost Breakdown")

        
        # Timeline
        st.subheader("Scaling Timeline")
        st.plotly_chart(plot_simulation_timeline_plotly(results, load_series, predictions=predictions), key="timeline", on_select="ignore")
        
        # Cost Chart
        st.plotly_chart(plot_cost_comparison_plotly(summary_df), key="cost_chart", on_select="ignore")

else:
    st.info("Click 'Run Simulation' to start.")
