import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import sys
import time

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoscaling.run_simulation import load_data
from autoscaling.cost_model import CostModel
from autoscaling.simulator import Simulator
from autoscaling.policies import ReactivePolicy, PredictivePolicy, HybridPolicy, OptimalFixedPolicy
from autoscaling.visualization.charts import plot_simulation_timeline_plotly, plot_cost_comparison_plotly
from autoscaling.forecaster_integration import get_forecasts
from autoscaling.bonus.anomaly_detector import AnomalyDetector

st.set_page_config(page_title="Autoscaling Control Center", layout="wide")

# --- CSS Styling for "Premium" Feel ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4e8cff;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Autoscaling Control Center")

# --- Config Loader ---
config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("Cluster Settings", expanded=True):
        capacity = st.number_input("Capacity Per Server (Bytes/min)", value=config["cost_parameters"]["capacity_per_server"], step=100000)
        min_servers = st.number_input("Min Servers", value=config["simulation"]["min_servers"], min_value=1)
        max_servers = st.number_input("Max Servers", value=config["simulation"]["max_servers"], min_value=1)
        
    with st.expander("Forecasting Model"):
        forecast_mode = st.radio("Prediction Source", ["Oracle (Actuals)", "Real Forecast (Bi-LSTM)"])
        
    # Update config locally
    config["cost_parameters"]["capacity_per_server"] = capacity
    config["simulation"]["min_servers"] = min_servers
    config["simulation"]["max_servers"] = max_servers

# --- Data Loading Helper ---
@st.cache_data
def load_and_prep_data():
    data_path = os.path.join(os.path.dirname(__file__), "../../data/processed/test.csv")
    load_series = load_data(data_path, interval="1min")
    return load_series

load_series = load_and_prep_data()
data_path = os.path.join(os.path.dirname(__file__), "../../data/processed/test.csv")

# --- Prediction Logic ---
if forecast_mode == "Oracle (Actuals)":
     predictions = load_series
else:
     # Real Forecast
     if "forecasting" not in config:
          config["forecasting"] = {"model": "bilstm_attention"}
     
     full_data_path = os.path.abspath(data_path)
     
     # Cache predictions to avoid re-running DL model on every rerun
     @st.cache_data
     def cached_forecast(f_path, f_config):
         return get_forecasts(f_path, f_config)
         
     forecasts, aligned_actuals = cached_forecast(full_data_path, config)
     
     if len(forecasts) > 0:
         # PIs Logic (inline for dashboard consistency)
         residuals = aligned_actuals - forecasts
         std_resid = np.std(residuals)
         predictions = {
            "mean": forecasts,
            "upper": forecasts + 1.96 * std_resid,
            "lower": np.maximum(forecasts - 1.96 * std_resid, 0)
         }
     else:
         st.sidebar.warning("Forecasting failed. Using Oracle.")
         predictions = load_series

# --- Anomaly Detection ---
@st.cache_resource
def load_anomaly_detector(data):
    detector = AnomalyDetector(contamination=0.05)
    model_path = os.path.join(os.path.dirname(__file__), "../bonus/anomaly_model.joblib")
    if os.path.exists(model_path):
        detector.load(model_path)
    else:
        detector.fit(data)
    return detector

detector = load_anomaly_detector(load_series)
anomalies = detector.detect_batch(load_series)
st.sidebar.info(f"Anomaly Detection Active: {np.sum(anomalies)} anomalies found.")

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["Overview & Analysis", "Live Simulation", "Sensitivity Lab"])

# ==========================================
# TAB 1: Overview (Static Analysis)
# ==========================================
with tab1:
    if st.button("Run Full Analysis", type="primary"):
        with st.spinner("Simulating Policies..."):
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
                results[name] = sim.run_scenario(name, load_series, predictions, anomalies=anomalies)
                
            # Metrics
            summary_rows = []
            for name, res in results.items():
                metrics = cost_model.calculate_total_cost(res)
                summary_rows.append({
                    "Policy": name,
                    "Total Cost": metrics["total_cost"],
                    "Server Cost": metrics["server_cost"],
                    "Scale Cost": metrics["scale_cost"],
                    "Violation Cost": metrics["violation_cost"],
                    "Dropped Reqs": metrics["total_dropped_requests"]
                })
            summary_df = pd.DataFrame(summary_rows)
            
            # Top Stats
            best_policy = summary_df.loc[summary_df["Total Cost"].idxmin()]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Policy", best_policy["Policy"])
            c2.metric("Lowest Cost", f"${best_policy['Total Cost']:.2f}")
            c3.metric("Avg Violations", f"${summary_df['Violation Cost'].mean():.2f}")
            
            st.plotly_chart(plot_cost_comparison_plotly(summary_df), key="cost_chart", on_select="ignore")
            st.plotly_chart(plot_simulation_timeline_plotly(results, load_series, predictions=predictions, anomalies=anomalies), key="timeline", on_select="ignore")

# ==========================================
# TAB 2: Live Simulation (Animation)
# ==========================================
with tab2:
    st.markdown("### Real-time Server Monitor")
    st.markdown("Simulate the cluster reacting to traffic in real-time.")
    
    col_sel, col_speed = st.columns([1, 2])
    selected_policy_name = col_sel.selectbox("Select Policy to Watch", ["Hybrid", "Reactive", "Predictive"])
    sim_speed = col_speed.slider("Simulation Speed (Steps/sec)", 1, 50, 10)
    
    if st.button("Start Live Simulation"):
        # Setup Simulation
        cost_model = CostModel(config)
        
        policy_map = {
            "Hybrid": HybridPolicy(config),
            "Reactive": ReactivePolicy(config),
            "Predictive": PredictivePolicy(config)
        }
        policy = policy_map[selected_policy_name]
        
        # We need to run step-by-step manually or pre-compute and replay.
        # Replaying is smoother for Streamlit.
        sim = Simulator(config, cost_model, {selected_policy_name: policy})
        full_results = sim.run_scenario(selected_policy_name, load_series, predictions, anomalies=anomalies)
        
        # Animation Loop
        placeholder = st.empty()
        bar = st.progress(0)
        
        # We simulate a window of the timeline (e.g., the gap area ~ minute 400-500)
        start_t = 350
        end_t = 550
        subset_results = full_results[start_t:end_t]
        
        total_steps = len(subset_results)
        
        for i, step in enumerate(subset_results):
            # Update Progress
            bar.progress((i + 1) / total_steps)
            
            # Extract metrics
            t = step["time_step"]
            load = step["load"]
            servers = step["servers_active"] # Active
            provisioned = step["servers_provisioned"]
            booting = provisioned - servers
            
            # construct status box
            with placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                
                c1.metric("Time", f"{t} min")
                c2.metric("Current Load", f"{load:.0f} B/m", delta=f"{load - subset_results[i-1]['load'] if i > 0 else 0:.0f}")
                
                c3.metric("Active Servers", f"{servers}", delta=f"{booting} booting" if booting > 0 else None)
                
                # Visualizing Server Racks
                rack_html = ""
                for _ in range(servers):
                    rack_html += "<span style='font-size: 2em; color: #4CAF50;'>■</span> " # Green Box
                for _ in range(booting):
                    rack_html += "<span style='font-size: 2em; color: #FFC107;'>□</span> " # Yellow Box (Hollow)
                for _ in range(max_servers - provisioned):
                    rack_html += "<span style='font-size: 2em; color: #E0E0E0;'>.</span> " # Gray Dot
                
                st.markdown(f"**Cluster Status:**<br>{rack_html}", unsafe_allow_html=True)
                
                # Check for gap
                if load == 0:
                    st.error("DATA GAP DETECTED - 0 TRAFFIC")
                elif booting > 0:
                    st.warning("SCALING UP - Servers Booting...")
                else:
                    st.success("System Stable")
                    
            time.sleep(1 / sim_speed)
            
        st.success("Simulation Sequence Complete.")

# ==========================================
# TAB 3: Sensitivity Lab
# ==========================================
with tab3:
    st.header("Sensitivity Analysis Lab")
    st.markdown("Experiment with **Violation Costs** and **Startup Time** to see impact on Hybrid Policy.")
    
    col1, col2 = st.columns(2)
    with col1:
        s_violation_cost = st.slider("Violation Cost ($/Byte)", 1e-6, 5e-5, config["cost_parameters"]["violation_cost"], step=1e-6, format="%.6f")
    with col2:
        s_startup_time = st.slider("Server Startup Time (min)", 1, 10, config["cost_parameters"]["startup_time"])
        
    if st.button("Run What-If Analysis"):
        wi_config = config.copy()
        wi_config["cost_parameters"] = config["cost_parameters"].copy()
        wi_config["cost_parameters"]["violation_cost"] = s_violation_cost
        wi_config["cost_parameters"]["startup_time"] = s_startup_time
        
        wi_cost_model = CostModel(wi_config)
        wi_policy = HybridPolicy(wi_config)
        wi_sim = Simulator(wi_config, wi_cost_model, {"Hybrid": wi_policy})
        
        res = wi_sim.run_scenario("Hybrid", load_series, predictions, anomalies=anomalies)
        metrics = wi_cost_model.calculate_total_cost(res)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"${metrics['total_cost']:.2f}")
        c2.metric("Violations", f"${metrics['violation_cost']:.2f}")
        c3.metric("Dropped Reqs", f"{metrics['total_dropped_requests']:,}")
        
        st.success("Analysis Complete.")
