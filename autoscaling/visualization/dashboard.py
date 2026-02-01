import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import sys
import time
import json
from typing import Dict, Any

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoscaling.run_simulation import load_data
from autoscaling.cost_model import CostModel
from autoscaling.simulator import Simulator
from autoscaling.policies import ReactivePolicy, PredictivePolicy, HybridPolicy, OptimalFixedPolicy
from autoscaling.dp_optimizer import DPOptimizer
from autoscaling.visualization.charts import plot_simulation_timeline_plotly, plot_cost_comparison_plotly
from autoscaling.forecaster_integration import get_forecasts
from autoscaling.bonus.anomaly_detector import AnomalyDetector

st.set_page_config(page_title="Autoscaling Control Center", layout="wide", initial_sidebar_state="expanded")

# --- Premium CSS Styling with Glassmorphism ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        margin-bottom: 10px;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 51, 234, 0.05) 100%);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .policy-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        display: inline-block;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        display: inline-block;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        display: inline-block;
    }
    
    .server-active {
        font-size: 2.5em;
        color: #10b981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    
    .server-booting {
        font-size: 2.5em;
        color: #f59e0b;
        text-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
    }
    
    .server-inactive {
        font-size: 2.5em;
        color: #e5e7eb;
        opacity: 0.3;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚡ Autoscaling Control Center</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent workload management with dynamic resource allocation</p>', unsafe_allow_html=True)

# --- Config Loader with Error Handling ---
try:
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("⚠️ Configuration file not found. Please ensure config.yaml exists.")
    st.stop()
except yaml.YAMLError as e:
    st.error(f"⚠️ Error parsing configuration: {e}")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    with st.expander("🖥️ Cluster Settings", expanded=True):
        capacity = st.number_input(
            "Capacity Per Server (Bytes/min)", 
            value=config["cost_parameters"]["capacity_per_server"], 
            step=100000,
            help="Maximum traffic each server can handle per minute"
        )
        min_servers = st.number_input(
            "Min Servers", 
            value=config["simulation"]["min_servers"], 
            min_value=1,
            help="Minimum number of servers to keep running"
        )
        max_servers = st.number_input(
            "Max Servers", 
            value=config["simulation"]["max_servers"], 
            min_value=1,
            help="Maximum number of servers allowed"
        )
        
    with st.expander("🔮 Forecasting Model"):
        forecast_mode = st.radio(
            "Prediction Source", 
            ["Oracle (Actuals)", "Real Forecast (Bi-LSTM)"],
            help="Oracle uses actual values (perfect foresight), Bi-LSTM uses trained forecasting model"
        )
        
    # Update config locally
    config["cost_parameters"]["capacity_per_server"] = capacity
    config["simulation"]["min_servers"] = min_servers
    config["simulation"]["max_servers"] = max_servers
    
    st.divider()
    
    # Info Section
    with st.expander("ℹ️ About Policies"):
        st.markdown("""
        **Reactive**: Scales based on current load
        
        **Predictive**: Uses forecasts to scale proactively
        
        **Hybrid**: Combines both with anomaly detection
        
        **DP Optimal**: Mathematical optimization for minimum cost
        
        **Fixed**: Always runs max servers (baseline)
        """)

# --- Data Loading Helper with Error Handling ---
@st.cache_data
def load_and_prep_data():
    """Load and prepare the traffic data."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), "../../data/processed/test.csv")
        load_series = load_data(data_path, interval="1min")
        if len(load_series) == 0:
            raise ValueError("Loaded data is empty")
        return load_series
    except FileNotFoundError:
        st.error("⚠️ Data file not found. Please ensure test.csv exists in data/processed/")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        st.stop()

load_series = load_and_prep_data()
data_path = os.path.join(os.path.dirname(__file__), "../../data/processed/test.csv")

# --- Prediction Logic with Error Handling ---
predictions = None
forecasts = None
aligned_actuals = None

if forecast_mode == "Oracle (Actuals)":
    predictions = load_series
    st.sidebar.success("✅ Using Oracle mode (perfect predictions)")
else:
    # Real Forecast
    if "forecasting" not in config:
        config["forecasting"] = {"model": "bilstm_attention", "window_size": 60}
    
    full_data_path = os.path.abspath(data_path)
    
    # Cache predictions to avoid re-running DL model on every rerun
    # Use immutable cache key (JSON string) instead of mutable dict
    @st.cache_data
    def cached_forecast(f_path, f_config_json):
        """Cached forecast function with immutable key."""
        f_config = json.loads(f_config_json)
        try:
            return get_forecasts(f_path, f_config)
        except Exception as e:
            st.warning(f"⚠️ Forecasting failed: {e}")
            return None, None
    
    config_json = json.dumps(config, sort_keys=True)
    forecasts, aligned_actuals = cached_forecast(full_data_path, config_json)
    
    if forecasts is not None and len(forecasts) > 0:
        # Calculate prediction intervals
        residuals = aligned_actuals - forecasts
        std_resid = np.std(residuals)
        predictions = {
            "mean": forecasts,
            "upper": forecasts + 1.96 * std_resid,
            "lower": np.maximum(forecasts - 1.96 * std_resid, 0)
        }
        st.sidebar.success("✅ Using Bi-LSTM forecasts with 95% confidence intervals")
    else:
        st.sidebar.warning("⚠️ Forecasting failed. Falling back to Oracle mode.")
        predictions = load_series

# --- Anomaly Detection with Error Handling ---
@st.cache_resource
def load_anomaly_detector(data):
    """Load or train anomaly detector."""
    try:
        detector = AnomalyDetector(contamination=0.05)
        model_path = os.path.join(os.path.dirname(__file__), "../bonus/anomaly_model.joblib")
        if os.path.exists(model_path):
            detector.load(model_path)
        else:
            detector.fit(data)
        return detector, True
    except Exception as e:
        st.sidebar.warning(f"⚠️ Anomaly detection unavailable: {e}")
        return None, False

detector, anomaly_available = load_anomaly_detector(load_series)
anomalies = None

if anomaly_available and detector is not None:
    try:
        anomalies = detector.detect_batch(load_series)
        num_anomalies = np.sum(anomalies)
        if num_anomalies > 0:
            st.sidebar.info(f"🔍 Anomaly Detection: **{num_anomalies}** anomalies detected ({num_anomalies/len(load_series)*100:.1f}%)")
        else:
            st.sidebar.success("✅ No anomalies detected in dataset")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Anomaly detection failed: {e}")
        anomalies = None

# --- Helper Functions ---
def calculate_savings(baseline_cost, policy_cost):
    """Calculate cost savings percentage."""
    if baseline_cost == 0:
        return 0
    return ((baseline_cost - policy_cost) / baseline_cost) * 100

def format_currency(value):
    """Format currency values."""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format percentage values."""
    return f"{value:+.1f}%"

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Overview & Analysis", "🔴 Live Simulation", "🧪 Sensitivity Lab"])

# ==========================================
# TAB 1: Overview (Static Analysis)
# ==========================================
@st.cache_data
def run_dp_optimization(config_json, data_series):
    """Run DP optimization with immutable cache key."""
    config_dict = json.loads(config_json)
    try:
        cm = CostModel(config_dict)
        dp = DPOptimizer(config_dict, cm, data_series)
        dp.optimize()
        return dp.reconstruct_path(), None
    except Exception as e:
        return None, str(e)

with tab1:
    st.markdown("### 📈 Policy Performance Analysis")
    st.markdown("Compare different autoscaling strategies and identify the optimal approach for your workload.")
    
    if st.button("🚀 Run Full Analysis", type="primary", width='stretch'):
        with st.spinner("⏳ Simulating policies and optimizing..."):
            try:
                cost_model = CostModel(config)
                
                # Fix: Use min_servers as baseline for Fixed policy
                policies = {
                    "Reactive": ReactivePolicy(config),
                    "Predictive": PredictivePolicy(config),
                    "Hybrid": HybridPolicy(config),
                    "Fixed": OptimalFixedPolicy(config, fixed_servers=min_servers)
                }
                
                sim = Simulator(config, cost_model, policies)
                results = {} 
                
                # Run simulations with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, name in enumerate(policies):
                    status_text.text(f"Running {name} policy...")
                    results[name] = sim.run_scenario(name, load_series, predictions, anomalies=anomalies)
                    progress_bar.progress((idx + 1) / (len(policies) + 1))
                
                # Run DP Optimal
                status_text.text("Running DP Optimization...")
                config_json = json.dumps(config, sort_keys=True)
                dp_results, dp_error = run_dp_optimization(config_json, load_series)
                
                if dp_results is not None:
                    results["DP Optimal"] = dp_results
                else:
                    st.warning(f"⚠️ DP Optimization failed: {dp_error}")
                
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                
                # Calculate Metrics
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
                
                # Find best and baseline policies
                best_policy = summary_df.loc[summary_df["Total Cost"].idxmin()]
                baseline_policy = summary_df[summary_df["Policy"] == "Fixed"].iloc[0] if "Fixed" in summary_df["Policy"].values else best_policy
                
                # Top Stats with Enhanced Metrics
                st.markdown("#### 🏆 Key Performance Indicators")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🥇 Best Policy", 
                        best_policy["Policy"],
                        help="Policy with lowest total cost"
                    )
                
                with col2:
                    savings = calculate_savings(baseline_policy["Total Cost"], best_policy["Total Cost"])
                    st.metric(
                        "💰 Cost Savings",
                        format_currency(best_policy["Total Cost"]),
                        f"{savings:.1f}% vs Fixed",
                        delta_color="inverse",
                        help="Total cost of best policy vs baseline"
                    )
                
                with col3:
                    avg_violations = summary_df["Violation Cost"].mean()
                    st.metric(
                        "⚠️ Avg Violations",
                        format_currency(avg_violations),
                        help="Average violation cost across all policies"
                    )
                
                with col4:
                    total_dropped = summary_df["Dropped Reqs"].sum()
                    st.metric(
                        "📉 Total Dropped",
                        f"{int(total_dropped):,}",
                        help="Total requests dropped across all policies"
                    )
                
                st.divider()
                
                # Enhanced Summary Table
                st.markdown("#### 📋 Detailed Cost Breakdown")
                
                # Add calculated columns
                summary_df["Savings vs Fixed"] = summary_df["Total Cost"].apply(
                    lambda x: format_percentage(calculate_savings(baseline_policy["Total Cost"], x))
                )
                summary_df["Cost/Request"] = (summary_df["Total Cost"] / len(load_series)).apply(format_currency)
                
                # Format display dataframe
                display_df = summary_df.copy()
                for col in ["Total Cost", "Server Cost", "Scale Cost", "Violation Cost"]:
                    display_df[col] = display_df[col].apply(format_currency)
                display_df["Dropped Reqs"] = display_df["Dropped Reqs"].apply(lambda x: f"{int(x):,}")
                
                st.dataframe(
                    display_df,
                    width='stretch',
                    hide_index=True
                )
                
                # Visualizations
                st.markdown("#### 📊 Cost Comparison")
                cost_chart = plot_cost_comparison_plotly(summary_df)
                if cost_chart:
                    st.plotly_chart(cost_chart, key="cost_chart", width='stretch')
                
                st.markdown("#### 📈 Simulation Timeline")
                st.markdown("Interactive view of traffic load, forecasts, anomalies, and server allocation over time.")
                timeline_chart = plot_simulation_timeline_plotly(
                    results, 
                    load_series, 
                    predictions=predictions, 
                    anomalies=anomalies
                )
                st.plotly_chart(timeline_chart, key="timeline", width='stretch')
                
                # Insights Section
                st.markdown("#### 💡 Insights & Recommendations")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.markdown(f"**Best Performer**: {best_policy['Policy']}")
                    st.markdown(f"- Total Cost: {format_currency(best_policy['Total Cost'])}")
                    st.markdown(f"- Dropped Requests: {int(best_policy['Dropped Reqs']):,}")
                    st.markdown(f"- Savings: {calculate_savings(baseline_policy['Total Cost'], best_policy['Total Cost']):.1f}% vs baseline")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with insights_col2:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    worst_policy = summary_df.loc[summary_df["Total Cost"].idxmax()]
                    st.markdown(f"**Worst Performer**: {worst_policy['Policy']}")
                    st.markdown(f"- Total Cost: {format_currency(worst_policy['Total Cost'])}")
                    st.markdown(f"- Dropped Requests: {int(worst_policy['Dropped Reqs']):,}")
                    st.markdown(f"- Additional Cost: {calculate_savings(worst_policy['Total Cost'], best_policy['Total Cost']):.1f}% more expensive")
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"⚠️ Error during analysis: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

# ==========================================
# TAB 2: Live Simulation (Animation)
# ==========================================
with tab2:
    st.markdown("### 🎬 Real-Time Server Monitor")
    st.markdown("Watch the autoscaling system respond to traffic in real-time with live server allocation visualization.")
    
    col_sel, col_speed = st.columns([1, 2])
    selected_policy_name = col_sel.selectbox(
        "Select Policy to Watch", 
        ["Hybrid", "Reactive", "Predictive", "DP Optimal"],
        help="Choose which autoscaling policy to visualize"
    )
    sim_speed = col_speed.slider(
        "Simulation Speed (Steps/sec)", 
        1, 50, 10,
        help="Higher values = faster simulation"
    )
    
    # Simulation window controls
    st.markdown("**⏱️ Simulation Window**")
    col_start, col_duration = st.columns(2)
    with col_start:
        window_start = st.number_input(
            "Start at minute", 
            min_value=0, 
            max_value=max(0, len(load_series) - 20),
            value=350,
            step=10,
            help="Starting time for simulation window"
        )
    with col_duration:
        window_duration = st.slider(
            "Duration (minutes)",
            min_value=1,
            max_value=20,
            value=10,
            help="How many minutes to simulate (1-20 min)"
        )
    
    start_simulation = st.button("▶️ Start Live Simulation", type="primary", width='stretch')
    
    if start_simulation:
        try:
            # Setup Simulation
            cost_model = CostModel(config)
            
            # Handle DP Optimal differently - it requires full DP optimization
            if selected_policy_name == "DP Optimal":
                with st.spinner("⏳ Running DP Optimization (this may take 1-2 minutes)..."):
                    config_json = json.dumps(config, sort_keys=True)
                    full_results, dp_error = run_dp_optimization(config_json, load_series)
                    
                    if full_results is None:
                        st.error(f"⚠️ DP Optimization failed: {dp_error}")
                        st.stop()
            else:
                # Regular policy simulation
                policy_map = {
                    "Hybrid": HybridPolicy(config),
                    "Reactive": ReactivePolicy(config),
                    "Predictive": PredictivePolicy(config)
                }
                policy = policy_map[selected_policy_name]
                
                # Run full simulation
                sim = Simulator(config, cost_model, {selected_policy_name: policy})
                with st.spinner(f"⏳ Running {selected_policy_name} simulation..."):
                    full_results = sim.run_scenario(selected_policy_name, load_series, predictions, anomalies=anomalies)
            
            # Animation Loop
            placeholder = st.empty()
            bar = st.progress(0)
            
            # Define window with user-specified duration
            start_t = max(0, int(window_start))
            end_t = min(len(full_results), start_t + int(window_duration))
            subset_results = full_results[start_t:end_t]
            
            total_steps = len(subset_results)
            accumulated_cost = 0
            
            # Show simulation info
            st.info(f"📊 Simulating **{selected_policy_name}** from minute **{start_t}** to **{end_t}** ({total_steps} steps)")
            
            for i, step in enumerate(subset_results):
                # Update Progress
                bar.progress((i + 1) / total_steps)
                
                # Extract metrics with bounds checking
                t = step["time_step"]
                load = step["load"]
                servers = step["servers_active"]
                provisioned = step["servers_provisioned"]
                booting = max(0, provisioned - servers)
                
                # Accumulate cost
                if "costs" in step:
                    accumulated_cost += step["costs"].get("total", 0)
                
                # Calculate capacity
                total_capacity = servers * capacity
                utilization = (load / total_capacity * 100) if total_capacity > 0 else 0
                headroom = max(0, total_capacity - load)
                
                # Construct status display
                with placeholder.container():
                    # Metrics Row
                    metric_cols = st.columns(5)
                    
                    # Fix: Proper delta calculation with bounds checking
                    prev_load = subset_results[i-1]['load'] if i > 0 else load
                    load_delta = load - prev_load
                    
                    metric_cols[0].metric("⏰ Time", f"{t} min")
                    metric_cols[1].metric(
                        "📊 Current Load", 
                        f"{load:,.0f} B/m",
                        f"{load_delta:+,.0f}" if i > 0 else None
                    )
                    metric_cols[2].metric("🖥️ Active Servers", f"{servers}", f"+{booting} booting" if booting > 0 else None)
                    metric_cols[3].metric("📈 Utilization", f"{utilization:.1f}%")
                    metric_cols[4].metric("💵 Accumulated Cost", format_currency(accumulated_cost))
                    
                    st.divider()
                    
                    # Server Rack Visualization
                    st.markdown("**🏢 Cluster Status**")
                    
                    rack_html = ""
                    for _ in range(servers):
                        rack_html += '<span class="server-active">■</span> '
                    for _ in range(booting):
                        rack_html += '<span class="server-booting">□</span> '
                    for _ in range(max(0, max_servers - provisioned)):
                        rack_html += '<span class="server-inactive">·</span> '
                    
                    st.markdown(rack_html, unsafe_allow_html=True)
                    
                    # Capacity Bar
                    st.markdown("**⚡ Capacity Buffer**")
                    capacity_pct = min(100, (headroom / total_capacity * 100) if total_capacity > 0 else 0)
                    st.progress(capacity_pct / 100)
                    st.caption(f"{headroom:,.0f} B/m headroom ({capacity_pct:.1f}% buffer)")
                    
                    # Status Alerts
                    if load == 0:
                        st.error("🚫 DATA GAP DETECTED - Zero traffic")
                    elif anomalies is not None and t < len(anomalies) and anomalies[t]:
                        st.error("🚨 ANOMALY DETECTED - Traffic spike!")
                    elif booting > 0:
                        st.warning(f"⚠️ SCALING UP - {booting} servers booting...")
                    elif utilization > 90:
                        st.warning("⚠️ HIGH UTILIZATION - Nearing capacity")
                    elif utilization < 30:
                        st.info("ℹ️ LOW UTILIZATION - Consider scaling down")
                    else:
                        st.success("✅ System Stable")
                    
                time.sleep(1 / sim_speed)
            
            bar.empty()
            st.success(f"✅ Simulation Complete! Total Cost: {format_currency(accumulated_cost)}")
            
        except Exception as e:
            st.error(f"⚠️ Simulation error: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

# ==========================================
# TAB 3: Sensitivity Lab
# ==========================================
with tab3:
    st.markdown("### 🧪 Sensitivity Analysis Lab")
    st.markdown("Experiment with cost parameters and startup time to understand their impact on the Hybrid Policy performance.")
    
    col1, col2 = st.columns(2)
    with col1:
        s_violation_cost = st.slider(
            "💰 Violation Cost ($/Byte)", 
            1e-6, 5e-5, 
            config["cost_parameters"]["violation_cost"], 
            step=1e-6, 
            format="%.6f",
            help="Cost penalty per byte of dropped traffic"
        )
    with col2:
        s_startup_time = st.slider(
            "⏱️ Server Startup Time (min)", 
            1, 10, 
            config["cost_parameters"]["startup_time"],
            help="Time for new servers to become active"
        )
    
    # Comparison mode
    comparison_mode = st.checkbox("📊 Compare with baseline", value=True, help="Show side-by-side comparison with current config")
    
    if st.button("🔬 Run What-If Analysis", type="primary", width='stretch'):
        try:
            with st.spinner("⏳ Running what-if scenario..."):
                # Create modified config
                wi_config = config.copy()
                wi_config["cost_parameters"] = config["cost_parameters"].copy()
                wi_config["cost_parameters"]["violation_cost"] = s_violation_cost
                wi_config["cost_parameters"]["startup_time"] = s_startup_time
                
                wi_cost_model = CostModel(wi_config)
                wi_policy = HybridPolicy(wi_config)
                wi_sim = Simulator(wi_config, wi_cost_model, {"Hybrid": wi_policy})
                
                wi_res = wi_sim.run_scenario("Hybrid", load_series, predictions, anomalies=anomalies)
                wi_metrics = wi_cost_model.calculate_total_cost(wi_res)
                
                # Baseline comparison
                if comparison_mode:
                    baseline_cost_model = CostModel(config)
                    baseline_policy = HybridPolicy(config)
                    baseline_sim = Simulator(config, baseline_cost_model, {"Hybrid": baseline_policy})
                    baseline_res = baseline_sim.run_scenario("Hybrid", load_series, predictions, anomalies=anomalies)
                    baseline_metrics = baseline_cost_model.calculate_total_cost(baseline_res)
                
                st.markdown("#### 📊 Results")
                
                if comparison_mode:
                    # Side-by-side comparison
                    col_baseline, col_whatif = st.columns(2)
                    
                    with col_baseline:
                        st.markdown("**📌 Baseline Configuration**")
                        st.metric("Total Cost", format_currency(baseline_metrics['total_cost']))
                        st.metric("Server Cost", format_currency(baseline_metrics['server_cost']))
                        st.metric("Violation Cost", format_currency(baseline_metrics['violation_cost']))
                        st.metric("Dropped Requests", f"{int(baseline_metrics['total_dropped_requests']):,}")
                    
                    with col_whatif:
                        st.markdown("**🧪 What-If Scenario**")
                        delta_total = wi_metrics['total_cost'] - baseline_metrics['total_cost']
                        delta_violations = wi_metrics['violation_cost'] - baseline_metrics['violation_cost']
                        delta_dropped = wi_metrics['total_dropped_requests'] - baseline_metrics['total_dropped_requests']
                        
                        st.metric(
                            "Total Cost", 
                            format_currency(wi_metrics['total_cost']),
                            f"{delta_total:+.2f}",
                            delta_color="inverse"
                        )
                        st.metric(
                            "Server Cost",
                            format_currency(wi_metrics['server_cost'])
                        )
                        st.metric(
                            "Violation Cost",
                            format_currency(wi_metrics['violation_cost']),
                            f"{delta_violations:+.2f}",
                            delta_color="inverse"
                        )
                        st.metric(
                            "Dropped Requests",
                            f"{int(wi_metrics['total_dropped_requests']):,}",
                            f"{int(delta_dropped):+,}",
                            delta_color="inverse"
                        )
                    
                    # Impact Summary
                    st.divider()
                    st.markdown("#### 💡 Impact Summary")
                    
                    pct_change = calculate_savings(baseline_metrics['total_cost'], wi_metrics['total_cost'])
                    
                    if pct_change > 5:
                        st.error(f"⚠️ **Increased cost by {abs(pct_change):.1f}%** - Consider reverting changes")
                    elif pct_change < -5:
                        st.success(f"✅ **Reduced cost by {abs(pct_change):.1f}%** - Recommended configuration!")
                    else:
                        st.info(f"ℹ️ **Minimal impact ({pct_change:+.1f}%)** - Changes are cost-neutral")
                    
                else:
                    # Simple display
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Total Cost", format_currency(wi_metrics['total_cost']))
                    metric_cols[1].metric("Server Cost", format_currency(wi_metrics['server_cost']))
                    metric_cols[2].metric("Violations", format_currency(wi_metrics['violation_cost']))
                    metric_cols[3].metric("Dropped Reqs", f"{int(wi_metrics['total_dropped_requests']):,}")
                
                st.success("✅ Analysis Complete!")
                
        except Exception as e:
            st.error(f"⚠️ Analysis error: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown(
    '<p style="text-align: center; color: #6b7280; font-size: 0.9em;">⚡ Autoscaling Control Center | Powered by Dynamic Programming & ML Forecasting</p>',
    unsafe_allow_html=True
)
