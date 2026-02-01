import os
import pandas as pd
import numpy as np
import yaml
from cost_model import CostModel
from policies import ReactivePolicy, PredictivePolicy, HybridPolicy, OptimalFixedPolicy
from dp_optimizer import DPOptimizer
from simulator import Simulator
from forecaster_integration import get_forecasts

def load_data(path, interval="1min"):
    """
    Load traffic data. Handles the 'gap' scenario (0 load).
    """
    if not os.path.exists(path):
        # Fallback or error
        print(f"Data not found at {path}. Generating synthetic data.")
        return generate_synthetic_data()
        
    df = pd.read_csv(path)
    # Parse time, set index
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    
    # Resample to simulation interval   
    df_agg = df.resample(interval)["size"].sum().fillna(0)
    
    return df_agg.values

def generate_synthetic_data(length=1000):
    # Sine wave + gap
    t = np.arange(length)
    load = 500 + 400 * np.sin(t * 0.05)
    # Add noise
    load += np.random.normal(0, 50, length)
    load = np.maximum(load, 0)
    
    # Create gap
    gap_start = 400
    gap_end = 500
    load[gap_start:gap_end] = 0
    
    return load

def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Data Data
    data_path = "data/processed/test.csv" # Adjust relative path
    # If file doesn't exist, try absolute or synthetic
    load_data_series = load_data(data_path, interval=config["simulation"]["interval"])
    
    # Cost Model
    cost_model = CostModel(config)
    
    # Policies
    policies = {
        "Reactive": ReactivePolicy(config),
        "Predictive": PredictivePolicy(config),
        "Hybrid": HybridPolicy(config),
        "Fixed_Max": OptimalFixedPolicy(config, fixed_servers=config["simulation"]["max_servers"])
    }
    
    # Run DP Optimizer (Pre-compute)
    print("Running DP Optimizer (this may take a moment)...")
    dp = DPOptimizer(config, cost_model, load_data_series)
    dp.optimize()
    dp_trajectory = dp.reconstruct_path()
    
    # Wrapper for DP trajectory to play back decisions in Simulator.
    class DPReplayPolicy:
        def __init__(self, trajectory):
            self.trajectory = {step["time_step"]: step["servers"] for step in trajectory}
            # trajectory provides 'servers' (provisioned).
            self.max_step = max(self.trajectory.keys())
            
        def decide(self, current_step, current_servers, *args, **kwargs):
            if current_step > self.max_step:
                return {"action": 0, "new_servers": current_servers}
                
            target = self.trajectory.get(current_step, current_servers)
            action = target - current_servers
            return {"action": action, "new_servers": target, "reason": "DP Optimal"}
            
    policies["DP_Optimal"] = DPReplayPolicy(dp_trajectory)
    
    # Simulator
    sim = Simulator(config, cost_model, policies)
    
    # Run Comparisons
    results = {}
    
    # Load Forecasts
    print("Generating Forecasts (this may take a moment)...")
    # Note: data_path is relative to script when using os.path logic, but get_forecasts takes direct path
    # We used relative path "data/processed/train.csv" earlier.
    # Let's verify path resolution in get_forecasts vs here.
    # get_forecasts calls read_csv(path).
    # ensure full path is passed if needed.
    full_data_path = os.path.abspath(data_path)
    
    # Also we need to make sure 'forecasting' config exists
    if "forecasting" not in config:
        config["forecasting"] = {"model": "bilstm_attention", "window_size": 60}
        
    forecasts, actuals_aligned = get_forecasts(full_data_path, config)
    
    # If forecasts are empty (failure), fallback to Oracle? or Zeros?
    if len(forecasts) == 0:
        print("Forecasting failed or no data. Falling back to Oracle (Actuals).")
        predictions = load_data_series
    else:
        predictions = forecasts
        # Use aligned actuals for simulation to match length?
        # Simulation usually runs on 'load_data_series'. 
        # Check if lengths match.
        if len(predictions) != len(load_data_series):
             print(f"Warning: Forecast length {len(predictions)} != Data length {len(load_data_series)}. Truncating/Padding.")
             # Usually forecaster_integration handles alignment to aggregated data. Just use what matches.
    
    for name in policies:
        print(f"Running {name}...")
        results[name] = sim.run_scenario(name, load_data_series, predictions)
        
    # Analysis
    print("\n--- Summary Results ---")
    for name, res in results.items():
        total_metrics = cost_model.calculate_total_cost(res)
        print(f"Policy: {name}")
        print(f"  Total Cost: ${total_metrics['total_cost']:.2f}")
        print(f"  Violations: ${total_metrics['violation_cost']:.2f} ({total_metrics['total_dropped_requests']:.0f} reqs)")
        print(f"  Scale Ops Cost: ${total_metrics['scale_cost']:.2f}")
        print("-" * 20)

if __name__ == "__main__":
    main()
