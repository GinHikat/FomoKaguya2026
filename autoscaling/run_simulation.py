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
    data_path = "data/processed/test.csv" 
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
    print("Running DP Optimizer...")
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
    
    # Anomaly Detection
    # Load Pre-trained model
    from autoscaling.bonus.anomaly_detector import AnomalyDetector
    print("Loading Anomaly Detector...")
    detector = AnomalyDetector(contamination=0.05) 
    
    # Path to saved model in bonus directory
    bonus_dir = os.path.join(script_dir, "bonus")
    model_path = os.path.join(bonus_dir, "anomaly_model.joblib")
    
    if os.path.exists(model_path):
        detector.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Fitting on test data (fallback).")
        detector.fit(load_data_series)
    
    # Run Comparisons
    results = {}
    
    # Load Forecasts
    print("Generating Forecasts...")
    full_data_path = os.path.abspath(data_path)
        
    forecasts, actuals_aligned = get_forecasts(full_data_path, config)
    
    if len(forecasts) == 0:
        print("Forecasting failed or no data. Falling back to Oracle (Actuals).")
        predictions = load_data_series
    else:
        print(f"Using Real Forecast Model: {config.get('forecasting', {}).get('model', 'Unknown')}")
        
        # Compute Prediction Intervals (Empirical Residuals on current batch)
        # Note: Ideally residuals are computed on validation set.
        residuals = actuals_aligned - forecasts
        std_resid = np.std(residuals)
        
        # 95% Confidence Interval
        z_score = 1.96
        upper_bound = forecasts + z_score * std_resid
        lower_bound = np.maximum(forecasts - z_score * std_resid, 0)
        
        print(f"Forecast Stats: Mean Residual={np.mean(residuals):.2f}, Std={std_resid:.2f}")
        
        predictions = {
            "mean": forecasts,
            "upper": upper_bound,
            "lower": lower_bound
        }
        
        # Check alignment with simulation data
        if len(forecasts) != len(load_data_series):
             print(f"Warning: Forecast length {len(forecasts)} != Data length {len(load_data_series)}. Truncating/Padding.")
    
    # Batch detect anomalies
    anomalies = detector.detect_batch(load_data_series)
    
    for name in policies:
        print(f"Running {name}...")
        if name == "DP_Optimal":
            pass
            
        # Pass pre-computed anomalies
        results[name] = sim.run_scenario(name, load_data_series, predictions, anomalies=anomalies)
        
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
