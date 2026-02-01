import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cost_model import CostModel
from simulator import Simulator
from policies import HybridPolicy, ReactivePolicy
from run_simulation import load_data

def run_sensitivity_analysis(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        
    # Load base config
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Load Data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/processed/test.csv")
    load_series = load_data(data_path, interval="1min")
    # Use Oracle predictions for sensitivity analysis to isolate cost param effects
    predictions = load_series 
    
    # Parameters to sweep
    violation_costs = [1e-6, 5e-6, 1e-5, 5e-5] 
    startup_times = [1, 3, 5, 10]
    
    results = []
    
    # Sweep
    print("Running Sensitivity Analysis...")
    total_iterations = len(violation_costs) * len(startup_times)
    pbar = tqdm(total=total_iterations)
    
    for v_cost in violation_costs:
        for s_time in startup_times:
            # Create config variant
            config = base_config.copy()
            config["cost_parameters"] = base_config["cost_parameters"].copy()
            config["cost_parameters"]["violation_cost"] = v_cost
            config["cost_parameters"]["startup_time"] = s_time
            
            # Re-init CostModel with new params
            cost_model = CostModel(config)
            
            # Policy to test (Hybrid is our best candidate)
            policy = HybridPolicy(config)
            
            # Run Simulation
            sim = Simulator(config, cost_model, {"Hybrid": policy})
            sim_timeline = sim.run_scenario("Hybrid", load_series, predictions)
            
            # Calculate Total Cost
            metrics = cost_model.calculate_total_cost(sim_timeline)
            
            results.append({
                "Violation Cost": v_cost,
                "Startup Time": s_time,
                "Total Cost": metrics["total_cost"],
                "Violation $": metrics["violation_cost"],
                "Dropped Reqs": metrics["total_dropped_requests"]
            })
            pbar.update(1)
            
    pbar.close()
    
    df_results = pd.DataFrame(results)
    print("\nSensitivity Analysis Results:")
    print(df_results)
    
    # Plot Heatmap
    try:
        pivot_table = df_results.pivot(index="Violation Cost", columns="Startup Time", values="Total Cost")
        
        plt.figure(figsize=(10, 8), dpi=200)
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="viridis_r")
        plt.title("Total Cost Sensitivity: Hybrid Policy")
        plt.xlabel("Startup Time (min)")
        plt.ylabel("Violation Cost ($/Byte)")
        
        output_path = "sensitivity_heatmap.png"
        plt.savefig(output_path)
        print(f"Heatmap saved to {output_path}")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_sensitivity_analysis()
