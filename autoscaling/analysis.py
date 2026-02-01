import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, cost_model):
        self.cost_model = cost_model
        
    def compare_policies_summary(self, results_dict):
        """
        Create a summary DataFrame comparing policies.
        """
        summary = []
        for name, timeline in results_dict.items():
            metrics = self.cost_model.calculate_total_cost(timeline)
            
            # Additional Stats
            actions = sum(1 for t in timeline if t["action"] != 0)
            avg_util = np.mean([t["costs"]["utilization"] for t in timeline])
            
            row = {
                "Policy": name,
                "Total Cost": metrics["total_cost"],
                "Violation Cost": metrics["violation_cost"],
                "Scale Cost": metrics["scale_cost"],
                "Server Cost": metrics["server_cost"],
                "Dropped Reqs": metrics["total_dropped_requests"],
                "Scale Events": actions,
                "Avg Utilization": avg_util
            }
            summary.append(row)
            
        return pd.DataFrame(summary)
        
    def plot_timeline(self, results_dict, load_data, title="Scaling Decisions Timeline"):
        """
        Plot servers vs load for each policy.
        """
        plt.figure(figsize=(15, 8), dpi=200)
        
        # Plot Load
        plt.plot(load_data, label="Load (Requests)", color="black", alpha=0.3, linewidth=1)
        
        # Plot Servers
        for name, timeline in results_dict.items():
            servers = [t["servers_active"] for t in timeline]
            plt.plot(servers, label=f"{name} Servers", drawstyle="steps-post", linewidth=2)
            
        plt.ylabel("Number of Active Servers")
        plt.xlabel("Time Step")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_cost_breakdown(self, summary_df):
        """
        Stacked bar chart of costs.
        """
        df_plot = summary_df.set_index("Policy")[["Server Cost", "Scale Cost", "Violation Cost"]]
        df_plot.plot(kind="bar", stacked=True, figsize=(10, 6), dpi=200)
        plt.title("Cost Breakdown by Policy")
        plt.ylabel("Cost ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
