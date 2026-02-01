import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_simulation_timeline_plotly(results_dict, load_data, predictions=None):
    """
    Interactive Plotly version of simulation timeline.
    Splits Load/Forecast and Server Counts into two synchronized subplots.
    """
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=("Traffic Load & Forecast", "Active Servers by Policy")
    )
    
    # 1. Plot Load (Row 1)
    fig.add_trace(
        go.Scatter(y=load_data, name="Traffic Load (Actual)", 
                   line=dict(color='gray', width=1), fill='tozeroy', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )
    
    # 2. Plot Predictions (Row 1)
    if predictions is not None:
        fig.add_trace(
            go.Scatter(y=predictions, name="Forecast", 
                       line=dict(color='blue', width=1.5, dash='dash')),
            row=1, col=1
        )

    # 3. Plot Servers (Row 2)
    colors = px.colors.qualitative.Plotly
    for idx, (name, timeline) in enumerate(results_dict.items()):
        servers = [step["servers_active"] for step in timeline]
        
        # Make "Fixed" policy distinct/less intrusive
        line_style = dict(color=colors[idx % len(colors)], width=2, shape='hv')
        if name == "Fixed":
            line_style['dash'] = 'dot'
            line_style['width'] = 1.5
            
        fig.add_trace(
            go.Scatter(y=servers, name=f"{name} Servers", 
                       line=line_style),
            row=2, col=1
        )
        
    fig.update_layout(
        title_text="Autoscaling Simulation Timeline",
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text="Bytes / Min", row=1, col=1)
    fig.update_yaxes(title_text="Server Count", row=2, col=1)
    fig.update_xaxes(title_text="Time (Minutes)", row=2, col=1)
    
    return fig

def plot_cost_comparison_plotly(summary_df):
    """
    Interactive Plotly bar chart for costs.
    """
    if summary_df.empty:
        return None
        
    df = summary_df.copy()
    
    # Stacked Bar Chart
    fig = px.bar(df, x="Policy", y=["Server Cost", "Scale Cost", "Violation Cost"],
                 title="Total Cost Breakdown by Policy",
                 labels={"value": "Cost ($)", "variable": "Cost Component"},
                 height=500)
    
    fig.update_layout(barmode='stack')
    return fig

def plot_simulation_timeline(results_dict, load_data, predictions=None, save_path=None):
    """
    Plot Load vs Active Servers for all policies.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=200)
    
    ax2 = ax1.twinx()
    
    # Plot Load
    ax2.plot(load_data, color="black", alpha=0.2, label="Traffic Load (Actual)", linewidth=1)
    
    # Plot Predictions if provided
    if predictions is not None:
        ax2.plot(predictions, color="blue", alpha=0.3, label="Forecast", linestyle="--", linewidth=1)

    ax2.set_ylabel("Traffic Load (Bytes)", color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    
    # Plot Servers
    colors = sns.color_palette("husl", len(results_dict))
    for idx, (name, timeline) in enumerate(results_dict.items()):
        servers = [step["servers_active"] for step in timeline]
        ax1.plot(servers, label=f"{name}", color=colors[idx], linewidth=2, drawstyle="steps-post")
        
    ax1.set_xlabel("Time (Minutes)")
    ax1.set_ylabel("Active Servers")
    ax1.set_title("Autoscaling Simulation: Policy Comparison")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    if save_path:
        fig.savefig(save_path)
    return fig

def plot_cost_comparison(summary_df, save_path=None):
    """
    Bar chart for costs.
    """
    if summary_df.empty:
        return None
        
    df = summary_df.copy()
    if "Policy" in df.columns:
        df.set_index("Policy", inplace=True)
    cost_cols = ["Server Cost", "Scale Cost", "Violation Cost"]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    df[cost_cols].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Total Cost Breakdown by Policy")
    ax.set_ylabel("Cost ($)")
    plt.xticks(rotation=45)
    ax.legend(title="Cost Component")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    return fig
