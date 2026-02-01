import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Modern color palette
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'gray': '#6b7280'
}

def plot_simulation_timeline_plotly(results_dict, load_data, predictions=None, anomalies=None):
    """
    Enhanced interactive Plotly version of simulation timeline.
    Splits Load/Forecast and Server Counts into two synchronized subplots with improved styling.
    """
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>Traffic Load & Forecast</b>", 
            "<b>Active Servers by Policy</b>"
        )
    )
    
    # 1. Plot Load (Row 1) with area fill
    fig.add_trace(
        go.Scatter(
            y=load_data, 
            name="Traffic Load (Actual)", 
            line=dict(color=COLORS['gray'], width=2),
            fill='tozeroy',
            fillcolor='rgba(107, 114, 128, 0.15)',
            hovertemplate='<b>Actual Load</b><br>Time: %{x} min<br>Load: %{y:,.0f} B/m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Plot Predictions (Row 1)
    if predictions is not None:
        if isinstance(predictions, dict):
            # PI Mode - Enhanced visualization
            mean_pred = predictions.get("mean", [])
            upper = predictions.get("upper", [])
            lower = predictions.get("lower", [])
            
            # Plot Confidence Interval (Shaded Area)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(upper))) + list(range(len(lower)))[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Plot Upper Bound
            fig.add_trace(
                go.Scatter(
                    y=upper,
                    name="Upper Bound (97.5%)",
                    line=dict(color=COLORS['info'], width=1, dash='dot'),
                    hovertemplate='<b>Upper Bound</b><br>Time: %{x} min<br>Load: %{y:,.0f} B/m<extra></extra>',
                    visible='legendonly'
                ),
                row=1, col=1
            )
            
            # Plot Lower Bound  
            fig.add_trace(
                go.Scatter(
                    y=lower,
                    name="Lower Bound (2.5%)",
                    line=dict(color=COLORS['info'], width=1, dash='dot'),
                    hovertemplate='<b>Lower Bound</b><br>Time: %{x} min<br>Load: %{y:,.0f} B/m<extra></extra>',
                    visible='legendonly'
                ),
                row=1, col=1
            )
            
            # Plot Mean Forecast
            fig.add_trace(
                go.Scatter(
                    y=mean_pred,
                    name="Forecast (Mean)",
                    line=dict(color=COLORS['primary'], width=2.5, dash='dash'),
                    hovertemplate='<b>Forecast</b><br>Time: %{x} min<br>Load: %{y:,.0f} B/m<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            # Simple Array Mode
            fig.add_trace(
                go.Scatter(
                    y=predictions,
                    name="Forecast",
                    line=dict(color=COLORS['primary'], width=2.5, dash='dash'),
                    hovertemplate='<b>Forecast</b><br>Time: %{x} min<br>Load: %{y:,.0f} B/m<extra></extra>'
                ),
                row=1, col=1
            )
            
    # 2.5 Plot Anomalies (Row 1) - Enhanced markers
    if anomalies is not None:
        anomaly_indices = np.where(anomalies)[0]
        if len(anomaly_indices) > 0:
            anomaly_vals = load_data[anomaly_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=anomaly_indices,
                    y=anomaly_vals,
                    mode='markers',
                    name='⚠️ Anomalies (Spike/DDoS)',
                    marker=dict(
                        color=COLORS['danger'],
                        size=10,
                        symbol='x',
                        line=dict(color='white', width=1)
                    ),
                    showlegend=True,
                    hovertemplate='<b>⚠️ ANOMALY DETECTED</b><br>Time: %{x} min<br>Load: %{y:,.0f} B/m<extra></extra>'
                ),
                row=1, col=1
            )

    # 3. Plot Servers (Row 2) - Enhanced styling for each policy type
    policy_colors = {
        'Reactive': '#ef4444',      # Red
        'Predictive': '#3b82f6',    # Blue
        'Hybrid': '#8b5cf6',        # Purple
        'DP Optimal': '#10b981',    # Green
        'Fixed': '#6b7280'          # Gray
    }
    
    for name, timeline in results_dict.items():
        servers = [step["servers_active"] for step in timeline]
        
        # Get color for this policy
        color = policy_colors.get(name, px.colors.qualitative.Plotly[len(policy_colors) % 10])
        
        # Style adjustments
        line_style = dict(color=color, width=2.5, shape='hv')
        
        if name == "Fixed":
            line_style['dash'] = 'dot'
            line_style['width'] = 2
        elif name == "DP Optimal":
            line_style['width'] = 3
            
        fig.add_trace(
            go.Scatter(
                y=servers,
                name=f"{name}",
                line=line_style,
                hovertemplate=f'<b>{name}</b><br>Time: %{{x}} min<br>Servers: %{{y}}<extra></extra>'
            ),
            row=2, col=1
        )
    
    
    # Enhanced Layout with separate legends for each subplot
    fig.update_layout(
        title={
            'text': "<b>Autoscaling Simulation Timeline</b>",
            'font': {'size': 24, 'family': 'Inter, sans-serif'}
        },
        height=800,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            # This will be overridden by trace-specific legend assignments
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=10)
        ),
        plot_bgcolor='rgba(250,250,250,0.5)',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=12),
        margin=dict(t=80, b=120, r=150)  # More space for legend on right
    )
    
    # Assign traces to specific legend groups
    # Row 1 traces (Traffic & Forecast) -> legend 1
    # Row 2 traces (Servers) -> legend 2
    
    # Update all traces to use separate legend groups
    for trace in fig.data:
        if 'Traffic' in trace.name or 'Forecast' in trace.name or 'Bound' in trace.name or 'Confidence' in trace.name or 'Anomal' in trace.name:
            # Traffic/Forecast traces - show in top legend
            trace.update(
                legendgroup='traffic',
                legendgrouptitle_text='<b>Traffic & Forecast</b>'
            )
        else:
            # Server traces - show in separate legend group
            trace.update(
                legendgroup='servers',
                legendgrouptitle_text='<b>Server Allocation</b>'
            )
    
    # Update axes styling
    fig.update_yaxes(
        title_text="<b>Traffic Load (Bytes/min)</b>",
        row=1, col=1,
        gridcolor='rgba(200,200,200,0.3)',
        showgrid=True
    )
    fig.update_yaxes(
        title_text="<b>Server Count</b>",
        row=2, col=1,
        gridcolor='rgba(200,200,200,0.3)',
        showgrid=True
    )
    fig.update_xaxes(
        title_text="<b>Time (Minutes)</b>",
        row=2, col=1,
        gridcolor='rgba(200,200,200,0.3)',
        showgrid=True,
        title_standoff=10  # More space between axis and label
    )
    
    # Update subplot title styling
    for annotation in fig.layout.annotations:
        annotation.font.size = 16
        annotation.font.family = 'Inter, sans-serif'
    
    return fig

def plot_cost_comparison_plotly(summary_df):
    """
    Enhanced interactive Plotly bar chart for costs with better color scheme.
    """
    if summary_df.empty:
        return None
        
    df = summary_df.copy()
    
    # Define color scheme for cost components
    color_map = {
        "Server Cost": COLORS['primary'],
        "Scale Cost": COLORS['warning'],
        "Violation Cost": COLORS['danger']
    }
    
    # Create stacked bar chart with custom colors
    fig = go.Figure()
    
    cost_components = ["Server Cost", "Scale Cost", "Violation Cost"]
    
    for component in cost_components:
        fig.add_trace(go.Bar(
            name=component,
            x=df["Policy"],
            y=df[component],
            marker_color=color_map[component],
            hovertemplate=f'<b>%{{x}}</b><br>{component}: $%{{y:,.2f}}<extra></extra>'
        ))
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': "<b>Total Cost Breakdown by Policy</b>",
            'font': {'size': 20, 'family': 'Inter, sans-serif'}
        },
        barmode='stack',
        height=500,
        xaxis_title="<b>Policy</b>",
        yaxis_title="<b>Cost ($)</b>",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        plot_bgcolor='rgba(250,250,250,0.5)',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=12),
        hovermode='x unified'
    )
    
    # Add total cost annotations on top of bars
    totals = df["Total Cost"].values
    for i, (policy, total) in enumerate(zip(df["Policy"], totals)):
        fig.add_annotation(
            x=policy,
            y=total,
            text=f"${total:,.2f}",
            showarrow=False,
            yshift=10,
            font=dict(size=11, color='black', family='Inter, sans-serif')
        )
    
    return fig

def plot_simulation_timeline(results_dict, load_data, predictions=None, save_path=None):
    """
    Matplotlib version - Enhanced styling with modern color palette.
    Plot Load vs Active Servers for all policies.
    """
    fig, ax1 = plt.subplots(figsize=(16, 7), dpi=200)
    
    ax2 = ax1.twinx()
    
    # Plot Load with enhanced styling
    ax2.fill_between(
        range(len(load_data)),
        load_data,
        alpha=0.15,
        color=COLORS['gray'],
        label="Traffic Load (Actual)"
    )
    ax2.plot(
        load_data,
        color=COLORS['gray'],
        alpha=0.6,
        linewidth=1.5,
        label="_nolegend_"
    )
    
    # Plot Predictions if provided
    if predictions is not None:
        pred_data = predictions if not isinstance(predictions, dict) else predictions.get("mean", predictions)
        ax2.plot(
            pred_data,
            color=COLORS['primary'],
            alpha=0.5,
            label="Forecast",
            linestyle="--",
            linewidth=2
        )

    ax2.set_ylabel("Traffic Load (Bytes/min)", color=COLORS['gray'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])
    ax2.grid(False)
    
    # Plot Servers with enhanced colors
    policy_colors = {
        'Reactive': COLORS['danger'],
        'Predictive': COLORS['info'],
        'Hybrid': COLORS['secondary'],
        'DP Optimal': COLORS['success'],
        'Fixed': COLORS['gray']
    }
    
    for name, timeline in results_dict.items():
        servers = [step["servers_active"] for step in timeline]
        color = policy_colors.get(name, '#000000')
        linewidth = 3 if name == "DP Optimal" else 2
        linestyle = ':' if name == "Fixed" else '-'
        
        ax1.plot(
            servers,
            label=f"{name}",
            color=color,
            linewidth=linewidth,
            drawstyle="steps-post",
            linestyle=linestyle
        )
        
    ax1.set_xlabel("Time (Minutes)", fontweight='bold')
    ax1.set_ylabel("Active Servers", fontweight='bold')
    ax1.set_title("Autoscaling Simulation: Policy Comparison", fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legends
    ax1.legend(loc="upper left", framealpha=0.95, edgecolor='lightgray')
    ax2.legend(loc="upper right", framealpha=0.95, edgecolor='lightgray')
    
    # Grid styling
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
    
    # Set background color
    ax1.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
    return fig

def plot_cost_comparison(summary_df, save_path=None):
    """
    Matplotlib version - Enhanced bar chart for costs.
    """
    if summary_df.empty:
        return None
        
    df = summary_df.copy()
    if "Policy" in df.columns:
        df.set_index("Policy", inplace=True)
    
    cost_cols = ["Server Cost", "Scale Cost", "Violation Cost"]
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    
    # Custom color palette
    colors = [COLORS['primary'], COLORS['warning'], COLORS['danger']]
    
    df[cost_cols].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        edgecolor='white',
        linewidth=1.5
    )
    
    ax.set_title("Total Cost Breakdown by Policy", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Cost ($)", fontweight='bold')
    ax.set_xlabel("Policy", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Cost Component", framealpha=0.95, edgecolor='lightgray')
    
    # Grid styling
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
    return fig
