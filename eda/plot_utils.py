import pandas as pd


def print_macro_overview(df):
    """
    Prints a text-based Executive Summary of the entire dataset.
    """
    import pandas as pd
    
    # Basic Time
    start = df.index.min()
    end = df.index.max()
    duration = end - start
    
    # Volume
    total_hits = len(df)
    total_bytes = df['size'].sum()
    total_gb = total_bytes / (1024**3)
    
    # Users
    if 'ip' in df.columns:
        unique_ips = df['ip'].nunique()
        avg_requests_per_user = total_hits / unique_ips
    else:
        unique_ips = 0
        avg_requests_per_user = 0
        
    # Reliability
    if 'status_label' in df.columns:
        errors = df[df['status_label'] != 'Success']
    else:
        errors = df[df['status'] >= 400]
        
    error_count = len(errors)
    error_rate = (error_count / total_hits) * 100
    
    print("="*50)
    print("           MACRO DATASET OVERVIEW           ")
    print("="*50)
    print(f"Time Range     : {start} to {end}")
    print(f"Duration       : {duration.days} Days")
    print("-" * 50)
    print(f"Total Requests : {total_hits:,.0f}")
    print(f"Avg Daily Hits : {total_hits/duration.days:,.0f}")
    print("-" * 50)
    print(f"Total Traffic  : {total_gb:.2f} GB")
    print(f"Avg Daily MB   : {(total_gb*1024)/duration.days:.2f} MB")
    print("-" * 50)
    print(f"Unique Users   : {unique_ips:,.0f}")
    print(f"Avg Req/User   : {avg_requests_per_user:.2f}")
    print("-" * 50)
    print(f"Error Rate     : {error_rate:.2f}% ({error_count:,.0f} failed requests)")
    print("="*50)


def plot_weekly_heatmap(df, interval='1H', figsize=(20, 8), title=None):
    """
    Plots a heatmap of request intensity (Day of Week vs Time of Day).
    
    Args:
        df (pd.DataFrame): The dataframe with a DatetimeIndex.
        interval (str): The time frequency (e.g., '1H', '30T', '15T').
        figsize (tuple): Figure size (width, height).
        title (str): Output chart title.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Resample to the desired interval
    resampled = df.resample(interval).size()
    
    # 2. Group by Day Name and Time
    heatmap_data = resampled.groupby([
        resampled.index.day_name(),
        resampled.index.time
    ]).sum().unstack(fill_value=0)
    
    # 3. Sort rows by Day of Week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    # 4. Calculate sensible xtick spacing based on total columns
    # We generally aim for ~24 labels on the x-axis to represent hours
    total_cols = heatmap_data.shape[1]
    xticks_step = max(1, total_cols // 24)

    # 5. Plot
    plt.figure(figsize=figsize)
    if title is None:
        title = f'Request Intensity by {interval} Interval'
        
    sns.heatmap(heatmap_data, cmap='inferno', xticklabels=xticks_step)
    
    plt.title(title)
    plt.xlabel('Time of Day')
    plt.xticks(rotation=45)
    plt.ylabel('Day of Week')
    plt.yticks(rotation=0)
    plt.show()

def analyze_status_distribution(df, interval='5T'):
    """
    Analyzes and plots the distribution of 'Success' vs 'Error' requests for a specified time interval.

    Args:
        df (pd.DataFrame): The dataframe with a DatetimeIndex and 'status_label' column.
        interval (str): Time frequency to analyze (e.g., '1min', '5T').
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    print(f"Analyzing Status Distribution for {interval} Interval...")
    
    # 1. Resample and count occurrences of each status_label
    # Group by Time + Status Label, then unstack to get columns
    col = 'status_label' if 'status_label' in df.columns else 'status'
    status_counts = df.groupby([pd.Grouper(freq=interval), col]).size().unstack(fill_value=0)
    

    # 2. Plot Histograms for 'Success' and 'Error'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Histogram for Success ---
    if 'Success' in status_counts.columns:
        # Drop 0s to see distribution of ACTIVE activity
        data = status_counts['Success']
        data.plot(kind='hist', bins=50, ax=axes[0], color='green', alpha=0.7)
        axes[0].set_title(f'Distribution of Success Requests per {interval}')
        axes[0].set_xlabel('Requests Count per Interval')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No 'Success' Data", ha='center')

    # --- Histogram for Error ---
    error_cols = [c for c in status_counts.columns if c != 'Success']
    # If we have explicit 'Error' column or multiple distinct error codes
    # The original logic looked for 'Error' label specifically, assuming 'status_label' mapped things to Success/Error/etc.
    # Let's try to find 'Error' column if it exists, or sum others? 
    # Original code: error_col = 'Error' if 'Error' in status_counts.columns else None
    
    error_col = 'Error' if 'Error' in status_counts.columns else None
    
    # If no explicit 'Error' column, but we have others (like 404, 500 etc if status_label wasn't used), 
    # we might want to sum them? 
    # For now, let's stick to original logic: check for 'Error'.
    
    if error_col and status_counts[error_col].sum() > 0:
        error_data = status_counts[error_col]
        nonzero_errors = error_data[error_data > 0]
        
        if len(nonzero_errors) > 0:
            axes[1].hist(nonzero_errors, bins=30, color='red', alpha=0.7)
            axes[1].set_title(f'Distribution of Errors (Excluding Zero-Error Intervals)')
            axes[1].set_xlabel('Error Count (when errors occur)')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
            # Add text about how many intervals were error-free
            zero_count = (error_data == 0).sum()
            total_count = len(error_data)
            axes[1].text(0.95, 0.95, f"{zero_count}/{total_count} intervals\nhad 0 errors", 
                         transform=axes[1].transAxes, ha='right', va='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
             axes[1].text(0.5, 0.5, "No Errors Found in Range", ha='center')
    else:
        # If we didn't find 'Error' col but have other columns, maybe show them?
        # But to be safe and consistent with previous behavior:
        axes[1].text(0.5, 0.5, "No 'Error' labeled data found", ha='center')

    plt.tight_layout()
    plt.show()
    
    # Optional: Print summary stats
    cols_to_show = [c for c in ['Success', error_col] if c]
    if cols_to_show:
        print(f"Stats for {interval}:")
        display(status_counts[cols_to_show].describe())


def plot_weekly_patterns(df, interval='1H', figsize=(20, 24)):
    """
    Plots 3 heatmaps: Average Hits, Average Size, and Average Success Rate.
    Aggregates historical data by (Day of Week, Time of Day) and takes the mean.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    print(f"Processing data for {interval} patterns...")

    # 1. Prepare Base Metrics (Resampled Time Series)
    # We need to calculate these for every specific timestamp in history first.
    resampler = df.resample(interval)
    
    # A. Total Hits (Count)
    ts_hits = resampler.size()
    
    # B. Total Size (Sum)
    ts_size = resampler['size'].sum()
    
    # C. Success Rate (Ratio)
    # Count only 'Success' rows per interval
    # Assuming 'status_label' exists. If not, use status code logic.
    if 'status_label' in df.columns:
        ts_success = df[df['status_label'] == 'Success'].resample(interval).size()
    else:
        # Fallback: assume status 200-299 is success
        ts_success = df[(df['status'] >= 200) & (df['status'] < 300)].resample(interval).size()
    
    # Calculate ratio (handle division by zero if an interval has 0 hits)
    # We use div and fillna(0) or NaN depending on preference. 
    # Usually 0 hits = NaN success rate, but for heatmap 0 or 1 might be assumed. 
    # Let's keep it as NaN for empty times to distinguish from 0% success.
    ts_rate = ts_success.div(ts_hits)

    # 2. Group by (Day Name, Time) and Take Mean
    # We create a helper function to reshape for heatmap
    def get_heatmap_matrix(series, agg_func='mean'):
        grouped = series.groupby([series.index.day_name(), series.index.time])
        # We take the mean across the weeks
        agg_data = getattr(grouped, agg_func)().unstack()
        
        # Sort days
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return agg_data.reindex(days)

    matrix_hits = get_heatmap_matrix(ts_hits, 'mean')
    matrix_size = get_heatmap_matrix(ts_size, 'mean')
    matrix_rate = get_heatmap_matrix(ts_rate, 'mean')

    # 3. Plotting
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # X-ticks formatting
    total_cols = matrix_hits.shape[1]
    xticks_step = max(1, total_cols // 24)

    # Heatmap 1: Average Hits
    sns.heatmap(matrix_hits, ax=axes[0], cmap='inferno', xticklabels=xticks_step)
    axes[0].set_title(f'Average Total Hits per {interval} (Load Pattern)')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Day of Week')
    axes[0].tick_params(axis='y', rotation=0)
    axes[0].tick_params(axis='x', rotation=45)

    # Heatmap 2: Average Total Size
    sns.heatmap(matrix_size, ax=axes[1], cmap='viridis', xticklabels=xticks_step)
    axes[1].set_title(f'Average Total File Size (Bytes) per {interval} (Bandwidth Pattern)')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Day of Week')
    axes[1].tick_params(axis='y', rotation=0)
    axes[1].tick_params(axis='x', rotation=45)

    # Heatmap 3: Success Rate
    # success rate is 0.0 to 1.0. Use a diverging or distinct map.
    sns.heatmap(matrix_rate, ax=axes[2], cmap='RdYlGn', xticklabels=xticks_step, vmin=0.8, vmax=1)
    axes[2].set_title(f'Average Success Rate (0-1) per {interval} (Reliability Pattern)')
    axes[2].set_xlabel('Time of Day')
    axes[2].set_ylabel('Day of Week')
    axes[2].tick_params(axis='y', rotation=0)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_daily_profile(df, interval='30T', figsize=(15, 12)):
    """
    Plots the Average Daily Profile (00:00 to 23:59) aggregated across all days.
    Shows 3 Line Plots: Average Hits, Average Size, and Average Success Rate.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    print(f"Generating daily profile (Time of Day analysis) for {interval} interval...")
    
    # 1. Resample first to chunk data correctly into history buckets
    resampler = df.resample(interval)
    
    # Calculate base metrics for every interval in history
    ts_hits = resampler.size()
    ts_size = resampler['size'].sum()
    
    # Success Rate calculation
    if 'status_label' in df.columns:
        ts_success = df[df['status_label'] == 'Success'].resample(interval).size()
    else:
        # Fallback
        ts_success = df[(df['status'] >= 200) & (df['status'] < 300)].resample(interval).size()
        
    # Calculate rate for each historical interval
    ts_rate = ts_success.div(ts_hits)
    
    # 2. Group by Time of Day and Aggregate (Mean)
    # This collapses "July 1 10:00", "July 2 10:00"... into just "10:00"
    profile_hits = ts_hits.groupby(ts_hits.index.time).mean()
    profile_size = ts_size.groupby(ts_size.index.time).mean()
    profile_rate = ts_rate.groupby(ts_rate.index.time).mean()
    
    # 3. Plotting
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Prepare X-axis labels (convert time objects to strings)
    times = [str(t) for t in profile_hits.index]
    x_indices = range(len(times))
    
    # Plot 1: Hits
    axes[0].plot(x_indices, profile_hits.values, color='#1f77b4', linewidth=2)
    axes[0].set_title(f'Average Hits by Time of Day (Aggregated per {interval})', fontsize=14)
    axes[0].set_ylabel('Avg Hits')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Size
    axes[1].plot(x_indices, profile_size.values, color='#ff7f0e', linewidth=2)
    axes[1].set_title('Average Total Size (Bytes)', fontsize=14)
    axes[1].set_ylabel('Bytes')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: Success Rate
    axes[2].plot(x_indices, profile_rate.values, color='#2ca02c', linewidth=2)
    axes[2].set_title('Average Success Rate (Reliability)', fontsize=14)
    axes[2].set_ylabel('Rate (0-1)')
    axes[2].set_ylim(0.84, 0.96)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    # Smart X-Tick Formatting
    # Ensure we don't have too many labels overlapping
    step = max(1, len(times) // 24)
    axes[2].set_xticks(x_indices[::step])
    axes[2].set_xticklabels(times[::step], rotation=45)
    
    plt.xlabel('Time of Day')
    plt.tight_layout()
    plt.show()


def plot_file_type_stats(df, top_n=10, regex=r'\.([a-zA-Z0-9]+)$'):
    """
    Analyzes requests by File Extension (extracted from URL/Resource).
    Plots distinct Count and Total Size for top file types.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    
    print("Analyzing File Types...")
    
    # Extract extension. This assumes 'resource' or 'request' column exists with path
    # We will use string manipulation for speed
    if 'resource' not in df.columns:
        print("Error: 'resource' column required for file analysis")
        return

    # Extract extension using str.extract
    # The regex looks for a dot followed by alphanumeric chars at the end of string
    extensions = df['resource'].str.extract(regex, expand=False).str.lower().fillna('unknown')
    
    # Create a wrapper DF for aggregation
    # We assign it temporarily to avoid modifying the original deeply
    stats = pd.DataFrame({'ext': extensions, 'size': df['size']})
    
    # Aggregate
    # Count = Popularity
    # Sum = Bandwidth
    agg = stats.groupby('ext')['size'].agg(['count', 'sum'])
    
    # Sort and Take Top N
    top_by_count = agg.sort_values('count', ascending=False).head(top_n)
    top_by_size = agg.sort_values('sum', ascending=False).head(top_n)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Popularity
    top_by_count['count'].plot(kind='bar', ax=axes[0], color='#1f77b4')
    axes[0].set_title(f'Top {top_n} File Types by Request Count')
    axes[0].set_ylabel('Number of Requests')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Bandwidth
    # Convert bytes to MB or GB for readability
    (top_by_size['sum'] / 1024**2).plot(kind='bar', ax=axes[1], color='#ff7f0e')
    axes[1].set_title(f'Top {top_n} File Types by Total Bandwidth (MB)')
    axes[1].set_ylabel('Total Size (MB)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_top_users(df, top_n=10):
    """
    Identifies top users (IPs) by Request Count and Traffic Volume.
    """
    import matplotlib.pyplot as plt
    
    print(f"Identifying Top {top_n} Users...")
    
    # 1. Identify Top IPs just by count first
    user_counts = df['ip'].value_counts().head(top_n)
    top_ips = user_counts.index.tolist()
    
    # 2. Filter data for these IP only and Pivot by Status
    # We want rows=IPs, cols=Status Label
    col_status = 'status_label' if 'status_label' in df.columns else 'status'
    
    subset = df[df['ip'].isin(top_ips)]
    
    # Cross-tabulation: Count of each status per IP
    stacked_data = pd.crosstab(subset['ip'], subset[col_status])
    
    # Re-sort match the order of top_ips (usually Crosstab sorts alphabetically)
    stacked_data = stacked_data.reindex(top_ips)
    
    # 3. Plot Stacked Bar
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a distinct colormap like 'tab20' to handle many potential status codes
    stacked_data.plot(kind='barh', stacked=True, ax=ax, cmap='tab20', edgecolor='none')
    
    ax.set_title(f'Top {top_n} Domains by Total Request Count (Breakdown by Request Status)')
    ax.set_xlabel('Number of Requests')
    ax.set_ylabel('IP Address')
    ax.invert_yaxis() # Top user at top
    ax.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_status_breakdown(df):
    """
    Visualizes the specific breakdown of HTTP Status Codes (not just Success/Error).
    """
    import matplotlib.pyplot as plt
    try:
        # Use simple status code if label doesn't exist
        col = 'status_label' if 'status_label' in df.columns else 'status'
        counts = df[col].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Pie Chart (Cleaned up)
        # We only label slices > 2% to avoid clutter, put rest in legend
        def autopct_filter(pct):
            return ('%1.1f%%' % pct) if pct > 2 else ''

        counts.plot(kind='pie', ax=axes[0], autopct=autopct_filter, startangle=90, cmap='Pastel1', labels=None)
        axes[0].set_ylabel('')
        axes[0].set_title('Status Code Distribution')
        axes[0].legend(labels=counts.index, loc="best", bbox_to_anchor=(1, 0.5))
        
        # Log Scale Bar Chart (to see rare errors)
        counts.plot(kind='bar', ax=axes[1], color='teal', log=True)
        axes[1].set_title('Status Counts (Log Scale)')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot status breakdown: {e}")

def plot_rolling_statistics(df, window='1H', figsize=(20, 8)):
    """
    Plots the Total Hits per interval with Rolling Mean and Rolling Std Dev bands (Bollinger Bands logic).
    Useful for spotting trends and volatility changes.
    """
    import matplotlib.pyplot as plt
    
    print(f"Calculating Rolling Statistics (window={window})...")
    
    # 1. Resample
    # Ensure no gaps, fill with 0
    ts = df.resample('10T').size().fillna(0) # Base resolution 10 mins for better rolling smoothness
    
    # 2. Calculate Rolling
    # We use a window corresponding to the input argument string logic or simple integer
    # Let's convert window string to integer bins if possible, or just use time-based string
    roller = ts.rolling(window=window)
    roll_mean = roller.mean()
    roll_std = roller.std()
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Raw Data (faint)
    ax.plot(ts.index, ts, label='Raw Traffic (10min)', color='gray', alpha=0.3)
    
    # Moving Average
    ax.plot(roll_mean.index, roll_mean, label=f'Rolling Mean ({window})', color='blue', linewidth=2)
    
    # Bands (+/- 2 or 3 STD)
    upper_band = roll_mean + (3 * roll_std)
    lower_band = roll_mean - (3 * roll_std)
    
    ax.fill_between(roll_mean.index, lower_band, upper_band, color='blue', alpha=0.1, label='3-Sigma Band')
    
    ax.set_title(f'Traffic Volatility & Trend Analysis (Window: {window})')
    ax.set_ylabel('Hits per 10min')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

def plot_anomaly_spikes(df, interval='5T', threshold_z=3):
    """
    Detects and plots anomalies using Z-Score method for:
    1. Traffic Volume (Hits)
    2. Total Bandwidth (Size)
    3. Success Rate
    
    Anomalies are points where (Value - Mean) / Std > threshold.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display
    
    print(f"Detecting Anomalies (Interval: {interval}, Z-Threshold: {threshold_z})...")
    
    # 1. Prep Data
    resampler = df.resample(interval)
    
    # A. Hits
    ts_hits = resampler.size().fillna(0)
    
    # B. Size
    ts_size = resampler['size'].sum().fillna(0)
    
    # C. Rate
    if 'status_label' in df.columns:
        ts_success = df[df['status_label'] == 'Success'].resample(interval).size()
    else:
        ts_success = df[(df['status'] >= 200) & (df['status'] < 300)].resample(interval).size()
        
    ts_rate = ts_success.div(ts_hits)
    
    # Define metrics to loop over
    # (Title, Series, Unit, Color)
    metrics = [
        ('Traffic Volume (Hits)', ts_hits, 'Hits', '#1f77b4'),
        ('Total Bandwidth (Bytes)', ts_size, 'Bytes', '#ff7f0e'),
        ('Success Rate', ts_rate, 'Rate (0-1)', '#2ca02c')
    ]
    
    # 2. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    
    for i, (name, series, unit, color) in enumerate(metrics):
        ax = axes[i]
        
        # Clean series for stats (ignore NaNs in Rate which come from 0 hits)
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue
            
        mean_val = clean_series.mean()
        std_val = clean_series.std()
        
        # Z-Score
        if std_val > 0:
            z_scores = (clean_series - mean_val) / std_val
            anomalies = clean_series[np.abs(z_scores) > threshold_z]
        else:
            anomalies = pd.Series(dtype='float64')

        # Plot Normal
        ax.plot(series.index, series, label='Normal', color=color, alpha=0.6)
        
        # Plot Anomalies
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies, color='red', s=50, label=f'Anomalies (Z > {threshold_z})', zorder=5)
            
            # Annotate Top 3
            top_3 = anomalies.abs().sort_values(ascending=False).head(3) # Sort by magnitude (for Z-score, usually magnitude from mean)
            # Actually, for the Series value itself, just sorting by value is often fine, 
            # but for "Anomalies", usually the most extreme Z-scores are interesting.
            # Let's just grab the 3 with highest deviation from mean.
            # We already have the series values in `anomalies`.
            
            # Re-calculate specific Zs for sorting or just sort by value?
            # For Rate: low is bad. For Hits: high is weird.
            # Let's just sort by ABS(Z-score).
            if std_val > 0:
                z_anom = (anomalies - mean_val) / std_val
                # Sort by absolute Z
                top_3_idx = z_anom.abs().sort_values(ascending=False).head(3).index
                top_3_vals = anomalies.loc[top_3_idx]
                
                for date, val in top_3_vals.items():
                    ax.annotate(f'{val:.2f}' if unit=='Rate (0-1)' else f'{val:,.0f}', 
                                xy=(date, val), 
                                xytext=(0, 10), 
                                textcoords='offset points', 
                                ha='center', fontsize=9, color='red', weight='bold')

        # Threshold Lines
        ax.axhline(mean_val, color='black', linestyle='-', alpha=0.3, label='Mean')
        ax.axhline(mean_val + (threshold_z * std_val), color='red', linestyle='--', alpha=0.3, label='Upper Bound')
        ax.axhline(mean_val - (threshold_z * std_val), color='red', linestyle='--', alpha=0.3, label='Lower Bound')
        
        ax.set_title(f'{name} - Anomaly Detection')
        ax.set_ylabel(unit)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Print Top List
        if not anomalies.empty:
            print(f"\nTop Anomalies for {name}:")
            # Sort by deviation magnitude
            if std_val > 0:
                z_anom = (anomalies - mean_val) / std_val
                sorted_anoms = anomalies.loc[z_anom.abs().sort_values(ascending=False).index].head(5)
                # Format
                if unit == 'Rate (0-1)':
                    display(sorted_anoms.map('{:.4f}'.format).to_frame(name=unit))
                else:
                    display(sorted_anoms.map('{:,.0f}'.format).to_frame(name=unit))

    plt.xlabel('Date/Time')
    plt.tight_layout()
    plt.show()

def plot_status_evolution(df, interval='1D'):
    """
    Plots the evolution of HTTP status codes over time using a Stacked Area Chart.
    Useful for seeing how the proportion of 200 vs 404 vs 500 changes.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print(f"Calculating Status Evolution (interval={interval})...")
    
    # Group by Time + Status
    # We use unstack to create columns for each status code (or label)
    col = 'status_label' if 'status_label' in df.columns else 'status'
    evolution = df.groupby([pd.Grouper(freq=interval), col]).size().unstack(fill_value=0)
    
    # Sort columns by total frequency so major statuses are at the bottom
    top_statuses = evolution.sum().sort_values(ascending=False).index
    evolution = evolution[top_statuses]
    
    # Plot
    fig, ax = plt.subplots(figsize=(20, 8))
    
    evolution.plot(kind='area', stacked=True, ax=ax, cmap='tab20', alpha=0.9)
    
    ax.set_title(f'Evolution of HTTP Status Codes over Time ({interval})')
    ax.set_ylabel('Request Count')
    ax.set_xlabel('Date')
    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Status')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def plot_size_distribution(df):
    """
    Plots the distribution of Response Sizes (Bytes) using Log Scales.
    Helps distinguish between 'Tiny Responses' (Redirects/Errors) vs 'Large Assets'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Analyzing Response Size Distribution...")
    
    # Filter out 0-byte responses (usually 304s or errors) for the log plot
    sizes = df['size']
    nonzero_sizes = sizes[sizes > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Linear Scale (Good for identifying if everything is small)
    axes[0].hist(sizes, bins=50, color='gray', alpha=0.7)
    axes[0].set_title('File Size Distribution (Linear Scale)')
    axes[0].set_xlabel('Bytes')
    axes[0].set_ylabel('Frequency')
    
    # 2. Log Scale (Good for spanning orders of magnitude)
    if len(nonzero_sizes) > 0:
        log_bins = np.logspace(np.log10(max(1, nonzero_sizes.min())), np.log10(nonzero_sizes.max()), 50)
        axes[1].hist(nonzero_sizes, bins=log_bins, color='purple', alpha=0.7)
        axes[1].set_title('File Size Distribution (Log-X Scale)')
        axes[1].set_xlabel('Bytes (Log Scale)')
        axes[1].set_xscale('log')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print quick stats
    print(f"Zero-byte responses: {(sizes == 0).sum()} ({ (sizes == 0).mean():.1%})")
    print(f"Small files (<1KB): {(sizes < 1024).sum()} ({ (sizes < 1024).mean():.1%})")
    print(f"Large files (>1MB): {(sizes > 1024**2).sum()} ({ (sizes > 1024**2).mean():.1%})")


def plot_autocorrelation(df, metric='hits', interval='1H', lags=72, figsize=(20, 6)):
    """
    Plots the Autocorrelation Function (ACF) of various metrics to confirm cyclic patterns.
    Args:
        metric (str): 'hits', 'size', or 'rate'.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    print(f"Calculating Autocorrelation for {metric}...")


def plot_timeline_overview(df, interval='30T', figsize=(20, 12)):
    """
    Plots the Macro Overview over the entire timeline.
    Shows 3 Line Plots: Total Hits, Total Size, and Success Rate over time.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print(f"Generating timeline overview for {interval} interval...")
    
    # 1. Resample
    resampler = df.resample(interval)
    
    # Calculate metrics
    ts_hits = resampler.size()
    ts_size = resampler['size'].sum()
    
    # Success Rate
    if 'status_label' in df.columns:
        ts_success = df[df['status_label'] == 'Success'].resample(interval).size()
    else:
        # Fallback
        ts_success = df[(df['status'] >= 200) & (df['status'] < 300)].resample(interval).size()
        
    # Calculate rate
    # Fill NaN with 1.0 or 0.0? If no hits, rate is undefined. 
    # Let's fill with forward fill or just leave gaps. Gaps are better.
    ts_rate = ts_success.div(ts_hits)
    
    # 2. Plotting
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Hits
    axes[0].plot(ts_hits.index, ts_hits.values, color='#1f77b4', linewidth=2)
    axes[0].set_title(f'Total Hits per {interval}', fontsize=14)
    axes[0].set_ylabel('Hits')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Size
    axes[1].plot(ts_size.index, ts_size.values, color='#ff7f0e', linewidth=2)
    axes[1].set_title('Total Size (Bytes)', fontsize=14)
    axes[1].set_ylabel('Bytes')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: Success Rate
    axes[2].plot(ts_rate.index, ts_rate.values, color='#2ca02c', linewidth=2)
    axes[2].set_title('Success Rate (Reliability)', fontsize=14)
    axes[2].set_ylabel('Rate (0-1)')
    # axes[2].set_ylim(0.0, 1.05) # Optional: fix range if desired
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.xlabel('Date/Time')
    plt.tight_layout()
    plt.show()

    # Prepare Time Series based on metric
    if metric == 'hits':
        ts = df.resample(interval).size().fillna(0)
        title = f'Autocorrelation of Traffic Hits (Interval={interval})'
    elif metric == 'size':
        ts = df.resample(interval)['size'].sum().fillna(0)
        title = f'Autocorrelation of Total File Size (Interval={interval})'
    elif metric == 'rate':
        # Result rate = Success / Total
        if 'status_label' in df.columns:
            ts_success = df[df['status_label'] == 'Success'].resample(interval).size()
        else:
            ts_success = df[(df['status'] >= 200) & (df['status'] < 300)].resample(interval).size()
        
        ts_total = df.resample(interval).size()
        # fillna(0) for intervals with no traffic
        ts = ts_success.div(ts_total).fillna(0)
        title = f'Autocorrelation of Success Rate (Interval={interval})'
    else:
        raise ValueError(f"Unknown metric: {metric}")

    fig, ax = plt.subplots(figsize=figsize)
    plot_acf(ts, lags=lags, ax=ax, title=title)
    ax.set_xlabel(f'Lags ({interval})')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)
    
    # Highlight 24h periods if interval is 1H
    if interval == '1H':
        for i in range(24, lags + 1, 24):
            ax.axvline(x=i, color='red', linestyle='--', alpha=0.5)
            ax.text(i, 0.5, f'{i}h', color='red', ha='center', va='bottom')

    plt.show()


def plot_top_ips_activity(df, top_n=10, figsize=(20, 10)):
    """
    Visualizes WHEN the top active IPs are accessing the server.
    Uses a scatter plot (Time of Day vs IP).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"Analyzing Time-of-Day Activity for Top {top_n} IPs...")

    # 1. Identify Top IPs
    top_ips = df['ip'].value_counts().head(top_n).index.tolist()
    
    # 2. Filter Data
    subset = df[df['ip'].isin(top_ips)].copy()
    
    # 3. Extract Time of Day (Hour + Minute/60)
    subset['hour_float'] = subset.index.hour + subset.index.minute / 60.0
    
    # 4. Plot
    plt.figure(figsize=figsize)
    # Using stripplot for scatter-like categorical plotting
    sns.stripplot(x='hour_float', y='ip', data=subset, jitter=0.2, alpha=0.5, hue='ip', palette='tab10', legend=False)
    
    plt.title(f'Activity Times of Top {top_n} IPs')
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('IP Address')
    plt.xticks(range(0, 25, 1))
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.xlim(0, 24)
    plt.show()


def plot_frequency_vs_size(df, figsize=(12, 8)):
    """
    Testing Hypothesis: "High frequency -> large file size, or vice versa".
    Scatter plot of Resource Hit Count vs Average Resource Size.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print("Correlating Resource Frequency vs Size...")

    if 'resource' not in df.columns:
        print("Error: 'resource' column missing.")
        return

    # 1. Group by Resource
    # Calculate Count and Mean Size
    resource_stats = df.groupby('resource')['size'].agg(['count', 'mean'])
    
    # Filter out 0-size if needed (optional)
    # resource_stats = resource_stats[resource_stats['mean'] > 0]
    
    # 2. Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scales because counts and sizes span orders of magnitude
    ax.scatter(resource_stats['count'], resource_stats['mean'], alpha=0.3, color='purple', edgecolors='w', s=30)
    
    ax.set_title('Hypothesis Check: Resource Frequency vs File Size')
    ax.set_xlabel('Hit Count (Log Scale)')
    ax.set_ylabel('Average File Size Bytes (Log Scale)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add trend line?
    # Simple correlation
    corr = resource_stats.corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes, 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.show()


def plot_weekday_vs_weekend(df, interval='1H', figsize=(12, 6)):
    """
    Hypothesis: Traffic at weekday < traffic at weekend.
    Plots distribution of hits comparing Weekdays vs Weekends.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    print("Comparing Weekday vs Weekend Traffic...")

    # 1. Resample counts
    ts = df.resample(interval).size()
    
    # 2. Create Analysis DataFrame
    analysis = pd.DataFrame({'hits': ts})
    analysis['day_name'] = analysis.index.day_name()
    # 5=Saturday, 6=Sunday
    analysis['is_weekend'] = analysis.index.dayofweek >= 5 
    analysis['Category'] = analysis['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
    
    # 3. Calculate Means
    means = analysis.groupby('Category')['hits'].mean()
    print("Mean Traffic per Interval:")
    print(means)
    
    # 4. Plot Boxplot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Boxplot
    sns.boxplot(x='Category', y='hits', data=analysis, ax=axes[0], palette='Set2')
    axes[0].set_title(f'Traffic Distribution ({interval})')
    axes[0].set_ylabel('Hits')
    
    # Barplot for Means
    means.plot(kind='bar', ax=axes[1], color=['#66c2a5', '#fc8d62'], alpha=0.8)
    axes[1].set_title('Average Traffic: Weekday vs Weekend')
    axes[1].set_ylabel('Average Hits')
    axes[1].tick_params(axis='x', rotation=0)
    
    # Annotate means
    for i, v in enumerate(means):
        axes[1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
