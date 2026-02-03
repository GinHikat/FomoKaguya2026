import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import inspect
try:
    from . import plot_utils
except ImportError:
    import plot_utils

def _create_interactive_wrapper(func):
    """
    Creates an interactive wrapper for a plotting function.
    Adds a time range slider to filter the DataFrame before plotting.
    Also adds a dropdown for 'interval' or 'window' if the function supports it.
    """
    def wrapper(df, *args, **kwargs):
        # validation
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: DataFrame index is not DatetimeIndex. Interactivity disabled.")
            return func(df, *args, **kwargs)
        
        if df.empty:
            print("DataFrame is empty.")
            return
            
        start_date = df.index.min()
        end_date = df.index.max()
        
        if pd.isna(start_date) or pd.isna(end_date):
            print("Invalid date index.")
            return func(df, *args, **kwargs)

        # 1. Create Time Range Slider
        # We generate 100 discrete steps for the slider.
        # We use (label, value) pairs to ensure the display is readable (short strings).
        slider_steps = 100
        slider_timestamps = pd.date_range(start=start_date, end=end_date, periods=slider_steps)
        
        # Format labels: 'YYYY-MM-DD HH:MM'
        slider_options = [(t.strftime('%Y-%m-%d %H:%M'), t) for t in slider_timestamps]
        
        slider = widgets.SelectionRangeSlider(
            options=slider_options,
            index=(0, slider_steps - 1),
            description='Time Range',
            orientation='horizontal',
            layout=widgets.Layout(width='95%'),
            continuous_update=False,
            readout=False # Disable truncated readout
        )
        
        # Label to show full range
        s_init = slider_options[0][0]
        e_init = slider_options[-1][0]
        range_label = widgets.Label(value=f"{s_init}  Wait...  {e_init}")
        # We will update it immediately anyway
        
        # 2. Check for Tunable Parameters (interval, window)
        sig = inspect.signature(func)
        params = sig.parameters
        
        extra_widgets = []
        param_widget = None
        param_name = None
        
        # Define common options for aggregation windows
        # Note: 'T' is minute, 'H' is hour, 'D' is day
        time_options = ['1T', '5T', '10T', '15T', '30T', '1H', '2H', '4H', '6H', '12H', '1D', '7D']

        if 'interval' in params:
            param_name = 'interval'
            # Get default or fallback
            default_val = params['interval'].default
            if default_val == inspect.Parameter.empty or default_val not in time_options:
                default_val = '1H'
                
            param_widget = widgets.Dropdown(
                options=time_options,
                value=default_val,
                description='Interval:',
                layout=widgets.Layout(width='200px')
            )
            
        elif 'window' in params:
            param_name = 'window'
            default_val = params['window'].default
            if default_val == inspect.Parameter.empty or default_val not in time_options:
                default_val = '1H'
                
            param_widget = widgets.Dropdown(
                options=time_options,
                value=default_val,
                description='Window:',
                layout=widgets.Layout(width='200px')
            )
            
        if param_widget:
            extra_widgets.append(param_widget)

        # Output widget to capture plot
        out = widgets.Output()
        
        def update_plot(change=None):
            with out:
                clear_output(wait=True)
                # Get time range (actual Timestamps)
                start, end = slider.value
                
                # Update Label
                range_label.value = f"Selected: {start.strftime('%Y-%m-%d %H:%M')}  to  {end.strftime('%Y-%m-%d %H:%M')}"
                
                # Filter logic
                mask = (df.index >= start) & (df.index <= end)
                filtered_df = df.loc[mask]
                
                if filtered_df.empty:
                    print(f"No data in selected range: {start} to {end}")
                    return

                # Prepare updated kwargs
                local_kwargs = kwargs.copy()
                if param_widget and param_name:
                    local_kwargs[param_name] = param_widget.value
                
                # Call the original function
                try:
                    func(filtered_df, *args, **local_kwargs)
                except Exception as e:
                    print(f"Error generating plot: {e}")
        
        # Attach observers
        slider.observe(update_plot, names='value')
        if param_widget:
            param_widget.observe(update_plot, names='value')
        
        # Initial Display
        update_plot()
            
        # Layout: Control bar (dropdowns) above, then slider info, then slider, then plot
        controls = [range_label, slider]
        if extra_widgets:
            controls.insert(0, widgets.HBox(extra_widgets))
            
        display(widgets.VBox(controls + [out]))
        
    # Copy docstring from original function
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__ + "_interactive"
    return wrapper

# Create interactive versions of all functions in plot_utils
print_macro_overview = _create_interactive_wrapper(plot_utils.print_macro_overview)
plot_weekly_heatmap = _create_interactive_wrapper(plot_utils.plot_weekly_heatmap)
analyze_status_distribution = _create_interactive_wrapper(plot_utils.analyze_status_distribution)
plot_weekly_patterns = _create_interactive_wrapper(plot_utils.plot_weekly_patterns)
plot_daily_profile = _create_interactive_wrapper(plot_utils.plot_daily_profile)
plot_file_type_stats = _create_interactive_wrapper(plot_utils.plot_file_type_stats)
plot_top_users = _create_interactive_wrapper(plot_utils.plot_top_users)
plot_status_breakdown = _create_interactive_wrapper(plot_utils.plot_status_breakdown)
plot_rolling_statistics = _create_interactive_wrapper(plot_utils.plot_rolling_statistics)
plot_anomaly_spikes = _create_interactive_wrapper(plot_utils.plot_anomaly_spikes)
plot_status_evolution = _create_interactive_wrapper(plot_utils.plot_status_evolution)
plot_size_distribution = _create_interactive_wrapper(plot_utils.plot_size_distribution)
plot_timeline_overview = _create_interactive_wrapper(plot_utils.plot_timeline_overview)
