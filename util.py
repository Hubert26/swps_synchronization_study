# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpld3
import re
pio.renderers.default='browser'
import math
import os
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from IPython.display import display
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
cooperation_1_time_intervals = {
    (0, 8000): "1_z1_instr",
    (8000, 28000): "1_z1_1_k",
    (28000, 48000): "1_z1_2_m",
    (48000, 68000): "1_z1_3_k",
    (68000, 88000): "1_z1_4_m",
    (88000, 108000): "1_z1_5_k",
    (108000, 128000): "1_z1_6_m",
    (128000, 136000): "1_z1_odp_idle",
    (226000, 226000): "1_z1_odp",
    (246000, 286000): "1_pause",
    (286000, 294000): "1_z2_instr",
    (294000, 314000): "1_z2_1_m",
    (314000, 334000): "1_z2_2_k",
    (334000, 354000): "1_z2_3_m",
    (354000, 374000): "1_z2_4_k",
    (374000, 394000): "1_z2_5_m",
    (394000, 414000): "1_z2_6_k",
    (414000, 422000): "1_z2_odp_idle",
    (422000, 512000): "1_z2_odp",
    (512000, 547000): "1_baseline2_idle",
    (547000, 787000): "1_baseline2",
}

cooperation_2_time_intervals = {
    (0, 8000): "2_z1_instr",
    (8000, 28000): "2_z1_1_k",
    (28000, 48000): "2_z1_2_m",
    (48000, 68000): "2_z1_3_k",
    (68000, 88000): "2_z1_4_m",
    (88000, 108000): "2_z1_5_k",
    (108000, 128000): "2_z1_6_m",
    (128000, 136000): "2_z1_odp_idle",
    (226000, 226000): "2_z1_odp",
    (246000, 286000): "2_pause",
    (286000, 294000): "2_z2_instr",
    (294000, 314000): "2_z2_1_m",
    (314000, 334000): "2_z2_2_k",
    (334000, 354000): "2_z2_3_m",
    (354000, 374000): "2_z2_4_k",
    (374000, 394000): "2_z2_5_m",
    (394000, 414000): "2_z2_6_k",
    (414000, 422000): "2_z2_odp_idle",
    (422000, 512000): "2_z2_odp",
    (512000, 547000): "2_baseline2_idle",
    (547000, 787000): "2_baseline2",
}

baseline_1_time_intervals = {
    (0, 20000): "1_baseline1_idle",
    (20000, 260000): "1_baseline1",
}

baseline_2_time_intervals = {
    (0, 20000): "2_baseline1_idle",
    (20000, 260000): "2_baseline1",
}

#%%
def create_results_folder(base_name="out"):
    """
    Creates a new results folder in the current working directory, with a unique name 
    based on the current date and time.

    Parameters:
    base_name (str, optional): The base name for the new folder. Defaults to "out".

    Returns:
    str: The full path to the newly created folder.
    
    Raises:
    OSError: If there is an error creating the directory.
    """
    
    # Set the current working directory as the output directory
    output_dir = os.getcwd()
    
    # Get the current date and time in the format "YYYYMMDD_HHMMSS"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the name for the new results folder
    new_folder_name = f"{base_name}_{current_time}"
    
    # Generate the full path for the new folder
    folder_path = os.path.join(output_dir, new_folder_name)
    
    # Create the new folder if it doesn't already exist
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path

output_folder = create_results_folder()

#%%
def get_info_from_path(file_path, part=0):
    """
    Extracts a specific part of the file name from a given file path.

    This function retrieves a segment of the file name based on the specified index 
    from the file name (excluding the extension). The file name is split by whitespace 
    or underscores into parts, and the function returns the part corresponding to 
    the given index.

    Parameters:
    file_path (str): The full path to the file from which the information is to be extracted.
    part (int, optional): The index of the part of the file name to return. Defaults to 0.

    Returns:
    str or None: The requested part of the file name if available; otherwise, None.
    
    Raises:
    FileNotFoundError: If the specified file path does not exist.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        # Extract the file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Split the file name into parts based on whitespace or underscores
        index = file_name.split()
        
        # Return the requested part of the file name if it exists
        if len(index) > part:
            return index[part]
        else:
            return None
    except Exception as e:
        # Print any exception that occurs and return None
        print(f"An error occurred: {e}")
        return None
    
#%%
def extract_data_from_file(file_path, df=None):
    """
    Extracts data from a specified file and appends it to a DataFrame.

    This function reads numeric data from a file, processes it, and updates a DataFrame
    with the extracted data. The file should contain numeric data separated by new lines.
    The function also retrieves metadata from the file name to create appropriate time stamps.

    Parameters:
    file_path (str): The path to the file containing the data to be extracted.
    df (pd.DataFrame, optional): The DataFrame to which the extracted data will be appended.
                                  If not provided, a new DataFrame will be created.

    Returns:
    pd.DataFrame: The updated DataFrame containing the extracted and processed data.

    Raises:
    FileNotFoundError: If the file specified by `file_path` does not exist.
    ValueError: If the file contains non-numeric data.
    """
    
    # Initialize DataFrame if not provided
    if df is None:
        df = pd.DataFrame()

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        # Read the file and split its content into lines
        with open(file_path, "r") as file:
            y_data = file.read().split('\n')

        # Validate that all data lines are numeric
        if not all(map(str.isdigit, y_data)):
            raise ValueError("File contains non-numeric data.")

        # Convert the list of string data to float numpy array
        y_data = np.array(y_data).astype(float)

        # Compute cumulative sum for x_data
        x_data = np.cumsum(y_data)

        # Extract metadata from the file name
        shift = 0
        meas_name = get_info_from_path(file_path).split('_')[0]
        start_timestamp = get_info_from_path(file_path, 1) + get_info_from_path(file_path, 2)
        
        # Convert start timestamp to pandas datetime object
        try:
            df_timestamp = pd.to_datetime(start_timestamp, format='%Y-%m-%d%H-%M-%S', errors='raise')
        except ValueError as e:
            print(f"Error: Invalid timestamp format for {start_timestamp} in file {file_path}")
            print(f"Details: {e}")
            
        # Create a new index name for the row
        index_name = f"{get_info_from_path(file_path)}_{str(shift)}"

        # Create a DataFrame with the new row of data
        new_row_data = pd.DataFrame({
            'x': [x_data],
            'y': [y_data],
            'shift': [shift],
            'meas_name': [meas_name],
            'starttime': [df_timestamp],
            'endtime': [df_timestamp + pd.to_timedelta(x_data[-1], unit='ms')],
            'duration': [(df_timestamp + pd.to_timedelta(x_data[-1], unit='ms') - df_timestamp).total_seconds() / 60]
        }, index=[index_name])

        # Append the new row to the existing DataFrame
        df = pd.concat([df, new_row_data])
        return df

    except Exception as e:
        # Print any exception that occurs and return the unchanged DataFrame
        print(f"An error occurred: {e}")
        return df

#%%
def scatter_plot(series_df: pd.DataFrame, title: str = None):
    """
    Creates a scatter plot from the data in the provided DataFrame.

    This function generates a scatter plot where each series in the DataFrame is plotted as a separate trace.
    The x-axis represents time in milliseconds, and the y-axis represents the time between heartbeats (RR intervals).

    Parameters:
    - series_df (pd.DataFrame): DataFrame containing time series data. 
                                The DataFrame should have columns 'x' (time points) and 'y' (RR intervals).
    - title (str, optional): The title of the plot. If not provided, a default title will be generated based on the data.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The generated scatter plot figure.
    - title (str): The title used for the plot.
    
    Raises:
    - TypeError: If `series_df` is not a DataFrame or the 'x' and 'y' columns do not contain numpy arrays.
    - ValueError: If the DataFrame is empty or lacks required columns.
    """

    # Validate the input DataFrame
    if not validate_series_df(series_df):
        print("Validation failed in scatter_plot function.")

    # Determine the overall time range of the series
    stop = max(series_df['x'].apply(lambda arr: arr[-1] if len(arr) > 0 else float('-inf')))
    start = min(series_df['x'].apply(lambda arr: arr[0] if len(arr) > 0 else float('inf')))

    # Generate names from the DataFrame index
    names = series_df.index.tolist()

    # Initialize the scatter plot
    fig = px.scatter()

    # Add each series as a scatter trace
    for i in range(series_df.shape[0]):
        name = names[i].split()[0]  # Use the first part of the index name
        
        fig.add_scatter(
            x=series_df.iloc[i]['x'],
            y=series_df.iloc[i]['y'],
            mode='markers',
            name=name
        )

    # Update plot layout
    fig.update_layout(
        xaxis_title="Time [ms]",
        yaxis_title="Time Between Heartbeats [ms]",
    )

    # Set the title if not provided
    if not title:
        title = f"{names} RANGE from {start} to {stop}"

    fig.update_layout(title=title)

    return fig, title

#%%
def density_plot(series_df: pd.DataFrame, sd_threshold: float = 3, title: str = None):
    """
    Creates a density plot (histogram) for each time series in the provided DataFrame.

    This function generates a density plot where each series in the DataFrame is plotted as a separate histogram trace.
    Vertical lines indicating outliers (based on a specified standard deviation threshold) are added to each plot.

    Parameters:
    - series_df (pd.DataFrame): DataFrame containing time series data.
                                The DataFrame should have an index with names and a 'y' column containing the RR intervals.
    - sd_threshold (float, optional): The standard deviation multiplier for determining outliers. Default is 3.
    - title (str, optional): The title of the plot. If not provided, a default title will be generated.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The generated density plot figure.
    - title (str): The title used for the plot.

    Raises:
    - TypeError: If `series_df` is not a DataFrame, or if the 'y' column does not contain numpy arrays.
    - ValueError: If the DataFrame is empty or lacks the required 'y' column.
    - RuntimeError: If `series_df` contains non-numeric data that cannot be processed.
    """
    
    
    # Validate the input DataFrame
    if not validate_series_df(series_df):
        print("Validation failed in density_plot function.")
    
    # Generate names from the DataFrame index
    names = series_df.index.tolist()

    # Initialize the density plot
    fig = go.Figure()

    # Add each series as a histogram trace
    for i, y_data in enumerate(series_df['y']):
        if not np.issubdtype(y_data.dtype, np.number):
            raise RuntimeError(f"Non-numeric data found in 'y' column for index {names[i]}.")
        
        # Check if y_data is empty or contains only NaN
        if y_data.size == 0 or np.all(np.isnan(y_data)):
            print(f"No valid data for index {names[i]}, skipping this series.")
            continue

        # Create the histogram trace
        trace = go.Histogram(
            x=y_data, 
            histnorm='probability density', 
            name=names[i],
            opacity=0.75
        )
        fig.add_trace(trace)
        
        # Calculate and add vertical lines for outliers
        outlier_low = round(np.nanmean(y_data) - sd_threshold * np.nanstd(y_data))
        outlier_high = round(np.nanmean(y_data) + sd_threshold * np.nanstd(y_data))
        trace_color = "gray"
        
        fig.add_vline(
            x=outlier_low, 
            line_dash="dash", 
            line_color=trace_color,
            annotation_text=f"Outlier Low {names[i]}: {outlier_low}", 
            annotation_position="top left", 
            annotation_textangle=90
        )
        
        fig.add_vline(
            x=outlier_high, 
            line_dash="dash", 
            line_color=trace_color,
            annotation_text=f"Outlier High {names[i]}: {outlier_high}", 
            annotation_position="top right",  
            annotation_textangle=90
        )
    
    # Update plot layout
    fig.update_layout(
        xaxis_title="RR-interval [ms]",
        yaxis_title="Density",
        barmode='overlay'  # Overlay histograms for better comparison
    )
    
    # Set the title if not provided
    if title is None:
        title = "Distribution of RR-intervals"
    
    fig.update_layout(title=title)

    return fig, title

#%%
# =============================================================================
# def corr_heatmap(df, title=None, color='viridis'):
#     # Tworzenie własnej mapy kolorów z 20 odcieniami od -1 do 1
#     colors = sns.color_palette(color, 20)
#     cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=20)
#     
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(10, 10))
#         sns.heatmap(df,
# # =============================================================================
# # to annotate on heatmap you need previous version of matplotlib              
# # pip install matplotlib==3.7.3
# # =============================================================================
#                     annot=df.round(2),
#                     vmax=1,
#                     vmin=-1,
#                     center=0,
#                     square=True,
#                     xticklabels=df.columns,
#                     yticklabels=df.index,
#                     cmap=cmap,
#                     linewidths=.5,
#                     cbar_kws={"shrink": 0.7, 'ticks': np.linspace(-1, 1, 21)})
#         # Ustawienie rotacji etykiet
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#     
#     if not title:
#         title = 'heatmap'
#     
#     plt.title(title)
# 
#     return f, title
# =============================================================================
#%%
def save_plot(fig, file_name: str, folder_name: str = "out", format: str = "png") -> None:
    """
    Save a given plot (either Plotly or Matplotlib) to a file in the specified format and directory.

    This function saves a plot to a specified file format (such as PNG, JPG, SVG, PDF, or HTML) 
    and ensures that the output directory exists. It supports both Plotly and Matplotlib figures.

    Parameters:
    fig (go.Figure or plt.Figure): The plot object to be saved. Can be a Plotly or Matplotlib figure.
    file_name (str): The name of the file (without extension) to save the plot as.
    folder_name (str): The name of the folder where the file will be saved. Defaults to "out".
    format (str): The format in which to save the plot. Supported formats include "png", "jpg", 
                  "svg", "pdf", and "html". Defaults to "png".

    Returns:
    None
    """

    # Ensure the output directory exists; create it if it does not
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create the full path for the output file
    output_file_path = os.path.join(folder_name, f"{file_name}.{format}")
    
    # Check if the figure is a Plotly figure
    if isinstance(fig, go.Figure):
        if format == "html":
            # Save the Plotly figure as an HTML file
            pio.write_html(fig, output_file_path)
        else:
            # Save the Plotly figure as an image file (PNG, JPG, SVG, PDF)
            fig.write_image(output_file_path, format=format)
    
    # Check if the figure is a Matplotlib figure
    elif isinstance(fig, plt.Figure):
        if format == "html":
            # Save the Matplotlib figure as an HTML file using mpld3
            mpld3.save_html(fig, output_file_path)
        else:
            # Save the Matplotlib figure as an image file (PNG, JPG, SVG, PDF)
            fig.savefig(output_file_path, format=format)
    
    # Raise an error if the figure is not a recognized type
    else:
        raise TypeError("The 'fig' parameter must be either a Plotly 'go.Figure' or a Matplotlib 'plt.Figure'.")
    
#%%
def trim(series_df, start=None, end=None):
    """
    Trims the time series data within the specified start and end times.
    
    Parameters:
    - series_df (pd.DataFrame): DataFrame containing time series data with columns 'x', 'y', 'starttime', and 'endtime'.
    - start (int, optional): The start time in milliseconds for trimming the series. Defaults to the maximum start time in the series if not provided.
    - end (int, optional): The end time in milliseconds for trimming the series. Defaults to the minimum end time in the series if not provided.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the trimmed series.
    """
    
    
    # Validate the input DataFrame
    if not validate_series_df(series_df):
        print("Validation failed in trim function.")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    series_df = series_df.copy()
    
    # Determine the maximum start time and minimum end time from the data
    max_start = max(series_df['x'].apply(lambda arr: arr[0]))
    min_stop = min(series_df['x'].apply(lambda arr: arr[-1]))
    
    # Set default values for new_start and new_end
    new_start = max_start if start is None else start
    new_end = min_stop if end is None else end
        
    # Initialize lists to store trimmed series
    trimmed_series = []
    
    # Iterate over each row in the DataFrame
    for i in range(series_df.shape[0]):
        x = series_df.iloc[i]['x']
        y = series_df.iloc[i]['y']
        
        index = series_df.index[i]
        
        # Select indices within the specified time range
        selected_indices_x = np.where((x >= new_start) & (x <= new_end))[0]
        
        new_x = x[selected_indices_x]
        new_y = y[selected_indices_x]
        starttime = series_df.iloc[i]['starttime']
        
        # Check if new_x is empty
        if new_x.size == 0:
            print(f"Trimmed new_x is empty for {index} and range {new_start} to {new_end}.")
            new_starttime = starttime + pd.to_timedelta(new_start, unit='ms')
            new_endtime = starttime + pd.to_timedelta(new_end, unit='ms')
            new_duration = (new_endtime - new_starttime).total_seconds() / 60
        else:   
            new_starttime = starttime + pd.to_timedelta(new_x[0], unit='ms')
            new_endtime = starttime + pd.to_timedelta(new_x[-1], unit='ms')
            new_duration = (new_endtime - new_starttime).total_seconds() / 60
            
        # Update the row using the update_series function
        updated_serie = update_series(
            serie=series_df.iloc[[i]], 
            x=new_x, 
            y=new_y, 
            starttime=new_starttime, 
            endtime=new_endtime, 
            duration=new_duration
        )
        
        # Append the updated row
        trimmed_series.append(updated_serie.iloc[0])
        
    trimmed_series = pd.DataFrame(trimmed_series)
    
    return trimmed_series

#%%
def update_series(
    serie: pd.DataFrame, 
    x: np.ndarray = None, 
    y: np.ndarray = None, 
    shift: int = None, 
    meas_name: str = None, 
    starttime: pd.Timestamp = None, 
    endtime: pd.Timestamp = None, 
    duration: float = None, 
    index: str = None
) -> pd.DataFrame:
    """
    Updates a single-row DataFrame with new values for specified columns.
    
    Parameters:
    - serie (pd.DataFrame): Input single-row DataFrame to be updated.
    - x (np.ndarray, optional): New 'x' value.
    - y (np.ndarray, optional): New 'y' value.
    - shift (int, optional): New 'shift' value.
    - meas_name (str, optional): New 'meas_name' value.
    - starttime (pd.Timestamp, optional): New 'starttime' value.
    - endtime (pd.Timestamp, optional): New 'endtime' value.
    - duration (float, optional): New 'duration' value.
    - index (str, optional): New index for the DataFrame.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with the new values.
    """
    
    # Validate that serie is a single-row DataFrame
    if not validate_series_df(serie, min_size=1, max_size=1):
        print("Validation failed in update_series function.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    serie = serie.copy()
    
    if x is not None:
        if not isinstance(x, np.ndarray):
            value_type = type(x)
            raise TypeError("Input x must be a np.ndarray but is {value_type}.")
# =============================================================================
#         if x.size == 0:
#             raise ValueError("Input x is empty.")
#         if np.all(np.isnan(x)):
#             raise ValueError("Input x is NaN.")
# =============================================================================
        new_x = x
    else:
        new_x = serie.iloc[0]['x']
        
    if y is not None:
        if not isinstance(y, np.ndarray):
            value_type = type(y)
            raise TypeError(f"Input y must be a np.ndarray but is {value_type}.")
# =============================================================================
#         if y.size == 0:
#             raise ValueError("Input y is empty.")
#         if np.all(np.isnan(y)):
#             raise ValueError("Input y is NaN.")
# =============================================================================
        new_y = y
    else:
        new_y = serie.iloc[0]['y']
    
    if shift is not None:
        if not isinstance(shift, (int, float)):
            value_type = type(shift)
            raise TypeError(f"Input shift must be a (int, float) but is {value_type}.")
        if shift == '':
            raise ValueError("Input shift is empty.")
        new_shift = shift
    else:
        new_shift = serie.iloc[0]['shift']
        
    if meas_name is not None:
        if not isinstance(meas_name, str):
            value_type = type(meas_name)
            raise TypeError(f"Input meas_name must be a str but is {value_type}.")
        if meas_name == '':
            raise ValueError("Input meas_name is empty.")
        new_meas_name = meas_name
    else:
        new_meas_name = serie.iloc[0]['meas_name']
        
    if starttime is not None:
        if not isinstance(starttime, pd.Timestamp):
            value_type = type(starttime)
            raise TypeError(f"Input starttime must be a pd.Timestamp but is {value_type}.")
        if pd.isna(starttime):
            raise ValueError("Input starttime is empty.")
        new_starttime = starttime
    else:
        new_starttime = serie.iloc[0]['starttime']

    if endtime is not None:
        if not isinstance(endtime, pd.Timestamp):
            value_type = type(endtime)
            raise TypeError(f"Input endtime must be a pd.Timestamp but is {value_type}.")
        if pd.isna(endtime):
            raise ValueError("Input endtime is empty.")
        new_endtime = endtime
    else:
        new_endtime = serie.iloc[0]['endtime']

    if duration is not None:
        if not isinstance(duration, (int, float)):
            value_type = type(duration)
            raise TypeError(f"Input duration must be a (int, float) but is {value_type}.")
        if math.isnan(duration):
            raise ValueError("Input duration is NaN.")
        new_duration = duration
    else:
        new_duration = serie.iloc[0]['duration']
        
    if index is not None:
        if not isinstance(index, str):
            value_type = type(index)
            raise TypeError(f"Input index must be a str but is {value_type}.")
        if index == '':
            raise ValueError("Input index is empty.")
        new_index = index
    else:
        new_index = serie.index[0]
        
    # Create a DataFrame with the new row of data
    updated_serie = pd.DataFrame({
        'x': [new_x],
        'y': [new_y],
        'shift': [new_shift],
        'meas_name': [new_meas_name],
        'starttime': [new_starttime],
        'endtime': [new_endtime],
        'duration': [new_duration]
    }, index=[new_index])  
    
   
    return updated_serie

#%%
def validate_series_df(series_df: pd.DataFrame, min_size: int = None, max_size: int = None, threshold_samples: float = None) -> bool:
    """
    Validates if a given DataFrame meets the required criteria for measurement data.

    Parameters:
    - series_df (pd.DataFrame): The DataFrame to be validated.
    - min_size (int, optional): The minimum number of allowed rows in the DataFrame. If `None`, no size limit is applied.
    - max_size (int, optional): The maximum number of allowed rows in the DataFrame. If `None`, no size limit is applied.
    - threshold_samples (float, optional): The minimum ratio of the actual number of samples in 'x' 
                                           compared to the estimated number of samples. If `None`, no checking sample ratio applied.

    Returns:
    - bool: True if the DataFrame meets all the criteria, False otherwise.
    """
    errors = []
    
    pd.set_option('display.max_columns', None)
    
    try:
        # Validate input type
        if not isinstance(series_df, pd.DataFrame):
            errors.append(f"Input must be a pd.DataFrame but is {type(series_df)}.")
        
        # Validate DataFrame size
        series_size = series_df.shape[0]
        if max_size is not None and series_size > max_size:
            errors.append(f"The input DataFrame must contain no more than {max_size} rows but has {series_size} rows.")
        if min_size is not None and series_size < min_size:
            errors.append(f"The input DataFrame must contain no less than {min_size} rows but has {series_size} rows.")
        
        # Define required columns and their expected types
        expected_columns = {
            'x': np.ndarray,
            'y': np.ndarray,
            'shift': (int, float),
            'meas_name': str,
            'starttime': pd.Timestamp,
            'endtime': pd.Timestamp,
            'duration': (int, float)
        }
        
        # Check for the presence of required columns and validate their data types
        for col, expected_type in expected_columns.items():
            if col not in series_df.columns:
                errors.append(f"'{col}' does not exist in the DataFrame.")
                continue  # Skip the type check if the column is missing
            
            # Validate the type of data in the column
            for i, value in enumerate(series_df[col]):
                if not isinstance(value, expected_type):
                    errors.append(f"The value in column '{col}' at row {i} must be of type {expected_type} but is {type(value)}.")
                
                # Additional checks for columns 'x' and 'y' to ensure np.ndarray size > 3
                if col in ['x', 'y']:
                    if isinstance(value, np.ndarray):
                        # Check for valid elements (non-NaN and non-empty)
                        valid_elements = value[~np.isnan(value) & (value != '')]
                        if valid_elements.size <= 3:
                            errors.append(f"The array in column '{col}' at row {i} must have more than 3 valid elements but has {valid_elements.size}.")
                            
    except Exception as e:
        errors.append(f"Unexpected error: {e}")
    
    # If there are any errors so far, print them and return False
    if errors:
        print(f"Validation errors found for indices: {series_df.index.tolist()}")
        display(series_df)
        for error in errors:
            print(f"- {error}")
        return False

    # Additional validation for 'x' length compared to estimated samples
    for i, row in series_df.iterrows():
        x_array = row['x']
        y_array = row['y']
        starttime = row['starttime']
        endtime = row['endtime']
    
        if threshold_samples is not None:
            if isinstance(x_array, np.ndarray) and isinstance(y_array, np.ndarray):
                # Calculate estimated_samples based on the duration and average y value
                duration_ms = (endtime - starttime).total_seconds() * 1000
                avg_y_interval = np.mean(y_array)
                if not np.isnan(avg_y_interval) and avg_y_interval > 0:
                    estimated_samples = duration_ms / avg_y_interval
                    
                    # Check if the length of 'x' is at least the threshold ratio of estimated_samples
                    if x_array.size < threshold_samples * estimated_samples:
                        errors.append(f"The length of 'x' array at row {i} is {x_array.size}, which is less than {threshold_samples * 100:.0f}% of estimated_samples ({estimated_samples:.2f}).")
                else:
                    errors.append(f"Invalid average y interval at row {i}.")
            else:
                errors.append(f"Invalid data types in row {i} for 'x' or 'y'.")
    
    # If there are any additional errors, print them and return False
    if errors:
        print(f"Validation errors found for indices: {series_df.index.tolist()}")
        display(series_df)
        for error in errors:
            print(f"- {error}")
        return False
    
    pd.reset_option('display.max_columns')
    
    return True
#%%
def shift_series(series_df: pd.DataFrame, shift_time_ms: float = 0) -> pd.DataFrame:
    """
    Shifts the x-values (time series) in a DataFrame by a specified amount of time.

    This function adjusts the x-values in the time series by adding a specified 
    shift (in milliseconds) and updates other associated fields like `shift` 
    and the row index accordingly.

    Parameters:
    series_df (pd.DataFrame): The input DataFrame containing time series data. 
                              Each row represents a different series with columns 
                              like 'x', 'y', 'meas_name', and others.
    shift_time_ms (float): The amount of time (in milliseconds) by which to shift 
                         the x-values.

    Returns:
    pd.DataFrame: A new DataFrame with the x-values shifted by the specified 
                  amount and other associated fields updated.
    """

    # Validate the input DataFrame
    if not validate_series_df(series_df):
        print("Validation failed in shift_series function.")
            
    # Create a copy of the DataFrame to avoid modifying the original
    series_df = series_df.copy()
    #series_df.reset_index(drop=True, inplace=True)

    # Initialize an empty DataFrame with the same columns as the original
    shifted_df = pd.DataFrame(columns=series_df.columns)

    # Iterate over each row in the DataFrame
    for i in range(series_df.shape[0]):
        # Shift the 'x' values by the specified time in milliseconds
        new_x = series_df.iloc[i]['x'] + shift_time_ms

        # Calculate the new shift value in seconds
        new_shift = int(pd.to_timedelta(shift_time_ms, unit='ms').total_seconds())

        # Create a new index based on the measurement name and new shift
        new_index = f"{series_df.iloc[i]['meas_name']}_{new_shift}"

        # Update the current row with the shifted data
        updated_row = update_series(
            serie=series_df.iloc[[i]],  # Wrapping in double brackets to ensure it remains a DataFrame
            x=new_x,
            y=None,
            shift=new_shift,
            meas_name=None,
            starttime=None,
            endtime=None,
            duration=None,
            index=new_index
        )

        # Ensure that updated_row is a DataFrame (handling case where it might be a Series)
        if isinstance(updated_row, pd.Series):
            updated_row = pd.DataFrame([updated_row])  # Convert to DataFrame if necessary
        
        # Append the updated row to the result DataFrame
        shifted_df = pd.concat([shifted_df, updated_row], ignore_index=False)
        
        # Ensure that updated_row is a DataFrame (handling case where it might be a Series)
        if isinstance(shifted_df, pd.Series):
            shifted_df = pd.DataFrame([shifted_df])  # Convert to DataFrame if necessary
        
        # Validate that series_df is a DataFrame
        validate_series_df(series_df, min_size=None, max_size=None)
        
    return shifted_df

#%%
def fisher_transform(x: float) -> float:
    """
    Applies the Fisher transformation to a correlation coefficient value.

    The Fisher transformation is used to transform Pearson correlation 
    coefficients into values that are approximately normally distributed, 
    making them more suitable for hypothesis testing or confidence interval 
    calculations.

    Parameters:
    x (float): The correlation coefficient value to transform. The value 
               should be in the range [-1, 1].

    Returns:
    float: The transformed value. Returns `np.inf` if the input is 1, 
           and `-np.inf` if the input is -1.

    Notes:
    - The function rounds the result to 4 decimal places for precision.
    - The function will return positive or negative infinity if the input 
      is exactly 1 or -1, respectively, as these values correspond to 
      perfect correlation, where the Fisher transform is undefined.
    """

    # Handle special cases where the correlation coefficient is exactly 1 or -1
    if x == 1:
        return np.inf
    elif x == -1: 
        return -np.inf
    
    # Apply the Fisher transformation and round the result to 4 decimal places
    return round(0.5 * np.log((1 + x) / (1 - x)), 4)

#%%
def interp_pair_uniform_time(pair_df: pd.DataFrame, ix_step: float, method: str = 'linear') -> pd.DataFrame:
    """
    Interpolates two signals to a common uniform time axis within their overlapping time range.
    
    Parameters:
    pair_df (pd.DataFrame): DataFrame containing exactly two rows, each representing a signal with 'x' and 'y' values.
    ix_step (float): Step size for the uniform time axis (in the same units as the 'x' values).
    method (str): Interpolation method, e.g., 'linear', 'nearest', 'cubic', 'quadratic'.
    
    Returns:
    pd.DataFrame: DataFrame containing the two signals interpolated to a common time axis.
    
    Raises:
    ValueError: If the input DataFrame does not contain exactly two rows for comparison.
    """
    
    # Ensure the DataFrame contains exactly two rows
    if not validate_series_df(pair_df, min_size=2, max_size=2):
        print("Validation failed in interp_pair_uniform_time function.")
    
    # Extract x and y data for both signals
    x1, y1 = pair_df.iloc[0]['x'], pair_df.iloc[0]['y']
    x2, y2 = pair_df.iloc[1]['x'], pair_df.iloc[1]['y']

    # Convert to numpy arrays if they are not already
    x1 = np.array(x1) if not isinstance(x1, np.ndarray) else x1
    y1 = np.array(y1) if not isinstance(y1, np.ndarray) else y1
    x2 = np.array(x2) if not isinstance(x2, np.ndarray) else x2
    y2 = np.array(y2) if not isinstance(y2, np.ndarray) else y2

    # Determine the common time range between the two signals
    max_common_x = min(np.max(x1), np.max(x2))
    min_common_x = max(np.min(x1), np.min(x2))
    
    # Generate a new uniform time axis within the common range
    ix = np.arange(min_common_x, max_common_x + ix_step, ix_step)
    
    # Interpolate both signals to the uniform time axis
    iy1 = interp1d(x1, y1, kind=method, fill_value='extrapolate')(ix)
    iy2 = interp1d(x2, y2, kind=method, fill_value='extrapolate')(ix)
    
    # Update the original rows with the interpolated data
    interp_meas1 = update_series(pair_df.iloc[[0]], x=ix, y=iy1)
    interp_meas2 = update_series(pair_df.iloc[[1]], x=ix, y=iy2)
    
    return pd.concat([interp_meas1, interp_meas2], ignore_index=False)
    
#%%
def merge_meas(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges measurements in the provided DataFrame by concatenating 
    time series data for entries with the same 'meas_name' and updates 
    the metadata accordingly.

    Parameters:
    series_df (pd.DataFrame): A DataFrame containing time series data 
                              with associated metadata, including 
                              'meas_name', 'starttime', and 'endtime'.

    Returns:
    pd.DataFrame: A DataFrame with merged measurements where possible,
                  with updated series data and metadata.
    
    Notes:
    - The function assumes that the input DataFrame `series_df` contains 
      columns 'x', 'y', 'starttime', 'endtime', 'meas_name', and 'shift'.
    - Measurements with the same 'meas_name' are merged based on their 
      time series data and start/end times.
    """

    # Validate the input DataFrame
    if not validate_series_df(series_df):
        print("Validation failed in merge_meas function.")
    
    # Sort the DataFrame by start time to ensure proper ordering for merging
    series_df = series_df.sort_values(by='starttime')
    
    # Initialize an empty DataFrame with the same columns as the input
    merged_df = pd.DataFrame(columns=series_df.columns)
    
    # Group the DataFrame by 'meas_name'
    for meas_name, group in series_df.groupby('meas_name'):
        if group.shape[0] > 1:
            # Calculate the time difference between consecutive measurements
            group['diff_ms'] = group['starttime'].diff().fillna(pd.Timedelta(seconds=0)).apply(lambda x: x.total_seconds() * 1000)
            
            # Adjust 'x' values by adding the cumulative sum of time differences
            group['x'] = group['x'] + group['diff_ms'].cumsum().astype(int)
            
            # Merge the data by concatenating 'x' and 'y' values
            updated_row = update_series(
                serie=group.iloc[[0]],  # Use the first row as the base
                x=np.concatenate(group['x'].values), 
                y=np.concatenate(group['y'].values),
                shift=group.iloc[0]['shift'],
                meas_name=meas_name,
                start_time=group['starttime'].min(), 
                end_time=group['endtime'].max(), 
                duration=(group['endtime'].max() - group['starttime'].min()).total_seconds() / 60,
                index=f"{meas_name}_{group.iloc[0]['shift']}"
            )
        else:
            # If there is only one measurement, just update its index
            updated_row = update_series(
                serie=group.iloc[[0]],
                index=f"{meas_name}_{group.iloc[0]['shift']}"
            )
        
        # Append the updated row to the merged DataFrame
        merged_df = pd.concat([merged_df, updated_row], ignore_index=False)
    
    return merged_df

#%%
def extract_numeric_suffix(text: str):
    """
    Extracts the numeric suffix from a given string.

    This function searches for a sequence of digits at the end of the input string. 
    If found, it returns the numeric suffix as an integer. If no numeric suffix is found, it returns None.

    Parameters:
    text (str): The input string from which the numeric suffix is to be extracted.

    Returns:
    int or None: The numeric suffix as an integer if found, otherwise None.
    """
    match = re.search(r'\d+$', text)  # Search for digits at the end of the string
    if match:
        return int(match.group())  # Return the matched digits as an integer
    else:
        return None  # Return None if no digits are found

#%%
def find_pairs(series_df: pd.DataFrame, serie_1_letter_suffix: str = 'm', serie_2_letter_suffix: str = 'k', index: bool = False):
    """
    Finds matching pairs of series in a DataFrame based on their measurement names.

    This function identifies pairs of series whose measurement names share the same prefix and numeric suffix,
    but differ by a specific letter suffix. It returns the matching pairs along with any unmatched measurements.

    Parameters:
    series_df (pd.DataFrame): The DataFrame containing the series data. It should have a column 'meas_name' 
                              with measurement names to be compared.
    serie_1_letter_suffix (str, optional): The letter suffix to be matched in the first series. Defaults to 'm'.
    serie_2_letter_suffix (str, optional): The letter suffix to be matched in the second series. Defaults to 'k'.
    index (bool, optional): If True, returns pairs of indices. If False, returns pairs of measurement names. Defaults to False.

    Returns:
    tuple:
        - list of tuples: Each tuple contains a pair of matching indices or measurement names based on the `index` parameter.
        - list of str: Measurement names that do not have a matching pair.
    """
    
    # Validate the input DataFrame
    if not validate_series_df(series_df):
        print("Validation failed in find_pairs function.")
    
    matching_pairs = set()  # Use a set to store unique pairs
    
    for index_1, row_1 in series_df.iterrows():
        for index_2, row_2 in series_df.iterrows():
            serie_1_numeric_suffix = extract_numeric_suffix(row_1['meas_name'])
            serie_2_numeric_suffix = extract_numeric_suffix(row_2['meas_name'])
            
            # Check if the measurements match the specified criteria
            if (row_1['meas_name'][:2] == row_2['meas_name'][:2] and
                    row_1['meas_name'][2] == serie_1_letter_suffix and row_2['meas_name'][2] == serie_2_letter_suffix and
                    serie_1_numeric_suffix is not None and serie_2_numeric_suffix is not None and
                    serie_1_numeric_suffix == serie_2_numeric_suffix):
                
                if index:
                    matching_pairs.add((index_1, index_2))  # Add index pair to the set
                else:
                    matching_pairs.add((row_1['meas_name'], row_2['meas_name']))  # Add measurement name pair to the set
    
    # Convert set to list of tuples
    matching_pairs = [tuple(pair) for pair in matching_pairs]
    
    # Gather all unique measurement names
    all_meas_names = set(series_df['meas_name'])
    
    # Identify measurements that do not have a matching pair
    unmatched_meas = all_meas_names - set(pair[0] for pair in matching_pairs) - set(pair[1] for pair in matching_pairs)
    
    return matching_pairs, list(unmatched_meas)
    
#%%
def calculate_pair_time_difference(pair_df: pd.DataFrame, time: str) -> pd.DataFrame:
    """
    Calculate the time difference in milliseconds between two signals based on their start or end times.
    
    Parameters:
    pair_df (pd.DataFrame): DataFrame containing exactly two rows representing the two signals to compare.
    time (str): A string specifying the time to compare ('starttime' or 'endtime').
    
    Returns:
    pd.DataFrame: A DataFrame containing the names of the signals and the time difference in milliseconds.
    
    Raises:
    ValueError: If the input DataFrame does not contain exactly two rows or if the specified time is invalid.
    """
    
    # Ensure the DataFrame contains exactly two rows
    if not validate_series_df(pair_df, min_size=2, max_size=2):
        print("Validation failed in calculate_pair_time_difference function.")
    
    # Extract the two series from the DataFrame
    serie_1 = pair_df.iloc[[0]]
    serie_2 = pair_df.iloc[[1]]
    
    # Check and calculate time difference based on the specified time ('starttime' or 'endtime')
    if time == 'starttime':
        time1 = pd.to_datetime(serie_1.iloc[0]['starttime'])
        time2 = pd.to_datetime(serie_2.iloc[0]['starttime'])
    elif time == 'endtime':
        time1 = pd.to_datetime(serie_1.iloc[0]['endtime'])
        time2 = pd.to_datetime(serie_2.iloc[0]['endtime'])
    else:
        raise ValueError("The 'time' parameter must be either 'starttime' or 'endtime'.")
    
    # Calculate the time difference in milliseconds if both times are valid Timestamps
    if isinstance(time1, pd.Timestamp) and isinstance(time2, pd.Timestamp):
        diff_time_ms = (time1 - time2).total_seconds() * 1000
    else:
        diff_time_ms = np.nan  # Handle invalid Timestamps by setting the difference to NaN
    
    # Create a DataFrame to store the results
    pair_time_diff = pd.DataFrame({
        'meas_name_1': [serie_1.iloc[0]['meas_name']],
        'meas_name_2': [serie_2.iloc[0]['meas_name']],
        'diff_time_ms': [diff_time_ms]
    })
    
    return pair_time_diff

#%%
def time_align_pair(pair_df: pd.DataFrame, time_ms_threshold: float) -> pd.DataFrame:
    """
    Align the start times of two signals within a specified time threshold by shifting one of the signals.
    
    Parameters:
    pair_df (pd.DataFrame): DataFrame containing exactly two rows representing the two signals to be aligned.
    time_ms_threshold (float): The time difference threshold in milliseconds within which the signals should be aligned.
    
    Returns:
    pd.DataFrame: A DataFrame containing the aligned signals.
    
    Raises:
    ValueError: If the input DataFrame does not contain exactly two rows for comparison.
    """
    
    # Ensure the DataFrame contains exactly two rows
    if not validate_series_df(pair_df, min_size=2, max_size=2):
        print("Validation failed in time_align_pair function.")
    
    # Calculate the time difference between the start times of the two signals
    diff_start_time_ms = calculate_pair_time_difference(pair_df, time='starttime')
    
    # Extract the two series from the DataFrame
    serie_1 = pd.DataFrame([pair_df.iloc[0]])
    serie_2 = pd.DataFrame([pair_df.iloc[1]])
    
    # Save the indexes of the two series into separate variables
    index_serie_1 = serie_1.index[0]
    index_serie_2 = serie_2.index[0]
    
    shift_serie_1 = float(serie_1.iloc[0]['shift'])
    shift_serie_2 = float(serie_2.iloc[0]['shift'])
    
    # Align the start times by shifting one of the signals if the difference exceeds the threshold
    if diff_start_time_ms['diff_time_ms'].iloc[0] > time_ms_threshold:
        serie_2 = shift_series(serie_2, shift_time_ms=diff_start_time_ms['diff_time_ms'].iloc[0].item() - time_ms_threshold)
    elif diff_start_time_ms['diff_time_ms'].iloc[0] < -time_ms_threshold: 
        serie_1 = shift_series(serie_1, shift_time_ms=abs(diff_start_time_ms['diff_time_ms'].iloc[0].item()) - time_ms_threshold)
    
    validate_series_df(serie_1, min_size=1, max_size=1)
    validate_series_df(serie_2, min_size=1, max_size=1)
    
    serie_1 = update_series(
        serie=serie_1,
        shift=shift_serie_1,
        index=index_serie_1        
        )
    
    serie_2 = update_series(
        serie=serie_2,
        shift=shift_serie_2,
        index=index_serie_2        
        )
    
    # Concatenated DataFrame of the aligned signals
    aligned_df = pd.concat([serie_1, serie_2]) 
    
    return aligned_df

#%%
def filter_rr_intervals_by_sd(rr: np.ndarray, sd_threshold: float = 3) -> np.ndarray:
    """
    Filters RR intervals by replacing outliers with NaN based on a standard deviation threshold.

    This function identifies outliers in the RR intervals based on a given standard deviation threshold.
    Outliers are defined as values that fall outside the range of mean ± (sd_threshold * standard deviation).

    Parameters:
    rr (np.ndarray): An array of RR intervals.
    sd_threshold (int, optional): The number of standard deviations used to define outliers. Default is 3.

    Returns:
    np.ndarray: The filtered array with outlier values replaced by NaN.
    """
    # Ensure the input is a numpy array of floats
    if not isinstance(rr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    # Convert to float if necessary to handle NaN values
    if not np.issubdtype(rr.dtype, np.floating):
        rr = rr.astype(float)
    
    # Compute the mean and standard deviation of the RR intervals, ignoring NaNs
    mean_rr = np.nanmean(rr)
    std_rr = np.nanstd(rr)
    
    # Calculate the lower and upper bounds for outliers
    outlier_low = mean_rr - sd_threshold * std_rr
    outlier_high = mean_rr + sd_threshold * std_rr
    
    # Identify and replace outliers with NaN
    mask = (rr < outlier_low) | (rr > outlier_high)
    rr[mask] = np.nan
    
    return rr

#%%
def filter_rr_intervals_by_relative_mean(rr: np.ndarray, threshold_factor: float) -> np.ndarray:
    """
    Filters RR intervals by replacing values that deviate more than a specified percentage from the average
    of the previous and next RR intervals with NaN. This function handles NaN values in the input array.

    Parameters:
    rr (np.ndarray): An array of RR intervals, which may contain NaN values.
    threshold_factor (float): The percentage threshold used to define outliers. For example, 0.1 means that
                              intervals deviating more than 10% from the average of their neighbors are considered outliers.

    Returns:
    np.ndarray: The filtered array with outlier values replaced by NaN.
    """
    
    # Ensure the input is a numpy array
    if not isinstance(rr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    if rr.size < 3:
        raise ValueError("Input array must contain at least three valid elements.")
    
    # Create a copy of the array to avoid modifying the original data
    filtered_rr = rr.copy()
    
    # Calculate the mean of previous and next RR intervals, ignoring NaN values
    mean_prev_next = np.full_like(rr, np.nan, dtype=float)  # Initialize with NaN
    
    for i in range(1, len(rr) - 1):
        prev_rr = rr[i - 1]
        next_rr = rr[i + 1]
        
        if not np.isnan(prev_rr) and not np.isnan(next_rr):
            mean_prev_next[i] = np.nanmean([prev_rr, next_rr])
    
    # Replace outliers with NaN
    for i in range(1, len(rr) - 1):
        if not np.isnan(mean_prev_next[i]):
            if rr[i] > (1 + threshold_factor) * mean_prev_next[i] or rr[i] < (1 - threshold_factor) * mean_prev_next[i]:
                filtered_rr[i] = np.nan
    
    return filtered_rr

#%%
def interpolate_nan_values(rr: np.ndarray) -> np.ndarray:
    """
    Interpolates NaN values in a NumPy array using linear interpolation based on non-NaN values.
    
    Parameters:
    rr (np.ndarray): A 1D NumPy array containing RR intervals, which may include NaN values.
    
    Returns:
    np.ndarray: The input array with NaN values replaced by interpolated values.
    
    Notes:
    - The function uses linear interpolation to estimate NaN values based on their surrounding non-NaN values.
    - If the input array does not contain any NaN values, the function will return the array unchanged.
    """
    
    # Find indices of NaN values
    nans = np.isnan(rr)
    
    if np.any(nans):
        # Indices that are not NaN
        not_nans = ~nans
        x = np.arange(len(rr))
        
        # Interpolate NaN values using linear interpolation
        rr[nans] = np.interp(x[nans], x[not_nans], rr[not_nans])
    
    return rr

#%%
def filter_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and updates each series in the provided DataFrame by applying 
    several processing steps including outlier removal, interpolation, and 
    updating series attributes.

    Parameters:
    series_df (pd.DataFrame): A DataFrame where each row contains a time series
                              with RR intervals and metadata.

    Returns:
    pd.DataFrame: A DataFrame containing the filtered and updated series.
    
    Notes:
    - The function assumes that the DataFrame `series_df` has columns 'x', 'y', 
      'starttime', and 'endtime', where 'x' and 'y' are NumPy arrays of RR intervals.
    - The functions `filter_rr_intervals_by_sd` and `filter_rr_intervals_by_relative_mean` 
      are used to filter out RR intervals based on standard deviation and relative mean.
    - The function `interpolate_nan_values` is used to handle NaN values in the RR intervals.
    - The function `update_series` is used to create a DataFrame with updated series data.
    """
    # Validate that serie is a DataFrame
    if not validate_series_df(series_df, min_size=None, max_size=None):
        print("Validation failed in filter_series function.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    series_df = series_df.copy()
    
    # Initialize lists to store filtered series
    filtered_series = []
    
    for i in range(len(series_df)):
        # Extract the RR intervals for the current row
        rr = series_df.iloc[i]['y']
        
        if not isinstance(rr, np.ndarray):
            raise TypeError("RR intervals must be a numpy array.")
        
        # Apply filtering to remove outliers based on standard deviation and relative mean
        filtered_rr = filter_rr_intervals_by_sd(rr, sd_threshold=3)
        filtered_rr = filter_rr_intervals_by_relative_mean(filtered_rr, threshold_factor=0.2)
        
        # Interpolate NaN values in the filtered RR intervals
        nn = interpolate_nan_values(filtered_rr)
                
        # Update the row with new data and append to the result DataFrame
        updated_serie = update_series(
            serie=series_df.iloc[[i]],
            x=None,
            y=nn,
            shift=None,
            meas_name=None,  # Assuming no change in measurement name
            starttime=None,  # Assuming no change in start time
            endtime=None,
            duration=None,
            index=None  # Assuming no change in index
        )
        
        filtered_series.append(updated_serie.iloc[0])
        
    filtered_series = pd.DataFrame(filtered_series)
    
    return filtered_series

#%%
def calc_corr(pair_df: pd.DataFrame):
    """
    Calculates the Pearson correlation coefficient and p-value for two time series 
    in a DataFrame, after trimming and interpolating them to a uniform time grid.

    Parameters:
    -----------
    pair_df : pd.DataFrame
        A DataFrame containing two time series to be compared. 
        The DataFrame should have columns 'x' (time), 'y' (signal values), 
        and 'shift' (time shift) for each series.

    Returns:
    --------
    corr_result : pd.DataFrame
        A DataFrame containing the correlation coefficient (`corr`), p-value (`p_val`), 
        the indices of the two series (`serie_1`, `serie_2`), and the difference 
        in time shifts (`shift_diff`) between the two series.
    
    interp_pair_df : pd.DataFrame
        The interpolated DataFrame containing the two series after trimming and 
        interpolation on a uniform time grid.
    """
    # Validate that serie is a 2 row DataFrame
    if not validate_series_df(pair_df, min_size=2, max_size=2, threshold_samples=0.7):
        print("Validation failed in calc_corr function.")
    
    # Trim the pair of series to the common time range
    #trimmed_pair_df = trim(pair_df)
    
    # Interpolate both series to a uniform time grid
    interp_pair_df = interp_pair_uniform_time(pair_df, ix_step=250, method='quadratic')
    
    # Extract the interpolated series
    serie_1 = interp_pair_df.iloc[[0]]
    serie_2 = interp_pair_df.iloc[[1]]
    
    # Calculate the Pearson correlation coefficient and p-value
    corr, p_val = pearsonr(serie_1['y'].iloc[0], serie_2['y'].iloc[0])
    
    # Prepare the result DataFrame with correlation results and series information
    corr_result = pd.DataFrame({
    'corr': [corr],
    'p_val': [p_val],
    'serie_1': [str(serie_1.index[0])],  # Index of the first series
    'serie_2': [str(serie_2.index[0])],  # Index of the second series
    'shift_diff': [serie_1['shift'].iloc[0] - serie_2['shift'].iloc[0]]  # Time shift difference
    })
    
    return corr_result, interp_pair_df

#%%
# =============================================================================
# def process_rr_data(series_df: pd.DataFrame, group_label: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
#     """
#     Processes RR interval data by finding matching pairs, aligning them in time,
#     trimming to specific time frames, and calculating correlations. Generates and 
#     saves various plots during the process.
# 
#     Parameters:
#     - series_df (pd.DataFrame): The input DataFrame containing series data.
#     - group_label (str): The label used for grouping in the output files.
#     - start_time_ms (int): The start time for trimming the series (in milliseconds).
#     - end_time_ms (int): The end time for trimming the series (in milliseconds).
# 
#     Returns:
#     - pd.DataFrame: DataFrame containing the best correlation results.
#     """
# 
#     # Validate that series_df is a DataFrame and has at least 2 rows for pairing
#     if not validate_series_df(series_df, min_size=2):
#         raise ValueError("Input DataFrame failed validation.")
#     
#     # Find matching pairs and unmatched series
#     matching_pairs, unmatched_series = find_pairs(series_df)
#     
#     best_corr_results = []
#     
#     # Process each matching pair
#     for pair in matching_pairs:
#         # Extract the corresponding rows from series_df for the given pair
#         pair_df = series_df.loc[series_df['meas_name'].isin(pair)].iloc[[0, 1]]
#         
#         # Align the pair in time
#         pair_df = time_align_pair(pair_df, time_ms_threshold=1000)
#         
#         # Trim the pair to the specified time frame
#         trimmed_pair_df = trim(pair_df, start_time_ms, end_time_ms)
#         
#         if not validate_series_df(series_df = trimmed_pair_df, threshold_samples = 0.7):
#             print(f"Validation failed in proces_rr_data for pair: {pair}. Skipping this pair.")
#             continue
#         
#         # Generate and save a histogram plot for the trimmed pair
#         output_plot_folder = os.path.join(output_folder, "hist_pairs_trimmed", group_label)
#         fig_hist, title_hist = density_plot(trimmed_pair_df, title=f"{pair}_Distribution of RR-intervals")
#         save_plot(fig_hist, title_hist, folder_name=output_plot_folder, format="html")
#         
#         # Generate and save a scatter plot for the trimmed pair
#         output_plot_folder = os.path.join(output_folder, "scatter_pairs_trimmed", group_label)
#         fig_scatter, title_scatter = scatter_plot(trimmed_pair_df)
#         save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
#         
#         # Filter the trimmed pair to remove outliers or unwanted data
#         filtered_pair_df = filter_series(trimmed_pair_df)
#         
#         # Generate and save a histogram plot for the filtered pair
#         output_plot_folder = os.path.join(output_folder, "hist_pairs_filtered", group_label)
#         fig_hist, title_hist = density_plot(filtered_pair_df, title=f"{pair}_Distribution of RR-intervals")
#         save_plot(fig_hist, title_hist, folder_name=output_plot_folder, format="html")
#         
#         # Generate and save a scatter plot for the filtered pair
#         output_plot_folder = os.path.join(output_folder, "scatter_pairs_filtered", group_label)
#         fig_scatter, title_scatter = scatter_plot(filtered_pair_df)
#         save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
#         
#         # Initialize lists to store correlation results
#         calc_corr_results = []
#     
#         # Calculate correlation for the initial pair without time shift
#         calc_corr_result, interp_pair_df = calc_corr(filtered_pair_df)
#         calc_corr_results.append(calc_corr_result)
#         
#         # Generate and save a scatter plot for the interp pair
#         output_plot_folder = os.path.join(output_folder, "scatter_pairs_interp", group_label, str(pair))
#         fig_scatter, title_scatter = scatter_plot(interp_pair_df)
#         save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
#     
#         # Obtain the two series from the trimmed pair
#         serie_1 = trimmed_pair_df.iloc[[0]]
#         serie_2 = trimmed_pair_df.iloc[[1]]
#         
#         # Iterate over various time shift values
#         for shift_ms in range(1000, 5001, 1000):
#             # Shift and calculate correlations for serie 1 and shifted series 2
#             shifted_serie_2 = shift_series(serie_2, shift_ms)    
#             calc_corr_result, interp_pair_df = calc_corr(pd.concat([serie_1, shifted_serie_2], ignore_index=False))
#             calc_corr_results.append(calc_corr_result)
#             
#             # Generate and save a scatter plot for the interp pair
#             output_plot_folder = os.path.join(output_folder, "scatter_pairs_interp", group_label, str(pair))
#             fig_scatter, title_scatter = scatter_plot(interp_pair_df)
#             save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
#     
#             # Shift and calculate correlations for serie 2 and shifted series 1
#             shifted_serie_1 = shift_series(serie_1, shift_ms)
#             calc_corr_result, interp_pair_df = calc_corr(pd.concat([shifted_serie_1, serie_2], ignore_index=False))
#             calc_corr_results.append(calc_corr_result)
#             
#             # Generate and save a scatter plot for the interp pair
#             output_plot_folder = os.path.join(output_folder, "scatter_pairs_interp", group_label, str(pair))
#             fig_scatter, title_scatter = scatter_plot(interp_pair_df)
#             save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
#         
#         # Concatenate the results for the current pair
#         calc_corr_results_df = pd.concat(calc_corr_results, ignore_index=False)
#         
#         # Find the best correlation result
#         max_abs_corr = calc_corr_results_df['corr'].abs().max()
#         max_corr_rows = calc_corr_results_df[calc_corr_results_df['corr'].abs() == max_abs_corr]
#         
#         if len(max_corr_rows) > 1:
#             best_corr_result = max_corr_rows.loc[max_corr_rows['shift_diff'].abs().idxmin()]
#         else:
#             best_corr_result = max_corr_rows.iloc[0]
#         
#         # Save the best correlation result
#         best_corr_results.append(best_corr_result)
#         
#     # Final concatenation of the best results
#     best_corr_results = pd.DataFrame(best_corr_results)
#     best_corr_results['group_label'] = group_label
#     
#     
#     return best_corr_results
# =============================================================================


#%%
def process_rr_data(series_df: pd.DataFrame, group_label: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
    """
    Processes RR interval data by finding matching pairs, aligning them in time,
    trimming to specific time frames, filtering, and calculating correlations.
    Generates and saves various plots during the process.

    Parameters:
    - series_df (pd.DataFrame): The input DataFrame containing series data. It should have at least two rows for pairing.
    - group_label (str): The label used for grouping in the output files.
    - start_time_ms (int): The start time for trimming the series, in milliseconds.
    - end_time_ms (int): The end time for trimming the series, in milliseconds.

    Returns:
    - pd.DataFrame: DataFrame containing the best correlation results.
    """

    # Validate that series_df is a DataFrame and has at least 2 rows for pairing
    if not validate_series_df(series_df, min_size=2):
        raise ValueError("Input DataFrame failed validation. It must have at least 2 rows.")

    # Find matching pairs and unmatched series
    matching_pairs, unmatched_series = find_pairs(series_df)

    best_corr_results = []

    # Process each matching pair
    for pair in matching_pairs:
        # Extract the corresponding rows from series_df for the given pair
        pair_df = series_df.loc[series_df['meas_name'].isin(pair)].iloc[[0, 1]]

        # Align the pair in time
        pair_df = time_align_pair(pair_df, time_ms_threshold=1000)

        # Trim the pair to the specified time frame
        trimmed_pair_df = trim(pair_df, start_time_ms, end_time_ms)

        # Validate trimmed pair DataFrame
        if not validate_series_df(trimmed_pair_df, threshold_samples=0.7):
            print(f"Validation failed for pair {pair} in time frame {start_time_ms} to {end_time_ms}. Skipping this pair.")
            continue

        # Generate and save plots for the TRIMMED pair
        output_plot_folder = os.path.join(output_folder, group_label, 'trimmed')
        
        # Histogram plot for the TRIMMED pair
        fig_hist, title_hist = density_plot(trimmed_pair_df, title=f"{pair}_Distribution of RR-intervals")
        save_plot(fig_hist, title_hist, folder_name=os.path.join(output_plot_folder, "hist_pairs_trimmed"), format="html")

        # Scatter plot for the TRIMMED pair
        fig_scatter, title_scatter = scatter_plot(trimmed_pair_df)
        save_plot(fig_scatter, title_scatter, folder_name=os.path.join(output_plot_folder, "scatter_pairs_trimmed"), format="html")

        # Filter the trimmed pair to remove outliers or unwanted data
        filtered_pair_df = filter_series(trimmed_pair_df)

        # Generate and save plots for the FILTERED pair
        output_plot_folder = os.path.join(output_folder, group_label, 'filtered')
        
        # Histogram plot for the FILTERED pair
        fig_hist, title_hist = density_plot(filtered_pair_df, title=f"{pair}_Distribution of RR-intervals")
        save_plot(fig_hist, title_hist, folder_name=os.path.join(output_plot_folder, "hist_pairs_filtered"), format="html")

        # Scatter plot for the FILTERED pair
        fig_scatter, title_scatter = scatter_plot(filtered_pair_df)
        save_plot(fig_scatter, title_scatter, folder_name=os.path.join(output_plot_folder, "scatter_pairs_filtered"), format="html")

        # Initialize lists to store correlation results
        calc_corr_results = []

        # Calculate correlation for the initial pair
        calc_corr_result, interp_pair_df = calc_corr(filtered_pair_df)
        calc_corr_results.append(calc_corr_result)

        # Generate and save scatter plot for the interpolated pair
        output_plot_folder = os.path.join(output_folder, group_label, "interp", str(pair))
        fig_scatter, title_scatter = scatter_plot(interp_pair_df)
        save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")

        # Obtain the two series from the filtered pair
        serie_1 = filtered_pair_df.iloc[[0]]
        serie_2 = filtered_pair_df.iloc[[1]]
        
        # Iterate over various time shift values
        for shift_ms in range(1000, 5001, 1000):
            # Shift and trim
            shifted_serie = shift_series(pair_df, shift_ms)
            shifted_serie = trim(shifted_serie, start_time_ms, end_time_ms)
            shifted_serie_1 = shifted_serie.iloc[[0]]
            shifted_serie_2 = shifted_serie.iloc[[1]]

            # Process shifted series 1
            if validate_series_df(shifted_serie_1, threshold_samples=0.7):
                shifted_serie_1 = filter_series(shifted_serie_1)
                calc_corr_result, interp_pair_df = calc_corr(pd.concat([shifted_serie_1, serie_2], ignore_index=False))
                calc_corr_results.append(calc_corr_result)
                
                # Generate and save scatter plot for the INTERP pair
                fig_scatter, title_scatter = scatter_plot(interp_pair_df)
                save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
            else:
                print(f"Validation failed for shifted series 1 for pair {pair} in time frame {start_time_ms} to {end_time_ms}. Skipping this pair.")

            # Process shifted series 2
            if validate_series_df(shifted_serie_2, threshold_samples=0.7):
                shifted_serie_2 = filter_series(shifted_serie_2)
                calc_corr_result, interp_pair_df = calc_corr(pd.concat([serie_1, shifted_serie_2], ignore_index=False))
                calc_corr_results.append(calc_corr_result)
                
                # Generate and save scatter plot for the INTERP pair
                fig_scatter, title_scatter = scatter_plot(interp_pair_df)
                save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
            else:
                print(f"Validation failed for shifted series 2 for pair {pair} in time frame {start_time_ms} to {end_time_ms}. Skipping this pair.")

        # Concatenate the results for the current pair
        calc_corr_results_df = pd.concat(calc_corr_results, ignore_index=False)

        # Find the best correlation result
        max_abs_corr = calc_corr_results_df['corr'].abs().max()
        max_corr_rows = calc_corr_results_df[calc_corr_results_df['corr'].abs() == max_abs_corr]

        if len(max_corr_rows) > 1:
            best_corr_result = max_corr_rows.loc[max_corr_rows['shift_diff'].abs().idxmin()]
        else:
            best_corr_result = max_corr_rows.iloc[0]

        # Save the best correlation result
        best_corr_results.append(best_corr_result)

    # Final concatenation of the best results
    best_corr_results = pd.DataFrame(best_corr_results)
    best_corr_results['group_label'] = group_label

    return best_corr_results



