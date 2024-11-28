# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:31:13 2024

@author: huber
"""
import numpy as np
import pandas as pd
import copy
from pathlib import Path
import matplotlib.pyplot as plt

from config import *
from classes import *
from data_management_functions import merge_grouped, merge_meas
from time_functions import time_align_pair

from utils.plotly_utils import save_html_plotly, create_multi_series_scatter_plot_plotly, create_multi_series_histogram_plotly
from utils.matplotlib_utils import save_fig_matplotlib, create_heatmap_matplotlib
from utils.file_utils import create_directory
from utils.general_utils import group_object_list

#%%
def scatter_plot(meas_list: list[Meas], title: str = "Scatter Plot", x_label: str = "Time [ms]", y_label: str = "Value"):
    """
    Creates a scatter plot from a list of Meas objects.

    Args:
        meas_list (list[Meas]): List of Meas objects containing the data to plot.
        title (str, optional): The title of the plot. Default is "Scatter Plot".
        x_label (str, optional): Label for the x-axis. Default is "Time [ms]".
        y_label (str, optional): Label for the y-axis. Default is "Value".

    Returns:
        fig: Plotly figure with the created scatter plot.
        plot_title: Title used for the plot.
    """
    
    # Determine the overall time range of the series
    stop = max(meas.data.x_data[-1] for meas in meas_list if len(meas.data.x_data) > 0)
    start = min(meas.data.x_data[0] for meas in meas_list if len(meas.data.x_data) > 0)

    # Prepare data for plotting
    data = []
    legend_labels = []
    scatter_colors = []  # Optionally define colors for each series

    for meas in meas_list:
        data.append({'x': meas.data.x_data, 'y': meas.data.y_data})
        legend_labels.append(str(meas))  # Use the __repr__ or __str__ method for labeling
        scatter_colors.append(None)  # You can customize colors if needed

    # Set the plot title
    plot_title = f"{title} Range from {start} to {stop} for {', '.join(legend_labels)}"
    
    # Create the scatter plot using the existing function
    fig = create_multi_series_scatter_plot_plotly(
        data,
        legend_labels=legend_labels,
        plot_title=plot_title,
        x_label=x_label,
        y_label=y_label,
        scatter_colors=scatter_colors,
        mode='markers'
    )

    return fig, plot_title

#%%
def density_plot(meas_list: list['Meas'], sd_threshold: float = 3, title: str = "Density Plot", x_label: str = "Value", y_label: str = "Density"):
    """
    Creates a density plot (histogram) for each time series in the provided list of Meas objects using Plotly.

    Args:
        meas_list (list[Meas]): List of Meas objects containing the data to plot.
        sd_threshold (float, optional): The standard deviation multiplier for determining outliers. Default is 3.
        title (str, optional): The title of the plot. Default is "Density Plot".
        x_label (str, optional): Label for the x-axis. Default is "Value".
        y_label (str, optional): Label for the y-axis. Default is "Density".

    Returns:
        fig: Plotly figure with the created density plot.
        title: Title used for the plot.
    """
     
    # Determine the overall time range of the series
    stop = max(meas.data.x_data[-1] for meas in meas_list if len(meas.data.x_data) > 0)
    start = min(meas.data.x_data[0] for meas in meas_list if len(meas.data.x_data) > 0)
    
    # Prepare data for plotting
    data = []
    legend_labels = []
    
    for meas in meas_list:
        data.append({'x': meas.data.y_data})  # Use the meas representation for labeling
        legend_labels.append(str(meas))

    # Calculate outlier thresholds
    outlier_info = {}
    for meas in meas_list:
        y_data = meas.data.y_data
        outlier_low = round(np.nanmean(y_data) - sd_threshold * np.nanstd(y_data))
        outlier_high = round(np.nanmean(y_data) + sd_threshold * np.nanstd(y_data))
        outlier_info[str(meas)] = (outlier_low, outlier_high)

    # Set the plot title
    plot_title = f"{title} Range from {start} to {stop} for {', '.join(legend_labels)}"

    # Create the density plot using the histogram function in Plotly
    fig = create_multi_series_histogram_plotly(
        data,
        legend_labels=legend_labels,
        plot_title=plot_title,
        x_label=x_label,  # Use custom x_label
        y_label=y_label,  # Use custom y_label
        show_grid=True
    )

    for meas_label, (outlier_low, outlier_high) in outlier_info.items():
        # Add the low outlier line
        fig.add_vline(x=outlier_low, line=dict(color='gray', dash='dash'))
    
        # Add annotation for the low outlier
        fig.add_annotation(
            x=outlier_low,
            y=0.1,  # Set a small value above 0 to ensure it doesn't overlap with the axis
            text=f'Outlier Low {meas_label}: {outlier_low}',
            showarrow=False,
            yanchor="bottom",  # Align text to be above the line
            xshift=-10,  # Shift text slightly left so it doesn't overlap with the line
            textangle=-90  # Rotate text vertically
        )
    
        # Add the high outlier line
        fig.add_vline(x=outlier_high, line=dict(color='gray', dash='dash'))
    
        # Add annotation for the high outlier
        fig.add_annotation(
            x=outlier_high,
            y=0.1,  # Keep y above 0 to avoid going below the axis
            text=f'Outlier High {meas_label}: {outlier_high}',
            showarrow=False,
            yanchor="bottom",  # Align text to be above the line
            xshift=10,  # Shift text slightly right to avoid overlap with the line
            textangle=-90  # Rotate text vertically
        )
    
    return fig, title

#%%
def meas_plot_from(meas_list: list[Meas], folder_name: str, title_label: str = "", value_label: str = "Value"):
    """
    Generates and saves histogram and scatter plots for merged Meas object 
    from a list of Meas objects and saves them in the specified folder.

    Parameters:
    -----------
    meas_list : list[Meas]
        A list of Meas objects to be plotted. Each Meas object contains signal data (x_data, y_data) 
        and associated metadata.

    folder_name : str
        The name of the folder where the plots will be saved. A new subdirectory structure will be created 
        under each measurement type and number.

    title_label : str, optional
        A label to be used in the title of the plots. If not provided, a default title will be generated 
        using the without title_label in titles.

    value_label : str, optional
        A label for the y-axis of the plots. Defaults to "Value", but can be replaced with a more specific 
        description (e.g., "HR", "RR-interval").

    Returns:
    --------
    None
        The function generates HTML files for both histogram and scatter plots for each Meas object. 
        These plots are saved in the specified folder structure, but nothing is returned.
    """
    plot_list = copy.deepcopy(meas_list)
    
    # Group measurements by their type and number
    grouped_meas = group_object_list(plot_list, ["metadata.condition", "metadata.meas_number"])

    for key, group in grouped_meas.items():
        merged_meas = merge_grouped(group)

        condition, meas_number = key
        folder_path = PLOTS_DIR / condition / str(meas_number) / folder_name

        # Create directories for saving plots
        create_directory(folder_path / 'HISTOGRAM')
        create_directory(folder_path / 'SCATTER')

        for meas in merged_meas:
            # Determine the overall time range of the series
            stop = meas.data.x_data[-1]
            start = meas.data.x_data[0]
            
            # Generate file name based on the measurement and its data range
            file_name = str(meas) + str(start) + str(stop) + ".html"

            # Create and save histogram plot
            fig_hist, title_hist = density_plot([meas], title=f"Density plot of {title_label}", x_label=value_label)
            save_html_plotly(fig_hist, folder_path / 'HISTOGRAM' / file_name)

            # Create and save scatter plot
            fig_scatter, title_scatter = scatter_plot([meas], title=f"Scatter plot of {title_label}", y_label=value_label)
            save_html_plotly(fig_scatter, folder_path / 'SCATTER' / file_name)

#%%
def plot_histogram_pair(meas_pair: tuple['Meas', 'Meas'], folder_path: Path, title_label: str = "", value_label: str = "Value"):
    """
    Plots and saves a density histogram for a pair of Meas objects.

    Args:
        meas_pair (tuple): A tuple containing two Meas objects (meas1, meas2).
        
        folder_path (Path): Path to the directory where the histogram will be saved.
        
        title_label : str, optional
            A label to be used in the title of the plots. If not provided, a default title will be generated 
            using the without title_label in titles.

        value_label : str, optional
            A label for the y-axis of the plots. Defaults to "Value", but can be replaced with a more specific 
            description (e.g., "HR", "RR-interval").
    
    Returns:
        None
    """
    meas1, meas2 = meas_pair
    
    # Align signals if necessary
    time_align_pair(meas1, meas2)
    
    # Determine the overall time range of the series
    stop = max(meas.data.x_data[-1] for meas in meas_pair)
    start = min(meas.data.x_data[0] for meas in meas_pair)
    
    # Generate the file name for saving the plot
    file_name = (
        f"{str(meas1)};{str(meas2)}_"
        f"{start}-"
        f"{stop}.html"
    )
    
    # Create directory for histogram plots if it doesn't exist
    create_directory(folder_path / 'HISTOGRAM')
    
    # Generate the histogram figure
    fig_hist, title_hist = density_plot([meas1, meas2], title=f"Density plot of {title_label}", x_label=value_label)
    
    # Save the plot as an HTML file
    save_html_plotly(fig_hist, folder_path / 'HISTOGRAM' / file_name)
    
#%%
def plot_scatter_pair(meas_pair: tuple['Meas', 'Meas'], folder_path: Path, title_label: str = "", value_label: str = "Value"):
    """
    Plots and saves a scatter plot for a pair of Meas objects.

    Args:
        meas_pair (tuple): A tuple containing two Meas objects (meas1, meas2).
        
        folder_path (Path): Path to the directory where the scatter plot will be saved.
        
        title_label : str, optional
            A label to be used in the title of the plots. If not provided, a default title will be generated 
            using the without title_label in titles.

        value_label : str, optional
            A label for the y-axis of the plots. Defaults to "Value", but can be replaced with a more specific 
            description (e.g., "HR", "RR-interval").
    Returns:
        None
    """
    meas1, meas2 = meas_pair
    
    # Align signals if necessary
    time_align_pair(meas1, meas2)
    
    # Determine the overall time range of the series
    stop = max(meas.data.x_data[-1] for meas in meas_pair)
    start = min(meas.data.x_data[0] for meas in meas_pair)
    
    # Generate the file name for saving the plot
    file_name = (
        f"{str(meas1)};{str(meas2)}_"
        f"{start}-"
        f"{stop}.html"
    )
     
    # Create directory for scatter plots if it doesn't exist
    create_directory(folder_path / 'SCATTER')
    
    # Generate the scatter plot figure
    fig_scatter, title_scatter = scatter_plot([meas1, meas2], title=f"Scatter plot of {title_label}", y_label=value_label)
    
    # Save the plot as an HTML file
    save_html_plotly(fig_scatter, folder_path / 'SCATTER' / file_name)
    
#%%
def pair_plots_from(meas_list: list[Meas], folder_name: str, title_label: str = "", value_label: str = "Value"):
    """
    Groups Meas objects, merges them, then creates and saves histogram and scatter plots for each pair.

    Parameters:
    -----------
    meas_list : list[Meas]
        A list of Meas objects to process and plot.

    folder_name : str
        The name of the folder where the plots will be saved.
        
    title_label : str, optional
        A label to be used in the title of the plots. If not provided, a default title will be generated 
        using the without title_label in titles.

    value_label : str, optional
        A label for the y-axis of the plots. Defaults to "Value", but can be replaced with a more specific 
        description (e.g., "HR", "RR-interval").

    Returns:
    --------
    None
    """
    
    plot_list = copy.deepcopy(meas_list)
    
    grouped_meas = group_object_list(plot_list, ["metadata.meas_number", "metadata.condition", "metadata.pair_number"])

    # Iterate over each group of measurements
    for key, group in grouped_meas.items():      
        meas_number, condition, pair_number = key
        
        person_meas1 = [meas for meas in group if meas.metadata.gender == 'M']
        person_meas2 = [meas for meas in group if meas.metadata.gender == 'F']
        
        plot_meas1 = merge_meas(person_meas1)
        plot_meas2 = merge_meas(person_meas2)
        
        folder_path = PLOTS_DIR / condition / str(meas_number) / folder_name

        plot_histogram_pair((plot_meas1, plot_meas2), folder_path, title_label=title_label, value_label=value_label)
        plot_scatter_pair((plot_meas1, plot_meas2), folder_path, title_label=title_label, value_label=value_label)

#%%
def save_final_pairs_plots(measurement_records, folder_name, title_label="", value_label="Value"):
    """
    Saves histogram and scatter plots for each MeasurementRecord, extracting details directly from the records.

    Args:
    -----
    measurement_records : list
        List of MeasurementRecord objects. Each record contains information about correlation results
        and associated pairs of Meas objects.
    
    folder_name : str
        Name of the folder where plots will be saved. This folder is organized by measurement type, number,
        and state within the predefined structure.
    
    title_label : str, optional
        Label to include in plot titles. Default is an empty string.
    
    value_label : str, optional
        Label for the y-axis of the plots. Default is "Value".
    
    Returns:
    --------
    None
    """
    for record in measurement_records:
        # Extract details from the MeasurementRecord object
        condition = record.condition
        meas_number = record.meas_number
        task = record.task
        
        # Define the base folder for saving the plots
        folder_path = Path(PLOTS_DIR) / condition / str(meas_number) / "intervals" / task / folder_name

        # Make deep copies of the Meas objects to avoid modifying original data
        plot_meas1_copy = copy.deepcopy(record.meas1)
        plot_meas2_copy = copy.deepcopy(record.meas2)
        
        # Plot and save the histogram and scatter plots
        plot_histogram_pair((plot_meas1_copy, plot_meas2_copy), folder_path, title_label=title_label, value_label=value_label)
        plot_scatter_pair((plot_meas1_copy, plot_meas2_copy), folder_path, title_label=title_label, value_label=value_label)

#%%
def save_corr_heatmap_by_pair_and_shift(df, folder_name, title_label=""):
    """
    Creates a heatmap for each group of records in the DataFrame where:
    - Columns are sorted `shift_diff` values.
    - Rows are `pair_number` as strings, sorted numerically.
    - Values are `corr`, with `NaN` where no data is available.

    Args:
        df (pd.DataFrame): 
            DataFrame containing the data with columns:
            ['meas_number', 'condition', 'task', 'pair_number', 'shift_diff', 'corr'].
        folder_name (str): 
            Name of the folder to save the heatmaps.
        title_label (str, optional): 
            Title for the heatmaps. Defaults to "".
    """
    # Ensure necessary columns exist
    required_columns = {'meas_number', 'condition', 'task', 'pair_number', 'shift_diff', 'corr'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Calculate global vmin and vmax for all records
    all_corr_values = df['corr'].dropna()
    vmin, vmax = all_corr_values.min(), all_corr_values.max()

    # Group the records by `meas_number`, `condition`, and `task`
    grouped_records = df.groupby(['meas_number', 'condition', 'task'])

    # Iterate over each group
    for group_key, group_df in grouped_records:
        # Pivot the DataFrame to create the heatmap data
        heatmap_data = group_df.pivot(index='pair_number', columns='shift_diff', values='corr')

        # Ensure the index and columns are sorted
        heatmap_data = heatmap_data.sort_index(axis=0, key=lambda x: pd.to_numeric(x, errors='coerce'))  # Sort Y (pair_number) numerically
        heatmap_data = heatmap_data.sort_index(axis=1)  # Sort X (shift_diff) naturally

        # Prepare data for the heatmap
        heatmap_array = heatmap_data.to_numpy()
        x_labels = heatmap_data.columns.tolist()
        y_labels = heatmap_data.index.tolist()

        # Extract group keys for file naming
        meas_number, condition, task = group_key

        # Generate folder path
        folder_path = Path(PLOTS_DIR) / 'HEATMAPS' / folder_name / "pair_and_shift"
        create_directory(folder_path)

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        create_heatmap_matplotlib(
            heatmap_array,
            ax=ax,
            axis_props={
                "x_title": "Shift Diff",
                "y_title": "Pair Number",
                "x_ticklabels": x_labels,
                "y_ticklabels": y_labels,
                "x_tickangle": 45,
            },
            title_props={
                "text": f"{title_label} ({meas_number}, {condition}, {task})",
                "font_size": 14,
            },
            heatmap_props={
                "vmin": vmin,
                "vmax": vmax,
                "cmap": "viridis",
            },
        )

        fig.tight_layout()

        # Save the plot as a PNG file
        file_name = f"{meas_number}_{condition}_{task}.png"
        save_fig_matplotlib(fig, folder_path / file_name)
        plt.close('all')


#%%
def save_corr_heatmap_by_task_and_shift(df, folder_name, title_label=""):
    """
    Creates a heatmap for each group of records in the DataFrame where:
    - Columns are sorted `shift_diff` values.
    - Rows are `task` values, sorted by a predefined `task_order`.
    - Values are `corr`, with `NaN` where no data is available.

    Args:
        df (pd.DataFrame): 
            DataFrame containing the data with columns:
            ['meas_number', 'condition', 'pair_number', 'task', 'shift_diff', 'corr'].
        folder_name (str): 
            Name of the folder to save the heatmaps.
        title_label (str, optional): 
            Title for the heatmaps. Defaults to "".
    """
    # Ensure necessary columns exist
    required_columns = {'meas_number', 'condition', 'pair_number', 'task', 'shift_diff', 'corr'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Define the task order for sorting
    task_order = [
        "baseline1",
        "z",
        "z1", "z1_1_f", "z1_2_m", "z1_3_f", "z1_4_m", "z1_5_f", "z1_6_m",
        "z2", "z2_1_m", "z2_2_f", "z2_3_m", "z2_4_f", "z2_5_m", "z2_6_f",
        "baseline2"
    ]

    # Calculate global vmin and vmax for the heatmap
    all_corr_values = df['corr'].dropna()
    vmin, vmax = all_corr_values.min(), all_corr_values.max()

    # Group the records by `meas_number`, `condition`, and `pair_number`
    grouped_records = df.groupby(['meas_number', 'condition', 'pair_number'])
    
    for group_key, group_df in grouped_records:
        # Pivot the DataFrame to create the heatmap data
        heatmap_data = group_df.pivot(index='task', columns='shift_diff', values='corr')

        # Sort rows (tasks) according to the predefined order
        heatmap_data = heatmap_data.reindex(index=task_order).dropna(how='all', axis=0)

        # Sort columns (shift_diff) by natural order
        heatmap_data = heatmap_data.sort_index(axis=1)

        # Prepare data for the heatmap
        heatmap_array = heatmap_data.to_numpy()
        x_labels = heatmap_data.columns.tolist()
        y_labels = heatmap_data.index.tolist()
        
        # Extract group keys for file naming
        meas_number, condition, pair_number = group_key

        # Generate folder path
        folder_path = Path(PLOTS_DIR) / 'HEATMAPS' / folder_name / "task_and_shift"
        create_directory(folder_path)

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        create_heatmap_matplotlib(
            heatmap_array,
            ax=ax,
            axis_props={
                "x_title": "Shift Diff",
                "y_title": "Task",
                "x_ticklabels": x_labels,
                "y_ticklabels": y_labels,
                "x_tickangle": 45,
            },
            title_props={
                "text": f"{title_label} ({meas_number}, {condition}, {pair_number})",
                "font_size": 14,
            },
            heatmap_props={
                "vmin": vmin,
                "vmax": vmax,
                "cmap": "viridis",
            },
        )

        fig.tight_layout()

        # Save the plot as a PNG file
        file_name = f"{meas_number}_{condition}_{pair_number}.png"
        save_fig_matplotlib(fig, folder_path / file_name)
        plt.close('all')
