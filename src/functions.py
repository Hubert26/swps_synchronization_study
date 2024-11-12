# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.stats import pearsonr, combine_pvalues, chi2

from config import *
from classes import *
from utils.plotly_utils import save_html_plotly, create_multi_series_scatter_plot_plotly, create_multi_series_histogram_plotly
from utils.file_utils import read_text_file, extract_file_name
from utils.string_utils import extract_numeric_suffix, extract_numeric_prefix, remove_digits
from utils.signal_utils import filter_values_by_sd, filter_values_by_relative_mean, interpolate_missing_values, interp_signals_uniform_time, validate_array, overlapping_sd, overlapping_rmssd
from utils.math_utils import fisher_transform

#%%
def assign_meas_type(meas_name: str) -> str:
    """
    Assigns a measurement type based on the provided measurement name using predefined regex patterns.

    Args:
        meas_name (str): The name of the measurement.

    Returns:
        str: The assigned measurement type (e.g., 'Relaxation', 'Baseline', 'Cooperation').

    Raises:
        ValueError: If no matching pattern is found.
    """
    for meas_type, pattern in MEAS_TYPES:
        if re.match(pattern, meas_name):
            return meas_type
    raise ValueError(f"Could not determine meas_type for '{meas_name}'")

#%%
def extract_data_from_file(file_path):
    """
    Extracts data from a text file, creates a Meas object based on the data and metadata extracted 
    from the file name.
    
    Args:
        file_path (str): Path to the text file containing numerical data.

    Returns:
        Meas: A Meas object containing the data from the file and metadata extracted from the filename.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains non-numeric data or the filename is invalid.
        ValueError: If there is an error in parsing the timestamp in the filename.
    """
    
    logger.info(f"Starting to process the file: {file_path}")
    
    # Check if the file exists
    if not Path(file_path).is_file():
        logger.error(f"File '{file_path}' not found.")
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    logger.info(f"File '{file_path}' exists, proceeding with data extraction.")
    
    # Read the contents of the file
    y_data = read_text_file(file_path).split('\n')

    # Ensure all data in the file is numeric
    if not all(map(str.isdigit, y_data)):
        logger.error(f"File '{file_path}' contains non-numeric data.")
        raise ValueError(f"File '{file_path}' contains non-numeric data.")
    
    logger.info(f"Data successfully read and validated as numeric from file '{file_path}'.")

    # Convert the data to a numpy array of floats
    y_data = np.array(y_data).astype(float)
    x_data = np.cumsum(y_data)

    # Extract metadata from the file name
    file_name = extract_file_name(file_path)
    splitted_file_name = file_name.split()

    if len(splitted_file_name) != 3:
        logger.error(f"Filename '{file_name}' is invalid. Expected format: 'name date time'.")
        raise ValueError(f"Filename '{file_name}' is invalid. Expected format: 'name date time'.")
    
    logger.info(f"Filename '{file_name}' is valid, proceeding with metadata extraction.")

    # Combine the date and time for parsing
    start_timestamp = splitted_file_name[1] + splitted_file_name[2]
    try:
        starttime = datetime.strptime(start_timestamp, '%Y-%m-%d%H-%M-%S')
    except ValueError as e:
        logger.error(f"Error parsing start_timestamp '{start_timestamp}': {e}")
        raise ValueError(f"Error parsing start_timestamp '{start_timestamp}': {e}")
    
    logger.info(f"Start time '{starttime}' successfully parsed from the filename.")

    # Calculate the end time based on the cumulative sum of x_data
    endtime = starttime + timedelta(milliseconds=x_data[-1])

    logger.info(f"End time calculated as '{endtime}'.")

    # Extract components from the file name
    meas_name = splitted_file_name[0]
    meas_number = extract_numeric_prefix(meas_name)
    pair_number = extract_numeric_suffix(meas_name.split('_')[0])
    meas_type = assign_meas_type(meas_name)
    gender = remove_digits(meas_name.split('_')[0])[1]
    gender = 'F' if gender == 'k' else ('M' if gender == 'm' else gender)

    logger.info(f"Metadata extracted from filename: "
                f"meas_number={meas_number}, pair_number={pair_number}, "
                f"meas_type={meas_type}, gender={gender}.")

    # Create a new Meas object
    new_meas = Meas(
        x_data=x_data,
        y_data=y_data,
        meas_number=meas_number,
        meas_type=meas_type,
        gender=gender,
        pair_number=pair_number,
        shift=0.0,  # Assuming shift is 0.0, can be updated as needed
        starttime=starttime,
        endtime=endtime
    )
    
    logger.info(f"Meas object created successfully for file '{file_name}'.")
    
    return new_meas

#%%
def load_data(file_paths):
    """
    Load data from the given file paths.
    """
    meas_list = []
    for path in file_paths:
        meas_list.append(extract_data_from_file(path))
    return meas_list

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
    grouped_meas = group_meas(plot_list, ["meas_type", "meas_number"])

    for key, group in grouped_meas.items():
        merged_meas = merge_grouped(group)

        meas_type, meas_number = key
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / folder_name

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
    
    grouped_meas = group_meas(meas_list, ["meas_number", "meas_type", "pair_number"])

    # Iterate over each group of measurements
    for key, group in grouped_meas.items():      
        meas_number, meas_type, pair_number = key
        
        person_meas1 = [meas for meas in group if meas.metadata.gender == 'M']
        person_meas2 = [meas for meas in group if meas.metadata.gender == 'F']
        
        plot_meas1 = merge_meas(person_meas1)
        plot_meas2 = merge_meas(person_meas2)
        
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / folder_name

        plot_histogram_pair((plot_meas1, plot_meas2), folder_path, title_label=title_label, value_label=value_label)
        plot_scatter_pair((plot_meas1, plot_meas2), folder_path, title_label=title_label, value_label=value_label)

#%%
def find_meas(meas_list: list['Meas'], **criteria) -> list['Meas']:
    """
    Find Meas objects in meas_list that match specified criteria.
    
    Args:
        meas_list (list[Meas]): List of Meas objects to search.
        **criteria: Key-value pairs of Meas attributes to filter by.
            For example: meas_number=1, gender='M', meas_type='Baseline'.
    
    Returns:
        list[Meas]: A list of Meas objects that match the given criteria.
    """
    matched_meas = []
    
    for meas in meas_list:
        match = True
        
        # Iterate through the criteria to check if the Meas object matches
        for key, value in criteria.items():
            # Get the value of the attribute from the Meas object
            if hasattr(meas.metadata, key):
                meas_value = getattr(meas.metadata, key)
                # Check if the value matches the criteria
                if meas_value != value:
                    match = False
                    break
            else:
                # If Meas object does not have the specified attribute, skip this Meas
                match = False
                break
        
        if match:
            matched_meas.append(meas)
    
    return matched_meas

#%%
def validate_meas_metadata(meas_list: list['Meas'], metadata_fields: list[str]) -> bool:
    """
    Validates if all Meas objects in the list have the same metadata values for specified fields.

    Args:
        meas_list (list[Meas]): List of Meas objects to validate.
        metadata_fields (list[str]): List of metadata field names to check for equality.

    Returns:
        bool: True if all Meas objects have the same metadata, False otherwise.
    """
    if len(meas_list) < 2:
        return True  # No need to compare if there's only one Meas

    # Get the reference metadata from the first Meas object
    reference_metadata = meas_list[0].metadata

    for meas in meas_list[1:]:
        for field in metadata_fields:
            if getattr(meas.metadata, field) != getattr(reference_metadata, field):
                return False
    return True

#%%
def validate_meas_data(meas: Meas, min_lenght: int = 3):
    if not isinstance(meas, Meas):
        return 0
    return (validate_array(meas.data.x_data, min_length=min_lenght) and validate_array(meas.data.y_data, min_length=min_lenght))

#%%
def merge_meas(meas_list: list['Meas']) -> 'Meas':
    """
    Merges a list of Meas objects into a single Meas object.
    The merging is done such that the younger Meas is added to the older one.
    Assumes that the provided Meas objects have the same metadata attributes.

    Args:
        meas_list (list[Meas]): List of Meas objects to be merged.

    Returns:
        Meas: A single Meas object that is the result of merging the provided list.
    """
    assert all(isinstance(meas, Meas) for meas in meas_list), "meas_list contains non-Meas objects."

    # Create a deep copy of the list to avoid modifying the original objects
    meas_to_merge = copy.deepcopy(meas_list)

    if not validate_meas_metadata(meas_list, ["gender", "meas_number", "meas_type", "pair_number", "shift"]):
        raise ValueError(f"Validation MEAS_LIST failed in merge_meas function") 
                
    # Sort the group by starttime to ensure younger is added to older
    sorted_meas_list = sorted(meas_list, key=lambda m: m.metadata.starttime)

    # Initialize with the oldest Meas object
    merged_meas = sorted_meas_list[0]

    # Merge all remaining Meas objects into the first one
    for meas in sorted_meas_list[1:]:
        merged_meas += meas

    return merged_meas

#%%
def group_meas(meas_list: list['Meas'], attributes: list[str]) -> dict[tuple, list['Meas']]:
    """
    Groups Meas objects from the given list based on specified metadata attributes.

    Args:
        meas_list (list[Meas]): List of Meas objects to group.
        attributes (list[str]): List of metadata attributes to group by (e.g., ["gender", "meas_number"]).

    Returns:
        dict[tuple, list[Meas]]: Dictionary where the keys are tuples of attribute values, and the values are lists of Meas objects.
    """
    grouped_meas = defaultdict(list)
    
    assert all(isinstance(meas, Meas) for meas in meas_list), "meas_list contains non-Meas objects."
            
    for meas in meas_list:
        try:
            # Create a tuple with the selected metadata fields for grouping
            metadata_tuple = tuple(getattr(meas.metadata, attr) for attr in attributes)
            grouped_meas[metadata_tuple].append(meas)
        except AttributeError as e:
            print(f"Error accessing metadata: {e}, meas: {meas}")

    return grouped_meas

#%%
def merge_grouped(meas_list: list['Meas']) -> list['Meas']:
    """
    Merges Meas objects in the provided list that share the same metadata attributes.
    The merging is done in such a way that the younger Meas is added to the older one.
    The function operates on a deep copy of the input list, leaving the original data unchanged.

    Args:
        meas_list (list[Meas]): List of Meas objects to be merged.

    Returns:
        list[Meas]: List of merged Meas objects.
    """
    
    # Group Meas objects by relevant metadata fields
    grouped_meas = group_meas(meas_list, ["gender", "meas_number", "meas_type", "pair_number", "shift"])
    
    merged_meas_list = []
    
    # Iterate over each group and merge the Meas objects
    for group in grouped_meas.values():
        if len(group) > 1:  # Only merge if there are multiple Meas objects in the group
            merged_meas = merge_meas(group)
            
            merged_meas_list.append(merged_meas)
        else:
            # If the group has only one Meas object, add it directly to the result
            merged_meas_list.append(group[0])
    
    return merged_meas_list

#%%
def calculate_pair_time_difference(meas1: 'Meas', meas2: 'Meas', based_on: str = 'starttime') -> float:
    """
    Calculate the time difference in milliseconds between two signals based on their start or end times.
    
    Args:
        meas1 (Meas): The first Meas object.
        meas2 (Meas): The second Meas object.
        based_on (str): Specifies whether to calculate the time difference based on 'starttime' or 'endtime'. 
                        Default is 'starttime'.
    
    Returns:
        float: The time difference in milliseconds between the two signals.
    
    Raises:
        ValueError: If based_on is not 'starttime' or 'endtime'.
    """
    if based_on not in ['starttime', 'endtime']:
        raise ValueError("The argument 'based_on' must be either 'starttime' or 'endtime'.")
    
    time1 = getattr(meas1.metadata, based_on)
    time2 = getattr(meas2.metadata, based_on)
    
    # Calculate the time difference in milliseconds
    time_difference = abs((time2 - time1).total_seconds() * 1000)
    
    return time_difference
    
#%%
def time_align_pair(meas1: 'Meas', meas2: 'Meas') -> None: 
    """
    Align the starttime of two Meas objects by adding the time difference 
    to the x_data of the younger measurement, so that both have the same starttime.
    
    Args:
        meas1 (Meas): The first Meas object.
        meas2 (Meas): The second Meas object.
    
    Raises:
        ValueError: If both Meas objects have the same starttime.
    """
    logger.info(f"Aligned {meas1.metadata} with {meas2.metadata}.")
    # Calculate the time difference in milliseconds
    time_diff_ms = calculate_pair_time_difference(meas1, meas2, based_on='starttime')
    logger.info(f"Time difference in ms between the two measurements: {time_diff_ms}")

    
    # Identify the younger and older measurement
    if meas1.metadata.starttime < meas2.metadata.starttime:
        younger_meas, older_meas = meas2, meas1
    elif meas1.metadata.starttime > meas2.metadata.starttime:
        younger_meas, older_meas = meas1, meas2
    else:
        logger.info("Both Meas objects have the same starttime; alignment is not needed.")
        return None
    
    # Update the x_data of the younger Meas by adding the rounded time difference
    # Ensure the x_data is adjusted based on the time difference
    younger_meas.data.x_data += time_diff_ms
    # Optionally update the starttime of the younger meas to match the older one
    younger_meas.metadata.starttime = older_meas.metadata.starttime


#%%
def filter_rr_meas(meas_list: list['Meas'], sd_threshold: float = 3, threshold_factor: float = 0.2, interp_method: str = 'linear') -> list['Meas']:
    """
    Filters and processes a list of Meas objects by removing outliers and interpolating missing values.
    The function operates on a deep copy of the input list, leaving the original data unchanged.
    Each Meas object is processed to filter outliers based on standard deviation and relative mean,
    and to interpolate any missing values in the data.

    Parameters:
    -----------
    meas_list : list[Meas]
        A list of Meas objects to process.

    sd_threshold : float
        The number of standard deviations used to define outliers for the initial filtering step. Default is 3.

    threshold_factor : float
        The percentage threshold used in the second filtering step, where values are compared to the relative mean of neighboring values. 
        Default is 0.2 (20%).

    interp_method : str
        The method used to interpolate missing (NaN) values. Default is 'linear'.

    Returns:
    --------
    list[Meas]
        A list of processed and merged Meas objects, with filtered and interpolated data.
    """
    
    # Ensure all items in meas_list are instances of the Meas class
    assert all(isinstance(meas, Meas) for meas in meas_list), "meas_list contains non-Meas objects."
    
    # Create a deep copy of the list to avoid modifying the original objects
    meas_list_copy = copy.deepcopy(meas_list)
    filtered_meas_list = []
    
    for meas in meas_list_copy:
        # Step 1: Filter y_data based on standard deviation
        filtered_y_data = filter_values_by_sd(meas.data.y_data, sd_threshold)

        # Step 2: Further filter y_data based on the relative mean of neighboring values
        try:
            filtered_y_data = filter_values_by_relative_mean(filtered_y_data, threshold_factor)
        except ValueError as e:
            logger.warning(f"Skipping relative mean filter for {meas} due to: {e}")
            continue

        # Step 3: Interpolate missing (NaN) values in the filtered data
        filtered_y_data = interpolate_missing_values(filtered_y_data, method=interp_method)
        
        # Step 4: Calculate new x_data as the cumulative sum of the filtered y_data
        new_x_data = np.cumsum(filtered_y_data)
        
        # Step 5: Update the Meas object with new x_data, filtered y_data, and updated end time
        meas.update(new_x_data=new_x_data,
                    new_y_data=filtered_y_data,
                    new_endtime=meas.metadata.starttime + timedelta(milliseconds=new_x_data[-1]))
                
        # Add the processed Meas object to the filtered list
        filtered_meas_list.append(meas)
        
    return filtered_meas_list


#%%
def trim_meas(meas: Meas, start_ms: float, end_ms: float, starttime: datetime = None) -> Meas:
    """
    Trims the Meas object to the specified time range, based on the provided or default start time.
    If the provided range is outside the actual range of x_data, it adjusts to the nearest valid range.
    If the range becomes invalid (start >= end), it raises a ValueError.

    Args:
        meas (Meas): The Meas object to be trimmed.
        start_ms (float): The desired start time in milliseconds relative to the provided or default start time.
        end_ms (float): The desired end time in milliseconds relative to the provided or default start time.
        starttime (datetime, optional): A reference start time. If None, meas.metadata.starttime is used.

    Returns:
        Meas: A trimmed copy of the Meas object within the valid x_data range.

    Raises:
        ValueError: If no valid range exists after adjusting the start and end times.
    """
    # Create a deep copy to avoid modifying the original Meas object
    meas_copy = copy.deepcopy(meas)

    # Use provided starttime or default to meas.metadata.starttime
    if starttime is None:
        starttime = meas.metadata.starttime

    # Calculate the offset between the provided starttime and meas.metadata.starttime
    start_offset = (starttime - meas.metadata.starttime).total_seconds() * 1000  # convert to milliseconds

    # Adjust start_ms and end_ms by the offset (can be positive or negative)
    adjusted_start_ms = start_offset + start_ms
    adjusted_end_ms = start_offset + end_ms

    # Get the actual range of the x_data
    x_data_start = meas_copy.data.x_data[0]
    x_data_end = meas_copy.data.x_data[-1]

    # Adjust start and end times if they're outside the actual range
    valid_start = max(adjusted_start_ms, x_data_start)
    valid_end = min(adjusted_end_ms, x_data_end)

    # If no valid range exists, raise an error
    if valid_start >= valid_end:
        raise ValueError(f"Invalid time range for {meas} in {start_ms} to {end_ms}. No data in this range.")

    # Trim the Meas object to the adjusted range
    meas_copy.trim(valid_start, valid_end)

    return meas_copy

#%%
def trim_to_common_range(meas_pair: tuple['Meas', 'Meas']) -> tuple['Meas', 'Meas']:
    """
    Trims the x_data and y_data arrays in two Meas objects to their common x_data range.
    The reference for trimming is the earlier starttime from the pair of Meas objects.

    Args:
        meas_pair (tuple[Meas, Meas]): A tuple containing two Meas objects to trim.

    Returns:
        tuple[Meas, Meas]: A tuple containing the trimmed versions of the two Meas objects 
                           within their common x_data range.
    """
    meas1, meas2 = meas_pair
    
    # Determine the earlier starttime to use as reference
    ref_starttime = min(meas1.metadata.starttime, meas2.metadata.starttime)
    start1 = (meas1.metadata.starttime - ref_starttime).total_seconds() * 1000 + meas1.data.x_data[0]
    start2 = (meas2.metadata.starttime - ref_starttime).total_seconds() * 1000 + meas2.data.x_data[0]
    
    end1 = (meas1.metadata.starttime - ref_starttime).total_seconds() * 1000 + meas1.data.x_data[-1]
    end2 = (meas2.metadata.starttime - ref_starttime).total_seconds() * 1000 + meas2.data.x_data[-1]

    # Get the common range by finding the overlap between the x_data arrays of both Meas objects
    start = max(start1, start2)
    end = min(end1, end2)

    # Trim both Meas objects to the common range, using the earlier starttime
    trimmed_meas1 = trim_meas(meas1, start, end, starttime=ref_starttime)
    trimmed_meas2 = trim_meas(meas2, start, end, starttime=ref_starttime)

    return trimmed_meas1, trimmed_meas2

#%%
def interp_meas_pair_uniform_time(meas_pair: tuple['Meas', 'Meas'], ix_step: int = 1000) -> tuple['Meas', 'Meas']:
    """
    Interpolates a pair of Meas objects to a common uniform time axis within their overlapping x_data range.

    Args:
        meas_pair (tuple[Meas, Meas]): A tuple containing two Meas objects to interpolate.
        ix_step (int, optional): Time step for the uniform axis in milliseconds. Default is 1000 ms.
    
    Returns:
        tuple[Meas, Meas]: A tuple containing the updated Meas objects with interpolated x_data and y_data.
    """
    # Trim the Meas objects to their common x_data range
    meas1, meas2 = trim_to_common_range(meas_pair)

    # Validate the trimmed Meas pair before proceeding
    if not (validate_meas_data(meas1, 3) and validate_meas_data(meas2, 3)):
        logger.info(f"Validation failed in interp_meas_pair_uniform_time for {meas1} and {meas2}")
        return None, None
    
    time_align_pair(meas1, meas2)
    
    # Extract the x_data and y_data from both trimmed Meas objects
    signals = [
        (meas1.data.x_data, meas1.data.y_data),
        (meas2.data.x_data, meas2.data.y_data)
    ]

    # Interpolate the signals to a uniform time axis
    ix, interpolated_signals = interp_signals_uniform_time(signals, ix_step=ix_step)

    # Update the Meas objects with interpolated x_data and y_data
    meas1.update_data(new_x_data=ix, new_y_data=interpolated_signals[0])
    meas2.update_data(new_x_data=ix, new_y_data=interpolated_signals[1])

    # Return the updated Meas objects
    return meas1, meas2


#%%
def calc_corr_weighted(meas_pair_lists: tuple[list[Meas], list[Meas]]) -> tuple[dict, tuple['Meas', 'Meas']]:
    """
    Calculates the weighted Pearson correlation coefficient and combines p-values for each pair of 
    Meas objects from two lists, based on their interpolated signals. Handles trimming, 
    validating, interpolating, and merging of the signals.

    Parameters:
    -----------
    meas_pair_lists : tuple[list[Meas], list[Meas]]
        A tuple containing two lists of Meas objects, where each list corresponds to one person in the pair.
        Each list will be iterated to compute correlations between all possible combinations of the Meas objects.

    Returns:
    --------
    avg_corr_result : dict
        A dictionary containing the weighted average correlation coefficient (`corr`), 
        combined p-value (`p_val`), the string representations of the Meas objects (`meas1`, `meas2`), 
        and their shift difference (`shift_diff`).

    merged_meas_pair : tuple[Meas, Meas]
        The merged and interpolated Meas objects after trimming, interpolation, and merging on a uniform time grid.

    Notes:
    ------
    - The weighted average correlation is calculated based on the length of the interpolated signal as weight.
    - P-values are combined using Fisher’s method via `combine_pvalues`.
    - If no valid correlations are computed, the function will return (None, None).
    """
    meas1_list, meas2_list = meas_pair_lists
    
    corr_list = []
    p_val_list = []
    weights = []

    interp_meas1_list = []
    interp_meas2_list = []
    
    # Iterate over each Meas object from both lists to calculate pairwise correlation
    for meas1 in meas1_list:
        for meas2 in meas2_list:            
            try:
                # Interpolate both Meas objects onto a uniform time grid for correlation
                interp_meas1, interp_meas2 = interp_meas_pair_uniform_time((meas1, meas2), ix_step=250)
            except ValueError as e:
                logger.info(f"Skipping interpolation due to error: {e}.")
                continue

            # If interpolation failed, skip to the next pair
            if interp_meas1 is None or interp_meas2 is None:
                logger.info(f"Skipping interpolation due to failure. For: {meas1}, {meas2}")
                continue

            # Collect the successfully interpolated Meas objects
            interp_meas1_list.append(interp_meas1)
            interp_meas2_list.append(interp_meas2)

            # Calculate the Pearson correlation coefficient and p-value between the interpolated signals
            corr, p_val = pearsonr(interp_meas1.data.y_data, interp_meas2.data.y_data)
            
            # Use fisfer transformation on pearson correlation result
            corr = fisher_transform(corr)
            
            # Calculate the length of the interpolated signal as weight
            weight = len(interp_meas1.data.x_data)

            # Append results to their respective lists
            corr_list.append(corr)
            p_val_list.append(p_val)
            weights.append(weight)

    # If no valid correlations were computed, return None
    if not corr_list:
        logger.info(f"No valid correlations were computed for {meas1_list} and {meas2_list}")
        return None, None

    try:
        # Merge the interpolated Meas objects from both lists
        merged_meas1 = merge_meas(interp_meas1_list)
        merged_meas2 = merge_meas(interp_meas2_list)
    except ValueError as e:
        # Log details of the Meas objects to assist debugging in case of a merge error
        for meas in interp_meas1_list:
            logger.warning(f"Meas: {str(meas)}, Start time: {meas.metadata.starttime}, End time: {meas.metadata.endtime}")
        for meas in interp_meas2_list:
            logger.warning(f"Meas: {str(meas)}, Start time: {meas.metadata.starttime}, End time: {meas.metadata.endtime}")
        logger.warning(f"Error occurred during merging: {e}")
        raise e

    # Calculate the weighted average of correlation coefficients
    avg_corr = np.average(corr_list, weights=weights)
    
    # Combine p-values using Fisher's method, then convert the chi-square statistic to a p-value
    fisher_stat, _ = combine_pvalues(p_val_list, method='fisher')

    # Convert Fisher's method chi-square statistic to p-value
    combined_p_val = 1 - chi2.cdf(fisher_stat, 2 * len(p_val_list))  # Degrees of freedom = 2 * number of p-values

    # Prepare the result dictionary with correlation, combined p-value, and other metadata
    avg_corr_result = {
        'corr': avg_corr,
        'p_val': combined_p_val,  # Combined p-value
        'meas1': str(meas1),
        'meas2': str(meas2),
        'shift_diff': meas1.metadata.shift - meas2.metadata.shift
    }

    return avg_corr_result, (merged_meas1, merged_meas2)

#%%
def process_meas_and_find_corr(meas_list: list[Meas]) -> tuple[pd.DataFrame, list[tuple[Meas, Meas]]]:
    """
    Calculates the best weighted correlation between pairs of measurements, including shifted versions, 
    for various time intervals. Also identifies the best pair with the maximum correlation and smallest 
    shift difference. Returns a DataFrame of correlation results and a list of best interpolated measurement pairs.

    Parameters:
    -----------
    filtered_meas_list : list[Meas]
        A list of filtered Meas objects to group, correlate, and evaluate.

    Returns:
    --------
    final_corr_results_df : pd.DataFrame
        A DataFrame containing the correlation results for each measurement pair.
    
    final_interp_pairs_list : list[tuple[Meas, Meas]]
        A list of tuples, where each tuple contains two interpolated Meas objects 
        representing the best-correlated pair for each time interval.
    """
    
    final_corr_results_df = pd.DataFrame(columns=['meas1', 'meas2', 'corr', 'shift_diff', 'meas_number', 'meas_type', 'pair_number', 'meas_state'])
    final_interp_pairs_list = []
    
    grouped_meas = group_meas(meas_list, ["meas_number", "meas_type", "pair_number"])

    # Iterate over each group of measurements
    for key, group in grouped_meas.items():      
        meas_number, meas_type, pair_number = key
                
        person_meas1 = [meas for meas in group if meas.metadata.gender == 'M']
        person_meas2 = [meas for meas in group if meas.metadata.gender == 'F']
        
        if not person_meas1 or not person_meas2:
            logger.warning(f"{meas_number}, {meas_type}, {pair_number}: Skipping pair. Lack of meas1 or meas2.")
            continue
            
        # Get time intervals for the given measurement type
        selected_intervals = get_time_intervals(meas_type)
        
        # Find the oldest starttime among all measurements in the pair
        oldest_starttime = min(meas.metadata.starttime for meas in group)
        
        # Create shifted versions of the measurement pair
        shifted_meas1_list = []
        shifted_meas2_list = []
        for shift_ms in range(SHIFT_MIN_MS, SHIFT_MAX_MS + 1, SHIFT_STEP_MS):
            # Deep copy first, then shift the copies
            shifted_meas1 = [copy.deepcopy(meas) for meas in person_meas1]
            shifted_meas2 = [copy.deepcopy(meas) for meas in person_meas2]
        
            # Apply shift on each copied meas object
            for meas in shifted_meas1:
                meas.shift_right(shift_ms)
            for meas in shifted_meas2:
                meas.shift_right(shift_ms)
            
            # Add the shifted versions to the final lists
            shifted_meas1_list.extend(shifted_meas1)
            shifted_meas2_list.extend(shifted_meas2)
        
        best_corr_results_list = []
        best_interp_pairs_list = []
        
        # Iterate through the selected time intervals
        for (start_ms, end_ms), meas_state in selected_intervals.items():
            
            if (meas_number == 1 and meas_type=="Cooperation" and pair_number==11 and meas_state=="z1_1_f"):
                print(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}")
                
            # Calculate the time bounds for selecting based on oldest_starttime
            lower_bound = oldest_starttime + pd.Timedelta(milliseconds=start_ms)
            upper_bound = oldest_starttime + pd.Timedelta(milliseconds=end_ms)
            
            # Select measurements based on starttime and endtime
            selected_person_meas1 = [meas for meas in person_meas1 if lower_bound <= meas.metadata.endtime and meas.metadata.starttime <= upper_bound]
            selected_person_meas2 = [meas for meas in person_meas2 if lower_bound <= meas.metadata.endtime and meas.metadata.starttime <= upper_bound]  
            
            selected_shifted_meas1_list = [meas for meas in shifted_meas1_list if lower_bound <= meas.metadata.endtime and meas.metadata.starttime <= upper_bound]
            selected_shifted_meas2_list = [meas for meas in shifted_meas2_list if lower_bound <= meas.metadata.endtime and meas.metadata.starttime <= upper_bound]
            
            # Trim measurements to the given interval
            trimmed_meas1_list = []
            trimmed_meas2_list = []
            trimmed_shifted_meas1_list = []
            trimmed_shifted_meas2_list = []
            
            for loop_meas1 in selected_person_meas1:
                try:
                    tm1 = trim_meas(loop_meas1, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}: Trimming meas to interval faild. Skipping meas due to invalid time range: {e}")
                    continue
                trimmed_meas1_list.append(tm1)
            
            for loop_meas2 in selected_person_meas2:
                try:
                    tm2 = trim_meas(loop_meas2, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}: Trimming meas to interval faild. Skipping meas due to invalid time range: {e}")
                    continue
                trimmed_meas2_list.append(tm2)
                
            for loop_meas1 in selected_shifted_meas1_list:
                try:
                    tsm1 = trim_meas(loop_meas1, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}: Trimming shifted meas to interval faild. Skipping shifted meas due to invalid time range: {e}")
                    continue
                trimmed_shifted_meas1_list.append(tsm1)

            for loop_meas2 in selected_shifted_meas2_list:
                try:
                    tsm2 = trim_meas(loop_meas2, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}: Trimming shifted meas to interval faild. Skipping shifted meas due to invalid time range: {e}")
                    continue
                trimmed_shifted_meas2_list.append(tsm2)
                
            if not trimmed_meas1_list and not trimmed_shifted_meas1_list:
                logger.warning(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}: Skipping pair. Lack of trimmed meas1")
                continue
            
            if not trimmed_meas2_list and not trimmed_shifted_meas2_list:
                logger.warning(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}: Skipping pair. Lack of trimmed meas2")
                continue            
            
            # Lists to store correlation results
            corr_res_list = []
            interp_pair_list = []
            
            interval_duration_min = (end_ms - start_ms) / 1000 / 60

            # Calculate shift1=0 with shift2=0
            if trimmed_meas1_list and trimmed_meas2_list:
                corr_res, interp_pair = calc_corr_weighted((trimmed_meas1_list, trimmed_meas2_list))
                if corr_res is not None and interp_pair is not None:
                    interp_meas1, interp_meas2 = interp_pair
                    # Ensure that the duration of each interpolated measurement exceeds MIN_DURATION_RATIO * interval_duration_min
                    if interp_meas1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO and interp_meas2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO:
                        corr_res_list.append(pd.DataFrame([corr_res]))
                        interp_pair_list.append(interp_pair)                    
            
            # Group trimmed shifted_meas1 by their shift value
            if trimmed_shifted_meas1_list and trimmed_meas2_list:
                trimmed_shifted_gruped_meas1 = group_meas(trimmed_shifted_meas1_list, ["shift"])
                
                # Calculate shift2=0 with every shift1
                for shift1, meas1_group in trimmed_shifted_gruped_meas1.items():
                    
                    if (meas_number == 1 and meas_type=="Cooperation" and pair_number==11 and meas_state=="z1_1_f" and shift1==4.0):
                        print(f"{meas_number}, {meas_type}, {pair_number}, {meas_state}")
                    
                        
                    corr_res, interp_pair = calc_corr_weighted((meas1_group, trimmed_meas2_list))
                    if corr_res is not None and interp_pair is not None:
                        interp_meas1, interp_meas2 = interp_pair
                        # Ensure that the duration of each interpolated measurement exceeds MIN_DURATION_RATIO * interval_duration_min
                        if interp_meas1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO and interp_meas2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO:
                            corr_res_list.append(pd.DataFrame([corr_res]))
                            interp_pair_list.append(interp_pair) 
                        
            if trimmed_shifted_meas2_list and trimmed_meas1_list:
                trimmed_shifted_gruped_meas2 = group_meas(trimmed_shifted_meas2_list, ["shift"])
                
                # Calculate shift1=0 with every shift2
                for shift2, meas2_group in trimmed_shifted_gruped_meas2.items():
                    corr_res, interp_pair = calc_corr_weighted((trimmed_meas1_list, meas2_group))
                    if corr_res is not None and interp_pair is not None:
                        interp_meas1, interp_meas2 = interp_pair
                        # Ensure that the duration of each interpolated measurement exceeds MIN_DURATION_RATIO * interval_duration_min
                        if interp_meas1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO and interp_meas2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO:
                            corr_res_list.append(pd.DataFrame([corr_res]))
                            interp_pair_list.append(interp_pair) 
                                     
            # If correlations were calculated, find the best result
            if not corr_res_list or not interp_pair_list:
                logger.warning(f"{meas_number}, {meas_type}, {pair_number}: Skipping pair. Lack of valid correlations")
            else:
                corr_res_df = pd.concat(corr_res_list, ignore_index=False)
                
                # Find the best correlation result
                max_abs_corr = corr_res_df['corr'].abs().max()
                max_corr_rows = corr_res_df[corr_res_df['corr'].abs() == max_abs_corr]
                
                # If there are multiple results with the same correlation, choose the one with smallest shift difference
                if max_corr_rows.shape[0] > 1:
                    min_shift_index = max_corr_rows['shift_diff'].abs().idxmin()
                    best_corr_result = max_corr_rows.iloc[min_shift_index]
                else:
                    best_corr_result = max_corr_rows.iloc[0]
                
                best_corr_result['meas_state'] = meas_state
                best_corr_result['meas_number'] = meas_number
                best_corr_result['meas_type'] = meas_type
                best_corr_result['pair_number'] = pair_number
                
                # Find corresponding interp_pair for best_corr_result
                corr_res_df = corr_res_df.reset_index(drop=True)
                for idx, row in corr_res_df.iterrows():
                    if (row['corr'] == best_corr_result['corr']) and (row['shift_diff'] == best_corr_result['shift_diff']):
                        best_interp_pair = interp_pair_list[idx]
                        break

                # Save the best correlation result and corresponding interp_pair
                best_corr_results_list.append(best_corr_result)
                best_interp_pairs_list.append(best_interp_pair)                    
                                
                 # Combine the best results for the current pair
                if best_corr_results_list:
                    best_corr_results_df = pd.DataFrame(best_corr_results_list)
                else:
                    best_corr_results_df = pd.DataFrame(columns=['meas1', 'meas2', 'corr', 'shift_diff', "meas_number", "meas_type", "pair_number", 'meas_state'])
            
        # Append to final results
        final_corr_results_df = pd.concat([final_corr_results_df, best_corr_results_df], ignore_index=True)
        final_interp_pairs_list += best_interp_pairs_list

    return final_corr_results_df, final_interp_pairs_list

#%%
def save_final_pairs_plots(corr_results: pd.DataFrame, interp_pairs: list[tuple[Meas, Meas]], folder_name: str, title_label: str = "", value_label: str = "Value"):
    """
    Saves the histogram and scatter plots for each interpolated pair of Meas objects in the interp_pairs list
    and associates them with their corresponding correlation results in corr_results.

    Parameters:
    -----------
    corr_results : pd.DataFrame
        A DataFrame containing correlation results, including measurement state. 
        Each row represents the correlation data for a specific pair of Meas objects.

    interp_pairs : list[tuple[Meas, Meas]]
        A list of tuples where each tuple contains two interpolated Meas objects.
        These pairs are used for generating plots.

    folder_name : str
        The name of the folder where the generated plots will be saved.
        This folder should exist within a specific structure defined by the measurement type, number, 
        and state.
        
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
    # Iterate through both corr_results and interp_pairs
    for idx in range(len(corr_results)):
        corr_row = corr_results.iloc[idx]
        plot_meas1, plot_meas2 = interp_pairs[idx]

        # Extract measurement details from the metadata of plot_meas1
        meas_type = plot_meas1.metadata.meas_type  # Adjust if the attribute name is different
        meas_number = plot_meas1.metadata.meas_number  # Adjust if the attribute name is different
        meas_state = corr_row['meas_state']

        # Define the base folder for saving the plots
        folder_path = PLOTS_DIR / meas_type / str(meas_number) / "intervals" / meas_state / folder_name
        
        # Make a deep copy of the Meas objects to avoid modifying the original data
        plot_meas1_copy = copy.deepcopy(plot_meas1)
        plot_meas2_copy = copy.deepcopy(plot_meas2)
        
        # Plot and save the histogram and scatter plots
        plot_histogram_pair((plot_meas1_copy, plot_meas2_copy), folder_path, title_label=title_label, value_label=value_label)
        plot_scatter_pair((plot_meas1_copy, plot_meas2_copy), folder_path, title_label=title_label, value_label=value_label)

#%%
def calculate_instant_hr(nn_intervals: np.ndarray) -> np.ndarray:
    """
    Calculates the instantaneous heart rate (HR) for each NN interval in a series.

    Args:
        nn_intervals (np.ndarray): A numpy array of NN intervals in milliseconds.

    Returns:
        np.ndarray: An array of instantaneous heart rates in beats per minute (bpm), excluding any NaN values.
    
    Raises:
        ValueError: If the input is not a numpy ndarray.
    """
    # Check if nn_intervals is a numpy ndarray
    if not isinstance(nn_intervals, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    
    # Calculate HR for each interval, assuming intervals are in ms
    instant_hr = 60 * 1000 / nn_intervals
    instant_hr = instant_hr[~np.isnan(instant_hr)]  # Exclude any NaN values
    
    return instant_hr

#%%
def instant_hr_meas(meas_list: list[Meas]):
    """
    Calculates instantaneous heart rate (HR) for each Meas object in the copied list 
    and updates the data in the Meas objects.

    Parameters:
    -----------
    meas_list : list[Meas]
        A list of Meas objects. Each Meas object must have a y_data attribute containing 
        NN intervals in milliseconds.

    Returns:
    --------
    instant_hr_meas_list: list[Meas]
        A list of Meas objects. Each Meas object must have a y_data attribute containing 
        instant HR in bpm.

    Raises:
    -------
    ValueError: If any of the elements in meas_list is not an instance of Meas.
    """
    # Check if all elements are instances of Meas
    if not all(isinstance(meas, Meas) for meas in meas_list):
        raise ValueError("All elements in meas_list must be instances of Meas.")

    instant_hr_meas_list = copy.deepcopy(meas_list)
    
    # Iterate through each Meas object
    for meas in instant_hr_meas_list:
        # Access the NN intervals from meas.data.y_data
        nn_intervals = meas.data.y_data
        
        # Calculate instantaneous HR
        instant_hr = calculate_instant_hr(nn_intervals)
        
        # Update meas.data with the calculated instantaneous HR
        meas.data.update(new_y_data=instant_hr)
    
    return instant_hr_meas_list

#%%
def calculate_overlapping_sd(meas: Meas, window_ms: float, overlap: float, min_fraction: float) -> None:
    """
    Calculates the overlapping standard deviations for a Meas object and updates it with the new data.

    Parameters:
    -----------
    meas : Meas
        The Meas object whose data will be updated.
        
    window_ms : float
        The size of the sliding window in milliseconds.
        
    overlap : float
        Overlap between consecutive windows as a percentage (e.g., 0.5 for 50% overlap).

    min_fraction : float
        Minimum fraction of the average number of elements per window. If the number of data points 
        in a window is less than this threshold, the window is skipped.
        
    Returns:
    --------
    None
        The function updates the Meas object in place, modifying its x_data (time), y_data (standard deviations), 
        and endtime.
    """
    
    # Extract the time and signal values from the Meas object
    time = meas.data.x_data
    values = meas.data.y_data
    
    # Calculate the overlapping standard deviations
    new_y_data, new_x_data = overlapping_sd((time, values), window_time=window_ms, overlap=overlap, min_fraction=min_fraction)
    
    # Determine the new end time based on the last window center or the original time
    last_window_center = new_x_data[-1] if len(new_x_data) > 0 else time[-1]
    new_endtime = meas.metadata.starttime + timedelta(milliseconds=last_window_center)
    
    # Update the Meas object with the new time (x_data), new standard deviation (y_data), and new endtime
    meas.update(new_x_data=new_x_data, new_y_data=new_y_data, new_endtime=new_endtime)

#%%
def calculate_overlapping_sd_meas(meas_list: list[Meas], window_ms: float, overlap: float, min_fraction: float) -> list[Meas]:
    """
    Calculates overlapping standard deviations (SD) for each Meas object in the copied list 
    and updates the data in the Meas objects.

    Parameters:
    -----------
    meas_list : list[Meas]
        A list of Meas objects. Each Meas object must have x_data (time in milliseconds) and y_data (signal values).
        
    window_ms : float
        The size of the sliding window in milliseconds.
        
    overlap : float
        Overlap between consecutive windows as a percentage (e.g., 0.5 for 50% overlap).

    min_fraction : float
        Minimum fraction of the average number of elements per window. If the number of data points 
        in a window is less than this threshold, the window is skipped.

    Returns:
    --------
    sd_meas_list: list[Meas]
        A list of Meas objects with updated x_data (time) and y_data (standard deviations).

    Raises:
    -------
    ValueError: If any of the elements in meas_list is not an instance of Meas.
    """
    # Check if all elements are instances of Meas
    if not all(isinstance(meas, Meas) for meas in meas_list):
        raise ValueError("All elements in meas_list must be instances of Meas.")
    
    # Deep copy the Meas list to avoid modifying the original data
    sd_meas_list = copy.deepcopy(meas_list)
    
    # Iterate through each Meas object in the copied list
    for meas in sd_meas_list:
        # Calculate overlapping standard deviations and update the Meas object
        calculate_overlapping_sd(meas, window_ms=window_ms, overlap=overlap, min_fraction=min_fraction)
    
    return sd_meas_list

#%%
def calculate_overlapping_rmssd(meas: Meas, window_ms: float, overlap: float, min_fraction: float) -> None:
    """
    Calculates the overlapping Root Mean Square of Successive Differences (RMSSD) for a Meas object and updates it with the new data.

    Parameters:
    -----------
    meas : Meas
        The Meas object whose data will be updated.
        
    window_ms : float
        The size of the sliding window in milliseconds.
        
    overlap : float
        Overlap between consecutive windows as a percentage (e.g., 0.5 for 50% overlap).

    min_fraction : float
        Minimum fraction of the average number of elements per window. If the number of data points 
        in a window is less than this threshold, the window is skipped.
        
    Returns:
    --------
    None
        The function updates the Meas object in place, modifying its x_data (time in milliseconds), 
        y_data (RMSSD values), and endtime.
    """
    
    # Extract the time and signal values from the Meas object
    time = meas.data.x_data
    values = meas.data.y_data
    
    # Calculate the overlapping RMSSD values and window centers
    new_y_data, new_x_data = overlapping_rmssd((time, values), window_time=window_ms, overlap=overlap, min_fraction=min_fraction)
    
    # Determine the new end time based on the last window center or the original time
    last_window_center = new_x_data[-1] if len(new_x_data) > 0 else time[-1]
    new_endtime = meas.metadata.starttime + timedelta(milliseconds=last_window_center)
    
    # Update the Meas object with the new time (x_data), new RMSSD (y_data), and new endtime
    meas.update(new_x_data=new_x_data, new_y_data=new_y_data, new_endtime=new_endtime)

#%%
def calculate_overlapping_rmssd_meas(meas_list: list[Meas], window_ms: float, overlap: float, min_fraction: float) -> list[Meas]:
    """
    Calculates overlapping RMSSD (Root Mean Square of Successive Differences) for each Meas object in the copied list 
    and updates the data in the Meas objects.

    Parameters:
    -----------
    meas_list : list[Meas]
        A list of Meas objects. Each Meas object must have x_data (time in milliseconds) and y_data (signal values).
        
    window_ms : float
        The size of the sliding window in milliseconds.
        
    overlap : float
        Overlap between consecutive windows as a percentage (e.g., 0.5 for 50% overlap).

    min_fraction : float
        Minimum fraction of the average number of elements per window. If the number of data points 
        in a window is less than this threshold, the window is skipped.

    Returns:
    --------
    rmssd_meas_list: list[Meas]
        A list of Meas objects with updated x_data (time) and y_data (RMSSD values).

    Raises:
    -------
    ValueError: If any of the elements in meas_list is not an instance of Meas.
    """
    # Check if all elements are instances of Meas
    if not all(isinstance(meas, Meas) for meas in meas_list):
        raise ValueError("All elements in meas_list must be instances of Meas.")
    
    # Deep copy the Meas list to avoid modifying the original data
    rmssd_meas_list = copy.deepcopy(meas_list)
    
    # Iterate through each Meas object in the copied list
    for meas in rmssd_meas_list:
        # Calculate overlapping RMSSD and update the Meas object
        calculate_overlapping_rmssd(meas, window_ms=window_ms, overlap=overlap, min_fraction=min_fraction)
    
    return rmssd_meas_list

