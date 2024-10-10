# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import re
pio.renderers.default='browser'
import math
import copy
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from IPython.display import display
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from config import *

from config import logger, MEAS_TYPES
from classes import Data, Metadata, Meas
from utils.plotly_utils import save_html_plotly, create_multi_series_scatter_plot_plotly, create_multi_series_histogram_plotly
from utils.file_utils import read_text_file, extract_file_name
#from utils.matplotlib_utils import save_fig_matplotlib, create_subplots_matplotlib, create_multi_series_bar_chart_matplotlib
from utils.string_utils import extract_numeric_suffix, extract_numeric_prefix, remove_digits
from utils.math_utils import filter_values_by_sd, filter_values_by_relative_mean, interpolate_missing_values, interp_signals_uniform_time, validate_array


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
def scatter_plot(meas_list: list[Meas], title: str = None):
    """
    Creates a scatter plot from a list of Meas objects.

    Args:
        meas_list (list[Meas]): List of Meas objects containing the data to plot.
        title (str, optional): Title of the plot. Defaults to None.

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

    # Set the plot title if not provided
    plot_title = title if title else f"Range from {start} to {stop} for {', '.join(legend_labels)}"
    
    # Create the scatter plot using the existing function
    fig = create_multi_series_scatter_plot_plotly(
        data,
        legend_labels=legend_labels,
        plot_title=plot_title,
        x_label="Time [ms]",
        y_label="Time Between Heartbeats [ms]",
        scatter_colors=scatter_colors,
        mode='markers'
    )

    return fig, plot_title

#%%
def density_plot(meas_list: list['Meas'], sd_threshold: float = 3, title: str = None):
    """
    Creates a density plot (histogram) for each time series in the provided list of Meas objects using Plotly.

    Args:
        meas_list (list[Meas]): List of Meas objects containing the data to plot.
        sd_threshold (float, optional): The standard deviation multiplier for determining outliers. Default is 3.
        title (str, optional): The title of the plot. If not provided, a default title will be generated.

    Returns:
        fig: Plotly figure with the created density plot.
        title: Title used for the plot.
    """
    
    title = "Density Plot of RR-intervals" if title is None else title
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

    # Create the density plot using the histogram function in Plotly
    fig = create_multi_series_histogram_plotly(
        data,
        legend_labels=legend_labels,
        plot_title=title,
        x_label="RR-interval [ms]",
        y_label="Density",
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
    return (validate_array(meas.data.x_data) and validate_array(meas.data.y_data))

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
def find_pairs(meas_list: list['Meas']) -> list[tuple['Meas', 'Meas']]:
    """
    Finds pairs of Meas objects with different genders but identical
    meas_number, meas_type, pair_number, and shift attributes.

    Args:
        meas_list (list[Meas]): List of Meas objects to search for pairs.

    Returns:
        list[tuple[Meas, Meas]]: List of tuples, where each tuple contains a pair of Meas objects with different genders.
    """
    
    assert all(isinstance(meas, Meas) for meas in meas_list), "meas_list contains non-Meas objects."
    
    # Group Meas objects by the relevant metadata fields except for gender
    grouped_meas = group_meas(meas_list, ["meas_number", "meas_type", "pair_number", "shift"])
    
    pairs = []
    
    # Iterate through the groups to find pairs of different genders
    for group in grouped_meas.values():
        males = [meas for meas in group if meas.metadata.gender == 'M']
        females = [meas for meas in group if meas.metadata.gender == 'F']
        
        # Find pairs of male and female Meas objects
        for male in males:
            for female in females:
                pairs.append((male, female))
    
    return pairs

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
    # Calculate the time difference in milliseconds (rounded to the nearest thousand)
    time_diff_ms = calculate_pair_time_difference(meas1, meas2, based_on='starttime')
    time_diff_ms_rounded = round(time_diff_ms / 1000) * 1000
    logger.info(f"Time difference in ms between the two measurements: {time_diff_ms_rounded}")

    
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
    younger_meas.data.x_data += time_diff_ms_rounded
    # Optionally update the starttime of the younger meas to match the older one
    younger_meas.metadata.starttime = older_meas.metadata.starttime


#%%
def filter_meas(meas_list: list['Meas'], sd_threshold: float = 3, threshold_factor: float = 0.2, interp_method: str = 'linear') -> list['Meas']:
    """
    Filters and processes a list of Meas objects by removing outliers and interpolating missing values.
    The function operates on a deep copy of the input list, leaving the original data unchanged.
    After splitting each Meas object into sub-Meas, the function filters and interpolates the data, and
    finally merges the sub-Meas objects back together.

    Parameters:
    meas_list (list): A list of Meas objects to process.
    sd_threshold (float): The number of standard deviations used to define outliers for the first filtering step. Default is 3.
    threshold_factor (float): The percentage threshold used in the second filtering step, where values are compared to the relative mean of neighbors. Default is 0.2 (20%).
    interp_method (str): The method used to interpolate missing (NaN) values. Default is 'linear'.
    
    Returns:
    list[Meas]: A list of processed and merged Meas objects, with filtered and interpolated data.
    """
    
    assert all(isinstance(meas, Meas) for meas in meas_list), "meas_list contains non-Meas objects."
    
    # Create a deep copy of the list to avoid modifying the original objects
    meas_list_copy = copy.deepcopy(meas_list)
    filtered_meas_list = []
    
    for meas in meas_list_copy:
        # List to store sub-Meas objects for filtering and merging
        sub_meas_list_to_merge = []
        
        # Step 1: Split Meas into smaller sub-Meas objects based on x_data continuity
        splitted_meas = meas.split()
        
        if len(splitted_meas) > 5:
            raise ValueError("The lists (splitted_meas) has more than 5 elements.")
        
        assert all(isinstance(meas, Meas) for meas in splitted_meas), "meas_list contains non-Meas objects."
        for sub_meas in splitted_meas:
            # Step 2: Filter y_data to remove outliers based on standard deviation
            filtered_y_data = filter_values_by_sd(sub_meas.data.y_data, sd_threshold)
    
            # Step 3: Further filter y_data based on the relative mean of neighboring values
            try:
                filtered_y_data = filter_values_by_relative_mean(filtered_y_data, threshold_factor)
            except ValueError as e:
                logger.info(f"Skipping relative mean filter for {sub_meas} due to: {e}")

    
            # Step 4: Interpolate missing (NaN) values in the filtered data
            filtered_y_data = interpolate_missing_values(filtered_y_data, method=interp_method)
            new_x_data = np.cumsum(filtered_y_data)
            
            # Step 5: Update sub-Meas object with new x_data (cumulative sum of y_data) and y_data
            sub_meas.update(new_x_data=new_x_data,
                            new_y_data=filtered_y_data,
                            new_endtime=sub_meas.metadata.starttime + timedelta(milliseconds=new_x_data[-1]))
            
            # Add filtered sub-Meas to the list for later merging
            sub_meas_list_to_merge.append(sub_meas)
        
        # Step 6: Merge all filtered sub-Meas objects into one Meas object
        merged_meas = merge_meas(sub_meas_list_to_merge)
        
        filtered_meas_list.append(merged_meas)
        
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
def interp_meas_pair_uniform_time(meas_pair: tuple['Meas', 'Meas'], ix_step: int = 1000, method: str = 'linear', fill_value='extrapolate') -> tuple['Meas', 'Meas']:
    """
    Interpolates a pair of Meas objects to a common uniform time axis within their overlapping x_data range.

    Args:
        meas_pair (tuple[Meas, Meas]): A tuple containing two Meas objects to interpolate.
        ix_step (int, optional): Time step for the uniform axis in milliseconds. Default is 1000 ms.
        method (str, optional): Interpolation method ('linear', 'quadratic', 'cubic', etc.). Default is 'linear'.
        fill_value (str or float, optional): How to handle out-of-bounds values. Default is 'extrapolate'.
    
    Returns:
        tuple[Meas, Meas]: A tuple containing the updated Meas objects with interpolated x_data and y_data.
    """
    # Trim the Meas objects to their common x_data range
    meas1, meas2 = trim_to_common_range(meas_pair)

    # Validate the trimmed Meas pair before proceeding
    if not (validate_meas_data(meas1, 3) and validate_meas_data(meas2, 3)):
        logger.info(f"Validation failed in interp_meas_pair_uniform_time for {meas1} and {meas2}")
        return None, None

    # Extract the x_data and y_data from both trimmed Meas objects
    signals = [
        (meas1.data.x_data, meas1.data.y_data),
        (meas2.data.x_data, meas2.data.y_data)
    ]

    # Interpolate the signals to a uniform time axis
    ix, interpolated_signals = interp_signals_uniform_time(signals, ix_step=ix_step, method=method, fill_value=fill_value)

    # Update the Meas objects with interpolated x_data and y_data
    meas1.update_data(new_x_data=ix, new_y_data=interpolated_signals[0])
    meas2.update_data(new_x_data=ix, new_y_data=interpolated_signals[1])

    # Return the updated Meas objects
    return meas1, meas2


#%%
def calc_corr_weighted(meas_pair: tuple['Meas', 'Meas']) -> tuple[dict, tuple['Meas', 'Meas']]:
    """
    Calculates the weighted Pearson correlation coefficient and p-value for each pair of 
    split time series from two Meas objects. Trims and validates each split before calculating correlation.
    Collects all interpolated Meas objects, merges them, and returns the weighted average correlation.

    Parameters:
    -----------
    meas_pair : tuple[Meas, Meas]
        A tuple containing two Meas objects to be compared.

    Returns:
    --------
    avg_corr_result : dict
        A dictionary containing the weighted average correlation coefficient (`corr`), 
        p-value (`p_val`), and other relevant series information.

    merged_meas_pair : tuple[Meas, Meas]
        The merged and interpolated Meas objects after trimming, interpolation, and merging on a uniform time grid.
    """
    meas1, meas2 = meas_pair
    
    if str(meas1) == "1RelaxationM14_0.0" or str(meas2) == "1RelaxationM14_0.0":
        logger.info("Jeden z Meas to 1RelaxationM14_0.0")

    # Validate the meas pair before proceeding
    if not (validate_meas_data(meas1, 3) and validate_meas_data(meas2, 3)):
        raise DataValidationError("Input is incorrect")

    # Split the signals based on gaps
    split_meas1 = meas1.split()
    split_meas2 = meas2.split()
    
    n = 4
    if len(split_meas1) > n or len(split_meas2) > n:
        raise ValueError("One of the lists (split_meas1 or split_meas2) has more than {n} elements.")

    corr_list = []
    p_val_list = []
    weights = []

    interp_meas1_list = []
    interp_meas2_list = []

    # Iterate over splits and calculate correlation
    for split1 in split_meas1:
        for split2 in split_meas2:
            # Validate the individual splits
            if not (validate_meas_data(split1, 3) and validate_meas_data(split2, 3)):
                logger.info(f"Validation failed in calc_corr_weighted for splitted signals {split1} and {split2}. Skipping invalid split pair.")
                continue
            try:
                # Interpolate both series to a uniform time grid
                interp_meas1, interp_meas2 = interp_meas_pair_uniform_time((split1, split2), ix_step=250)
            except ValueError as e:
                logger.info(f"Skipping interpolation due to error {e}.")
                continue
                
            # Skip if interpolation failed
            if interp_meas1 is None or interp_meas2 is None:
                logger.info(f"Skipping interpolation due to failure. For splits: {split1}, {split2}")
                continue

            # Collect the interpolated Meas objects
            interp_meas1_list.append(interp_meas1)
            interp_meas2_list.append(interp_meas2)

            # Calculate the Pearson correlation coefficient and p-value
            corr, p_val = pearsonr(interp_meas1.data.y_data, interp_meas2.data.y_data)

            # Calculate the length of the interpolated signal as weight
            weight = len(interp_meas1.data.y_data)

            # Append the results to the lists
            corr_list.append(corr)
            p_val_list.append(p_val)
            weights.append(weight)

    if not corr_list:
        logger.info(f"No valid correlations were computed for {meas1} and {meas2}")
        return None, None
    
    try: 
        # Merge the collected interpolated Meas objects
        merged_meas1 = merge_meas(interp_meas1_list)
        merged_meas2 = merge_meas(interp_meas2_list)
    except ValueError as e:
        # Log the information about each Meas object in the list
        for meas in interp_meas1_list:
            logger.warning(f"Meas: {str(meas)}, Start time: {meas.metadata.starttime}, End time: {meas.metadata.endtime}")
        for meas in interp_meas2_list:
            logger.warning(f"Meas: {str(meas)}, Start time: {meas.metadata.starttime}, End time: {meas.metadata.endtime}")
        logger.info(f"Error occurred during merging: {e}")
        raise e

    # Calculate weighted average correlation and p-value
    avg_corr = np.average(corr_list, weights=weights)
    avg_p_val = np.average(p_val_list, weights=weights)

    # Prepare the result
    avg_corr_result = {
        'corr': avg_corr,
        'p_val': avg_p_val,
        'meas1': str(meas1),
        'meas2': str(meas2),
        'shift_diff': meas1.metadata.shift - meas2.metadata.shift
    }

    return avg_corr_result, (merged_meas1, merged_meas2)

#%%


#%%


#%%



#%%


#%%



#%%
def calculate_mean_hr(nn_intervals):
    # Sprawdzenie, czy nn_intervals jest numpy ndarray
    if not isinstance(nn_intervals, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    
    # Obliczenie średniej interwałów, pomijając NaN
    mean_interval = np.nanmean(nn_intervals)
    
    # Obliczenie tętna HR (bpm) na podstawie średniego interwału
    hr = 60 * 1000 / mean_interval  # interwały w ms, więc przeliczamy na bpm
    
    return hr

#%%
def calculate_instant_hr(nn_intervals):
    # Sprawdzenie, czy nn_intervals jest numpy ndarray
    if not isinstance(nn_intervals, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    
    # Obliczenie HR dla każdego interwału, pomijając NaN
    instant_hr = 60 * 1000 / nn_intervals  # Zakładamy, że interwały są w ms
    instant_hr = instant_hr[~np.isnan(instant_hr)]  # Pomijamy NaN-y, jeśli są
    
    return instant_hr

#%%



