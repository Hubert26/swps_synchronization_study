# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:50:19 2024

@author: huber
"""
import numpy as np
import copy
from datetime import timedelta

from config import *
from classes import *

from utils.signal_utils import filter_values_by_sd, filter_values_by_relative_mean, interpolate_missing_values, overlapping_sd, overlapping_rmssd

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
        meas.data.update(
            x_data=new_x_data,
            y_data=filtered_y_data
            )
        
        meas.metadata.update(
            endtime=meas.metadata.starttime + timedelta(milliseconds=new_x_data[-1])
            )
                
        # Add the processed Meas object to the filtered list
        filtered_meas_list.append(meas)
        
    return filtered_meas_list

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
        meas.data.update(y_data=instant_hr)
    
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
    meas.data.update(
        x_data=new_x_data,
        y_data=new_y_data
        )
        
    meas.metadata.update(
        endtime=new_endtime
        )

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
    meas.data.update(
        x_data=new_x_data,
        y_data=new_y_data)
    
    meas.metadata.update(
        endtime=new_endtime
        )

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