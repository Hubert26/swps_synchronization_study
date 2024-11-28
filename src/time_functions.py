# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:30:00 2024

@author: Hubert Szewczyk
"""
from datetime import datetime
import copy

from config import *
from classes import *
from data_management_functions import validate_meas_data

from utils.signal_utils import interp_signals_uniform_time

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
    meas1.data.update(x_data=ix, y_data=interpolated_signals[0])
    meas2.data.update(x_data=ix, y_data=interpolated_signals[1])

    # Return the updated Meas objects
    return meas1, meas2

