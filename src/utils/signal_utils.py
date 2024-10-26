# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:20:45 2024

@author: Hubert Szewczyk
"""
import numpy as np
from scipy.interpolate import interp1d, CubicSpline




#%%
def filter_values_by_sd(data: np.ndarray, sd_threshold: float = 3) -> np.ndarray:
    """
    Filters values in a dataset by replacing outliers with NaN based on a standard deviation threshold.

    This function identifies outliers in the dataset based on a given standard deviation threshold.
    Outliers are defined as values that fall outside the range of mean ± (sd_threshold * standard deviation).

    Parameters:
    data (np.ndarray): An array of values.
    sd_threshold (float, optional): The number of standard deviations used to define outliers. Default is 3.

    Returns:
    np.ndarray: The filtered array with outlier values replaced by NaN.
    """
    # Ensure the input is a numpy array of floats
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    # Convert to float if necessary to handle NaN values
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    
    # Compute the mean and standard deviation of the values, ignoring NaNs
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    
    # Calculate the lower and upper bounds for outliers
    outlier_low = mean_val - sd_threshold * std_val
    outlier_high = mean_val + sd_threshold * std_val
    
    # Identify and replace outliers with NaN
    mask = (data < outlier_low) | (data > outlier_high)
    data[mask] = np.nan
    
    return data

#%%
def filter_values_by_relative_mean(data: np.ndarray, threshold_factor: float) -> np.ndarray:
    """
    Filters values in a dataset by replacing values that deviate more than a specified percentage
    from the average of the previous and next values with NaN. This function handles NaN values in the input array.

    Parameters:
    data (np.ndarray): An array of values, which may contain NaN values.
    threshold_factor (float): The percentage threshold used to define outliers. For example, 0.1 means that
                              values deviating more than 10% from the average of their neighbors are considered outliers.

    Returns:
    np.ndarray: The filtered array with outlier values replaced by NaN.
    """
    # Ensure the input is a numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    if data.size < 3:
        raise ValueError("Input array must contain at least three valid elements.")
    
    # Create a copy of the array to avoid modifying the original data
    filtered_data = data.copy()
    
    # Calculate the mean of previous and next values, ignoring NaN values
    mean_prev_next = np.full_like(data, np.nan, dtype=float)  # Initialize with NaN
    
    for i in range(1, len(data) - 1):
        prev_value = data[i - 1]
        next_value = data[i + 1]
        
        if not np.isnan(prev_value) and not np.isnan(next_value):
            mean_prev_next[i] = np.nanmean([prev_value, next_value])
    
    # Replace outliers with NaN
    for i in range(1, len(data) - 1):
        if not np.isnan(mean_prev_next[i]):
            if data[i] > (1 + threshold_factor) * mean_prev_next[i] or data[i] < (1 - threshold_factor) * mean_prev_next[i]:
                filtered_data[i] = np.nan
    
    return filtered_data

#%%
def interpolate_missing_values(data_array: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolates missing (NaN) values in a 1D NumPy array using the specified interpolation method.
    
    Parameters:
    data_array (np.ndarray): A 1D NumPy array containing numerical values, which may include NaN values.
    method (str, optional): The interpolation method ('linear', 'quadratic', 'cubic', etc.). 
                            Default is 'linear'.
    
    Returns:
    np.ndarray: The input array with NaN values replaced by interpolated values.
    
    Notes:
    - The function uses interpolation to estimate missing (NaN) values based on surrounding values.
    - If the input array contains only NaN values or no NaN values, the function will return the array unchanged.
    - The method argument allows for flexible interpolation approaches (e.g., linear, cubic).
    """
    
    # Find indices of NaN values
    nan_indices = np.isnan(data_array)
    
    # If there are no NaN values, return the array unchanged
    if not np.any(nan_indices):
        return data_array
    
    # Find indices of non-NaN values
    valid_indices = ~nan_indices
    x = np.arange(len(data_array))  # Array of indices
    
    # If the entire array is NaN, return the array unchanged
    if np.all(nan_indices):
        return data_array
    
    # Create interpolation function based on non-NaN values
    interpolation_func = interp1d(
        x[valid_indices], data_array[valid_indices], kind=method, fill_value="extrapolate"
    )
    
    # Apply interpolation to replace NaN values
    data_array[nan_indices] = interpolation_func(x[nan_indices])
    
    return data_array
#%%
def interp_signals_uniform_time(signals: list[tuple[np.ndarray, np.ndarray]], ix_step: int = 1000) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Interpolates multiple signals to a common uniform time axis within their overlapping time range using CubicSpline.
    
    Args:
        signals (list[tuple[np.ndarray, np.ndarray]]): List of signals as tuples (x_data, y_data).
        ix_step (int, optional): Time step for the uniform axis in milliseconds. Default is 1000 ms.
    
    Returns:
        tuple[np.ndarray, list[np.ndarray]]: A tuple containing the new uniform time axis and the list of interpolated signals.
    """
    # Find the common time range
    max_common_x = min(np.max(x) for x, _ in signals)
    min_common_x = max(np.min(x) for x, _ in signals)
    
    # Generate the new uniform time axis
    ix = np.arange(min_common_x, max_common_x, ix_step)
    
    # Interpolate each signal to the uniform time axis using CubicSpline
    interpolated_signals = [
        CubicSpline(x, y)(ix) for x, y in signals
    ]
    
    return ix, interpolated_signals


#%%
def validate_array(array: np.ndarray, min_length: int = None, max_length: int = None, 
                   val_nan: bool = False, val_inf: bool = False, val_num: bool = False) -> bool:
    """
    Validates an input numpy array based on specified criteria:
    
    1. Checks if the input is a numpy array.
    2. Ensures there are no NaNs in the array if val_nan is True.
    3. Ensures there are no infinite values in the array if val_inf is True.
    4. Ensures the array length is greater than a specified minimum length, if min_length is provided.
    5. Ensures the array length is less than or equal to a specified maximum length, if max_length is provided.
    6. Verifies that the array contains numeric types only if val_num is True.

    Args:
        array (np.ndarray): The input data to validate.
        min_length (int, optional): The minimum length the array must have to be considered valid. Defaults to None.
        max_length (int, optional): The maximum length the array can have to be considered valid. Defaults to None.
        val_nan (bool, optional): If True, checks for NaNs in the array. Defaults to False.
        val_inf (bool, optional): If True, checks for infinite values in the array. Defaults to False.
        val_num (bool, optional): If True, checks that all elements are numeric. Defaults to False.

    Returns:
        bool: True if all specified validation checks pass, False otherwise.
    """
    # Check if the input is a numpy array
    if not isinstance(array, np.ndarray):
        return False
    
    # Check if the array length is greater than min_length if min_length is specified
    if min_length is not None and len(array) <= min_length:
        return False
    
    # Check if the array length is less than or equal to max_length if max_length is specified
    if max_length is not None and len(array) > max_length:
        return False
    
    # Check for NaNs in the array if val_nan is True
    if val_nan and np.isnan(array).any():
        return False
    
    # Check for infinite values in the array if val_inf is True
    if val_inf and np.isinf(array).any():
        return False
    
    # Check that all elements are numeric if val_num is True
    if val_num and not np.issubdtype(array.dtype, np.number):
        return False

    return True

#%%
def overlapping_sd(signal: tuple[np.ndarray, np.ndarray], window_time: float, overlap: float, min_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the standard deviation over overlapping windows in the signal, skipping windows with insufficient data points.

    Parameters:
    -----------
    signal : tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - signal[0]: time array (in seconds or another time unit)
        - signal[1]: corresponding signal values (e.g., heart rate or other measurements)
        
    window_time : float
        The size of the sliding window in the same time unit as the time array.
        
    overlap : float
        Overlap between consecutive windows as a percentage (e.g., 0.5 for 50% overlap).
    
    min_fraction : float
        Minimum fraction of the average number of elements per window. If the number of data points 
        in a window is less than this threshold, the window is skipped.

    Returns:
    --------
    sd_values : np.ndarray
        An array of calculated standard deviations for each window.
        
    window_centers : np.ndarray
        An array of the center points (times) for each window where SD was calculated.
    """
    
    time, values = signal
    step_time = window_time * (1 - overlap)  # Calculate the time step based on overlap
    
    # Initialize lists to store SD values and window centers
    sd_values = []
    window_centers = []
    
    # Estimate average number of elements in each window based on the first window
    avg_elements_per_window = np.mean(np.diff(np.searchsorted(time, [time[0], time[0] + window_time])))
    min_elements_threshold = avg_elements_per_window * min_fraction  # Set the threshold for minimum elements

    start_idx = 0  # Starting index for sliding window
    while start_idx < len(time):
        # Define window start and end times based on the current start_idx
        window_start_time = time[start_idx]
        window_end_time = window_start_time + window_time
        
        # Find the indices of the points that fall within the window
        selected_indices = np.where((time >= window_start_time) & (time < window_end_time))[0]
        
        # Skip windows that do not meet the minimum elements requirement
        if len(selected_indices) < min_elements_threshold:
            # Move the window forward by step_time
            start_idx = np.searchsorted(time, time[start_idx] + step_time)
            continue
        
        # Extract the signal values for the current window
        window_values = values[selected_indices]
        
        # If the window has more than 1 data point, calculate its standard deviation
        if len(window_values) > 1:
            sd = np.std(window_values)
            sd_values.append(sd)
            
            # Calculate the window center in terms of time
            window_center_time = (window_start_time + window_end_time) / 2
            window_centers.append(window_center_time)
        
        # Move the start index forward by the calculated step_time
        start_idx = np.searchsorted(time, time[start_idx] + step_time)

    return np.array(sd_values), np.array(window_centers)

#%%
def overlapping_rmssd(signal: tuple[np.ndarray, np.ndarray], window_time: float, overlap: float, min_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Root Mean Square of Successive Differences (RMSSD) over overlapping windows in the signal, skipping windows with insufficient data points.

    Parameters:
    -----------
    signal : tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - signal[0]: time array (in seconds or another time unit)
        - signal[1]: corresponding signal values (e.g., heart rate or other measurements)
        
    window_time : float
        The size of the sliding window in the same time unit as the time array.
        
    overlap : float
        Overlap between consecutive windows as a percentage (e.g., 0.5 for 50% overlap).
    
    min_fraction : float
        Minimum fraction of the average number of elements per window. If the number of data points 
        in a window is less than this threshold, the window is skipped.

    Returns:
    --------
    rmssd_values : np.ndarray
        An array of calculated RMSSD values for each window.
        
    window_centers : np.ndarray
        An array of the center points (times) for each window where RMSSD was calculated.
    """
    
    time, values = signal
    step_time = window_time * (1 - overlap)  # Calculate the time step based on overlap
    
    # Initialize lists to store RMSSD values and window centers
    rmssd_values = []
    window_centers = []
    
    # Estimate average number of elements in each window based on the first window
    avg_elements_per_window = np.mean(np.diff(np.searchsorted(time, [time[0], time[0] + window_time])))
    min_elements_threshold = avg_elements_per_window * min_fraction  # Set the threshold for minimum elements

    start_idx = 0  # Starting index for sliding window
    while start_idx < len(time):
        # Define window start and end times based on the current start_idx
        window_start_time = time[start_idx]
        window_end_time = window_start_time + window_time
        
        # Find the indices of the points that fall within the window
        selected_indices = np.where((time >= window_start_time) & (time < window_end_time))[0]
        
        # Skip windows that do not meet the minimum elements requirement
        if len(selected_indices) < min_elements_threshold:
            # Move the window forward by step_time
            start_idx = np.searchsorted(time, time[start_idx] + step_time)
            continue
        
        # Extract the signal values for the current window
        window_values = values[selected_indices]
        
        # If the window has more than 1 data point, calculate RMSSD
        if len(window_values) > 1:
            successive_diffs = np.diff(window_values)  # Calculate successive differences
            rmssd = np.sqrt(np.mean(successive_diffs ** 2))  # Root mean square of successive differences
            rmssd_values.append(rmssd)
            
            # Calculate the window center in terms of time
            window_center_time = (window_start_time + window_end_time) / 2
            window_centers.append(window_center_time)
        
        # Move the start index forward by the calculated step_time
        start_idx = np.searchsorted(time, time[start_idx] + step_time)

    return np.array(rmssd_values), np.array(window_centers)



