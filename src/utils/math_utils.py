# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:20:45 2024

@author: huber
"""
import numpy as np
from scipy.interpolate import interp1d




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
def interp_signals_uniform_time(signals: list[tuple[np.ndarray, np.ndarray]], ix_step: int = 1000, method: str = 'linear', fill_value='extrapolate') -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Interpolates multiple signals to a common uniform time axis within their overlapping time range.
    
    Args:
        signals (list[tuple[np.ndarray, np.ndarray]]): List of signals as tuples (x_data, y_data).
        ix_step (int, optional): Time step for the uniform axis in milliseconds. Default is 1000 ms.
        method (str, optional): Interpolation method ('linear', 'quadratic', 'cubic', etc.). Default is 'linear'.
        fill_value (str or float, optional): How to handle out-of-bounds values. Default is 'extrapolate'.
    
    Returns:
        tuple[np.ndarray, list[np.ndarray]]: A tuple containing the new uniform time axis and the list of interpolated signals.
    """
    # Find the common time range
    max_common_x = min(np.max(x) for x, _ in signals)
    min_common_x = max(np.min(x) for x, _ in signals)
    
    # Generate the new uniform time axis
    ix = np.arange(min_common_x, max_common_x, ix_step)
    
    # Interpolate each signal to the uniform time axis
    interpolated_signals = [
        interp1d(x, y, kind=method, fill_value=fill_value)(ix) for x, y in signals
    ]
    
    return ix, interpolated_signals

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
def validate_array(array: np.ndarray, min_length: int = 3) -> bool:
    """
    Validates an input numpy array based on certain criteria:
    
    1. Checks if the input is a numpy array.
    2. Ensures there are no NaNs in the array.
    3. Ensures the array length is greater than a specified minimum length.

    Args:
        array (np.ndarray): The input data to validate.
        min_length (int, optional): The minimum length the array must have to be considered valid. Defaults to 3.

    Returns:
        bool: True if all validation checks pass, False otherwise.
    """
    # Check if data is a numpy array
    if not isinstance(array, np.ndarray):
        return False
    
    # Check for NaNs in the array
    if np.isnan(array).any():
        return False
    
    # Check if array length is greater than min_length
    if len(array) <= min_length:
        return False

    return True