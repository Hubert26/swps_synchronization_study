# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:39:35 2024

@author: Hubert Szewczyk
"""
import numpy as np

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
