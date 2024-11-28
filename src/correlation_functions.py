# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import copy
from scipy.stats import pearsonr, combine_pvalues, chi2

from config import *
from classes import *
from data_management_functions import merge_meas
from time_functions import trim_meas, interp_meas_pair_uniform_time

from utils.math_utils import fisher_transform
from utils.general_utils import group_object_list

#%%
def calc_corr_weighted(meas_pair_lists: tuple[list[Meas], list[Meas]]) -> tuple:
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
    tuple:
        - avg_corr (float): The weighted average of the Fisher-transformed Pearson correlation coefficients.
        - combined_p_val (float): The combined p-value derived using Fisher's method.
        - shift_diff (float): The difference in shift values from the metadata of the two Meas objects.
        - meas1_name (str): String representation of the first Meas object in the pair.
        - meas2_name (str): String representation of the second Meas object in the pair.
        - merged_meas1 (Meas): Meas object from the interpolated and combined signals from input lists.
        - merged_meas2 (Meas): Meas object from the interpolated and combined signals from input lists.

    Notes:
    ------
    - If no valid correlations are computed, the function will return `(None, None)`.
    - The weighted average correlation is calculated based on the length of the interpolated signals as weights.
    - P-values are combined using Fisher’s method via `combine_pvalues`.
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
        return None, None, meas1.metadata.shift - meas2.metadata.shift, str(meas1), str(meas2), None, None

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

    return avg_corr, combined_p_val, meas1.metadata.shift - meas2.metadata.shift, str(meas1), str(meas2), merged_meas1, merged_meas2

#%%
def process_meas_and_find_corr(meas_list: list[Meas]) -> list[MeasurementRecord]:
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
    final_corr_results : list[MeasurementRecord]
        A list of MeasurementRecord objects containing the correlation results for each measurement pair.
    """
    
    final_corr_results = []
    
    grouped_meas = group_object_list(meas_list, ["metadata.meas_number", "metadata.condition", "metadata.pair_number"])

    # Iterate over each group of measurements
    for key, group in grouped_meas.items():      
        meas_number, condition, pair_number = key
                
        person_meas1 = [meas for meas in group if meas.metadata.gender == 'M']
        person_meas2 = [meas for meas in group if meas.metadata.gender == 'F']
        
        if not person_meas1 or not person_meas2:
            logger.warning(f"{meas_number}, {condition}, {pair_number}: Skipping pair. Lack of meas1 or meas2.")
            continue
            
        # Get time intervals for the given measurement type
        selected_intervals = get_time_intervals(condition)
        
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
        
        
        # Iterate through the selected time intervals
        for (start_ms, end_ms), task in selected_intervals.items():
                            
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
                    logger.info(f"{meas_number}, {condition}, {pair_number}, {task}: Trimming meas to interval faild. Skipping meas due to invalid time range: {e}")
                    continue
                trimmed_meas1_list.append(tm1)
            
            for loop_meas2 in selected_person_meas2:
                try:
                    tm2 = trim_meas(loop_meas2, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {condition}, {pair_number}, {task}: Trimming meas to interval faild. Skipping meas due to invalid time range: {e}")
                    continue
                trimmed_meas2_list.append(tm2)
                
            for loop_meas1 in selected_shifted_meas1_list:
                try:
                    tsm1 = trim_meas(loop_meas1, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {condition}, {pair_number}, {task}: Trimming shifted meas to interval faild. Skipping shifted meas due to invalid time range: {e}")
                    continue
                trimmed_shifted_meas1_list.append(tsm1)

            for loop_meas2 in selected_shifted_meas2_list:
                try:
                    tsm2 = trim_meas(loop_meas2, start_ms, end_ms, starttime=oldest_starttime)
                except ValueError as e:
                    logger.info(f"{meas_number}, {condition}, {pair_number}, {task}: Trimming shifted meas to interval faild. Skipping shifted meas due to invalid time range: {e}")
                    continue
                trimmed_shifted_meas2_list.append(tsm2)
                
            if not trimmed_meas1_list and not trimmed_shifted_meas1_list:
                logger.warning(f"{meas_number}, {condition}, {pair_number}, {task}: Skipping pair. Lack of trimmed meas1")
                continue
            
            if not trimmed_meas2_list and not trimmed_shifted_meas2_list:
                logger.warning(f"{meas_number}, {condition}, {pair_number}, {task}: Skipping pair. Lack of trimmed meas2")
                continue            
            
            
            interval_duration_min = (end_ms - start_ms) / 1000 / 60

            # Calculate shift1=0 with shift2=0
            if trimmed_meas1_list and trimmed_meas2_list:
                corr, p_val, shift_diff, name_meas1, name_meas2, interp_meas1, interp_meas2 = calc_corr_weighted((trimmed_meas1_list, trimmed_meas2_list))
                if corr is not None and interp_meas1 is not None and interp_meas2 is not None:
                    # Ensure that the duration of each interpolated measurement exceeds MIN_DURATION_RATIO * interval_duration_min
                    if interp_meas1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO and interp_meas2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO:
                        final_corr_results.append(
                            MeasurementRecord(
                                meas_number = meas_number,
                                condition = condition,
                                pair_number = pair_number,
                                task = task,
                                shift_diff = shift_diff,
                                corr = corr,
                                p_val = p_val,
                                name_meas1 = name_meas1,
                                name_meas2 = name_meas2,
                                meas1 = interp_meas1,
                                meas2 = interp_meas2
                                )
                            )
            
            # Group trimmed shifted_meas1 by their shift value
            if trimmed_shifted_meas1_list and trimmed_meas2_list:
                trimmed_shifted_gruped_meas1 = group_object_list(trimmed_shifted_meas1_list, ["metadata.shift"])
                
                # Calculate shift2=0 with every shift1
                for shift1, meas1_group in trimmed_shifted_gruped_meas1.items():                        
                    corr, p_val, shift_diff, name_meas1, name_meas2, interp_meas1, interp_meas2 = calc_corr_weighted((meas1_group, trimmed_meas2_list))
                    if corr is not None and interp_meas1 is not None and interp_meas2 is not None:
                        # Ensure that the duration of each interpolated measurement exceeds MIN_DURATION_RATIO * interval_duration_min
                        if interp_meas1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO and interp_meas2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO:
                            final_corr_results.append(
                                MeasurementRecord(
                                    meas_number = meas_number,
                                    condition = condition,
                                    pair_number = pair_number,
                                    task = task,
                                    shift_diff = shift_diff,
                                    corr = corr,
                                    p_val = p_val,
                                    name_meas1 = name_meas1,
                                    name_meas2 = name_meas2,
                                    meas1 = interp_meas1,
                                    meas2 = interp_meas2
                                    )
                                )
                        
            if trimmed_shifted_meas2_list and trimmed_meas1_list:
                trimmed_shifted_gruped_meas2 = group_object_list(trimmed_shifted_meas2_list, ["metadata.shift"])
                
                # Calculate shift1=0 with every shift2
                for shift2, meas2_group in trimmed_shifted_gruped_meas2.items():
                    corr, p_val, shift_diff, name_meas1, name_meas2, interp_meas1, interp_meas2 = calc_corr_weighted((trimmed_meas1_list, meas2_group))
                    if corr is not None and interp_meas1 is not None and interp_meas2 is not None:
                        # Ensure that the duration of each interpolated measurement exceeds MIN_DURATION_RATIO * interval_duration_min
                        if interp_meas1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO and interp_meas2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO:
                            final_corr_results.append(
                                MeasurementRecord(
                                    meas_number = meas_number,
                                    condition = condition,
                                    pair_number = pair_number,
                                    task = task,
                                    shift_diff = shift_diff,
                                    corr = corr,
                                    p_val = p_val,
                                    name_meas1 = name_meas1,
                                    name_meas2 = name_meas2,
                                    meas1 = interp_meas1,
                                    meas2 = interp_meas2
                                    )
                                )

    return final_corr_results





