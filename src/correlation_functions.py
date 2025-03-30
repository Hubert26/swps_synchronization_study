# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import copy
from scipy.stats import spearmanr, combine_pvalues, chi2

import logging
from logger_config import log_message
from config import (
    get_time_intervals,
    MIN_DURATION_RATIO,
    SHIFT_MAX_MS,
    SHIFT_MIN_MS,
    SHIFT_STEP_MS,
)
from classes import Meas, MeasurementRecord
from data_management_functions import merge_meas
from time_functions import trim_meas, interp_meas_pair_uniform_time

from utils.math_utils import fisher_transform
from utils.general_utils import group_object_list


class MeasurementProcessingError(Exception):
    """Custom exception for errors during measurement processing."""


# %%
def calc_corr_weighted(meas_pair_lists: tuple[list[Meas], list[Meas]]) -> tuple:
    """Calculates weighted correlation and combines p-values for measurement pairs."""
    meas1_list, meas2_list = meas_pair_lists

    corr_list, p_val_list, weights = [], [], []
    interp_meas1_list, interp_meas2_list = [], []

    for meas1 in meas1_list:
        for meas2 in meas2_list:
            try:
                interp_meas1, interp_meas2 = interp_meas_pair_uniform_time(
                    (meas1, meas2), ix_step=250
                )
            except ValueError as e:
                log_message(
                    logging.INFO,
                    "Interpolation failed",
                    meas1=str(meas1),
                    meas2=str(meas2),
                    error=str(e),
                )
                continue

            if not interp_meas1 or not interp_meas2:
                log_message(
                    logging.INFO,
                    "Skipping due to interpolation failure",
                    meas1=str(meas1),
                    meas2=str(meas2),
                )
                continue

            interp_meas1_list.append(interp_meas1)
            interp_meas2_list.append(interp_meas2)

            corr, p_val = spearmanr(interp_meas1.data.y_data, interp_meas2.data.y_data)
            corr = fisher_transform(corr)
            weight = len(interp_meas1.data.x_data)

            corr_list.append(corr)
            p_val_list.append(p_val)
            weights.append(weight)

    if not corr_list:
        log_message(
            logging.WARNING,
            "No valid correlations computed",
            meas1_list=str(meas1_list),
            meas2_list=str(meas2_list),
        )
        return None, None, None, None, None, None, None

    try:
        merged_meas1 = merge_meas(interp_meas1_list)
        merged_meas2 = merge_meas(interp_meas2_list)
    except ValueError as e:
        log_message(logging.ERROR, "Error during merging", error=str(e))
        raise MeasurementProcessingError("Merging failed.") from e

    avg_corr = np.average(corr_list, weights=weights)
    fisher_stat, _ = combine_pvalues(p_val_list, method="fisher")
    combined_p_val = 1 - chi2.cdf(fisher_stat, 2 * len(p_val_list))

    return (
        avg_corr,
        combined_p_val,
        meas1_list[0].metadata.shift - meas2_list[0].metadata.shift,
        str(meas1_list[0]),
        str(meas2_list[0]),
        merged_meas1,
        merged_meas2,
    )


# %%
def calc_corr_pair(person_meas1, person_meas2, meas_number, condition, pair_number):
    """Main processing function to calculate measurement correlations, including shifts."""
    try:
        if not person_meas1 or not person_meas2:
            raise MeasurementProcessingError(
                f"Missing data for meas_number: {meas_number}, "
                f"pair_number: {pair_number}"
            )

        log_message(
            logging.INFO,
            "Processing pair",
            meas_number=meas_number,
            condition=condition,
            pair_number=pair_number,
        )

        # Find the oldest starttime among all measurements
        oldest_starttime = min(
            min(meas.metadata.starttime for meas in person_meas1),
            min(meas.metadata.starttime for meas in person_meas2),
        )

        # Create shifted versions of the measurement pair
        shifted_meas1_list = []
        shifted_meas2_list = []

        for shift_ms in range(SHIFT_MIN_MS, SHIFT_MAX_MS + 1, SHIFT_STEP_MS):
            shifted_meas1 = [copy.deepcopy(meas) for meas in person_meas1]
            shifted_meas2 = [copy.deepcopy(meas) for meas in person_meas2]

            for meas in shifted_meas1:
                meas.shift_right(shift_ms)
            for meas in shifted_meas2:
                meas.shift_right(shift_ms)

            shifted_meas1_list.extend(shifted_meas1)
            shifted_meas2_list.extend(shifted_meas2)

        final_corr_results = []
        for (start_ms, end_ms), task in get_time_intervals(condition).items():
            lower_bound, upper_bound = get_time_bounds(
                oldest_starttime, start_ms, end_ms
            )

            # Select measurements based on time bounds
            selected_meas1 = [
                meas
                for meas in person_meas1
                if lower_bound <= meas.metadata.endtime
                and meas.metadata.starttime <= upper_bound
            ]
            selected_meas2 = [
                meas
                for meas in person_meas2
                if lower_bound <= meas.metadata.endtime
                and meas.metadata.starttime <= upper_bound
            ]

            selected_shifted_meas1 = [
                meas
                for meas in shifted_meas1_list
                if lower_bound <= meas.metadata.endtime
                and meas.metadata.starttime <= upper_bound
            ]
            selected_shifted_meas2 = [
                meas
                for meas in shifted_meas2_list
                if lower_bound <= meas.metadata.endtime
                and meas.metadata.starttime <= upper_bound
            ]

            trimmed_meas1 = trim_measurements(
                selected_meas1, start_ms, end_ms, oldest_starttime
            )
            trimmed_meas2 = trim_measurements(
                selected_meas2, start_ms, end_ms, oldest_starttime
            )
            trimmed_shifted_meas1 = trim_measurements(
                selected_shifted_meas1, start_ms, end_ms, oldest_starttime
            )
            trimmed_shifted_meas2 = trim_measurements(
                selected_shifted_meas2, start_ms, end_ms, oldest_starttime
            )

            if not trimmed_meas1 and not trimmed_shifted_meas1:
                log_message(
                    logging.WARNING,
                    "No valid meas1 data",
                    meas_number=meas_number,
                    task=task,
                )
                continue

            if not trimmed_meas2 and not trimmed_shifted_meas2:
                log_message(
                    logging.WARNING,
                    "No valid meas2 data",
                    meas_number=meas_number,
                    task=task,
                )
                continue

            interval_duration_min = (end_ms - start_ms) / 1000 / 60

            # Calculate correlation for shift=0 with shift=0
            if trimmed_meas1 and trimmed_meas2:
                results = calculate_correlations(
                    trimmed_meas1,
                    trimmed_meas2,
                    meas_number,
                    condition,
                    pair_number,
                    task,
                    interval_duration_min,
                )
                final_corr_results.extend(results)

            # Calculate shift=0 with all other shift1
            if trimmed_shifted_meas1 and trimmed_meas2:
                grouped_shifted_meas1 = group_object_list(
                    trimmed_shifted_meas1, ["metadata.shift"]
                )
                for shift1, meas1_group in grouped_shifted_meas1.items():
                    results = calculate_correlations(
                        meas1_group,
                        trimmed_meas2,
                        meas_number,
                        condition,
                        pair_number,
                        task,
                        interval_duration_min,
                    )
                    final_corr_results.extend(results)

            # Calculate shift=0 with all other shift2
            if trimmed_shifted_meas2 and trimmed_meas1:
                grouped_shifted_meas2 = group_object_list(
                    trimmed_shifted_meas2, ["metadata.shift"]
                )
                for shift2, meas2_group in grouped_shifted_meas2.items():
                    results = calculate_correlations(
                        trimmed_meas1,
                        meas2_group,
                        meas_number,
                        condition,
                        pair_number,
                        task,
                        interval_duration_min,
                    )
                    final_corr_results.extend(results)

        return final_corr_results

    except MeasurementProcessingError as e:
        log_message(logging.ERROR, "Measurement processing failed", error=str(e))
        raise

    except Exception as e:
        log_message(logging.CRITICAL, "Unexpected error", error=str(e))
        raise


# %%
def get_time_bounds(oldest_starttime, start_ms, end_ms):
    """Returns lower and upper bounds based on the provided time intervals."""
    lower_bound = oldest_starttime + pd.Timedelta(milliseconds=start_ms)
    upper_bound = oldest_starttime + pd.Timedelta(milliseconds=end_ms)
    return lower_bound, upper_bound


# %%
def trim_measurements(measurements, start_ms, end_ms, oldest_starttime):
    """Trims measurements to the specified interval."""
    trimmed_list = []
    for meas in measurements:
        try:
            trimmed_list.append(
                trim_meas(meas, start_ms, end_ms, starttime=oldest_starttime)
            )
        except ValueError as e:
            log_message(
                logging.INFO,
                "Trimming failed",
                meas_metadata=str(meas.metadata),
                error=str(e),
            )

            continue
    return trimmed_list


# %%
def calculate_correlations(
    meas1_list,
    meas2_list,
    meas_number,
    condition,
    pair_number,
    task,
    interval_duration_min,
):
    results = []

    # Compute weighted correlation and other statistics
    corr, p_val, shift_diff, name1, name2, interp1, interp2 = calc_corr_weighted(
        (meas1_list, meas2_list)
    )

    # Check if interpolation was successful
    if interp1 is None or interp2 is None:
        log_message(
            logging.WARNING,
            "No valid interpolated measurements",
            meas_number=meas_number,
            condition=condition,
            pair_number=pair_number,
            task=task,
        )
        return results

    # Validate if the interpolated data meets the minimum required duration ratio
    if (
        interp1.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO
        and interp2.metadata.duration_min > interval_duration_min * MIN_DURATION_RATIO
    ):
        results.append(
            MeasurementRecord(
                meas_number=meas_number,
                condition=condition,
                pair_number=pair_number,
                task=task,
                shift_diff=shift_diff,
                corr=corr,
                p_val=p_val,
                name_meas1=name1,
                name_meas2=name2,
                meas1=interp1,
                meas2=interp2,
            )
        )
    else:
        log_message(
            logging.INFO,
            "Interpolated data does not meet the minimum duration requirement",
            meas_number=meas_number,
            condition=condition,
            pair_number=pair_number,
            task=task,
            min_duration_ratio=MIN_DURATION_RATIO,
            measured_duration_1=interp1.metadata.duration_min,
            measured_duration_2=interp2.metadata.duration_min,
        )

    return results


# %%
def process_regular_pairs(meas_list: list[Meas]) -> list[MeasurementRecord]:
    """Process regular pairs of measurements."""

    final_corr_results = []

    grouped_meas = group_object_list(
        meas_list,
        ["metadata.meas_number", "metadata.condition", "metadata.pair_number"],
    )

    # Iterate over each group of measurements
    for key, group in grouped_meas.items():
        meas_number, condition, pair_number = key

        person_meas1 = [meas for meas in group if meas.metadata.gender == "M"]
        person_meas2 = [meas for meas in group if meas.metadata.gender == "F"]

        if not person_meas1 or not person_meas2:
            log_message(
                logging.WARNING,
                "Skipping measurement pair due to missing gender data",
                meas_number=meas_number,
                condition=condition,
                pair_number=pair_number,
            )
            continue

        log_message(
            logging.INFO,
            "Processing measurement pair",
            meas_number=meas_number,
            condition=condition,
            pair_number=pair_number,
            meas1_count=len(person_meas1),
            meas2_count=len(person_meas2),
        )

        try:
            final_corr_results.extend(
                calc_corr_pair(
                    person_meas1, person_meas2, meas_number, condition, pair_number
                )
            )
        except Exception as e:
            log_message(
                logging.ERROR,
                "Error processing measurement pair",
                meas_number=meas_number,
                condition=condition,
                pair_number=pair_number,
                error=str(e),
            )

    return final_corr_results


# %%
def process_baseline_pairs(meas_list: list[Meas]) -> list[MeasurementRecord]:
    """Process baseline pairs of measurements across different pair numbers."""

    final_corr_results = []

    # Group measurements based on meas_number, condition, pair_number, and gender
    grouped_meas = group_object_list(
        meas_list,
        [
            "metadata.meas_number",
            "metadata.condition",
            "metadata.pair_number",
            "metadata.gender",
        ],
    )

    # Separate measurements by gender across different pairs
    grouped_male_meas = {
        key: group for key, group in grouped_meas.items() if key[-1] == "M"
    }
    grouped_female_meas = {
        key: group for key, group in grouped_meas.items() if key[-1] == "F"
    }

    for (
        meas_number_m,
        condition_m,
        pair_number_m,
        _,
    ), person_meas1 in grouped_male_meas.items():
        for (
            meas_number_f,
            condition_f,
            pair_number_f,
            _,
        ), person_meas2 in grouped_female_meas.items():
            # Ensure we are comparing different pair_numbers
            if (
                pair_number_m != pair_number_f
                and meas_number_m == meas_number_f
                and condition_m == condition_f
            ):
                log_message(
                    logging.INFO,
                    "Processing baseline pair comparison",
                    meas_number=meas_number_m,
                    condition=condition_m,
                    pair_number_1=pair_number_m,
                    pair_number_2=pair_number_f,
                    num_male=len(person_meas1),
                    num_female=len(person_meas2),
                )

                try:
                    baseline_pair_id = pair_number_m * 1000 + pair_number_f
                    final_corr_results.extend(
                        calc_corr_pair(
                            person_meas1,
                            person_meas2,
                            meas_number_m,
                            condition_m,
                            pair_number=baseline_pair_id,
                        )
                    )
                except Exception as e:
                    log_message(
                        logging.ERROR,
                        "Error processing baseline pair",
                        meas_number=meas_number_m,
                        condition=condition_m,
                        pair_number_1=pair_number_m,
                        pair_number_2=pair_number_f,
                        error=str(e),
                    )

    return final_corr_results
