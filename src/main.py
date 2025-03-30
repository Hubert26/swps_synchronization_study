# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:45:31 2023

@author: Hubert Szewczyk
"""

import logging
from config import DATA_DIR, ANALYSIS_DATA_DIR
from logger_config import log_message
from correlation_functions import process_regular_pairs, process_baseline_pairs
from data_management_functions import load_data, records_to_dataframe, find_best_results
from charts_functions import *
from metrics_functions import (
    instant_hr_meas,
    calculate_overlapping_sd_meas,
    calculate_overlapping_rmssd_meas,
    filter_rr_meas,
)
from time_functions import normalize_meas_time

from utils.dataframe_utils import write_to_excel
from utils.file_utils import list_file_paths


# %%
def load_and_prepare_data():
    """
    Load and prepare data for processing.

    Returns:
        tuple: Lists of processed measurements (nn_list, hr_list, sdnn_list, rmssd_list).
    """
    logging.info("Loading and preprocessing data...")
    file_paths = list_file_paths(DATA_DIR)
    rr_list = load_data(file_paths)
    rr_list = normalize_meas_time(rr_list)

    nn_list = filter_rr_meas(rr_list)
    hr_list = instant_hr_meas(nn_list)
    sdnn_list = calculate_overlapping_sd_meas(
        nn_list, window_ms=10000, overlap=0.8, min_fraction=0.3
    )
    rmssd_list = calculate_overlapping_rmssd_meas(
        nn_list, window_ms=10000, overlap=0.8, min_fraction=0.3
    )
    log_message(logging.INFO, "Data loading and preprocessing completed.")
    return nn_list, hr_list, sdnn_list, rmssd_list


def process_regular_correlations(nn_list, hr_list, sdnn_list, rmssd_list):
    """
    Process regular correlations for NN, HR, SDNN, and RMSSD measurements.

    Args:
        nn_list (list): List of NN measurements.
        hr_list (list): List of HR measurements.
        sdnn_list (list): List of SDNN measurements.
        rmssd_list (list): List of RMSSD measurements.

    Returns:
        tuple: DataFrames with correlation results for NN, HR, SDNN, and RMSSD.
    """
    log_message(logging.INFO, "Calculating regular correlations...")

    nn_results_df = records_to_dataframe(process_regular_pairs(nn_list))
    hr_results_df = records_to_dataframe(process_regular_pairs(hr_list))
    sdnn_results_df = records_to_dataframe(process_regular_pairs(sdnn_list))
    rmssd_results_df = records_to_dataframe(process_regular_pairs(rmssd_list))

    log_message(logging.INFO, "Regular correlation calculations completed.")
    return nn_results_df, hr_results_df, sdnn_results_df, rmssd_results_df


def process_baseline_correlations(nn_list, hr_list, sdnn_list, rmssd_list):
    """
    Process baseline correlations for NN, HR, SDNN, and RMSSD measurements.

    Args:
        nn_list (list): List of NN measurements.
        hr_list (list): List of HR measurements.
        sdnn_list (list): List of SDNN measurements.
        rmssd_list (list): List of RMSSD measurements.

    Returns:
        tuple: DataFrames with baseline correlation results for NN, HR, SDNN, and RMSSD.
    """
    log_message(logging.INFO, "Calculating baseline correlations...")

    nn_baseline_df = records_to_dataframe(process_baseline_pairs(nn_list))
    hr_baseline_df = records_to_dataframe(process_baseline_pairs(hr_list))
    sdnn_baseline_df = records_to_dataframe(process_baseline_pairs(sdnn_list))
    rmssd_baseline_df = records_to_dataframe(process_baseline_pairs(rmssd_list))

    log_message(logging.INFO, "Baseline correlation calculations completed.")
    return nn_baseline_df, hr_baseline_df, sdnn_baseline_df, rmssd_baseline_df


def main():
    """
    Main function to orchestrate the processing of data and calculations.
    """
    nn_list, hr_list, sdnn_list, rmssd_list = load_and_prepare_data()

    # Process regular correlations
    # nn_regular_df, hr_regular_df, sdnn_regular_df, rmssd_regular_df = (
    #     process_regular_correlations(nn_list, hr_list, sdnn_list, rmssd_list)
    # )

    # Process baseline correlations
    nn_baseline_df, hr_baseline_df, sdnn_baseline_df, rmssd_baseline_df = (
        process_baseline_correlations(nn_list, hr_list, sdnn_list, rmssd_list)
    )

    # Saveing all corr beseline
    write_to_excel(nn_baseline_df, ANALYSIS_DATA_DIR / "nn_baseline.xlsx")
    write_to_excel(hr_baseline_df, ANALYSIS_DATA_DIR / "hr_baseline.xlsx")
    write_to_excel(sdnn_baseline_df, ANALYSIS_DATA_DIR / "sdnn_baseline.xlsx")
    write_to_excel(rmssd_baseline_df, ANALYSIS_DATA_DIR / "rmssd_baseline.xlsx")


# %%
if __name__ == "__main__":
    main()


# %%
# Ploting

# #Ploting full measurements
# meas_plot_from(rr_list, folder_name="rr_meas", title_label="RR-intervals", value_label="Time Between Heartbeats [ms]")
# pair_plots_from(rr_list, folder_name="rr_pairs", title_label="paired RR-intervals", value_label="Time Between Heartbeats [ms]")
# pair_plots_from(nn_list, folder_name="nn_pairs", title_label="paired NN-intervals", value_label="Time Between Heartbeats [ms]")
# pair_plots_from(hr_list, folder_name="hr_pairs", title_label="paired Heart Rate", value_label="Heart Rate [bpm]")
# pair_plots_from(sdnn_list, folder_name="sdnn_pairs", title_label="paired Standard Deviation of NN-intervals", value_label="SDNN [ms]")
# pair_plots_from(rmssd_list, folder_name="rmssd_pairs", title_label="paired Root Mean Square of Successive Differences of NN-intervals", value_label="RMSSD [ms]")

# %%
# #Ploting interpolated measurmenents
# save_final_pairs_plots(best_nn_results, folder_name="nn_results", title_label="interpolated and paired NN-intervals", value_label="Time Between Heartbeats [ms]")
# save_final_pairs_plots(best_hr_results, folder_name="hr_results", title_label="interpolated and paired Heart Rate", value_label="Heart Rate [bpm]")
# save_final_pairs_plots(best_sdnn_results, folder_name="sdnn_results", title_label="interpolated and paired Standard Deviation of NN-intervals", value_label="SDNN [ms]")
# save_final_pairs_plots(best_rmssd_results, folder_name="rmssd_results", title_label="interpolated and paired Successive Differences of NN-intervals", value_label="RMSSD [ms]")


# %%
# # Ploting corr heatmaps
# save_corr_heatmap_by_task_and_shift(
#     nn_results_df,
#     folder_name="nn_results",
#     title_label="Correlation heatmap of shifted NN-intervals",
# )
# save_corr_heatmap_by_task_and_shift(
#     hr_results_df,
#     folder_name="hr_results",
#     title_label="Correlation heatmap of shifted Heart Rate",
# )
# save_corr_heatmap_by_task_and_shift(
#     sdnn_results_df,
#     folder_name="sdnn_results",
#     title_label="Correlation heatmap of shifted Standard Deviation of NN-intervals",
# )
# save_corr_heatmap_by_task_and_shift(
#     rmssd_results_df,
#     folder_name="rmssd_results",
#     title_label="Correlation heatmap of shifted Successive Differences of NN-intervals",
# )

# # %%
# # Saveing all corr results
# write_to_excel(nn_results_df, ANALYSIS_DATA_DIR / "nn_results.xlsx")
# write_to_excel(hr_results_df, ANALYSIS_DATA_DIR / "hr_results.xlsx")
# write_to_excel(sdnn_results_df, ANALYSIS_DATA_DIR / "sdnn_results.xlsx")
# write_to_excel(rmssd_results_df, ANALYSIS_DATA_DIR / "rmssd_results.xlsx")
# # %%
# # Saveing best corr results
# write_to_excel(best_nn_results_df, ANALYSIS_DATA_DIR / "best_nn_results.xlsx")
# write_to_excel(best_hr_results_df, ANALYSIS_DATA_DIR / "best_hr_results.xlsx")
# write_to_excel(best_sdnn_results_df, ANALYSIS_DATA_DIR / "best_sdnn_results.xlsx")
# write_to_excel(best_rmssd_results_df, ANALYSIS_DATA_DIR / "best_rmssd_results.xlsx")
