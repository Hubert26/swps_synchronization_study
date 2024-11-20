# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:45:31 2023

@author: Hubert Szewczyk
"""

from config import *
from functions import *
from utils.dataframe_utils import write_to_excel
from utils.file_utils import list_file_paths
#%%



#%%
if __name__ == '__main__':
    
    #Loading data and calculating measurements
    file_paths = list_file_paths(DATA_DIR)
    rr_list = load_data(file_paths)
    nn_list = filter_rr_meas(rr_list)
    hr_list = instant_hr_meas(nn_list)
    sdnn_list = calculate_overlapping_sd_meas(nn_list, window_ms=10000, overlap=0.8, min_fraction=0.3)
    rmssd_list = calculate_overlapping_rmssd_meas(nn_list, window_ms=10000, overlap=0.8, min_fraction=0.3)
    
#%%
    #Calculate correlations    

    #nn_corelations
    nn_results = process_meas_and_find_corr(nn_list)
    nn_results_df = records_to_dataframe(nn_results)
    best_nn_results = find_best_results(nn_results)
    best_nn_results_df = records_to_dataframe(best_nn_results)
    
    #hr_correlations
    hr_results = process_meas_and_find_corr(hr_list)
    hr_results_df = records_to_dataframe(hr_results)
    best_hr_results = find_best_results(hr_results)
    best_hr_results_df = records_to_dataframe(best_hr_results)
    
    #sdnn_correlations
    sdnn_results = process_meas_and_find_corr(sdnn_list)
    sdnn_results_df = records_to_dataframe(sdnn_results)
    best_sdnn_results = find_best_results(sdnn_results)
    best_sdnn_results_df = records_to_dataframe(best_sdnn_results)
    
    #rmssd_correlations
    rmssd_results = process_meas_and_find_corr(rmssd_list)
    rmssd_results_df = records_to_dataframe(rmssd_results)
    best_rmssd_results = find_best_results(rmssd_results)
    best_rmssd_results_df = records_to_dataframe(best_rmssd_results)

#%%
    #Ploting

# =============================================================================
#     #Ploting full measurements
#     meas_plot_from(rr_list, folder_name="rr_meas", title_label="RR-intervals", value_label="Time Between Heartbeats [ms]")
#     pair_plots_from(rr_list, folder_name="rr_pairs", title_label="paired RR-intervals", value_label="Time Between Heartbeats [ms]")
#     pair_plots_from(nn_list, folder_name="nn_pairs", title_label="paired NN-intervals", value_label="Time Between Heartbeats [ms]")
#     pair_plots_from(hr_list, folder_name="hr_pairs", title_label="paired Heart Rate", value_label="Heart Rate [bpm]")
#     pair_plots_from(sdnn_list, folder_name="sdnn_pairs", title_label="paired Standard Deviation of NN-intervals", value_label="SDNN [ms]")
#     pair_plots_from(rmssd_list, folder_name="rmssd_pairs", title_label="paired Root Mean Square of Successive Differences of NN-intervals", value_label="RMSSD [ms]")
# =============================================================================

#%%
# =============================================================================
#     #Ploting interpolated measurmenents
#     save_final_pairs_plots(best_nn_results, folder_name="nn_results", title_label="interpolated and paired NN-intervals", value_label="Time Between Heartbeats [ms]")
#     save_final_pairs_plots(best_hr_results, folder_name="hr_results", title_label="interpolated and paired Heart Rate", value_label="Heart Rate [bpm]")
#     save_final_pairs_plots(best_sdnn_results, folder_name="sdnn_results", title_label="interpolated and paired Standard Deviation of NN-intervals", value_label="SDNN [ms]")
#     save_final_pairs_plots(best_rmssd_results, folder_name="rmssd_results", title_label="interpolated and paired Successive Differences of NN-intervals", value_label="RMSSD [ms]")
# =============================================================================
    
#%%
    #Ploting corr heatmaps
    save_corr_heatmap_by_pair_and_shift(nn_results_df, folder_name="nn_results", title_label="Correlation heatmap of shifted NN-intervals")
    save_corr_heatmap_by_pair_and_shift(hr_results_df, folder_name="hr_results", title_label="Correlation heatmap of shifted Heart Rate")
    save_corr_heatmap_by_pair_and_shift(sdnn_results_df, folder_name="sdnn_results", title_label="Correlation heatmap of shifted Standard Deviation of NN-intervals")
    save_corr_heatmap_by_pair_and_shift(rmssd_results_df, folder_name="rmssd_results", title_label="Correlation heatmap of shifted Successive Differences of NN-intervals")

#%%
    #Ploting corr heatmaps
    save_corr_heatmap_by_task_and_shift(nn_results_df, folder_name="nn_results", title_label="Correlation heatmap of shifted NN-intervals")
    save_corr_heatmap_by_task_and_shift(hr_results_df, folder_name="hr_results", title_label="Correlation heatmap of shifted Heart Rate")
    save_corr_heatmap_by_task_and_shift(sdnn_results_df, folder_name="sdnn_results", title_label="Correlation heatmap of shifted Standard Deviation of NN-intervals")
    save_corr_heatmap_by_task_and_shift(rmssd_results_df, folder_name="rmssd_results", title_label="Correlation heatmap of shifted Successive Differences of NN-intervals")

#%%
# =============================================================================
#     #Saveing corr results
#     write_to_excel(best_nn_results_df, ANALYSIS_DATA_DIR / "nn_results.xlsx")  
#     write_to_excel(best_hr_results_df, ANALYSIS_DATA_DIR / "hr_results.xlsx")
#     write_to_excel(best_sdnn_results_df, ANALYSIS_DATA_DIR / "sdnn_results.xlsx")
#     write_to_excel(best_rmssd_results_df, ANALYSIS_DATA_DIR / "rmssd_results.xlsx")
# 
# =============================================================================
