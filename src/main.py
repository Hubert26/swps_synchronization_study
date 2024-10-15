# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:45:31 2023

@author: Hubert Szewczyk
"""

import pandas as pd


from config import *
import importlib
from functions import *
from utils.file_utils import list_file_paths, create_directory, delete_directory
from utils.dataframe_utils import write_to_excel
#%%



#%%
if __name__ == '__main__':
    
    file_paths = list_file_paths(DATA_DIR)
    rr_list = load_data(file_paths)
    nn_list = filter_rr_meas(rr_list)
    hr_list = instant_hr_meas(nn_list)
    sd_list = calculate_overlapping_sd_meas(nn_list, window_ms=10000, overlap=0.5, min_fraction=0.3)
    rmssd_list = calculate_overlapping_rmssd_meas(nn_list, window_ms=10000, overlap=0.5, min_fraction=0.3)
    
#%%
    #Ploting
    meas_plot_from(rr_list, folder_name="rr_meas", title_label="RR-intervals", value_label="Time Between Heartbeats [ms]")
    pair_plots_from(rr_list, folder_name="rr_pairs", title_label="paired RR-intervals", value_label="Time Between Heartbeats [ms]")
    pair_plots_from(nn_list, folder_name="nn_pairs", title_label="paired NN-intervals", value_label="Time Between Heartbeats [ms]")
    pair_plots_from(hr_list, folder_name="hr_pairs", title_label="paired Heart Rate", value_label="Heart Rate [bpm]")
    pair_plots_from(sd_list, folder_name="sd_pairs", title_label="paired Standard Deviation of NN-intervals", value_label="STDNN [ms]")
    pair_plots_from(rmssd_list, folder_name="rmssd_pairs", title_label="paired Root Mean Square of Successive Differences of NN-intervals", value_label="RMSSD [ms]")

#%%
    #nn_corelations
    nn_results, nn_interp_pairs = process_meas_and_find_corr(nn_list)
    save_final_pairs_plots(nn_results, nn_interp_pairs, folder_name="nn_results", title_label="interpolated and paired NN-intervals", value_label="Time Between Heartbeats [ms]")
    write_to_excel(nn_results, ANALYSIS_DATA_DIR / "nn_results.xlsx")
    
#%%
    #hr_correlations
    hr_results, hr_interp_pairs = process_meas_and_find_corr(hr_list)
    save_final_pairs_plots(hr_results, hr_interp_pairs, folder_name="hr_results", title_label="interpolated and paired Heart Rate", value_label="Heart Rate [bpm]")
    write_to_excel(hr_results, ANALYSIS_DATA_DIR / "hr_results.xlsx")
#%%
    #sd_correlations
    sd_results, sd_interp_pairs = process_meas_and_find_corr(sd_list)
    save_final_pairs_plots(sd_results, sd_interp_pairs, folder_name="sd_results", title_label="interpolated and paired Standard Deviation of NN-intervals", value_label="STDNN [ms]")
    write_to_excel(sd_results, ANALYSIS_DATA_DIR / "sd_results.xlsx")
    
#%%
    #rmssd_correlations
    rmssd_results, rmssd_interp_pairs = process_meas_and_find_corr(rmssd_list)
    save_final_pairs_plots(rmssd_results, rmssd_interp_pairs, folder_name="rmssd_results", title_label="interpolated and paired Successive Differences of NN-intervals", value_label="RMSSD [ms]")
    write_to_excel(rmssd_results, ANALYSIS_DATA_DIR / "rmssd_results.xlsx")