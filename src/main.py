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
    
#%%
    #Ploting
    meas_plot_from(rr_list, folder_name="rr_meas")
    pair_plots_from(rr_list, folder_name="rr_pairs")
    pair_plots_from(nn_list, folder_name="nn_pairs")
    pair_plots_from(hr_list, folder_name="hr_pairs")

#%%
    #nn_corelations
    nn_results, nn_interp_pairs = process_meas_and_find_corr(nn_list)
    save_final_pairs_plots(nn_results, nn_interp_pairs, folder_name="nn_results")
    write_to_excel(nn_results, ANALYSIS_DATA_DIR / "nn_results.xlsx")
    
#%%
    #hr_correlations
    hr_results, hr_interp_pairs = process_meas_and_find_corr(hr_list)
    save_final_pairs_plots(hr_results, hr_interp_pairs, folder_name="hr_results")
    write_to_excel(nn_results, ANALYSIS_DATA_DIR / "hr_results.xlsx")