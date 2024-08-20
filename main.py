# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:45:31 2023

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'
import seaborn as sns
import glob
import os
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from collections import OrderedDict
import re
from IPython.display import display

from util import *

#%%
data_df = pd.DataFrame()
result_df = pd.DataFrame()


#%%
file_paths = glob.glob('data/cooperation/**')

for i in range(len(file_paths)):
    data_df = extract_data_from_file(file_paths[i], data_df)
    
#%%
# =============================================================================
# #Ploting oryginal signals
# merged_df = merge_meas(selected_df)
# matching_pairs, unmatched_series = find_pairs(merged_df)
# for pair in matching_pairs:
#     # Extract the corresponding rows from series_df for the given pair
#     pair_df = merged_df.loc[merged_df['meas_name'].isin(pair)].iloc[[0, 1]]
#         
#     # Generate and save a histogram plot for the trimmed pair
#     output_plot_folder = os.path.join(output_folder, 'oryginal', "hist_pairs_oryginal")
#     fig_hist, title_hist = density_plot(pair_df, title=f"{pair}_Distribution of RR-intervals")
#     save_plot(fig_hist, title_hist, folder_name=output_plot_folder, format="html")
#     
#     # Generate and save a scatter plot for the trimmed pair
#     output_plot_folder = os.path.join(output_folder, 'oryginal', "scatter_pairs_oryginal")
#     fig_scatter, title_scatter = scatter_plot(pair_df)
#     save_plot(fig_scatter, title_scatter, folder_name=output_plot_folder, format="html")
# =============================================================================

#%%
#1w BASELINE
#%%
meas_type = '1w'
selected_df = data_df[data_df.index.str.contains(meas_type + '.\d+_1', regex=True)]
merged_df = merge_meas(selected_df)   
for (start_time, end_time), meas_state in baseline_1_time_intervals.items():
    best_corr = process_rr_data(merged_df, group_label = meas_state, start_time_ms = start_time, end_time_ms = end_time)

    result_df = pd.concat([result_df, best_corr])
#%%




#%%
#1w COOPERATION
#%%
meas_type = '1w'
selected_df = filtered_df[filtered_df.index.str.contains(meas_type + '.\d+_(?!1)', regex=True)]
merged_df = merge_meas(selected_df)
for (start_time, end_time), meas_state in cooperation_1_time_intervals.items():
    best_corr = process_rr_pair_data(merged_df, folder_name = "merged_1_filter_1", group_label = meas_state, start_time = start_time, end_time = end_time)

    result_df = pd.concat([result_df, best_corr])
#%%




#%%
#2w BASELINE
#%%
meas_type = '2w'
selected_df = filtered_df[filtered_df.index.str.contains(meas_type + '.\d+_1', regex=True)]
merged_df = merge_meas(selected_df)
for (start_time, end_time), meas_state in baseline_2_time_intervals.items():
    best_corr = process_rr_pair_data(merged_df, folder_name = "merged_1_filter_1", group_label = meas_state, start_time = start_time, end_time = end_time)

    result_df = pd.concat([result_df, best_corr])
#%%




#%%
#2w COOPERATION
#%%
meas_type = '2w'
selected_df = filtered_df[filtered_df.index.str.contains(meas_type + '.\d+_(?!1)', regex=True)]
merged_df = merge_meas(selected_df)
for (start_time, end_time), meas_state in cooperation_2_time_intervals.items():
    best_corr = process_rr_pair_data(merged_df, folder_name = "merged_1_filter_1", group_label = meas_state, start_time = start_time, end_time = end_time)

    result_df = pd.concat([result_df, best_corr])


#%%
file_path = f'{output_folder}/dataset_cooperation.xlsx'
result_df.to_excel(file_path, index=False)






#%%
#TESTOWANIE
# =============================================================================
# meas_type = '1w'
# selected_df = data_df[data_df.index.str.contains(meas_type + '.9_1', regex=True)]
# 
# merged_df = merge_meas(selected_df)
# trim_merged_df = trim(merged_df, 0, 20000)
# shifted_df = shift_series(merged_df, 5000)
# trimed_df = trim(shifted_df, 0, 20000)
# filtered_df = filter_series(trimed_df)
# 
# meas_1_df = pd.concat([trim_merged_df.iloc[[0]], trimed_df.iloc[[0]], filtered_df.iloc[[0]]], ignore_index=False)
# 
# fig_scatter, title_scatter = scatter_plot(meas_1_df)
# save_plot(fig_scatter, title_scatter, folder_name=os.path.join(output_folder, f"scatter_pairs"), format="html")
# 
# 
# =============================================================================
