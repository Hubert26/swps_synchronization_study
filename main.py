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
# =============================================================================
# #TESTOWANIE
# meas_type = '2w'
# selected_df = filtered_df[filtered_df.index.str.contains(meas_type + '.5_(?!1)', regex=True)]
# 
# merged_df = merge_meas(selected_df)
# fig_scatter, title_scatter = scatter_plot(merged_df)
# save_plot(fig_scatter, title_scatter, folder_name=os.path.join(output_folder, f"scatter_pairs_merged_df"), format="html")
# 
# shifted_df = shift_series(merged_df, 2000)
# fig_scatter, title_scatter = scatter_plot(shifted_df)
# save_plot(fig_scatter, title_scatter, folder_name=os.path.join(output_folder, f"scatter_pairs_shifted_df"), format="html")
# 
# trimed_df = trim(merged_df, 10000, 20000)
# fig_scatter, title_scatter = scatter_plot(trimed_df)
# save_plot(fig_scatter, title_scatter, folder_name=os.path.join(output_folder, f"scatter_pairs_trimmed_df"), format="html")
# 
# =============================================================================
