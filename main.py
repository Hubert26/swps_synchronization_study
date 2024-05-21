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
#%%
file_paths = glob.glob('data/relaxation/**') + glob.glob('data/cooperation/**')

for i in range(len(file_paths)):
    data_df = extract_data_from_file(file_paths[i], data_df)

#%%



#%%
#1r
#%%
selected_df = data_df[data_df.index.str.contains('1r', regex=True)]
merged_df = merge_meas(selected_df)
matching_pairs, unmatched_meas = find_pairs(merged_df)

best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])

for pair in matching_pairs:
    meas_1 = merged_df.loc[merged_df.meas_name == pair[0]].iloc[[0]]
    meas_2 = merged_df.loc[merged_df.meas_name == pair[1]].iloc[[0]]

    diff_start_time_ms = calculate_time_difference(meas_1, meas_2)

    if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
        meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
    elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
        meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)

    meas_df = pd.concat([meas_1, meas_2])
    
# =============================================================================
#     fig_scatter, title_scatter = scatter_plot(meas_df)
#     fig_scatter.show()
#     save_plot(fig_scatter, title_scatter, folder_name="out/pairs_merged_1_shift_0_filter_0/1r", format="html")
# =============================================================================
    
    shifted_df = meas_df.copy()
    for i in range(1000, 10001, 1000):
        shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
    
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
    
    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
    
    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)
    diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)

# =============================================================================
#     fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
#     fig_heatmap.show()
#     save_plot(fig_heatmap, title_heatmap, folder_name="out/corr_heatmap/1r", format="png")
# =============================================================================

#%%
r1_best_corr = best_corr_df.copy()
r1_selected_df = selected_df.copy()
r1_shifted_df = shifted_df.copy()
r1_merged_df = merged_df.copy()
r1_unmatched_meas = unmatched_meas.copy()
r1_diff_start_time_ms_df = diff_start_time_ms_df.copy()

#%%
r1_m_tmp = merged_df[merged_df.meas_name.str.contains('1r.14', regex=True)]
r1_tmp = selected_df[selected_df.meas_name.str.contains('1r.14', regex=True)]

#%%





#%%
#2r 
#%%
selected_df = data_df[data_df.index.str.contains('2r', regex=True)]
merged_df = merge_meas(selected_df)
matching_pairs, unmatched_meas = find_pairs(merged_df)

best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])

for pair in matching_pairs:
    meas_1 = merged_df.loc[merged_df.meas_name == pair[0]].iloc[[0]]
    meas_2 = merged_df.loc[merged_df.meas_name == pair[1]].iloc[[0]]

    diff_start_time_ms = calculate_time_difference(meas_1, meas_2)

    if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
        meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
    elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
        meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)

    meas_df = pd.concat([meas_1, meas_2])
    
# =============================================================================
#     fig_scatter, title_scatter = scatter_plot(meas_df)
#     fig_scatter.show()
#     save_plot(fig_scatter, title_scatter, folder_name="out/pairs_merged_1_shift_0_filter_0/2r", format="html")
# =============================================================================
    
    shifted_df = meas_df.copy()
    for i in range(1000, 10001, 1000):
        shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
    
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
    
    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
    
    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)
    diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)

# =============================================================================
#     fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
#     fig_heatmap.show()
#     save_plot(fig_heatmap, title_heatmap, folder_name="out/corr_heatmap/2r", format="png")
# =============================================================================

#%%
r2_best_corr = best_corr_df.copy()
r2_selected_df = selected_df.copy()
r2_merged_df = merged_df.copy()
r2_unmatched_meas = unmatched_meas
r2_diff_start_time_ms_df = diff_start_time_ms_df.copy()

#%%
r2_tmp = r2_selected_df[r2_selected_df.meas_name.str.contains('2r.7', regex=True)]
r2_m_tmp = r2_merged_df[r2_merged_df.meas_name.str.contains('2r.7', regex=True)]

#%%





#%%
#1w RELAKCACJA
#%%
selected_df = data_df[data_df.index.str.contains('1w.\d+_1', regex=True)]
merged_df = merge_meas(selected_df)
matching_pairs, unmatched_meas = find_pairs(merged_df)

best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])

for pair in matching_pairs:
    meas_1 = merged_df.loc[merged_df.meas_name == pair[0]].iloc[[0]]
    meas_2 = merged_df.loc[merged_df.meas_name == pair[1]].iloc[[0]]

    diff_start_time_ms = calculate_time_difference(meas_1, meas_2)

    if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
        meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
    elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
        meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)

    meas_df = pd.concat([meas_1, meas_2])
    
# =============================================================================
#     fig_scatter, title_scatter = scatter_plot(meas_df)
#     fig_scatter.show()
#     save_plot(fig_scatter, title_scatter, folder_name="out/pairs_merged_1_shift_0_filter_0/1wb", format="html")
# =============================================================================
    
    shifted_df = meas_df.copy()
    for i in range(1000, 10001, 1000):
        shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
    
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
    
    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
    
    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)
    diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)

# =============================================================================
#     fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
#     fig_heatmap.show()
#     save_plot(fig_heatmap, title_heatmap, folder_name="out/corr_heatmap/1wb", format="png")
# =============================================================================

#%%
w1b_best_corr = best_corr_df.copy()
w1b_selected_df = selected_df.copy()
w1b_merged_df = merged_df.copy()
w1b_unmatched_meas = unmatched_meas
w1b_diff_start_time_ms_df = diff_start_time_ms_df.copy()

#%%
w1b_tmp = w1b_selected_df[w1b_selected_df.meas_name.str.contains('1w.8', regex=True)]
w1b_m_tmp = w1b_merged_df[w1b_merged_df.meas_name.str.contains('1w.8', regex=True)]

#%%




#%%
#1w WSPÓŁPRACA
#%%
selected_df = data_df[data_df.index.str.contains('1w.\d+_(?!1)', regex=True)]
merged_df = merge_meas(selected_df)
matching_pairs, unmatched_meas = find_pairs(merged_df)

best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])

for pair in matching_pairs:
    meas_1 = merged_df.loc[merged_df.meas_name == pair[0]].iloc[[0]]
    meas_2 = merged_df.loc[merged_df.meas_name == pair[1]].iloc[[0]]

    diff_start_time_ms = calculate_time_difference(meas_1, meas_2)

    if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
        meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
    elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
        meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)

    meas_df = pd.concat([meas_1, meas_2])
    
# =============================================================================
#     fig_scatter, title_scatter = scatter_plot(meas_df)
#     fig_scatter.show()
#     save_plot(fig_scatter, title_scatter, folder_name="out/pairs_merged_1_shift_0_filter_0/1w", format="html")
# =============================================================================
    
    shifted_df = meas_df.copy()
    for i in range(1000, 10001, 1000):
        shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
    
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
    
    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
    
    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)
    diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)

# =============================================================================
#     fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
#     fig_heatmap.show()
#     save_plot(fig_heatmap, title_heatmap, folder_name="out/corr_heatmap/1w", format="png")
# =============================================================================

#%%
w1_best_corr = best_corr_df.copy()
w1_selected_df = selected_df.copy()
w1_merged_df = merged_df.copy()
w1_unmatched_meas = unmatched_meas
w1_diff_start_time_ms_df = diff_start_time_ms_df.copy()

#%%
w1_tmp = w1_selected_df[w1_selected_df.meas_name.str.contains('1w.14', regex=True)]
w1_m_tmp = w1_merged_df[w1_merged_df.meas_name.str.contains('1w.14', regex=True)]

#%%




#%%
#2w RELAKSACJA
#%%
selected_df = data_df[data_df.index.str.contains('2w.\d+_1', regex=True)]
merged_df = merge_meas(selected_df)
matching_pairs, unmatched_meas = find_pairs(merged_df)

best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])

for pair in matching_pairs:
    meas_1 = merged_df.loc[merged_df.meas_name == pair[0]].iloc[[0]]
    meas_2 = merged_df.loc[merged_df.meas_name == pair[1]].iloc[[0]]

    diff_start_time_ms = calculate_time_difference(meas_1, meas_2)

    if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
        meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
    elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
        meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)

    meas_df = pd.concat([meas_1, meas_2])
    
# =============================================================================
#     fig_scatter, title_scatter = scatter_plot(meas_df)
#     fig_scatter.show()
#     save_plot(fig_scatter, title_scatter, folder_name="out/pairs_merged_1_shift_0_filter_0/2wb", format="html")
# =============================================================================
    
    shifted_df = meas_df.copy()
    for i in range(1000, 10001, 1000):
        shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
    
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
    
    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
    
    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)
    diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)

# =============================================================================
#     fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
#     fig_heatmap.show()
#     save_plot(fig_heatmap, title_heatmap, folder_name="out/corr_heatmap/2wb", format="png")
# =============================================================================

#%%
w2b_best_corr = best_corr_df.copy()
w2b_selected_df = selected_df.copy()
w2b_merged_df = merged_df.copy()
w2b_unmatched_meas = unmatched_meas
w2b_diff_start_time_ms_df = diff_start_time_ms_df.copy()

#%%
w2b_tmp = w2b_selected_df[w2b_selected_df.meas_name.str.contains('2w.16', regex=True)]
w2b_m_tmp = w2b_merged_df[w2b_merged_df.meas_name.str.contains('2w.16', regex=True)]


#%%
#scatter_plot(w2b_m_tmp)

#%%





#%%
#2w WSPÓŁPRACA
#%%
selected_df = data_df[data_df.index.str.contains('2w.\d+_(?!1)', regex=True)]
merged_df = merge_meas(selected_df)
matching_pairs, unmatched_meas = find_pairs(merged_df)

best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])

for pair in matching_pairs:
    meas_1 = merged_df.loc[merged_df.meas_name == pair[0]].iloc[[0]]
    meas_2 = merged_df.loc[merged_df.meas_name == pair[1]].iloc[[0]]

    diff_start_time_ms = calculate_time_difference(meas_1, meas_2)

    if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
        meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
    elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
        meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)

    meas_df = pd.concat([meas_1, meas_2])
    

# =============================================================================
#     fig_scatter, title_scatter = scatter_plot(meas_df)
#     fig_scatter.show()
#     save_plot(fig_scatter, title_scatter, folder_name="out/pairs_merged_1_shift_0_filter_0/2w", format="html")
# =============================================================================
    
    shifted_df = meas_df.copy()
    for i in range(1000, 10001, 1000):
        shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
    
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
    
    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
    
    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)
    diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)

# =============================================================================
#     fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
#     fig_heatmap.show()
#     save_plot(fig_heatmap, title_heatmap, folder_name="out/corr_heatmap/2w", format="png")
# =============================================================================
    
#%%
w2_best_corr = best_corr_df.copy()
w2_selected_df = selected_df.copy()
w2_merged_df = merged_df.copy()
w2_unmatched_meas = unmatched_meas
w2_diff_start_time_ms_df = diff_start_time_ms_df.copy()

#%%
w2_tmp = w2_selected_df[w2_selected_df.meas_name.str.contains('2w.16', regex=True)]
w2_m_tmp = w2_merged_df[w2_merged_df.meas_name.str.contains('2w.16', regex=True)]

