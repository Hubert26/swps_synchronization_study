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
selected_df = data_df[data_df.index.str.contains('2r', regex=True)]
merged_df = merge_meas(selected_df)
shifted_df = merged_df.copy()
for i in range(1000, 10001, 1000):
    shifted_df = pd.concat([shifted_df, shift_series(merged_df, i)])

matching_pairs = find_pairs(shifted_df)

#%%
best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])

for pair in matching_pairs:
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]

    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)

    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)

best_corr_df.reset_index(drop=True, inplace=True)

#%%
r2_best_corr = best_corr_df.copy()
r2_selected_df = selected_df.copy()
r2_merged_df = merged_df.copy()

#%%
r2_tmp = r2_selected_df[r2_selected_df.meas_name.str.contains('2r.7', regex=True)]
r2_m_tmp = r2_merged_df[r2_merged_df.meas_name.str.contains('2r.7', regex=True)]

#%%
#scatter_plot(tmp_merged_meas)

#%%



#%%
selected_df = data_df[data_df.index.str.contains('1r', regex=True)]
merged_df = merge_meas(selected_df)
shifted_df = merged_df.copy()
for i in range(1000, 10001, 1000):
    shifted_df = pd.concat([shifted_df, shift_series(merged_df, i)])

matching_pairs = find_pairs(shifted_df)

#%%
best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])

for pair in matching_pairs:
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]

    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)

    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)

best_corr_df.reset_index(drop=True, inplace=True)

#%%
r1_best_corr = best_corr_df.copy()
r1_selected_df = selected_df.copy()
r1_merged_df = merged_df.copy()

#%%
r1_tmp = selected_df[selected_df.meas_name.str.contains('1r.8', regex=True)]
r1_m_tmp = merged_df[merged_df.meas_name.str.contains('1r.8', regex=True)]


#%%
#scatter_plot(tmp_merged_meas)

#%%



#%%
selected_df = data_df[data_df.index.str.contains('1w.\d+_1', regex=True)]
merged_df = merge_meas(selected_df)
shifted_df = merged_df.copy()
for i in range(1000, 10001, 1000):
    shifted_df = pd.concat([shifted_df, shift_series(merged_df, i)])

matching_pairs = find_pairs(shifted_df)

#%%
best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])

for pair in matching_pairs:
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]

    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)

    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)

best_corr_df.reset_index(drop=True, inplace=True)

#%%
w1b_best_corr = best_corr_df.copy()
w1b_selected_df = selected_df.copy()
w1b_merged_df = merged_df.copy()

#%%
w1b_tmp = selected_df[selected_df.meas_name.str.contains('1w.8', regex=True)]
w1b_m_tmp = merged_df[merged_df.meas_name.str.contains('1w.8', regex=True)]


#%%
#scatter_plot(tmp_merged_meas)

#%%



#%%
selected_df = data_df[data_df.index.str.contains('1w.\d+_(?!0)', regex=True)]
merged_df = merge_meas(selected_df)
shifted_df = merged_df.copy()
for i in range(1000, 10001, 1000):
    shifted_df = pd.concat([shifted_df, shift_series(merged_df, i)])

matching_pairs = find_pairs(shifted_df)

#%%
best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])

for pair in matching_pairs:
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]

    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)

    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)

best_corr_df.reset_index(drop=True, inplace=True)

#%%
w1_best_corr = best_corr_df.copy()
w1_selected_df = selected_df.copy()
w1_merged_df = merged_df.copy()

#%%
w1_tmp = selected_df[selected_df.meas_name.str.contains('1w.8', regex=True)]
w1_m_tmp = merged_df[merged_df.meas_name.str.contains('1w.8', regex=True)]


#%%
#scatter_plot(tmp_merged_meas)

#%%


#%%
selected_df = data_df[data_df.index.str.contains('2w.\d+_1', regex=True)]
merged_df = merge_meas(selected_df)
shifted_df = merged_df.copy()
for i in range(1000, 10001, 1000):
    shifted_df = pd.concat([shifted_df, shift_series(merged_df, i)])

matching_pairs = find_pairs(shifted_df)

#%%
best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])

for pair in matching_pairs:
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]

    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)

    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)

best_corr_df.reset_index(drop=True, inplace=True)

#%%
w2b_best_corr = best_corr_df.copy()
w2b_selected_df = selected_df.copy()
w2b_merged_df = merged_df.copy()

#%%
w2b_tmp = selected_df[selected_df.meas_name.str.contains('1w.8', regex=True)]
w2b_m_tmp = merged_df[merged_df.meas_name.str.contains('1w.8', regex=True)]


#%%
#scatter_plot(tmp_merged_meas)

#%%



#%%
selected_df = data_df[data_df.index.str.contains('2w.\d+_(?!0)', regex=True)]
merged_df = merge_meas(selected_df)
shifted_df = merged_df.copy()
for i in range(1000, 10001, 1000):
    shifted_df = pd.concat([shifted_df, shift_series(merged_df, i)])

matching_pairs = find_pairs(shifted_df)

#%%
best_corr_df = pd.DataFrame(columns=['indeks_1', 'meas_name_1', 'shift_1', 'indeks_2', 'meas_name_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])

for pair in matching_pairs:
    meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
    meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]

    correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
    best_corr = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)

    best_corr_df = pd.concat([best_corr_df, best_corr], ignore_index=True)

best_corr_df.reset_index(drop=True, inplace=True)

#%%
w2_best_corr = best_corr_df.copy()
w2_selected_df = selected_df.copy()
w2_merged_df = merged_df.copy()

#%%
w2_tmp = selected_df[selected_df.meas_name.str.contains('1w.8', regex=True)]
w2_m_tmp = merged_df[merged_df.meas_name.str.contains('1w.8', regex=True)]


#%%
#scatter_plot(tmp_merged_meas)

#%%