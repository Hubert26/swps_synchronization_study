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


#%%
def extract_index_from_path(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        index = file_name.split()

        if len(index):
            return index[0]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
#%%
def extract_data_from_file(file_path, df=None):
    if df is None:
        df = pd.DataFrame()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        with open(file_path, "r") as file:
            y_data = file.read().split('\n')

        if not all(map(str.isdigit, y_data)):
            raise ValueError("File contains non-numeric data.")

        y_data = np.array(y_data).astype(int)
        x_data = np.cumsum(y_data)

        index_name = f"{extract_index_from_path(file_path)} {x_data[0]}_{x_data[-1]}"

        new_row_data = pd.DataFrame({'x': [x_data],
                                     'y': [y_data]},
                                    index=[index_name])

        df = pd.concat([df, new_row_data])
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return df

#%%
def scatter_plot(df, title=''):
    stop = max(df['x'].apply(lambda arr: arr[-1]))
    start = min(df['x'].apply(lambda arr: arr[0]))
    
    names = df.index.tolist()
    fig = px.scatter()
    
    for i in range(df.shape[0]):
        name = names[i].split()[0]
        
        fig.add_scatter(
            x=df.iloc[i]['x'],
            y=df.iloc[i]['y'],
            mode='markers',
            name=name
        )
        
    fig.update_layout(
        xaxis_title="Time [ms]",
        yaxis_title="Time Between Heartbeats [ms]",
        title=f"{name} RANGE from {start} to {stop}"
    )
    display(fig)
    
# =============================================================================
#     output_file_path = os.path.join("out", f"{name}_RANGE_from_{start}_to_{stop}.html")
#     pio.write_html(fig, output_file_path)
#     pio.write_html(fig, output_file_path)
# =============================================================================

#%%
def rename_index(df, idx, part, new_name):
    index_parts = df.index[idx].split()
    index_parts[part] = new_name
        
    new_index = list(df.index)
    new_index[idx] = ' '.join(index_parts)
    df.index = new_index
    
#%%
def trim(df, start=None, stop=None):
    df_copy = df.copy()
    max_start = max(df_copy['x'].apply(lambda arr: arr[0]))
    min_stop = min(df_copy['x'].apply(lambda arr: arr[-1]))
    
    if start is None or start < max_start:
        start = max_start
    
    if stop is None or stop > min_stop:
        stop = min_stop
        
    
    for i in range(df_copy.shape[0]):
        x=df_copy.iloc[i]['x']
        y=df_copy.iloc[i]['y']
        selected_indices_x = np.where((x >= start) & (x <= stop))[0]
        df_copy.iloc[i]['x'] = x[selected_indices_x]
        df_copy.iloc[i]['y'] = y[selected_indices_x]
        rename_index(df_copy, i, 1, str(df_copy.iloc[i]['x'][0]) + '_' + str(df_copy.iloc[i]['x'][-1]))
        
    return df_copy

#%%
def interpolate(x, y, ix, method='linear'):
    if not isinstance(x, (list, np.ndarray)):
        x = [x]
    if not isinstance(y, (list, np.ndarray)):
        y = [y]
    
    f = interp1d(x, y, kind=method, fill_value='extrapolate')
    return f(ix).tolist()
#%%
def shift_series(df, shift_time):
    df_copy = df.copy()
    for i in range(df_copy.shape[0]):
        df_copy.iloc[i]['x']+=shift_time
        rename_index(df_copy, i, 1, str(df_copy.iloc[i]['x'][0]) + '_' + str(df_copy.iloc[i]['x'][-1]))
    return df_copy

#%%
def fisher_transform(r):
    if r == 1:
        return np.inf
    elif r == -1: 
        return -np.inf
    return round(0.5 * np.log((1 + r) / (1 - r)),4)

#%%
def calculate_correlations(df):
    df = trim(df)
    n_rows = df.shape[0]
    correlation_matrix = np.empty((n_rows, n_rows))
    p_value_matrix = np.empty((n_rows, n_rows))
    
    # Pobranie nazw kolumn i indeksów
    index_names = df.index.values
    column_names = df.index.values
    
    for i, row1 in enumerate(df.iterrows()):
        _, data1 = row1
        x1, y1 = data1['x'], data1['y']
        
        for j, row2 in enumerate(df.iterrows()):
            _, data2 = row2
            x2, y2 = data2['x'], data2['y']
            
            common_x = np.union1d(x1, x2)
            y1_interp = interp1d(x1, y1, kind='linear', fill_value='extrapolate')(common_x)
            y2_interp = interp1d(x2, y2, kind='linear', fill_value='extrapolate')(common_x)
            
            correlation, p_value = pearsonr(y1_interp, y2_interp)
            correlation_matrix[i, j] = round(correlation,4)
            p_value_matrix[i, j] = round(p_value,4)
            
    correlation_df = pd.DataFrame(correlation_matrix, index=index_names, columns=column_names)
    p_value_df = pd.DataFrame(p_value_matrix, index=index_names, columns=column_names)
    
    return correlation_df, p_value_df
#%%
data_df = pd.DataFrame()
#%%
file_paths = glob.glob('data/pomiary relaksacji/**') + glob.glob('data/pomiary współpracy/**')

for i in range(len(file_paths)):
    data_df = extract_data_from_file(file_paths[i], data_df)

#%%
# =============================================================================
# filtered_df = data_df[data_df.index.str.contains('1rk1 |1rm1 ', regex=True)]
# =============================================================================

#%%
# =============================================================================
# timmed_df = trim(filtered_df, 2000, 5000)
# =============================================================================
#%%
# =============================================================================
# shifted_df = shift_series(filtered_df, 1000)
# =============================================================================
#%%
# =============================================================================
# correlation_matrix, p_value_matrix = calculate_correlations(filtered_df)
# =============================================================================

#%%
# =============================================================================
# fisher_correlation_matrix = correlation_matrix.applymap(fisher_transform)
# 
# =============================================================================
#%%

#%%
first_relaksation_df = data_df[data_df.index.str.contains('2w', regex=True)]
# Obliczenie średniej ostatnich elementów w kolumnie 'x'
first_relaksation_mean_stop = first_relaksation_df['x'].apply(lambda arr: arr[-1]).mean()

mean_stop_threshold = 0.9 * first_relaksation_mean_stop
filtered_df = first_relaksation_df[first_relaksation_df['x'].apply(lambda arr: arr[-1] < mean_stop_threshold)]
#%%
# =============================================================================
# for i in range(data_df.shape[0]):
#     scatter_plot(data_df.iloc[i:i+1])
# =============================================================================

#%%
# =============================================================================
# fisher_transform_vec = np.vectorize(fisher_transform)
# fisher_df = pd.DataFrame(
#     {'x': [np.arange(-1.0, 1.0, 0.01)],
#      'y': [fisher_transform_vec(np.arange(-1.0, 1.0, 0.01))]
#      },
#      index = ['fisher_tranform']
#      )
# scatter_plot(fisher_df)
# =============================================================================


