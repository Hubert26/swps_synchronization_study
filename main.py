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
    # display(fig)
    
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
    
    max_start = max(df['x'].apply(lambda arr: arr[0]))
    min_stop = min(df['x'].apply(lambda arr: arr[-1]))
    
    if start is None or start < max_start:
        start = max_start
    
    if stop is None or stop > min_stop:
        stop = min_stop
        
    
    for i in range(df.shape[0]):
        x=df.iloc[i]['x']
        y=df.iloc[i]['y']
        selected_indices_x = np.where((x >= start) & (x <= stop))[0]
        df.iloc[i]['x'] = x[selected_indices_x]
        df.iloc[i]['y'] = y[selected_indices_x]
        rename_index(df, i, 1, str(df.iloc[i]['x'][0]) + '_' + str(df.iloc[i]['x'][-1]))
        
    return df

#%%
data_df = pd.DataFrame()
#%%
file_paths = glob.glob('data/pomiary relaksacji/**') + glob.glob('data/pomiary współpracy/**')

for i in range(len(file_paths)):
    data_df = extract_data_from_file(file_paths[i], data_df)

#%%
filtered_df = data_df[data_df.index.str.contains('1rk1 |1rm1 ', regex=True)]

#%%
timmed_df = trim(filtered_df, 2000, 5000)
#%%
# =============================================================================
# for i in range(data_df.shape[0]):
#     scatter_plot(data_df.iloc[i:i+1])
# =============================================================================

#%%





