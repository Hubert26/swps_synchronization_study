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
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
import seaborn as sns
from IPython.display import display
import glob
import os
from scipy.interpolate import interp1d
import scipy.stats
from itertools import product

#%% Create datasets for inf, data and correlations
metadata_df = pd.DataFrame(columns=['TIME',
                                    'DATE' ,
                                    'NUMBER',
                                    'PAIR',
                                    'TYPE',
                                    'PARTICIPANT'])
metadata_df.index.name='Id'

data_df = pd.DataFrame()
data_df.index.name='Id'

correlation_coefficients = pd.DataFrame(columns=['correlation_iyA_yL',
                                                 'p_values__iyA_yL',
                                                 'correlation_iyL_yA',
                                                 'p_values__iyL_yA'])
correlation_coefficients.index.name='Id'
#%%
def append_data_from_file(file_path, result_df=None):
    if result_df is None:
        result_df = pd.DataFrame()

    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return result_df

    try:
        with open(file_path, "r") as file:
            tmp_data = file.read()
            tmp_data = tmp_data.split('\n')
            new_row_data = pd.DataFrame(tmp_data).T
            result_df = pd.concat([result_df, new_row_data], 
                                  ignore_index=True)
        return result_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return result_df
#%%
def extract_info_from_path(file_path):
    
   if not os.path.exists(file_path):
       print(f"File '{file_path}' not found.")
       return None, None, None, None, None, None
   
    # Delate file extention
   file_name = os.path.splitext(os.path.basename(file_path))[0]
   parts = file_name.split()
   
   if len(parts) < 3:
       return None, None, None, None, None, None

   meas_time = parts[2]  # 22-05-27
   meas_date = parts[1]  # 2023-08-22
   meas_num = parts[0][0]  # '2'
   meas_pair = parts[0][1]  # 'o'
   meas_type = parts[0][2] # 'r'
   participant = parts[0][3] # 'k'

   return meas_time, meas_date, meas_num, meas_pair, meas_type, participant 

#meas_time, meas_date, meas_num, meas_pair, meas_type, participant = extract_info_from_path("data\\2ork 2023-08-22 22-05-27.txt")
#%%
def append_metadata(result_df=pd.DataFrame(), *args):
    while len(args) < len(result_df.columns):
        args += (np.nan,)
    
    new_row_data = pd.DataFrame([args], columns=result_df.columns)
    result_df = pd.concat([result_df, new_row_data], ignore_index=True)

    return result_df
#%%
def find_indx(df, **kwargs):
    search_res = df.copy()

    for column, value in kwargs.items():
        search_res = search_res[search_res[column] == value]
    
    return search_res.index.tolist()
  
#%%

    
#%%
def find_serie(df, indx, start=0, stop=10000000):
    result = []
    maximum = []
    minimum = []
    
    for i in indx:
        serie = df.loc[i].dropna().astype('int')
        suma = serie.cumsum()
        max = suma.iloc[-1]
        min = suma.iloc[0]
        if stop > max:
            stop = max

        selected_columns = [column for column in suma.index if (start <= suma[column]) & (suma[column] <= stop)]
        result.append((serie[selected_columns].values.tolist(), suma[selected_columns].values.tolist()))
        maximum.append(max)
        minimum.append(min)
    
    return result, minimum, maximum


#%%
def scatter_plot(tuple_list, info_list, title=''):
    stop = max(sublist[-1] for sublist in info_list)
    start = min(sublist[-2] for sublist in info_list)
    
    fig = px.scatter()
    
    for i in range(len(tuple_list)):
        name = ' '.join(map(str, info_list[i]))
        
        fig.add_scatter(
            x=tuple_list[i][1],
            y=tuple_list[i][0],
            mode='markers',
            name=name
        )
        
    fig.update_layout(
        xaxis_title="Time [ms]",
        yaxis_title="Time Between Heartbeats [ms]",
        title=f"{title} RANGE from {start} to {stop}"
    )
    #display(fig)
    
#   output_file_path = os.path.join("out", f"{title}_RANGE_from_{start}_to_{stop}.html")
#   pio.write_html(fig, output_file_path)
#   pio.write_html(fig, output_file_path)

#%%
def trim(tuple_list, info_list):
    trimmed_info_list = copy.deepcopy(info_list)
    stop = min(sublist[-1] for sublist in trimmed_info_list)
    start = max(sublist[-2] for sublist in trimmed_info_list)    
    trimmed = []
    
    for i in range(len(tuple_list)):
        trimmed_series = [[], []]
        
        trimmed_series[1] = [val for val in tuple_list[i][1] if start <= val <= stop]
        trimmed_series[0] = [tuple_list[i][0][tuple_list[i][1].index(val)] for val in trimmed_series[1]]
        
        trimmed.append(trimmed_series)
        trimmed_info_list[i][-1] = trimmed_series[1][-1]
        trimmed_info_list[i][-2] = trimmed_series[1][0] 
        
    return trimmed, trimmed_info_list
#%%
def interpolate(x, y, ix, method='linear'):
    f = interp1d(x, y, kind=method, fill_value='extrapolate')
    return f(ix).tolist()
#%%
def calculate_correlation(tuple_list, info_list):
    correlation_matrix = np.zeros((len(tuple_list), len(tuple_list)))
    
    for i, (x1, y1) in enumerate(tuple_list):
        for j, (x2, y2) in enumerate(tuple_list):
            iy1 = interpolate(x1, y1, x2)
            iy2 = interpolate(x2, y2, x1)
            
            sorted_unique_pairs1 = list(set(sorted(zip(x1 + iy1, y1 + iy1))))
            #second_elements1 = [pair[1] for pair in sorted_unique_pairs1]
            
            sorted_unique_pairs2 = list(set(sorted(zip(x2 + iy2, y2 + iy2))))
            #second_elements2 = [pair[1] for pair in sorted_unique_pairs2]
            
            correlation_matrix[i, j] = calculate_correlation(sorted_unique_pairs1, sorted_unique_pairs2)
    
    return correlation_matrix
#%%
file_paths = glob.glob('data/**')

for i in range(len(file_paths)):
    data_df = append_data_from_file(file_paths[i], data_df)
    meas_time, meas_date, meas_num, meas_pair, meas_type, participant = extract_info_from_path(file_paths[i])
    metadata_df = append_metadata(metadata_df, meas_time, meas_date, meas_num, meas_pair, meas_type, participant)


#%%
indx = find_indx(metadata_df, NUMBER = meas_num, PAIR = meas_pair, TYPE = meas_type)

#%%
serie, minimum, maximum = find_serie(data_df, indx)

#%%
info_df = metadata_df.loc[indx]
info_df['Min'] = minimum
info_df['Max'] = maximum
info_list = info_df.values.tolist()

#%%
scatter_plot(serie, info_list, title = 'TEST')

#%%
trimmed, trimmed_info_list = trim(serie, info_list)
scatter_plot(trimmed, trimmed_info_list, title = 'Trimeed')
#%%
len(trimmed)
#%%
corr = calculate_correlation(trimmed, trimmed_info_list)










