# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:45:31 2023

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
def find_serie(df, indx, start=0, stop=100000):
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

        selected_columns = [column for column in suma.index if (start < suma[column]) & (suma[column] < stop)]
        result.append((serie[selected_columns].values.tolist(), suma[selected_columns].values.tolist()))
        maximum.append(max)
        minimum.append(min)
    
    return result, minimum, maximum


#%%
def scatter_plot(tuple_list, info_list_of_lists, title=''):
    stop = max(sublist[-1] for sublist in info_list_of_lists)
    start = min(sublist[-2] for sublist in info_list_of_lists)
    
    fig = px.scatter()
    
    for i in range(len(tuple_list)):
        name = ' '.join(map(str, info_list_of_lists[i]))
        
        fig.add_scatter(
            x=tuple_list[i][0],
            y=tuple_list[i][1],
            mode='markers',
            name=name
        )
        
    fig.update_layout(
        xaxis_title="Time [ms]",
        yaxis_title="Time Between Heartbeats [ms]",
        title=f"{title} RANGE from {start} to {stop}"
    )
    display(fig)
    #pio.write_html(fig, f"{title}, RANGE from {start} to {stop}.html")

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
info_list_of_lists = info_df.values.tolist()

#%%
scatter_plot(serie, info_list_of_lists, title = 'TEST')

#%%

x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 17, 20]

# Utwórz wykres liniowy
fig = px.line(x=x, y=y, labels={'x': 'X-Axis', 'y': 'Y-Axis'}, title='Prosty Wykres Liniowy')

# Wyświetl wykres
fig.show()








