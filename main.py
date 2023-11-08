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
file_paths = glob.glob('data/**')
#%%
for i in range(len(file_paths)):
    data_df = append_data_from_file(file_paths[i], data_df)
    meas_time, meas_date, meas_num, meas_pair, meas_type, participant = extract_info_from_path(file_paths[i])
    metadata_df = append_metadata(metadata_df, meas_time, meas_date, meas_num, meas_pair, meas_type, participant)


#%%














