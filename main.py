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
metadata_df = pd.DataFrame(columns=['POMIAR',
                                    'TYP' ,
                                    'PARA',
                                    'BADANY',
                                    'DATA',
                                    'CZAS'])
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
def append_metadata_from_path(file_path, result_df=None):
    if result_df is None:
        result_df = pd.DataFrame()
        
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return result_df
        
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    parts = file_name.split()
    
    while len(parts) < len(result_df.columns):
        parts.append(np.nan)
    
    if len(parts) >= 3:
        new_row_data = pd.DataFrame([parts], columns=result_df.columns)  # Utwórz nowy DataFrame z danymi
        result_df = pd.concat([result_df, new_row_data], ignore_index=True)  # Dodaj nowy wiersz do istniejącego DataFrame

    return result_df





#%%
file_paths = glob.glob('data/**')
#%%
for i in range(len(file_paths)):
    data_df = append_data_from_file(file_paths[i], data_df)
    metadata_df = append_metadata_from_path(file_paths[i], metadata_df)


#%%
append_metadata_from_path(file_paths[1], metadata_df)













