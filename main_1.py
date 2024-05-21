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


ctrl + 4/5
#%% Create datasets for inf, data and correlations
metadata_df = pd.DataFrame(columns=['TIME',
                                    'DATE' ,
                                    'NUMBER',
                                    'PAIR',
                                    'TYPE',
                                    'PARTICIPANT',
                                    ])
metadata_df.index.name='Id'

data_df = pd.DataFrame()
data_df.index.name='Id'

#%%
def extract_data_from_file(file_path, result_df=None):
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
def append_data_to_df(result_df=pd.DataFrame(), *args):
    while len(args) < len(result_df.columns):
        args += (np.nan,)
    
    new_row_data = pd.DataFrame([args], columns=result_df.columns)
    result_df = pd.concat([result_df, new_row_data], ignore_index=True)

    return result_df
#%%
def extract_info_from_path(file_path, result_df=None):
    
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

   return append_data_to_df(result_df, meas_time, meas_date, meas_num, meas_pair, meas_type, participant) 

#meas_time, meas_date, meas_num, meas_pair, meas_type, participant = extract_info_from_path("data\\2ork 2023-08-22 22-05-27.txt")

#%%
def find_indx(df, **kwargs):
    search_res = df.copy()
    if not kwargs:
        return search_res.index.tolist()

    for column, value in kwargs.items():
        search_res = search_res[search_res[column] == value]
    
    return search_res.index.tolist()
  
#%%
def create_serie(df_data, df_info, indx):
    data_list = []
    min_list = []
    max_list = []
    
    info_list = df_info.loc[indx].values.tolist()
    for i in indx:
        hr_values = df_data.loc[i].dropna().astype('int')
        research_time = hr_values.cumsum()
        research_time_max = research_time.iloc[-1]
        research_time_min = research_time.iloc[0]
                
        data_list.append((hr_values.values.tolist(), research_time.values.tolist()))
        min_list.append(research_time_min)
        max_list.append(research_time_max)
    
    new_values = list(zip(min_list, max_list))
    for info, (research_time_min, research_time_max) in zip(info_list, new_values):
        info.extend([research_time_min, research_time_max])
    
    return data_list, info_list

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
def trim(series_list, info_list, start=None, stop=None):
    trimmed_info_list = copy.deepcopy(info_list)
    
    max_start = max(sublist[-2] for sublist in trimmed_info_list)
    min_stop = min(sublist[-1] for sublist in trimmed_info_list)
    
    if start is None or start < max_start:
        start = max_start
    
    if stop is None or stop > min_stop:
        stop = min_stop
        
    trimmed_series_list = []
    
    for i in range(len(series_list)):
        trimmed_series = [[], []]
        
        trimmed_series[1] = [val for val in series_list[i][1] if start <= val <= stop]
        trimmed_series[0] = [series_list[i][0][series_list[i][1].index(val)] for val in trimmed_series[1]]
        
        trimmed_series_list.append(trimmed_series)
        trimmed_info_list[i][-1] = trimmed_series[1][-1]
        trimmed_info_list[i][-2] = trimmed_series[1][0] 
        
    return trimmed_series_list, trimmed_info_list
#%%
def interpolate(x, y, ix, method='linear'):
    if not isinstance(x, (list, np.ndarray)):
        x = [x]
    if not isinstance(y, (list, np.ndarray)):
        y = [y]
    
    f = interp1d(x, y, kind=method, fill_value='extrapolate')
    return f(ix).tolist()

#%%
def shift_series(series_list, info_list, shift_time):
    shifted_info_list = copy.deepcopy(info_list)
    
    shifted_series_list = []
    
    for i in range(len(series_list)):
        shifted_series = [[], []]
        shifted_series[1] = [x + shift_time for x in series_list[i][1]]
        shifted_series[0] = series_list[i][0]
        
        shifted_series_list.append(shifted_series)
        shifted_info_list[i][-1] = shifted_series[1][-1]
        shifted_info_list[i][-2] = shifted_series[1][0]
    return shifted_series_list, shifted_info_list
#%%
def calculate_correlation(tuple_list, info_list):
    correlation_matrix = np.zeros((len(tuple_list), len(tuple_list)))
    p_value_matrix = np.zeros((len(tuple_list), len(tuple_list)))
    
    for i, (x1, y1) in enumerate(tuple_list):
        for j, (x2, y2) in enumerate(tuple_list):
            common_x = list(set(x1) & set(x2))
            y1_interp = interp1d(x1, y1, kind='linear')(common_x)
            y2_interp = interp1d(x2, y2, kind='linear')(common_x)
            
            correlation_matrix[i, j], p_value_matrix[i, j] = pearsonr(y1_interp, y2_interp)
    
    return correlation_matrix, p_value_matrix

#%%
def create_correlation_dataframes(correlation_matrix, p_value_matrix, info_list):
    info_strings = ["_".join(map(str, info)) for info in info_list]
    
    correlation_df = pd.DataFrame(correlation_matrix, columns=info_strings, index=info_strings)
    p_value_df = pd.DataFrame(p_value_matrix, columns=info_strings, index=info_strings)
    return correlation_df, p_value_df
#%%
def matrix_heatmap(df, title='', color='viridis'):
    mask = np.zeros_like(df.corr().round(2), dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        cmap = sns.color_palette(color)
        sns.heatmap(df.corr(),
                    annot=df.corr().round(2),
                    mask=mask,
                    vmax=.3,
                    square=True,
                    xticklabels=df.columns,
                    yticklabels=df.columns,
                    cmap=cmap,
                    linewidths=.5,
                    cbar_kws={"shrink": 0.7})
    plt.title(title)
    #output_path = 'out/heatmap.png'
    #plt.savefig(output_path)
#%%
def fisher_transform(r):
    return 0.5 * np.log((1 + r) / (1 - r))

#%%
def find_correlation_pairs(correlation_df, min_corr, max_corr):
    corr_pairs = correlation_df[(abs(correlation_df) > min_corr) & (abs(correlation_df) < max_corr)]

    corr_pairs_df = pd.DataFrame(columns=['column1', 'column2', 'correlation'])

    added_pairs = []

    for index, column in corr_pairs.stack().items():
        if not pd.isna(column):
            pair = (index[0], index[1])
            reverse_pair = (index[1], index[0])

            if pair not in added_pairs and reverse_pair not in added_pairs:
                temp_df = pd.DataFrame({'column1': [index[0]], 'column2': [index[1]], 'correlation': [column]})
                added_pairs.append(pair)
                corr_pairs_df = pd.concat([corr_pairs_df, temp_df], ignore_index=True)

    return corr_pairs_df
#%% -??????????????????
def find_bests_correlations(correlation_df, p_value_df):
    tmp_df = copy.deepcopy(correlation_df)
    column_names = correlation_df.columns
    meas_names = [column_name[:27] for column_name in column_names]
    unique_meas_names = list(OrderedDict.fromkeys(meas_names))
    result_df = pd.DataFrame(columns=['MEASURE_1', 'MEASURE_2', 'corr', 'p_value'])
    
    for meas_name in unique_meas_names:
        pattern = re.compile(rf'{re.escape(meas_name)}\w+')
        filtered_columns = list(filter(pattern.search, correlation_df.columns))
        index_names = list(set(column_names) - set(filtered_columns))
        for indx in index_names:
            max_column_name = tmp_df[indx].idxmax()
            corr = correlation_df.at[indx, max_column_name]
            p_value = p_value_df.at[indx, max_column_name]
            result_df = pd.concat([result_df, pd.DataFrame({'MEASURE_1': [meas_name], 'MEASURE_2': [indx], 'corr': [corr], 'p_value': [p_value]})], ignore_index=True)
           
    return result_df
#%%








#%%
file_paths = glob.glob('data/**')

for i in range(len(file_paths)):
    data_df = extract_data_from_file(file_paths[i], data_df)
    metadata_df = extract_info_from_path(file_paths[i], metadata_df)


#%% Find all relaxation measures 1 and 2
#indx = find_indx(metadata_df, NUMBER = '2', PAIR = 'o', TYPE = 'r')
indx = find_indx(metadata_df, NUMBER = '1',TYPE = 'r')
r_1_series_list, r_1_series_info_list = create_serie(data_df, metadata_df, indx)

indx = find_indx(metadata_df, NUMBER = '2',TYPE = 'r')
r_2_series_list, r_2_series_info_list = create_serie(data_df, metadata_df, indx)

#%%
r_1_shift1000_series_list, r_1_shift1000_series_info_list = shift_series(r_1_series_list, r_1_series_info_list,1000)


#%%
trimmed_series_list, trimmed_info_list = trim(r_1_series_list + r_1_shift1000_series_list, r_1_series_info_list + r_1_shift1000_series_info_list)
scatter_plot(trimmed_series_list, trimmed_info_list, title = 'Trimeed')

#%%
correlation_matrix, p_value_matrix = calculate_correlation(trimmed_series_list, trimmed_info_list)
correlation_df, p_value_df = create_correlation_dataframes(correlation_matrix, p_value_matrix, r_1_series_info_list + r_1_shift1000_series_info_list)
#%%
corr_pairs = find_correlation_pairs(correlation_df, 0.0, 1)

#%%
nazwy_pomiarów = find_bests_correlations(correlation_df, p_value_df)
#%%
matrix_heatmap(correlation_df, "corr")
matrix_heatmap(p_value_df, "p_value")

#%%
#scatter_plot(r_1_series_list, r_1_series_info_list, title = 'TEST')
#%%
def get_ifo_from_path(file_path):
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
# Obliczenie średniej ostatnich elementów w kolumnie 'x'
#first_relaxation_mean_stop = first_relaxation_df['x'].apply(lambda arr: arr[-1]).mean()

#mean_stop_threshold = 0.9 * first_relaxation_mean_stop
#filtered_df = first_relaxation_df[first_relaxation_df['x'].apply(lambda arr: arr[-1] < mean_stop_threshold)]

#%%
wb1 = data_df[data_df.index.str.contains('1w.._1', regex=True)]
wb2 = data_df[data_df.index.str.contains('2w.._1', regex=True)]

#%%
w1 = data_df[data_df.index.str.contains('1w.._(?!1)', regex=True)]
w2 = data_df[data_df.index.str.contains('2w.._(?!1)', regex=True)]

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

#%%
def shift_series(df, shift_time_ms):
    df_copy = df.copy()
    for i in range(df_copy.shape[0]):
        df_copy.iloc[i]['x']+=shift_time_ms
        df_copy.iloc[i]['shift'] = int(pd.to_timedelta(shift_time_ms, unit='ms').total_seconds())
        rename_index(df_copy, i, df_copy.iloc[i]['meas_name'] + "_" + str(df_copy.iloc[i]['shift']))
    return df_copy

#%%
def shift_series(df, shift_time_ms):
    df_copy = df.copy()
    result_df = pd.DataFrame(columns=df.columns)
    new_index = []  # Lista przechowująca nowe indeksy
    
    for i in range(df_copy.shape[0]):
        # Przypisanie wartości do DataFrame za pomocą metody `at`
        result_df.at[i, 'x'] = df_copy.at[i, 'x'] + shift_time_ms
        result_df.at[i, 'y'] = df_copy.at[i, 'y']
        result_df.at[i, 'shift'] = int(pd.to_timedelta(shift_time_ms, unit='ms').total_seconds())
        result_df.at[i, 'starttime'] = df_copy.at[i, 'starttime']
        result_df.at[i, 'endtime'] = df_copy.at[i, 'endtime']
        result_df.at[i, 'duration'] = df_copy.at[i, 'duration']
        
        # Dodanie nowego indeksu do listy
        new_index.append(df_copy.at[i, 'meas_name'] + "_" + str(result_df.at[i, 'shift']))
    
    # Przypisanie nowej listy indeksów do DataFrame
    result_df.index = new_index
    
    return result_df

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
def calculate_correlations(df):
    df = trim(df)
    n_rows = df.shape[0]
    correlation_matrix = np.empty((n_rows, n_rows))
    p_value_matrix = np.empty((n_rows, n_rows))
    
    # Pobranie nazw kolumn i indeksów
    index_names = df.index.values
    column_names = df.index.values
    
    for i, (_, data1) in enumerate(df.iterrows()):
        for j, (_, data2) in enumerate(df.iterrows()):
            data = pd.DataFrame([data1, data2])
            data.reset_index(drop=True, inplace=True)   
            data = trim(data)
            
            x1 = data.iloc[0]['x']
            y1 = data.iloc[0]['y']
            x2 = data.iloc[1]['x']
            y2 = data.iloc[1]['y']
                        
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
def merge_meas(df):
    df = df.sort_values(by='starttime')
    result_df = pd.DataFrame(columns=df.columns)
    
    unique_meas_names = df['meas_name'].unique()
    
    for name in unique_meas_names:
        subset_df = df[df['meas_name'] == name].reset_index(drop=True)
        
        
        if subset_df.shape[0] > 1:
            diffs = [0] + [(subset_df.iloc[i+1]['starttime'] - subset_df.iloc[0]['starttime']).total_seconds() * 1000 
                           for i in range(subset_df.shape[0] - 1)]
            
            subset_df['x'] = subset_df.apply(lambda row: row['x'] + int(diffs[row.name]), axis=1)
            
            print(subset_df['x'])
            
            starttime = min(subset_df['starttime'])
            endtime = max(subset_df['endtime'])
            duration = (endtime - starttime).total_seconds() / 60
            
            tmp_row_data = pd.DataFrame({'x': [np.concatenate(subset_df['x'].values)],
                                         'y': [np.concatenate(subset_df['y'].values)],
                                         'shift': [subset_df.iloc[0]['shift']],
                                         'meas_name': [name],
                                         'starttime': [starttime],
                                         'endtime': [endtime],
                                         'duration': [duration]
                                         })
            
            print(tmp_row_data.iloc[0]['x'][-1])
            
            tmp_row_data.index = pd.Index([tmp_row_data.iloc[0]['meas_name'] + "_" + str(tmp_row_data.iloc[0]['shift'])])
            
            
        result_df = pd.concat([result_df, tmp_row_data])
    
    return result_df

#%%
tmp_selected_meas = data_df[data_df.index.str.contains('2r.\d+_(?!0)', regex=True)]

#%%
def save_plot(fig, file_name, folder_name="out"):
    # Sprawdzenie istnienia katalogu i jego utworzenie, jeśli nie istnieje
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # Utworzenie pełnej ścieżki do pliku wyjściowego
    output_file_path = os.path.join(folder_name, f"{file_name}.html")
    
    # Zapisanie wykresu do pliku HTML
    pio.write_html(fig, output_file_path)