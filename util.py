# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import re
pio.renderers.default='browser'

import os
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from IPython.display import display
#%%
def get_info_from_path(file_path, part = 0):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        index = file_name.split()

        if len(index) >= part:
            return index[part]
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
        shift = 0
        meas_name = get_info_from_path(file_path).split('_')[0]
        
        start_timestamp = get_info_from_path(file_path, 1) + get_info_from_path(file_path, 2)
        df_timestamp = pd.to_datetime(start_timestamp, format='%Y-%m-%d%H-%M-%S', errors='raise')
        
        index_name = f"{get_info_from_path(file_path)}_{str(shift)}"

        new_row_data = pd.DataFrame({'x': [x_data],
                                     'y': [y_data],
                                     'shift': [shift],
                                    'meas_name': [meas_name]},
                                    index=[index_name])

        new_row_data['starttime'] = df_timestamp
        new_row_data['endtime'] = df_timestamp + + pd.to_timedelta(x_data[-1], unit='ms')
        new_row_data['duration'] = (new_row_data['endtime'] - new_row_data['starttime']).dt.total_seconds() / 60

        df = pd.concat([df, new_row_data])
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return df

#%%
def scatter_plot(df, title=None):
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
    )
    
    if not title:
        title = f"{names} RANGE from {start} to {stop}"
    
    fig.update_layout(title=title)
    display(fig)
    
# =============================================================================
#     output_file_path = os.path.join("out", f"{title}.html")
#     pio.write_html(fig, output_file_path)
#     pio.write_html(fig, output_file_path)
# =============================================================================

#%%
def rename_index(df, idx, new_idx_name):      
    new_index = list(df.index)
    new_index[idx] = new_idx_name
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
def shift_series(df, shift_time_ms):
    df_copy = df.copy()
    df_copy.reset_index(drop=True, inplace=True)
    
    result_df = pd.DataFrame(columns=df.columns)
    new_index = []  # Lista przechowująca nowe indeksy
    
    for i in range(df_copy.shape[0]):
        # Przypisanie wartości do DataFrame za pomocą metody `at`
        result_df.at[i, 'x'] = df_copy.at[i, 'x'] + shift_time_ms
        result_df.at[i, 'y'] = df_copy.at[i, 'y']
        result_df.at[i, 'shift'] = int(pd.to_timedelta(shift_time_ms, unit='ms').total_seconds())
        result_df.at[i, 'meas_name'] = df_copy.at[i, 'meas_name']
        result_df.at[i, 'starttime'] = df_copy.at[i, 'starttime']
        result_df.at[i, 'endtime'] = df_copy.at[i, 'endtime']
        result_df.at[i, 'duration'] = df_copy.at[i, 'duration']
        
        # Dodanie nowego indeksu do listy
        new_index.append(df_copy.at[i, 'meas_name'] + "_" + str(result_df.at[i, 'shift']))
    
    # Przypisanie nowej listy indeksów do DataFrame
    result_df.index = new_index
    
    return result_df

#%%
def fisher_transform(r):
    if r == 1:
        return np.inf
    elif r == -1: 
        return -np.inf
    return round(0.5 * np.log((1 + r) / (1 - r)),4)

#%%
def calculate_correlations(df1, df2):
    n_rows = df1.shape[0]
    n_columns = df2.shape[0]
    correlation_matrix = np.empty((n_rows, n_columns))
    p_value_matrix = np.empty((n_rows, n_columns))
    
    # Pobranie nazw kolumn i indeksów
    index_names = df1.index.values
    column_names = df2.index.values
    
    for i, (_, data1) in enumerate(df1.iterrows()):
        for j, (_, data2) in enumerate(df2.iterrows()):
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
    
    tmp_row_data = pd.DataFrame(columns=df.columns)  # Utwórz pustą ramkę danych poza pętlą
    
    for name, group in df.groupby('meas_name'):
        if group.shape[0] > 1:
            group['diff_ms'] = group['starttime'].diff().fillna(pd.Timedelta(seconds=0)).apply(lambda x: x.total_seconds() * 1000)
            group['x'] = group['x'] + group['diff_ms'].cumsum().astype(int)
            
            starttime = group['starttime'].min()
            endtime = group['endtime'].max()
            duration = (endtime - starttime).total_seconds() / 60
            
            tmp_row_data = pd.DataFrame({'x': [np.concatenate(group['x'].values)],
                                         'y': [np.concatenate(group['y'].values)],
                                         'shift': [group.iloc[0]['shift']],
                                         'meas_name': [name],
                                         'starttime': [starttime],
                                         'endtime': [endtime],
                                         'duration': [duration]
                                         }, index=[f"{name}_{group.iloc[0]['shift']}"])
        else:
            tmp_row_data = group  # Ustaw grupę jako wartość tmp_row_data, jeśli warunek nie jest spełniony
            tmp_row_data.index = pd.Index([f"{name}_{group.iloc[0]['shift']}"], name='new_index_name')

            
        result_df = pd.concat([result_df, tmp_row_data])
    
    return result_df




#%%
def find_best_corr_pairs(correlation_df, p_value_df, meas_1, meas_2):
    abs_correlation_df = correlation_df.abs()
    max_value = abs_correlation_df.max().max()
    max_locations = (abs_correlation_df == max_value).stack().reset_index()
    max_locations = max_locations[max_locations[0]].iloc[:, :-1]

    # Wybierz odpowiednie pary z max_locations
    pairs = max_locations[(max_locations['level_0'].isin(meas_1.index)) & (max_locations['level_1'].isin(meas_2.index))]
    
    # Wybierz odpowiednie pomiary z meas_1 i meas_2
    selected_meas_1 = meas_1.loc[pairs['level_0']]
    selected_meas_2 = meas_2.loc[pairs['level_1']]
    
    result_df = pd.DataFrame({
        'indeks_1': selected_meas_1.index,
        'meas_name_1': selected_meas_1['meas_name'].values,
        'shift_1': selected_meas_1['shift'].values,
        'indeks_2': selected_meas_2.index,
        'meas_name_2': selected_meas_2['meas_name'].values,
        'shift_2': selected_meas_2['shift'].values,
        'shift_diff': selected_meas_1['shift'].values - selected_meas_2['shift'].values
    })
    
    # Znajdź indeksy kolumn i wierszy dla wybranych par
    row_indices = pairs['level_0'].values
    column_indices = pairs['level_1'].values
    

    correlation_values = []
    p_value_values = []
    
    # Pętla po wszystkich parach indeksów
    for row_index, column_index in zip(row_indices, column_indices):
        corr = correlation_df.at[row_index, column_index]
        p_val = p_value_df.at[row_index, column_index]
        
        correlation_values.append(corr)
        p_value_values.append(p_val)
    
    result_df['corr'] = correlation_values
    result_df['p_val'] = p_value_values

    # Znajdź indeks (wiersz) gdzie kolumna shift_diff ma wartość najbliższą 0
    min_diff = result_df['shift_diff'].abs().min()
    # Wybierz wszystkie wiersze, gdzie wartość shift_diff jest równa najmniejszej różnicy
    result_df = result_df[result_df['shift_diff'].abs() == min_diff]
    
    
    # Znajdź najmniejszą wartość spośród kolumn 'shift_1' i 'shift_2'
    min_shift_value = result_df[['shift_1', 'shift_2']].values.min()
    # Wybierz ten wiersz, w którym 'shift_1' lub 'shift_2' ma najmniejszą wartość
    result_df = result_df[(result_df['shift_1'] == min_shift_value) | (result_df['shift_2'] == min_shift_value)]

    return result_df

#%%
def extract_numeric_suffix(text):
    match = re.search(r'\d+$', text)
    if match:
        return int(match.group())
    else:
        return None

#%%
def find_pairs(df, index=0):
    matching_pairs = set()  # Używamy zbioru do przechowywania unikalnych par
    for index_1, row_1 in df.iterrows():
        for index_2, row_2 in df.iterrows():
            numeric_suffix_1 = extract_numeric_suffix(row_1['meas_name'])
            numeric_suffix_2 = extract_numeric_suffix(row_2['meas_name'])
            if (row_1['meas_name'][:2] == row_2['meas_name'][:2] and
                    row_1['meas_name'][2] == 'm' and row_2['meas_name'][2] == 'k' and
                    numeric_suffix_1 is not None and numeric_suffix_2 is not None and
                    numeric_suffix_1 == numeric_suffix_2):
                if index:
                    matching_pairs.add((index_1, index_2))  # Dodajemy parę indeksów do zbioru
                else:
                    matching_pairs.add((row_1['meas_name'], row_2['meas_name']))  # Dodajemy parę nazw do zbioru
    
    # Konwertujemy zestawy na krotki
    matching_pairs = [tuple(pair) for pair in matching_pairs]
    
    all_meas_names = set(df['meas_name'])  # Zbierz wszystkie unikalne nazwy pomiarów z df['meas_name']

    # Sprawdź, które pomiary nie są w `matching_pairs`
    unmatched_meas = all_meas_names - set(pair[0] for pair in matching_pairs) - set(pair[1] for pair in matching_pairs)
    
    return matching_pairs, list(unmatched_meas)
    




    










