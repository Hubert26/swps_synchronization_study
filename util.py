# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:05:29 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpld3
import re
pio.renderers.default='browser'

import os
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from IPython.display import display

#%%
cooperation_1_time_intervals = {
    (0, 8000): "1_z1_instr",
    (8000, 28000): "1_z1_1_k",
    (28000, 48000): "1_z1_2_m",
    (48000, 68000): "1_z1_3_k",
    (68000, 88000): "1_z1_4_m",
    (88000, 108000): "1_z1_5_k",
    (108000, 128000): "1_z1_6_m",
    (128000, 136000): "1_z1_odp_idle",
    (226000, 226000): "1_z1_odp",
    (246000, 286000): "1_pause",
    (286000, 294000): "1_z2_instr",
    (294000, 314000): "1_z2_1_m",
    (314000, 334000): "1_z2_2_k",
    (334000, 354000): "1_z2_3_m",
    (354000, 374000): "1_z2_4_k",
    (374000, 394000): "1_z2_5_m",
    (394000, 414000): "1_z2_6_k",
    (414000, 422000): "1_z2_odp_idle",
    (422000, 512000): "1_z2_odp",
    (512000, 547000): "1_baseline2_idle",
    (547000, 787000): "1_baseline2",
}

cooperation_2_time_intervals = {
    (0, 8000): "2_z1_instr",
    (8000, 28000): "2_z1_1_k",
    (28000, 48000): "2_z1_2_m",
    (48000, 68000): "2_z1_3_k",
    (68000, 88000): "2_z1_4_m",
    (88000, 108000): "2_z1_5_k",
    (108000, 128000): "2_z1_6_m",
    (128000, 136000): "2_z1_odp_idle",
    (226000, 226000): "2_z1_odp",
    (246000, 286000): "2_pause",
    (286000, 294000): "2_z2_instr",
    (294000, 314000): "2_z2_1_m",
    (314000, 334000): "2_z2_2_k",
    (334000, 354000): "2_z2_3_m",
    (354000, 374000): "2_z2_4_k",
    (374000, 394000): "2_z2_5_m",
    (394000, 414000): "2_z2_6_k",
    (414000, 422000): "2_z2_odp_idle",
    (422000, 512000): "2_z2_odp",
    (512000, 547000): "2_baseline2_idle",
    (547000, 787000): "2_baseline2",
}

baseline_1_time_intervals = {
    (0, 20000): "1_baseline1_idle",
    (20000, 260000): "1_baseline1",
}

baseline_2_time_intervals = {
    (0, 20000): "2_baseline1_idle",
    (20000, 260000): "2_baseline1",
}
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
    
    return fig, title

#%%
def density_plot(df, title=None):
    names = df.index.tolist()
    fig = go.Figure()
    
    for i, y_data in enumerate(df['y']):
        trace = go.Histogram(x=y_data, histnorm='probability density', name=names[i])
        fig.add_trace(trace)
        
        # Dodanie linii pionowych dla wartości odstających
        outlier_low = round(np.mean(y_data) - 2 * np.std(y_data))
        outlier_high = round(np.mean(y_data) + 2 * np.std(y_data))
        trace_color = "gray"
        fig.add_vline(x=outlier_low, line_dash="dash", line_color=trace_color, annotation_text=f"Outlier_Low_{names[i]} = {outlier_low}", annotation_position="top left", annotation_textangle=90)
        fig.add_vline(x=outlier_high, line_dash="dash", line_color=trace_color, annotation_text=f"Outlier_High_{names[i]} = {outlier_high}", annotation_position="top right",  annotation_textangle=90)
    
    fig.update_layout(
        xaxis_title="RR-interval [ms]",
        yaxis_title="Density",
    )
    
    if title is None:
        title = "Distribution of RR-intervals"
    
    fig.update_layout(title=title)
    
    return fig, title

#%%
def corr_heatmap(df, title=None, color='viridis'):
    # Tworzenie własnej mapy kolorów z 20 odcieniami od -1 do 1
    colors = sns.color_palette(color, 20)
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=20)
    
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df,
# =============================================================================
# to annotate on heatmap you need previous version of matplotlib              
# pip install matplotlib==3.7.3
# =============================================================================
                    annot=df.round(2),
                    vmax=1,
                    vmin=-1,
                    center=0,
                    square=True,
                    xticklabels=df.columns,
                    yticklabels=df.index,
                    cmap=cmap,
                    linewidths=.5,
                    cbar_kws={"shrink": 0.7, 'ticks': np.linspace(-1, 1, 21)})
        # Ustawienie rotacji etykiet
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if not title:
        title = 'heatmap'
    
    plt.title(title)

    return f, title
#%%
def save_plot(fig, file_name, folder_name="out", format="png"):
    # Sprawdzenie istnienia katalogu i jego utworzenie, jeśli nie istnieje
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # Utworzenie pełnej ścieżki do pliku wyjściowego
    output_file_path = os.path.join(folder_name, f"{file_name}.{format}")
    
    if isinstance(fig, go.Figure):
        if format == "html":
            # Zapisanie wykresu Plotly do pliku HTML
            pio.write_html(fig, output_file_path)
        else:
            # Zapisanie wykresu Plotly do pliku graficznego (PNG, JPG, SVG, PDF)
            fig.write_image(output_file_path, format=format)
    elif isinstance(fig, plt.Figure):
        if format == "html":
            # Zapisanie wykresu Matplotlib do pliku HTML za pomocą mpld3
            mpld3.save_html(fig, output_file_path)
        else:
            # Zapisanie wykresu Matplotlib do pliku graficznego (PNG, JPG, SVG, PDF)
            fig.savefig(output_file_path, format=format)
    
#%%
def rename_index(df, idx, new_idx_name):      
    new_index = list(df.index)
    new_index[idx] = new_idx_name
    df.index = new_index
    
#%%
def trim(df, start=None, end=None):
    df_copy = df.copy()
    max_start = max(df_copy['x'].apply(lambda arr: arr[0]))
    min_stop = min(df_copy['x'].apply(lambda arr: arr[-1]))
    
    # Ustawienie domyślnych wartości dla start_time i end_time
    start_time = max_start
    end_time = min_stop

    if start is None or start < max_start:
        start_time = max_start
    
    if end is None or end > min_stop:
        end_time = min_stop
        
    
    for i in range(df_copy.shape[0]):
        x=df_copy.iloc[i]['x']
        y=df_copy.iloc[i]['y']
        selected_indices_x = np.where((x >= start_time) & (x <= end_time))[0]
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
def shift_series(df, shift_time_ms, index=0):
    df_copy = df.copy()
    df_copy.reset_index(drop=True, inplace=True)
    
    result_df = pd.DataFrame(columns=df.columns)
    new_index = []  # Lista przechowująca nowe indeksy
    
    for i in range(df_copy.shape[0]):
        # Przypisanie wartości do DataFrame za pomocą metody `at`
        result_df.at[i, 'x'] = df_copy.at[i, 'x'] + shift_time_ms
        result_df.at[i, 'y'] = df_copy.at[i, 'y']
        if index:
            result_df.at[i, 'shift'] = int(pd.to_timedelta(shift_time_ms, unit='ms').total_seconds())
        else:
            result_df.at[i, 'shift'] = df_copy.at[i, 'shift']
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
            correlation_matrix[i, j] = fisher_transform(round(correlation,4))
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
        #'meas_name_1': selected_meas_1['meas_name'].values,
        'shift_1': selected_meas_1['shift'].values,
        'indeks_2': selected_meas_2.index,
        #'meas_name_2': selected_meas_2['meas_name'].values,
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
    
#%%
def calculate_time_difference(row1, row2):
    starttime1 = pd.to_datetime(row1['starttime'].iloc[0])
    starttime2 = pd.to_datetime(row2['starttime'].iloc[0])
    
    if isinstance(starttime1, pd.Timestamp) and isinstance(starttime2, pd.Timestamp):
        diff_start_time_ms = (starttime1 - starttime2).total_seconds() * 1000
    else:
        diff_start_time_ms = np.nan

    pair_df = pd.DataFrame({
        'meas_name_1': [row1['meas_name'].iloc[0]],
        'meas_name_2': [row2['meas_name'].iloc[0]],
        'diff_start_time_ms': [diff_start_time_ms]
    })
    
    return pair_df

#%%
def remove_outliers(rr):
    outlier_low = round(np.nanmean(rr) - 2 * np.nanstd(rr))
    outlier_high = round(np.nanmean(rr) + 2 * np.nanstd(rr))
        
    # Zmiana wartości odstających na np.nan
    mask = (rr < outlier_low) | (rr > outlier_high)
    rr[mask] = np.nan
        
    mean_prev_next = np.array([
        np.nan if i == 0 or i == len(rr) - 1 else np.nanmean([rr[i-1], rr[i+1]])
        for i in range(len(rr))
    ])
    
    for i in range(1, len(rr) - 1):
        if np.isnan(mean_prev_next[i]):
            continue
        if rr[i] > 1.2 * mean_prev_next[i] or rr[i] < 0.8 * mean_prev_next[i]:
            rr[i] = np.nan
            
    return rr

#%%
def interpolate_nan_values(rr: np.ndarray) -> np.ndarray:
    # Znalezienie indeksów nan
    nans = np.isnan(rr)
    if np.any(nans):
        # Indeksy, które nie są nan
        not_nans = ~nans
        x = np.arange(len(rr))
        # Interpolacja wartości nan
        rr[nans] = np.interp(x[nans], x[not_nans], rr[not_nans])
        
    return rr

#%%
def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    result_df = pd.DataFrame(columns=df.columns)  
    for i in range(len(df)):
        rr = df.iloc[i]['y']
        rr = np.array(rr, dtype=float)  # Konwertowanie rr na ndarray typu float
        
        rr = remove_outliers(rr)
        rr = interpolate_nan_values(rr)
        
        x = np.cumsum(rr)
        starttime = df.iloc[i]['starttime']
        endtime = starttime + pd.to_timedelta(x[-1], unit='ms')
        duration = (endtime - starttime).total_seconds() / 60
    
        tmp_row_data = pd.DataFrame({'x': [x],
                                     'y': [rr],
                                     'shift': [df.iloc[i]['shift']],
                                     'meas_name': [df.iloc[i]['meas_name']],
                                     'starttime': [starttime],
                                     'endtime': [endtime],
                                     'duration': [duration]
                                     }, index=[df.index[i]])
    
        result_df = pd.concat([result_df, tmp_row_data])
    
    return result_df


#%%
def process_rr_pair_data(df, folder_name, group_label, start_time, end_time):
    matching_pairs, unmatched_meas = find_pairs(df)
    
    result_df = pd.DataFrame(columns=['group_label', 'indeks_1', 'shift_1', 'indeks_2', 'shift_2', 'shift_diff', 'corr', 'p_val'])
    diff_start_time_ms_df = pd.DataFrame(columns=['meas_name_1', 'meas_name_2','diff_start_time_ms'])
    
    for pair in matching_pairs:
        meas_1 = df.loc[df.meas_name == pair[0]].iloc[[0]]
        meas_2 = df.loc[df.meas_name == pair[1]].iloc[[0]]
    
        diff_start_time_ms = calculate_time_difference(meas_1, meas_2)    
        if(diff_start_time_ms['diff_start_time_ms'].iloc[0] > 1000):
            meas_2 = shift_series(meas_2, shift_time_ms = diff_start_time_ms['diff_start_time_ms'].iloc[0].item() - 1000, index=0)
        elif(diff_start_time_ms['diff_start_time_ms'].iloc[0] <- 1000): 
            meas_1 = shift_series(meas_1, shift_time_ms=abs(diff_start_time_ms['diff_start_time_ms'].iloc[0].item()) - 1000, index=0)
        
        meas_1 = trim(meas_1, start_time, end_time)
        meas_2 = trim(meas_2, start_time, end_time)
        meas_df = pd.concat([meas_1, meas_2])
        

        
        fig_hist, title_hist = density_plot(meas_df, title=f"{pair}_Distribution of RR-intervals")
        save_plot(fig_hist, title_hist, folder_name=f"out/hist_pairs_{folder_name}/{group_label}", format="html")
        
    
        fig_scatter, title_scatter = scatter_plot(meas_df)
        save_plot(fig_scatter, title_scatter, folder_name=f"out/scatter_pairs_{folder_name}/{group_label}", format="html")
        
        shifted_df = meas_df.copy()
        for i in range(1000, 10001, 1000):
            shifted_df = pd.concat([shifted_df, shift_series(meas_df, i, index=1)])
        
        meas_1 = shifted_df.loc[shifted_df.meas_name == pair[0]]
        meas_2 = shifted_df.loc[shifted_df.meas_name == pair[1]]
        
        correlation_matrix, p_value_matrix = calculate_correlations(meas_1, meas_2)
        best_corr_row = find_best_corr_pairs(correlation_matrix, p_value_matrix, meas_1, meas_2)
        
        # Dodanie nowej kolumny 'group_label' z wartością ze zmiennej group_label
        best_corr_row['group_label'] = group_label
        
        result_df = pd.concat([result_df, best_corr_row], ignore_index=True)
        diff_start_time_ms_df = pd.concat([diff_start_time_ms_df, diff_start_time_ms], ignore_index=True)
    

        fig_heatmap, title_heatmap = corr_heatmap(correlation_matrix, title="corr_heatmap_" + '_'.join(pair), color='coolwarm')
        save_plot(fig_heatmap, title_heatmap, folder_name=f"out/corr_heatmap_{folder_name}/{group_label}", format="png")

    return result_df #selected_df, merged_df, unmatched_meas, diff_start_time_ms_df









