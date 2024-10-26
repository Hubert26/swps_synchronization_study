# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:41:20 2024

@author: Hubert Szewczyk
"""

import pandas as pd


from config import *
from utils.dataframe_utils import read_excel_file, write_to_excel

#%%
def process_measurement_data(df: pd.DataFrame, measure: str, value: str) -> pd.DataFrame:
    """
    Processes the input DataFrame to extract measurement types, pairs, and
    create a pivot table with combined column names.

    Parameters:
    - df (pd.DataFrame): DataFrame containing measurement data with necessary columns.
    - measure (str): The type of measure to be used in the DataFrame.
    - value (str): The value to use for the pivot table values (e.g., 'corr' or 'shift_diff').

    Returns:
    - pd.DataFrame: Processed DataFrame with pivoted columns and combined names.
    
    Raises:
    - ValueError: If required columns are not present in the DataFrame.
    """

    # Check if necessary columns are present in the DataFrame
    required_columns = ['meas1', 'meas_state', value]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Extract 'meas_type' and 'pair' from 'meas1'
    df[['meas_type', 'pair']] = df['meas1'].str.extract(r'(.+)[M](\d+)')

    # Convert 'pair' to numeric type
    df['pair'] = df['pair'].astype(int)

    # Replace specific measurement types with single letters
    df['meas_type'] = df['meas_type'].replace({
        r'(\d+)Cooperation': r'\g<1>C',
        r'(\d+)Baseline': r'\g<1>B',
        r'(\d+)Relaxation': r'\g<1>R'
    }, regex=True)

    # Add a new column for the measure and fill it with the specified value
    df['measure'] = measure  # This adds a new column named 'measure' filled with the given measure

    # Create a pivot table based on the specified value
    result = df.pivot_table(index='pair', 
                             columns=['meas_type', 'meas_state', 'measure'], 
                             values=value,  # Use 'value' as specified to pull correct column data
                             aggfunc='first')  # Use 'first' or another aggregation function as needed

    # Reset index to convert the pivot table back to a DataFrame
    result = result.reset_index()

    # Combine the columns into a single name using underscore as a separator
    result.columns = ['pair'] + ['_'.join(map(str, col)) for col in result.columns[1:]]

    return result

#%%
if __name__ == '__main__':
    # Read and process NN results
    nn_results = read_excel_file(RESULTS_DIR / "analysis_data" / "nn_results.xlsx")
    nn_corr_processed_data = process_measurement_data(nn_results, "NN", 'corr')
    nn_shift_processed_data = process_measurement_data(nn_results, "NN", 'shift_diff')
    
    # Read and process HR results
    hr_results = read_excel_file(RESULTS_DIR / "analysis_data" / "hr_results.xlsx")
    hr_corr_processed_data = process_measurement_data(hr_results, "HR", 'corr')
    hr_shift_processed_data = process_measurement_data(hr_results, "HR", 'shift_diff')
    
    # Read and process SD results
    sd_results = read_excel_file(RESULTS_DIR / "analysis_data" / "sd_results.xlsx")
    sd_corr_processed_data = process_measurement_data(sd_results, "SD", 'corr')
    sd_shift_processed_data = process_measurement_data(sd_results, "SD", 'shift_diff')
    
    # Read and process RMSSD results
    rmssd_results = read_excel_file(RESULTS_DIR / "analysis_data" / "rmssd_results.xlsx")
    rmssd_corr_processed_data = process_measurement_data(rmssd_results, "RMSSD", 'corr')
    rmssd_shift_processed_data = process_measurement_data(rmssd_results, "RMSSD", 'shift_diff')

    # Merge all correlation dataframes on 'pair'
    merged_corr_data = nn_corr_processed_data.merge(hr_corr_processed_data, on='pair', suffixes=('_NN', '_HR')) \
                                              .merge(sd_corr_processed_data, on='pair', suffixes=('', '_SD')) \
                                              .merge(rmssd_corr_processed_data, on='pair', suffixes=('', '_RMSSD'))

    # Merge all shift dataframes on 'pair'
    merged_shift_data = nn_shift_processed_data.merge(hr_shift_processed_data, on='pair', suffixes=('_NN', '_HR')) \
                                                .merge(sd_shift_processed_data, on='pair', suffixes=('', '_SD')) \
                                                .merge(rmssd_shift_processed_data, on='pair', suffixes=('', '_RMSSD'))

#%%
    write_to_excel(merged_corr_data, ANALYSIS_DATA_DIR / "corr_processed_data.xlsx")
    write_to_excel(merged_shift_data, ANALYSIS_DATA_DIR / "shift_processed_data.xlsx")