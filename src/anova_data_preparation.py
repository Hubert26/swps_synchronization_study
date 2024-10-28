# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:41:20 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import copy
import re


from config import *
from utils.dataframe_utils import read_excel_file, write_to_excel

#%%
regex_patterns = {
    'meas_number': r'^(1|2)',  # Zakładając, że tylko '1' i '2' są wartościami meas_number
    'condition': r'_(C|R)_',   # Zakładając, że tylko 'C' i 'R' są wartościami condition
    'task': r'_(baseline1|baseline2|z\d(?:_\d_[fm])?)_',
    'value_name': r'_(CORR|SHIFT)_',  # 'CORR' lub 'SHIFT'
    'measure_type': r'_(HR|SDNN|RMSSD)$'  # 'HR', 'SDNN' lub 'RMSSD'
}

task_order = [
    "baseline1",
    "z1", "z1_1_f", "z1_2_m", "z1_3_f", "z1_4_m", "z1_5_f", "z1_6_m",
    "z2", "z2_1_m", "z2_2_f", "z2_3_m", "z2_4_f", "z2_5_m", "z2_6_f",
    "baseline2"
]

sorting_rules = {
    'meas_number': ['1', '2'],
    'condition': ['R', 'C'],
    'task': task_order
}

#%%
def extract_column_parts(df: pd.DataFrame, patterns: dict) -> pd.DataFrame:
    """
    Rozdziela części nazwy kolumn na podstawie określonych regexów.
    
    Parameters:
    - df (pd.DataFrame): DataFrame z kolumnami do analizy.
    - patterns (dict): Słownik z regexami dla każdej części kolumny.
    
    Returns:
    - pd.DataFrame: DataFrame z kolumnami rozdzielonymi na części.
    """
    # DataFrame do przechowywania rozdzielonych części nazw kolumn
    parts_df = pd.DataFrame(index=df.columns)
    
    # Dopasowanie każdego regexa i wyciągnięcie wartości dla każdej części
    for part, pattern in patterns.items():
        # Użycie str.findall do uzyskania wszystkich dopasowań
        parts = df.columns.str.findall(pattern)

        # List comprehension, aby uzyskać pierwsze dopasowanie lub None
        parts_df[part] = [x[0] if len(x) > 0 else None for x in parts]
    
    return parts_df


#%%
def sort_columns_by_rules(df: pd.DataFrame, patterns: dict, sorting_rules: dict) -> pd.DataFrame:
    """
    Sortuje kolumny DataFrame’a na podstawie wyodrębnionych części nazw i reguł sortowania.
    
    Parameters:
    - df (pd.DataFrame): DataFrame do posortowania.
    - patterns (dict): Słownik z regexami dla każdej części nazwy.
    - sorting_rules (dict): Reguły sortowania dla wyodrębnionych części.
    
    Returns:
    - pd.DataFrame: DataFrame z kolumnami posortowanymi na podstawie sorting_rules.
    """
    # Wyodrębnienie części nazw kolumn
    parts_df = extract_column_parts(df, patterns)
    parts_df['original_column'] = df.columns  # Dodaj oryginalne nazwy kolumn

    # Zastosowanie reguł sortowania
    for col_name, order in sorting_rules.items():
        parts_df[col_name] = pd.Categorical(parts_df[col_name], categories=order, ordered=True)
    
    # Sortowanie
    sorted_parts_df = parts_df.sort_values(list(sorting_rules.keys()))
    
    # Zwrócenie DataFrame’a z posortowanymi kolumnami
    return df[sorted_parts_df['original_column'].values]

#%%
def process_measurement_data(df: pd.DataFrame, value: str) -> pd.DataFrame:
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
    required_columns = ['meas1', 'meas2', 'meas_number', 'meas_type', 'pair_number', 'meas_state', value]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    df = copy.deepcopy(df)

    # Convert 'pair' to numeric type
    df['pair_number'] = df['pair_number'].astype(int)

    # Replace specific measurement types with single letters
    df['meas_type'] = df['meas_type'].map({
        'Cooperation': 'C',
        'Baseline': 'C',
        'Relaxation': 'R'
    })

    # Create a pivot table based on the specified value
    result = df.pivot_table(index='pair_number', 
                             columns=['meas_number', 'meas_type', 'meas_state'], 
                             values=value,  # Use 'value' as specified to pull correct column data
                             aggfunc='first')  # Use 'first' or another aggregation function as needed

    # Reset index to convert the pivot table back to a DataFrame
    result = result.reset_index()

    # Combine the columns into a single name using underscore as a separator
    result.columns = ['pair_number'] + ['_'.join(map(str, col)) for col in result.columns[1:]]

    return result

#%%
if __name__ == '__main__':
    
# =============================================================================
#     # Read and process NN results
#     nn_results = read_excel_file(RESULTS_DIR / "analysis_data" / "nn_results.xlsx")
#     nn_corr_processed_data = process_measurement_data(nn_results, 'corr')
#     nn_shift_processed_data = process_measurement_data(nn_results, 'shift_diff')
#     nn_anova_data = nn_corr_processed_data.merge(nn_shift_processed_data, on='pair_number', suffixes=('_CORR', '_SHIFT'))
# =============================================================================
    
    # Read and process HR results
    hr_results = read_excel_file(RESULTS_DIR / "analysis_data" / "hr_results.xlsx")
    hr_corr_processed_data = process_measurement_data(hr_results, 'corr')
    hr_shift_processed_data = process_measurement_data(hr_results, 'shift_diff')
    hr_anova_data = hr_corr_processed_data.merge(hr_shift_processed_data, on='pair_number', suffixes=('_CORR', '_SHIFT'))
    hr_anova_data.set_index('pair_number', inplace=True)
    hr_anova_data = hr_anova_data.add_suffix('_HR')
    
    # Read and process SD results
    sdnn_results = read_excel_file(RESULTS_DIR / "analysis_data" / "sdnn_results.xlsx")
    sdnn_corr_processed_data = process_measurement_data(sdnn_results, 'corr')
    sdnn_shift_processed_data = process_measurement_data(sdnn_results, 'shift_diff')
    sdnn_anova_data = sdnn_corr_processed_data.merge(sdnn_shift_processed_data, on='pair_number', suffixes=('_CORR', '_SHIFT'))
    sdnn_anova_data.set_index('pair_number', inplace=True)
    sdnn_anova_data = sdnn_anova_data.add_suffix('_SDNN')
    
    # Read and process RMSSD results
    rmssd_results = read_excel_file(RESULTS_DIR / "analysis_data" / "rmssd_results.xlsx")
    rmssd_corr_processed_data = process_measurement_data(rmssd_results, 'corr')
    rmssd_shift_processed_data = process_measurement_data(rmssd_results, 'shift_diff')
    rmssd_anova_data = rmssd_corr_processed_data.merge(rmssd_shift_processed_data, on='pair_number', suffixes=('_CORR', '_SHIFT'))
    rmssd_anova_data.set_index('pair_number', inplace=True)
    rmssd_anova_data = rmssd_anova_data.add_suffix('_RMSSD')

    # Merge all anova dataframes on index
    anova_data = hr_anova_data.merge(sdnn_anova_data, left_index=True, right_index=True, suffixes=('_HR', '_SDNN')) \
                               .merge(rmssd_anova_data, left_index=True, right_index=True, suffixes=('', '_RMSSD'))

    # Dropping columns with "R" and "SHIFT" in name
    columns_to_drop = [col for col in anova_data.columns if "R" in col and "SHIFT" in col]
    anova_data = anova_data.drop(columns=columns_to_drop)
    anova_data = sort_columns_by_rules(anova_data, regex_patterns, sorting_rules)
    anova_data.reset_index(names='pair_number', inplace=True)
#%%
    write_to_excel(anova_data, ANALYSIS_DATA_DIR / "anova_data.xlsx")
    
#%%
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    
    # Load workbook and select the active sheet
    wb = openpyxl.load_workbook(ANALYSIS_DATA_DIR / "anova_data.xlsx")
    ws = wb.active
    
    note = (
        "Variable Naming Convention:\n"
        "Each variable name in this dataset follows a structured format to specify its measurement details.\n"
        "<meas_number>_<condition>_<task>_<value_name>_<measure_type>\n\n"
        "Example: '1_C_z1_1_f_CORR_HR'\n\n"
        "- meas_number (e.g., 1): Measurement number:\n"
        "    - '1' for first session before intervention\n"
        "    - '2' for second session after intervention\n"
        "- condition (e.g., C): Condition:\n"
        "    - 'C' for Cooperation\n"
        "    - 'R' for Relaxation\n"
        "- task (e.g., z1_1_f): The specific task performed by the participants.\n"
        "    - for condition 'C':\n"
        "       - z<exercise_number>_<exercise_round>_<gender>\n"
        "           - exercise_number:\n"
        "               - '1' for first exercise in condition\n"
        "               - '2' for second exercise in condition\n"
        "           - exercise_round:\n"
        "               - '1' to '6': indicates the sequential round of the exercise within the task (e.g., '1' for the first round, '2' for the second, etc.)\n"
        "           - gender:\n"
        "               - 'f' for female and 'm' for male: indicates who leads in specific exercise\n"
        "       - z<exercise_number>: all exercises parts in specific exercise\n"
        "           - exercise_number:\n"
        "               - '1' for first exercise in condition\n"
        "               - '2' for second exercise in condition\n"
        "    - for condition 'R':\n"
        "       - z\n"
        "    - for condition 'C' and 'R':\n"
        "       - baseline<baseline_number>\n"
        "           - baseline_number:\n"
        "               - '1' for baseline before tasks in condition\n"
        "               - '2' for baseline after tasks in condition\n"
        "- value_name (e.g., CORR): Value type\n"
        "    - 'CORR' for correlation value\n"
        "    - 'SHIFT' for shift difference\n"
        "- measure_type (e.g., HR): Type of measure\n"
        "    - 'HR' for Heart Rate\n"
        "    - 'SDNN' for Standard Deviation of NN-intervals\n"
        "    - 'RMSSD' for Root Mean Square of Successive Differences of NN-intervals\n"
    )
 
    # Add the note as a comment to cell A1
    comment = Comment(note, "HS")

    # Estimate width and height based on text length
    # Each character approximates about 5 pixels in width, and each line about 20 pixels in height
    approx_width = min(10000, max(1000, len(note) * 5 // 100))  # Width is capped between 300 and 800 points
    approx_height = min(5000, max(1000, note.count('\n') * 20 + 100))  # Height is capped between 300 and 600 points
    
    # Set comment dimensions
    comment.width = approx_width
    comment.height = approx_height

    ws["A1"].comment = comment
    
    # Save the workbook
    wb.save(ANALYSIS_DATA_DIR / "anova_data.xlsx")
    
    #%%
    # Save the note to a text file
    with open(ANALYSIS_DATA_DIR / "variable_naming_convention.txt", "w") as file:
        file.write(note)

       
