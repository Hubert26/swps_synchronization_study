# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:50:13 2024

@author: huber
"""

import os
from pathlib import Path
import pandas as pd
import openpyxl


#%%
def read_excel_file(file_path):
    """
    Opens an Excel file from the provided path.
    
    Args:
        file_path (str): The path to the Excel file.
    
    Returns:
        pd.DataFrame: The contents of the Excel file as a DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened or is not a valid Excel file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # Try to open the Excel file
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Failed to open the Excel file: {e}")
        
#%%
def read_csv_file(file_path, **kwargs):
    """
    Reads a CSV file into a DataFrame.

    This function uses pandas' read_csv method to load data from a CSV file into a DataFrame.
    It allows for various parameters used in pd.read_csv to be passed as keyword arguments.

    Args:
        file_path (str): The path to the CSV file to be read.
        **kwargs: Additional keyword arguments passed to pd.read_csv, e.g., dtype, sep, header, 
        index_col, na_values, low_memory.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is an error parsing the file.
    """
    try:
        # Load the CSV file into a DataFrame with additional parameters
        dataframe = pd.read_csv(file_path, **kwargs)
        return dataframe
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"File not found: {file_path}") from fnf_error
    except pd.errors.EmptyDataError as ede_error:
        raise ValueError(f"The file is empty: {file_path}") from ede_error
    except pd.errors.ParserError as pe_error:
        raise ValueError(f"Error parsing the file: {file_path}") from pe_error
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

#%%
def write_to_excel(dataframe, file_path, **kwargs):
    """
    Writes a DataFrame to an Excel file with optional parameters.

    Args:
        dataframe (pd.DataFrame): The DataFrame to write.
        file_path (str or Path): The path where the Excel file will be saved.
        **kwargs: Additional keyword arguments for `to_excel()`, such as 'index', 'sheet_name'.
                  'mode' is handled for `ExcelWriter`.

    Raises:
        ValueError: If the DataFrame is not valid or if there is an error during file writing.
        OSError: If the file cannot be created or written to.
    """
    # Validate the input dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The provided input is not a valid pandas DataFrame.")

    # Convert file_path to a string if it is a Path object
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    # Ensure the file path ends with .xlsx
    if not file_path.endswith('.xlsx'):
        file_path += '.xlsx'

    # Extract and remove 'index' and 'sheet_name' from kwargs for to_excel()
    index = kwargs.pop('index', False)  # Default is False if not provided
    sheet_name = kwargs.pop('sheet_name', 'Sheet1')

    # Extract 'mode' for ExcelWriter, defaulting to 'w' (write mode)
    mode = kwargs.pop('mode', 'w')

    # Attempt to write the DataFrame to an Excel file
    try:
        with pd.ExcelWriter(file_path,  engine='openpyxl', mode=mode) as writer:
            dataframe.to_excel(writer, sheet_name=sheet_name, index=index, **kwargs)
        print(f"DataFrame successfully written to {file_path}")
    except ValueError as ve:
        raise ValueError(f"Failed to write DataFrame to Excel: {ve}")
    except OSError as oe:
        raise OSError(f"Failed to create or write to file: {oe}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

#%%
def write_to_csv(dataframe, file_path, **kwargs):
    """
    Writes a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be written to the CSV file.
        file_path (str): The path where the CSV file will be saved.
        **kwargs: Additional keyword arguments passed to pd.DataFrame.to_csv, e.g., header, index.

    Raises:
        ValueError: If the DataFrame is invalid or if there is an error during file writing.
        OSError: If the file cannot be created or written to.
    """
    # Validate the input dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The provided input is not a valid pandas DataFrame.")
    
    # Ensure the directory exists
    dir_path = Path(file_path).parent
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Attempt to write the DataFrame to a CSV file
    try:
        dataframe.to_csv(file_path, **kwargs)
        print(f"DataFrame successfully written to {file_path}")
    except ValueError as ve:
        # Catch errors related to the DataFrame or file writing issues
        raise ValueError(f"Failed to write DataFrame to CSV: {ve}")
    except OSError as oe:
        # Catch errors related to file system issues
        raise OSError(f"Failed to create or write to file: {oe}")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"An unexpected error occurred: {e}")

#%%
def filter_dataframe(dataframe, **filters):
    """
    Filters the DataFrame based on the specified column criteria without resetting or changing the index.

    Args:
        dataframe (pd.DataFrame): The DataFrame to filter.
        **filters: Key-value pairs where the key is the column name and the value is the filter criterion.
                   The value can be a single value or a list of values for filtering.
                   Example: filter_dataframe(df, Age1=30, Gender=['F', 'M'])

    Returns:
        pd.DataFrame: The filtered DataFrame that matches all the filter conditions, 
                      with the original index preserved.
    """
    filtered_dataframe = dataframe.copy()

    # Apply each filter condition to the DataFrame
    for column, value in filters.items():
        # If the filter value is a list, use the isin() method for multiple values
        if isinstance(value, list):
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column].isin(value)]
        else:
            # Otherwise, filter for exact matches
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column] == value]
    
    return filtered_dataframe
