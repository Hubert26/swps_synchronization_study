# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:02:21 2024

@author: huber
"""
import re
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from config import *
from classes import *

from utils.file_utils import read_text_file, extract_file_name
from utils.string_utils import extract_numeric_suffix, extract_numeric_prefix, remove_digits
from utils.signal_utils import validate_array
from utils.general_utils import group_object_list

#%%
def assign_condition(meas_name: str) -> str:
    """
    Assigns a measurement type based on the provided measurement name using predefined regex patterns.

    Args:
        meas_name (str): The name of the measurement.

    Returns:
        str: The assigned measurement type (e.g., 'Relaxation', 'Baseline', 'Cooperation').

    Raises:
        ValueError: If no matching pattern is found.
    """
    for condition, pattern in CONDITIONS:
        if re.match(pattern, meas_name):
            return condition
    raise ValueError(f"Could not determine condition for '{meas_name}'")
    
#%%
def extract_data_from_file(file_path):
    """
    Extracts data from a text file, creates a Meas object based on the data and metadata extracted 
    from the file name.
    
    Args:
        file_path (str): Path to the text file containing numerical data.

    Returns:
        Meas: A Meas object containing the data from the file and metadata extracted from the filename.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains non-numeric data or the filename is invalid.
        ValueError: If there is an error in parsing the timestamp in the filename.
    """
    
    logger.info(f"Starting to process the file: {file_path}")
    
    # Check if the file exists
    if not Path(file_path).is_file():
        logger.error(f"File '{file_path}' not found.")
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    logger.info(f"File '{file_path}' exists, proceeding with data extraction.")
    
    # Read the contents of the file
    y_data = read_text_file(file_path).split('\n')

    # Ensure all data in the file is numeric
    if not all(map(str.isdigit, y_data)):
        logger.error(f"File '{file_path}' contains non-numeric data.")
        raise ValueError(f"File '{file_path}' contains non-numeric data.")
    
    logger.info(f"Data successfully read and validated as numeric from file '{file_path}'.")

    # Convert the data to a numpy array of floats
    y_data = np.array(y_data).astype(float)
    x_data = np.cumsum(y_data)

    # Extract metadata from the file name
    file_name = extract_file_name(file_path)
    splitted_file_name = file_name.split()

    if len(splitted_file_name) != 3:
        logger.error(f"Filename '{file_name}' is invalid. Expected format: 'name date time'.")
        raise ValueError(f"Filename '{file_name}' is invalid. Expected format: 'name date time'.")
    
    logger.info(f"Filename '{file_name}' is valid, proceeding with metadata extraction.")

    # Combine the date and time for parsing
    start_timestamp = splitted_file_name[1] + splitted_file_name[2]
    try:
        starttime = datetime.strptime(start_timestamp, '%Y-%m-%d%H-%M-%S')
    except ValueError as e:
        logger.error(f"Error parsing start_timestamp '{start_timestamp}': {e}")
        raise ValueError(f"Error parsing start_timestamp '{start_timestamp}': {e}")
    
    logger.info(f"Start time '{starttime}' successfully parsed from the filename.")

    # Calculate the end time based on the cumulative sum of x_data
    endtime = starttime + timedelta(milliseconds=x_data[-1])

    logger.info(f"End time calculated as '{endtime}'.")

    # Extract components from the file name
    meas_name = splitted_file_name[0]
    meas_number = extract_numeric_prefix(meas_name)
    pair_number = extract_numeric_suffix(meas_name.split('_')[0])
    condition = assign_condition(meas_name)
    gender = remove_digits(meas_name.split('_')[0])[1]
    gender = 'F' if gender == 'k' else ('M' if gender == 'm' else gender)

    logger.info(f"Metadata extracted from filename: "
                f"meas_number={meas_number}, pair_number={pair_number}, "
                f"condition={condition}, gender={gender}.")

    # Create a new Meas object
    new_meas = Meas(
        x_data=x_data,
        y_data=y_data,
        meas_number=meas_number,
        condition=condition,
        gender=gender,
        pair_number=pair_number,
        shift=0.0,  # Assuming shift is 0.0, can be updated as needed
        starttime=starttime,
        endtime=endtime
    )
    
    logger.info(f"Meas object created successfully for file '{file_name}'.")
    
    return new_meas

#%%
def load_data(file_paths):
    """
    Load data from the given file paths.
    """
    meas_list = []
    for path in file_paths:
        meas_list.append(extract_data_from_file(path))
    return meas_list

#%%
def find_meas(meas_list: list['Meas'], **criteria) -> list['Meas']:
    """
    Find Meas objects in meas_list that match specified criteria.
    
    Args:
        meas_list (list[Meas]): List of Meas objects to search.
        **criteria: Key-value pairs of Meas attributes to filter by.
            For example: meas_number=1, gender='M', condition='Baseline'.
    
    Returns:
        list[Meas]: A list of Meas objects that match the given criteria.
    """
    matched_meas = []
    
    for meas in meas_list:
        match = True
        
        # Iterate through the criteria to check if the Meas object matches
        for key, value in criteria.items():
            # Get the value of the attribute from the Meas object
            if hasattr(meas.metadata, key):
                meas_value = getattr(meas.metadata, key)
                # Check if the value matches the criteria
                if meas_value != value:
                    match = False
                    break
            else:
                # If Meas object does not have the specified attribute, skip this Meas
                match = False
                break
        
        if match:
            matched_meas.append(meas)
    
    return matched_meas

#%%
def validate_meas_metadata(meas_list: list['Meas'], metadata_fields: list[str]) -> bool:
    """
    Validates if all Meas objects in the list have the same metadata values for specified fields.

    Args:
        meas_list (list[Meas]): List of Meas objects to validate.
        metadata_fields (list[str]): List of metadata field names to check for equality.

    Returns:
        bool: True if all Meas objects have the same metadata, False otherwise.
    """
    if len(meas_list) < 2:
        return True  # No need to compare if there's only one Meas

    # Get the reference metadata from the first Meas object
    reference_metadata = meas_list[0].metadata

    for meas in meas_list[1:]:
        for field in metadata_fields:
            if getattr(meas.metadata, field) != getattr(reference_metadata, field):
                return False
    return True

#%%
def validate_meas_data(meas: Meas, min_lenght: int = 3):
    if not isinstance(meas, Meas):
        return 0
    return (validate_array(meas.data.x_data, min_length=min_lenght) and validate_array(meas.data.y_data, min_length=min_lenght))

#%%
def merge_meas(meas_list: list['Meas']) -> 'Meas':
    """
    Merges a list of Meas objects into a single Meas object.
    The merging is done such that the younger Meas is added to the older one.
    Assumes that the provided Meas objects have the same metadata attributes.

    Args:
        meas_list (list[Meas]): List of Meas objects to be merged.

    Returns:
        Meas: A single Meas object that is the result of merging the provided list.
    """
    assert all(isinstance(meas, Meas) for meas in meas_list), "meas_list contains non-Meas objects."

    if not validate_meas_metadata(meas_list, ["gender", "meas_number", "condition", "pair_number", "shift"]):
        raise ValueError(f"Validation MEAS_LIST failed in merge_meas function") 
        
    # Create a deep copy of the list to avoid modifying the original objects
    meas_to_merge = copy.deepcopy(meas_list)
                
    # Sort the group by starttime to ensure younger is added to older
    sorted_meas_list = sorted(meas_to_merge, key=lambda m: m.metadata.starttime)

    # Initialize with the oldest Meas object
    merged_meas = sorted_meas_list[0]

    # Merge all remaining Meas objects into the first one
    for meas in sorted_meas_list[1:]:
        merged_meas += meas

    return merged_meas

#%%
def merge_grouped(meas_list: list['Meas']) -> list['Meas']:
    """
    Merges Meas objects in the provided list that share the same metadata attributes.
    The merging is done in such a way that the younger Meas is added to the older one.
    The function operates on a deep copy of the input list, leaving the original data unchanged.

    Args:
        meas_list (list[Meas]): List of Meas objects to be merged.

    Returns:
        list[Meas]: List of merged Meas objects.
    """
    
    # Group Meas objects by relevant metadata fields
    grouped_meas = group_object_list(meas_list, ["metadata.gender", "metadata.meas_number", "metadata.condition", "metadata.pair_number", "metadata.shift"])
    
    merged_meas_list = []
    
    # Iterate over each group and merge the Meas objects
    for group in grouped_meas.values():
        if len(group) > 1:  # Only merge if there are multiple Meas objects in the group
            merged_meas = merge_meas(group)
            
            merged_meas_list.append(merged_meas)
        else:
            # If the group has only one Meas object, add it directly to the result
            merged_meas_list.append(group[0])
    
    return merged_meas_list

#%%
def find_best_results(measurement_records: list[MeasurementRecord]) -> list[MeasurementRecord]:
    """
    Finds the best correlation result in each group of MeasurementRecords.

    Args:
        measurement_records (list[MeasurementRecord]): 
            List of MeasurementRecord objects to process.

    Returns:
        list[MeasurementRecord]: List of MeasurementRecords with the highest `corr` in each group.
    """
    # Group the records by specified attributes
    grouped_records = group_object_list(
        measurement_records, 
        ["meas_number", "condition", "task", "pair_number"]
    )
    
    best_corr_results = []
    
    # Iterate over each group and find the record with the highest `corr`
    for group_key, records in grouped_records.items():
        # Sort by abs(corr) descending, then by shift_diff ascending
        best_record = min(
            records, 
            key=lambda record: (-abs(record.corr), record.shift_diff)
        )
        best_corr_results.append(best_record)
    
    return best_corr_results

#%%
def records_to_dataframe(measurement_records: list[MeasurementRecord]):
    """
    Converts a list of MeasurementRecord objects into a DataFrame, excluding Meas objects.

    Args:
        measurement_records (list):
            List of MeasurementRecord objects to process.

    Returns:
        DataFrame: A DataFrame containing non-Meas attributes as columns.
    """
    # Use the to_dict method to convert each record to a dictionary
    data_for_df = [record.to_dict() for record in measurement_records]
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data_for_df)
    # Replace specific measurement types with single letters
    df['condition'] = df['condition'].map({
        'Cooperation': 'C',
        'Baseline': 'C',
        'Relaxation': 'R'
    })
    
    return df