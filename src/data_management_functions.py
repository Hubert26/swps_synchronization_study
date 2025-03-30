# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:02:21 2024

@author: huber
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import logging
from logger_config import log_message
from config import CONDITIONS

from classes import Meas, MeasurementRecord

from utils.file_utils import read_text_file, extract_file_name
from utils.string_utils import (
    extract_numeric_suffix,
    extract_numeric_prefix,
)
from utils.signal_utils import validate_array
from utils.general_utils import group_object_list


# %%
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
            log_message(
                logging.INFO,
                f"Condition '{condition}' assigned to measurement '{meas_name}'",
            )
            return condition

    log_message(
        logging.ERROR, f"No matching condition found for measurement '{meas_name}'"
    )
    raise ValueError(f"Could not determine condition for '{meas_name}'")


# %%
def extract_data_from_file(file_path: str) -> Meas:
    """
    Extracts data from a text file and creates a Meas object.

    Args:
        file_path (str): Path to the text file containing numerical data.

    Returns:
        Meas: A Meas object containing the data and metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains non-numeric data or the filename is invalid.
    """
    log_message(logging.INFO, f"Processing file: {file_path}")

    if not Path(file_path).is_file():
        log_message(logging.ERROR, f"File '{file_path}' not found.")
        raise FileNotFoundError(f"File '{file_path}' not found.")

    raw_data = read_text_file(file_path).strip().split("\n")

    # Validate numeric data
    try:
        y_data = np.array(raw_data, dtype=float)
    except ValueError:
        log_message(logging.ERROR, f"File '{file_path}' contains non-numeric data.")
        raise ValueError(f"File '{file_path}' contains non-numeric data.")

    x_data = np.cumsum(y_data)

    file_name = extract_file_name(file_path)
    parts = file_name.split()

    if len(parts) != 3:
        raise ValueError(
            f"Invalid filename format '{file_name}'. Expected 'name date time'."
        )

    try:
        starttime = datetime.strptime(parts[1] + parts[2], "%Y-%m-%d%H-%M-%S")
    except ValueError as e:
        log_message(logging.ERROR, f"Invalid timestamp in filename '{file_name}': {e}")
        raise

    endtime = starttime + timedelta(milliseconds=x_data[-1])

    meas_name = parts[0]
    meas_number = extract_numeric_prefix(meas_name)
    pair_number = extract_numeric_suffix(meas_name.split("_")[0])
    condition = assign_condition(meas_name)
    gender = "F" if "k" in meas_name.lower() else "M"

    return Meas(
        x_data=x_data,
        y_data=y_data,
        meas_number=meas_number,
        condition=condition,
        gender=gender,
        pair_number=pair_number,
        shift=0.0,
        starttime=starttime,
        endtime=endtime,
    )


# %%
def load_data(file_paths: list[str]) -> list[Meas]:
    """
    Load data from the given file paths.

    Args:
        file_paths (list[str]): List of file paths to process.

    Returns:
        list[Meas]: List of Meas objects.
    """
    meas_list = []
    for path in file_paths:
        try:
            meas = extract_data_from_file(path)
            meas_list.append(meas)
        except Exception as e:
            log_message(logging.ERROR, f"Failed to process file {path}: {e}")
    return meas_list


# %%
def find_meas(meas_list: list[Meas], **criteria) -> list[Meas]:
    """
    Find Meas objects in meas_list that match specified criteria.

    Args:
        meas_list (list[Meas]): List of Meas objects to search.
        **criteria: Key-value pairs of Meas attributes to filter by.

    Returns:
        list[Meas]: A list of Meas objects that match the given criteria.
    """
    return [
        meas
        for meas in meas_list
        if all(
            getattr(meas.metadata, key, None) == value
            for key, value in criteria.items()
        )
    ]


# %%
def validate_meas_metadata(meas_list: list[Meas], metadata_fields: list[str]) -> bool:
    """
    Validates if all Meas objects in the list have the same metadata values.

    Args:
        meas_list (list[Meas]): List of Meas objects to validate.
        metadata_fields (list[str]): Metadata field names to check for equality.

    Returns:
        bool: True if all Meas objects have the same metadata, False otherwise.
    """
    if len(meas_list) < 2:
        return True

    reference_metadata = meas_list[0].metadata
    for meas in meas_list[1:]:
        for field in metadata_fields:
            if getattr(meas.metadata, field, None) != getattr(
                reference_metadata, field, None
            ):
                log_message(logging.WARNING, f"Metadata mismatch in field '{field}'.")
                return False
    return True


# %%
def validate_meas_data(meas: Meas, min_lenght: int = 3):
    if not isinstance(meas, Meas):
        return 0
    return validate_array(meas.data.x_data, min_length=min_lenght) and validate_array(
        meas.data.y_data, min_length=min_lenght
    )


# %%
def merge_meas(meas_list: list[Meas]) -> Meas:
    """
    Merges a list of Meas objects into a single Meas object.

    Args:
        meas_list (list[Meas]): List of Meas objects to be merged.

    Returns:
        Meas: A single Meas object.
    """
    if not validate_meas_metadata(
        meas_list, ["gender", "meas_number", "condition", "pair_number", "shift"]
    ):
        raise ValueError("Metadata mismatch in Meas objects.")

    sorted_meas_list = sorted(meas_list, key=lambda m: m.metadata.starttime)
    merged_meas = sorted_meas_list[0]

    for meas in sorted_meas_list[1:]:
        merged_meas += meas

    log_message(logging.INFO, "Meas objects successfully merged.")
    return merged_meas


# %%
def merge_grouped(meas_list: list["Meas"]) -> list["Meas"]:
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
    grouped_meas = group_object_list(
        meas_list,
        [
            "metadata.gender",
            "metadata.meas_number",
            "metadata.condition",
            "metadata.pair_number",
            "metadata.shift",
        ],
    )

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


# %%
def find_best_results(
    measurement_records: list[MeasurementRecord],
) -> list[MeasurementRecord]:
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
        measurement_records, ["meas_number", "condition", "task", "pair_number"]
    )

    best_corr_results = []

    # Iterate over each group and find the record with the highest `corr`
    for group_key, records in grouped_records.items():
        # Sort by abs(corr) descending, then by shift_diff ascending
        best_record = min(
            records, key=lambda record: (-abs(record.corr), record.shift_diff)
        )
        best_corr_results.append(best_record)

    return best_corr_results


# %%
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
    df["condition"] = df["condition"].map(
        {"Cooperation": "C", "Baseline": "C", "Relaxation": "R"}
    )

    return df
