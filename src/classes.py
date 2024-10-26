# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:17:46 2024

@author: Hubert Szewczyk
"""

import numpy as np
import pandas as pd
import copy

from datetime import datetime, timedelta
from dataclasses import field


    
#%%
class DataValidationError(Exception):
    pass

#%%
class Data:
    x_data: np.ndarray
    y_data: np.ndarray

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        if not isinstance(x_data, np.ndarray):
            raise TypeError(f"x_data must be a numpy array, got {type(x_data)} instead.")
        if not isinstance(y_data, np.ndarray):
            raise TypeError(f"y_data must be a numpy array, got {type(y_data)} instead.")
            
        self.x_data = x_data
        self.y_data = y_data
        
    def __repr__(self):
        x_shape = self.x_data.shape if self.x_data.ndim > 0 else (0,)
        y_shape = self.y_data.shape if self.y_data.ndim > 0 else (0,)
        return (f"Data(x_data: shape={x_shape}, dtype={self.x_data.dtype}, "
                f"y_data: shape={y_shape}, dtype={self.y_data.dtype})")
    

    def update(self, new_x_data: np.ndarray = None, new_y_data: np.ndarray = None):
        """Updates the x_data and/or y_data arrays."""
        if new_x_data is not None and not isinstance(new_x_data, np.ndarray):
            raise TypeError(f"new_x_data must be a numpy array, got {type(new_x_data)} instead.")
        if new_y_data is not None and not isinstance(new_y_data, np.ndarray):
                raise TypeError(f"new_y_data must be a numpy array, got {type(new_y_data)} instead.")
            
        if new_x_data is not None:
            self.x_data = new_x_data
        if new_y_data is not None:
            self.y_data = new_y_data
    
    def trim(self, new_start, new_end):
        """
        Trims x_data and y_data based on the specified range of x_data values.
    
        Args:
            new_start (float): The new starting value for x_data.
            new_end (float): The new ending value for x_data.
    
        Returns:
            tuple: The first and last values of the trimmed x_data (representing the new start and end of the trimmed range).
    
        Raises:
            ValueError: If the selected indices are empty, or if new_start >= new_end.
        """
        # Check if the provided range is valid
        if new_start >= new_end:
            raise ValueError(f"Invalid range: new_start ({new_start}) must be less than new_end ({new_end}).")
    
        # 1. Select the indices where x_data is within the specified range [new_start, new_end]
        selected_indices = np.where((self.x_data >= new_start) & (self.x_data <= new_end))[0]
    
        # Check if any indices are selected
        if selected_indices.size == 0:
            raise ValueError(f"No data points found in the range [{new_start}, {new_end}].")
    
        # 2. Extract the trimmed x_data and y_data using the selected indices
        trimmed_x = self.x_data[selected_indices]
        trimmed_y = self.y_data[selected_indices]
    
        # Ensure trimmed_x and trimmed_y are not empty or mismatched
        if len(trimmed_x) == 0 or len(trimmed_y) == 0:
            raise ValueError(f"Trimming resulted in empty data. Check the provided range [{new_start}, {new_end}].")
        if len(trimmed_x) != len(trimmed_y):
            raise ValueError("Mismatch between the lengths of trimmed x_data and y_data.")
    
        # 3. Update the x_data values by normalizing them to start from 0
        new_x_data = trimmed_x - trimmed_x[0]
    
        # 4. Update the internal x_data and y_data arrays using the update method
        self.update(new_x_data=new_x_data, new_y_data=trimmed_y)
    
        # 5. Return the new start (first value) and end (last value) of the trimmed x_data
        return trimmed_x[0], trimmed_x[-1]
     
#%%
class Metadata:
    meas_number: int
    meas_type: str
    gender: str
    pair_number: int
    shift: float = field(default=0.0)
    starttime: datetime
    endtime: datetime
    duration_min: float

    def __init__(self, meas_number: int, meas_type: str, gender: str, pair_number: int, shift: float, starttime: datetime, endtime: datetime):
        # Type checking for initialization
        if not isinstance(meas_number, int):
            raise TypeError(f"meas_number must be an int, got {type(meas_number)} instead.")
        if not isinstance(meas_type, str):
            raise TypeError(f"meas_type must be a string, got {type(meas_type)} instead.")
        if not isinstance(gender, str):
            raise TypeError(f"gender must be a string, got {type(gender)} instead.")
        if not isinstance(pair_number, int):
            raise TypeError(f"pair_number must be an int, got {type(pair_number)} instead.")
        if not isinstance(shift, (float, int)):
            raise TypeError(f"shift must be a float or int, got {type(shift)} instead.")
        if not isinstance(starttime, datetime):
            raise TypeError(f"starttime must be a datetime object, got {type(starttime)} instead.")
        if not isinstance(endtime, datetime):
            raise TypeError(f"endtime must be a datetime object, got {type(endtime)} instead.")

        self.meas_number = meas_number
        self.meas_type = meas_type
        self.gender = gender
        self.pair_number = pair_number
        self.shift = float(shift)  # Convert int to float if needed
        self.starttime = starttime
        self.endtime = endtime
        self.__post_init__()

    def __post_init__(self):
        # Automatically calculate duration in minutes
        self.duration_min = (self.endtime - self.starttime).total_seconds() / 60.0

    def update(self, new_meas_number: int = None, new_meas_type: str = None, new_gender: str = None, new_pair_number: int = None, new_shift: float = None,
               new_starttime: datetime = None, new_endtime: datetime = None):
        """Updates the metadata attributes with type checking."""
        if new_meas_number is not None:
            if not isinstance(new_meas_number, int):
                raise TypeError(f"new_meas_number must be an int, got {type(new_meas_number)} instead.")
            self.meas_number = new_meas_number
            
        if new_meas_type is not None:
            if not isinstance(new_meas_type, str):
                raise TypeError(f"new_meas_type must be a string, got {type(new_meas_type)} instead.")
            self.meas_type = new_meas_type

        if new_gender is not None:
            if not isinstance(new_gender, str):
                raise TypeError(f"new_gender must be a string, got {type(new_gender)} instead.")
            self.gender = new_gender
            
        if new_pair_number is not None:
            if not isinstance(new_pair_number, int):
                raise TypeError(f"new_pair_number must be an int, got {type(new_pair_number)} instead.")
            self.pair_number = new_pair_number

        if new_shift is not None:
            if not isinstance(new_shift, (float, int)):
                raise TypeError(f"new_shift must be a float or int, got {type(new_shift)} instead.")
            self.shift = float(new_shift)  # Convert int to float if needed

        if new_starttime is not None:
            if not isinstance(new_starttime, datetime):
                raise TypeError(f"new_starttime must be a datetime object, got {type(new_starttime)} instead.")
            self.starttime = new_starttime

        if new_endtime is not None:
            if not isinstance(new_endtime, datetime):
                raise TypeError(f"new_endtime must be a datetime object, got {type(new_endtime)} instead.")
            self.endtime = new_endtime

        # Recalculate duration when time is updated
        self.__post_init__()

    def __repr__(self):
        return (f"Metadata(meas_number={self.meas_number!r}, meas_type={self.meas_type!r}, gender={self.gender!r}, pair_number={self.pair_number!r}, "
                f"shift={self.shift}, starttime={self.starttime}, "
                f"endtime={self.endtime}, duration_min={self.duration_min})")

#%%
class Meas:
    data: Data
    metadata: Metadata
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, meas_number: int, meas_type: str, gender: str, pair_number: int, shift: float, starttime: datetime, endtime: datetime):
        self.data = Data(x_data, y_data)
        self.metadata = Metadata(meas_number, meas_type, gender, pair_number, shift, starttime, endtime)
        
    def __add__(self, other: 'Meas') -> 'Meas':
        """
        Merges two Meas objects by concatenating their x_data and y_data arrays, 
        only if their metadata matches and the second Meas starts after the first ends.
        """
        if not isinstance(other, Meas):
            raise TypeError("Can only merge with another Meas object.")

        # Ensure metadata fields are the same
        if (self.metadata.meas_number != other.metadata.meas_number or
            self.metadata.meas_type != other.metadata.meas_type or
            self.metadata.gender != other.metadata.gender or
            self.metadata.pair_number != other.metadata.pair_number or
            self.metadata.shift != other.metadata.shift):
            raise ValueError("Cannot merge Meas objects with different metadata attributes.")

        # Ensure the current Meas is older
        if self.metadata.endtime > other.metadata.starttime:
            raise ValueError("Cannot merge: other Meas must start after the current Meas ends.")

        # Calculate time difference in milliseconds
        diff_starttime = (other.metadata.starttime - self.metadata.starttime).total_seconds() * 1000

        # Concatenate y_data and adjust x_data by the time shift
        new_y_data = np.concatenate((self.data.y_data, other.data.y_data))
        new_x_data = np.concatenate((self.data.x_data, other.data.x_data + diff_starttime))

        # Return new Meas object with merged data
        return Meas(
            x_data=new_x_data,
            y_data=new_y_data,
            meas_number=self.metadata.meas_number,
            meas_type=self.metadata.meas_type,
            gender=self.metadata.gender,
            pair_number=self.metadata.pair_number,
            shift=self.metadata.shift,
            starttime=self.metadata.starttime,  # Keep the earliest start time
            endtime=other.metadata.endtime  # Use the latest end time
        )

    def update_data(self, new_x_data: np.ndarray = None, new_y_data: np.ndarray = None):
        """Updates the x_data and/or y_data arrays."""
        self.data.update(new_x_data, new_y_data)
        
    def update_metadata(self, new_meas_number: int = None, new_meas_type: str = None, new_gender: str = None, new_pair_number: int = None, new_shift: float = None,
               new_starttime: datetime = None, new_endtime: datetime = None):
        """Updates the metadata attributes."""
        self.metadata.update(new_meas_number, new_meas_type, new_gender, new_pair_number, new_shift, new_starttime, new_endtime)

    def update(self, new_x_data: np.ndarray = None, new_y_data: np.ndarray = None,
               new_meas_number: int = None, new_meas_type: str = None, new_gender: str = None, new_pair_number: int = None, new_shift: float = None,
               new_starttime: datetime = None, new_endtime: datetime = None):
        """
        Updates both data and metadata. Allows None values to be passed and only updates
        the attributes that are not None.
        """
        self.update_data(new_x_data, new_y_data)
        self.update_metadata(new_meas_number, new_meas_type, new_gender, new_pair_number, new_shift, new_starttime, new_endtime)
        
    def trim(self, start: float, end: float):
        """
        Trims the x_data and y_data arrays in the Data object to only include values within the provided range.
        It also updates the Metadata object with the new starttime and endtime.
    
        Args:
            start (float): The starting value for trimming the x_data.
            end (float): The ending value for trimming the x_data.
    
        Returns:
            None
        """
        # 1. Trim the data using the trim method of the Data class.
        #    This will return the new start and end values of x_data after trimming.
        new_start, new_end = self.data.trim(start, end)
        
        # 2. Calculate the new starttime based on the trimmed start value.
        #    The original starttime (in Metadata) is adjusted by the new_start (converted from milliseconds to timedelta).
        new_starttime = self.metadata.starttime + pd.to_timedelta(new_start, unit='ms')
        
        # 3. Calculate the new endtime similarly, based on the new_end value.
        new_endtime = self.metadata.starttime + pd.to_timedelta(new_end, unit='ms')
    
        # 4. Update the Metadata object with the new starttime and endtime.
        self.metadata.update(new_starttime=new_starttime, new_endtime=new_endtime)
        
    def shift_right(self, shift_ms: float):
        """
        Shifts the starttime of the Meas object to the right by the given amount of time (in milliseconds).
        The shift attribute is updated accordingly (in seconds).
        
        Args:
            shift_ms (float): The amount of time (in milliseconds) to shift the signal.
        """
        # Dodaj przesunięcie do starttime
        new_starttime = self.metadata.starttime + timedelta(milliseconds=shift_ms)
        new_endtime = self.metadata.endtime + timedelta(milliseconds=shift_ms)
    
        # Konwertuj milisekundy na sekundy do zapisu w shift
        shift_s = shift_ms / 1000.0
        new_shift = self.metadata.shift + shift_s
    
        # Zaktualizuj obiekt Meas z nowym starttime i shift
        self.update(new_starttime=new_starttime, new_endtime=new_endtime, new_shift=new_shift)    
          
    def __repr__(self):
        return f"Meas(data={self.data!r}, metadata={self.metadata!r})"
    
    def __str__(self):
        return f"{self.metadata.meas_number}{self.metadata.meas_type}{self.metadata.gender}{self.metadata.pair_number}_{self.metadata.shift}"

#%%
def split_nn_rr_meas(meas: 'Meas', threshold: float = 1e-2) -> list['Meas']:
    """
    Splits a Meas object (specifically for NN or RR intervals) into multiple Meas objects 
    wherever there's a mismatch between np.diff(x_data) and y_data, indicating a gap in the data.

    Args:
        meas (Meas): The Meas object containing NN or RR interval data.
        threshold (float, optional): The tolerance level for detecting gaps based on differences 
                                     in x_data and y_data. Default is 1e-2.
    
    Returns:
        list[Meas]: A list of Meas objects, each representing a contiguous segment of the input data.
    """

    def recursive_split(x_data, y_data, starttime, threshold):
        """
        Helper function that performs the recursive splitting of the data.

        Args:
            x_data (ndarray): The x-axis data (typically time) of the measurement.
            y_data (ndarray): The y-axis data (typically the measured intervals).
            starttime (datetime): The start time for the data segment.
            threshold (float): The tolerance level for detecting gaps.

        Returns:
            list[Meas]: A list of Meas objects for each detected segment.
        """
        # Base case: if the segment is too small to split
        if len(x_data) <= 1:
            norm_x_data = x_data - x_data[0] + y_data[0]
            return [Meas(
                x_data=norm_x_data,
                y_data=y_data,
                meas_number=meas.metadata.meas_number,
                meas_type=meas.metadata.meas_type,
                gender=meas.metadata.gender,
                pair_number=meas.metadata.pair_number,
                shift=meas.metadata.shift,
                starttime=starttime,
                endtime=starttime + timedelta(milliseconds=norm_x_data[-1])
            )]

        # Calculate the differences between consecutive x_data elements
        diff_x_data = np.diff(x_data)

        # Identify indices where the diff in x_data does not match y_data
        mismatch_indices = np.where(np.abs(diff_x_data - y_data[1:]) > threshold)[0]

        # Normalize x_data
        norm_x_data = x_data - x_data[0] + y_data[0]

        if len(mismatch_indices) == 0:
            # No gaps detected, return a single Meas object
            return [Meas(
                x_data=norm_x_data,
                y_data=y_data,
                meas_number=meas.metadata.meas_number,
                meas_type=meas.metadata.meas_type,
                gender=meas.metadata.gender,
                pair_number=meas.metadata.pair_number,
                shift=meas.metadata.shift,
                starttime=starttime,
                endtime=starttime + timedelta(milliseconds=norm_x_data[-1])
            )]

        # First mismatch (gap detected)
        first_mismatch = mismatch_indices[0] + 1

        # Split x_data and y_data into two parts at the gap
        y_data_1, y_data_2 = y_data[:first_mismatch], y_data[first_mismatch:]
        x_data_1, x_data_2 = x_data[:first_mismatch], x_data[first_mismatch:]

        # If the second part is empty, return only the first part
        if len(x_data_2) == 0 or len(y_data_2) == 0:
            return [Meas(
                x_data=x_data_1,
                y_data=y_data_1,
                meas_number=meas.metadata.meas_number,
                meas_type=meas.metadata.meas_type,
                gender=meas.metadata.gender,
                pair_number=meas.metadata.pair_number,
                shift=meas.metadata.shift,
                starttime=starttime,
                endtime=starttime + timedelta(milliseconds=norm_x_data[-1])
            )]

        # Normalize the first part of x_data
        x_data_1 = x_data_1 - x_data_1[0] + y_data_1[0]

        # Calculate the new start time for the second part
        starttime_2 = starttime + timedelta(milliseconds=(norm_x_data[first_mismatch] - y_data[first_mismatch]))

        # Recursively split both parts
        return (
            recursive_split(x_data_1, y_data_1, starttime, threshold) +
            recursive_split(x_data_2, y_data_2, starttime_2, threshold)
        )

    # Start the recursive splitting process
    return recursive_split(meas.data.x_data, meas.data.y_data, meas.metadata.starttime, threshold)




#%%
if __name__ == '__main__':

    #Testing
    y_data_1 = np.array([100, 100, 100, 100, 100])
    x_data_1 = np.cumsum(y_data_1, dtype=float)
    endtime_1 = timedelta(milliseconds=x_data_1[-1])

    meas1 = Meas(
        x_data=x_data_1, 
        y_data=y_data_1, 
        meas_number=1, 
        meas_type="HR", 
        gender="M", 
        pair_number=1, 
        shift=0.0, 
        starttime=datetime(2024, 9, 24, 10, 0, 0), 
        endtime=datetime(2024, 9, 24, 10, 0, 0) + endtime_1
    )
    
    y_data_2 = np.array([101, 101, 101, 101, 101])
    x_data_2 = np.cumsum(y_data_2, dtype=float)
    endtime_2 = timedelta(milliseconds=x_data_2[-1])
    
    meas2 = Meas(
        x_data=x_data_2, 
        y_data=y_data_2, 
        meas_number=1, 
        meas_type="HR", 
        gender="M", 
        pair_number=1, 
        shift=0.0, 
        starttime=datetime(2024, 9, 24, 10, 5, 1), 
        endtime=datetime(2024, 9, 24, 10, 5, 1) + endtime_2
    )
    
    y_data_3 = np.array([110, 110, 110, 110, 110])
    x_data_3 = np.cumsum(y_data_3, dtype=float)
    endtime_3 = timedelta(milliseconds=x_data_3[-1])
    
    meas3 = Meas(
        x_data=x_data_3, 
        y_data=y_data_3, 
        meas_number=1, 
        meas_type="HR", 
        gender="M", 
        pair_number=1, 
        shift=0.0, 
        starttime=datetime(2024, 9, 24, 10, 10, 10), 
        endtime=datetime(2024, 9, 24, 10, 10, 10) + endtime_3
    )
    
    merged_meas = meas1 + meas2 + meas3
    
    splitted_meas_list = merged_meas.split()
    
    merged_x = np.concatenate((x_data_1, x_data_2, x_data_3))
    merged_y = np.concatenate((y_data_1, y_data_2, y_data_3))
    