# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:17:46 2024

@author: Hubert Szewczyk
"""

import numpy as np
import pandas as pd
import copy
from datetime import datetime, timedelta
import attr

#%%
@attr.define
class Data:
    x_data: np.ndarray = attr.field(validator=attr.validators.instance_of(np.ndarray))
    y_data: np.ndarray = attr.field(validator=attr.validators.instance_of(np.ndarray)) 

# =============================================================================
#     def update(self, new_x_data: np.ndarray = None, new_y_data: np.ndarray = None):
#         """Updates the x_data and/or y_data arrays."""
#         if new_x_data is not None:
#             self.x_data = new_x_data
#         if new_y_data is not None:
#             self.y_data = new_y_data
# =============================================================================
    def update(self, **kwargs):
        """Updates the data attributes with type checking."""
        for field_name, value in kwargs.items():
           if hasattr(self, field_name):
               setattr(self, field_name, value)
                   
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
        self.update(x_data=new_x_data, y_data=trimmed_y)
    
        # 5. Return the new start (first value) and end (last value) of the trimmed x_data
        return trimmed_x[0], trimmed_x[-1]
     
#%%
@attr.define
class Metadata:
    meas_number: int = attr.field(validator=attr.validators.instance_of(int))
    condition: str = attr.field(validator=attr.validators.instance_of(str))
    gender: str = attr.field(validator=attr.validators.instance_of(str))
    pair_number: int = attr.field(validator=attr.validators.instance_of(int))
    shift: float = attr.field(validator=attr.validators.instance_of(float))
    starttime: datetime = attr.field(validator=attr.validators.instance_of(datetime))
    endtime: datetime = attr.field(validator=attr.validators.instance_of(datetime))
    duration_min: float = attr.field(init=False)

    @duration_min.default
    def _calculate_duration_min(self) -> float:
        return (self.endtime - self.starttime).total_seconds() / 60.0

    def update(self, **kwargs):
        """Updates the metadata attributes with type checking."""
        for field_name, value in kwargs.items():
           if hasattr(self, field_name):
               setattr(self, field_name, value)
        
        if "starttime" in kwargs or "endtime" in kwargs:
            self.duration_min = self._calculate_duration_min()

#%%
@attr.define
class Meas:
    data: Data = attr.field(validator=attr.validators.instance_of(Data))
    metadata: Metadata = attr.field(validator=attr.validators.instance_of(Metadata))
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, meas_number: int, condition: str, gender: str, pair_number: int, shift: float, starttime: datetime, endtime: datetime):
       self.data = Data(x_data, y_data)
       self.metadata = Metadata(meas_number, condition, gender, pair_number, shift, starttime, endtime)
    
    def __add__(self, other: 'Meas') -> 'Meas':
        """
        Merges two Meas objects by concatenating their x_data and y_data arrays, 
        only if their metadata matches and the second Meas starts after the first ends.
        """
        if not isinstance(other, Meas):
            raise TypeError("Can only merge with another Meas object.")

        # Ensure metadata fields are the same
        if (self.metadata.meas_number != other.metadata.meas_number or
            self.metadata.condition != other.metadata.condition or
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
            condition=self.metadata.condition,
            gender=self.metadata.gender,
            pair_number=self.metadata.pair_number,
            shift=self.metadata.shift,
            starttime=self.metadata.starttime,  # Keep the earliest start time
            endtime=other.metadata.endtime  # Use the latest end time
        )

# =============================================================================
#     def update_data(self, new_x_data: np.ndarray = None, new_y_data: np.ndarray = None):
#         """Updates the x_data and/or y_data arrays."""
#         self.data.update(new_x_data = new_x_data, new_y_data = new_y_data)
# =============================================================================
        
# =============================================================================
#     def update_metadata(self, new_meas_number: int = None, new_condition: str = None, new_gender: str = None, new_pair_number: int = None, new_shift: float = None,
#                new_starttime: datetime = None, new_endtime: datetime = None):
#         """Updates the metadata attributes."""
#         self.metadata.update(
#             new_meas_number = new_meas_number,
#             new_condition = new_condition,
#             new_gender = new_gender,
#             new_pair_number = new_pair_number,
#             new_shift = new_shift,
#             new_starttime = new_starttime,
#             new_endtime = new_endtime
#             )
# =============================================================================

# =============================================================================
#     def update(self, new_x_data: np.ndarray = None, new_y_data: np.ndarray = None,
#                new_meas_number: int = None, new_condition: str = None, new_gender: str = None, new_pair_number: int = None, new_shift: float = None,
#                new_starttime: datetime = None, new_endtime: datetime = None):
#         """
#         Updates both data and metadata. Allows None values to be passed and only updates
#         the attributes that are not None.
#         """
#         self.data.update(
#             new_x_data = new_x_data,
#             new_y_data = new_y_data
#             )
#         
#         self.metadata.update(
#             new_meas_number = new_meas_number,
#             new_condition = new_condition,
#             new_gender = new_gender,
#             new_pair_number = new_pair_number,
#             new_shift = new_shift,
#             new_starttime = new_starttime,
#             new_endtime = new_endtime
#             )
# =============================================================================
        
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
        self.metadata.update(starttime=new_starttime, endtime=new_endtime)
        
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
        self.metadata.update(
            starttime=new_starttime,
            endtime=new_endtime,
            shift=new_shift)    
          
    def __repr__(self):
        return f"Meas(data={self.data!r}, metadata={self.metadata!r})"
    
    def __str__(self):
        return f"{self.metadata.meas_number}{self.metadata.condition}{self.metadata.gender}{self.metadata.pair_number}_{self.metadata.shift}"

#%%
@attr.define
class MeasurementRecord:
    meas_number: int = attr.field(validator=attr.validators.instance_of(int))
    condition: str = attr.field(validator=attr.validators.instance_of(str))
    pair_number: int = attr.field(validator=attr.validators.instance_of(int))
    task: str = attr.field(validator=attr.validators.instance_of(str))
    shift_diff: float = attr.field(validator=attr.validators.instance_of(float))
    corr: float = attr.field(validator=attr.validators.instance_of(float))
    p_val: float = attr.field(validator=attr.validators.instance_of(float))
    name_meas1: str = attr.field(validator=attr.validators.instance_of(str))
    name_meas2: str = attr.field(validator=attr.validators.instance_of(str))
    meas1: Meas = attr.field(validator=attr.validators.instance_of(Meas))
    meas2: Meas = attr.field(validator=attr.validators.instance_of(Meas))


#%%
if __name__ == '__main__':
    print("classes.py")
    