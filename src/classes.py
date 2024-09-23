# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:17:46 2024

@author: huber
"""

import numpy as np
import pandas as pd
import copy

from datetime import datetime
from typing import Protocol
from dataclasses import dataclass, field

#%%
class DeepCopyable(Protocol):
    def copy(self) -> 'DeepCopyable':
        ...

#%%
def copy_df_with_immutable(df, obj_column: str):
    """
    Creates a deep copy of a DataFrame with object references in a specific column,
    using the deep_copy method defined by the DeepCopyable protocol.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame to be copied.
    obj_column (str): The name of the column containing objects adhering to the DeepCopyable protocol.
    
    Returns:
    pd.DataFrame: A deep copy of the DataFrame where objects in the specified column are deep copied.
    
    Raises:
    KeyError: If the specified obj_column does not exist in the DataFrame.
    TypeError: If an object in obj_column does not adhere to the DeepCopyable protocol.
    """
    # Check if the specified column exists in the DataFrame
    if obj_column not in df.columns:
        raise KeyError(f"Column '{obj_column}' not found in DataFrame.")
    
    # Create a new DataFrame (shallow copy of the DataFrame structure)
    df_copy = df.copy()

    # Deep copy objects in the specified column
    def deep_copy_obj(obj):
        if obj is None:
            return None
        if isinstance(obj, DeepCopyable):
            return obj.deep_copy()  # Call the deep_copy method defined in the protocol
        else:
            raise TypeError(f"Object of type {type(obj).__name__} does not implement the DeepCopyable protocol.")
    
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

    def copy(self):
        """Creates a deep copy of the Data object."""
        return Data(np.copy(self.x_data), np.copy(self.y_data))
    
    def range(self):
        """Returns the range of x_data and y_data as a tuple."""
        x_range = (np.min(self.x_data), np.max(self.x_data)) if self.x_data.size > 0 else (None, None)
        y_range = (np.min(self.y_data), np.max(self.y_data)) if self.y_data.size > 0 else (None, None)
        return x_range, y_range

    def count_nan(self):
        """Returns the number of NaN values in x_data and y_data."""
        x_nan_count = np.isnan(self.x_data).sum()
        y_nan_count = np.isnan(self.y_data).sum()
        return x_nan_count, y_nan_count
    
    def length(self):
        """Returns the lengths of x_data and y_data."""
        return len(self.x_data), len(self.y_data)
    
    def count_type(self, expected_type: type):
        """Checks for expected type in x_data and y_data and returns their counts."""
        # Check if elements in x_data and y_data are of expected type
        type_count_x = np.sum(isinstance(val, expected_type) for val in self.x_data)
        type_count_y = np.sum(isinstance(val, expected_type) for val in self.y_data)
        
        return type_count_x, type_count_y


    
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

    def update(self, meas_number: int = None, meas_type: str = None, gender: str = None, pair_number: int = None, shift: float = None,
               starttime: datetime = None, endtime: datetime = None):
        """Updates the metadata attributes with type checking."""
        if meas_number is not None:
            if not isinstance(meas_number, int):
                raise TypeError(f"meas_number must be an int, got {type(meas_number)} instead.")
            self.meas_number = meas_number
            
        if meas_type is not None:
            if not isinstance(meas_type, str):
                raise TypeError(f"meas_type must be a string, got {type(meas_type)} instead.")
            self.meas_type = meas_type

        if gender is not None:
            if not isinstance(gender, str):
                raise TypeError(f"gender must be a string, got {type(gender)} instead.")
            self.gender = gender
            
        if pair_number is not None:
            if not isinstance(pair_number, int):
                raise TypeError(f"pair_number must be an int, got {type(pair_number)} instead.")
            self.pair_number = pair_number

        if shift is not None:
            if not isinstance(shift, (float, int)):
                raise TypeError(f"shift must be a float or int, got {type(shift)} instead.")
            self.shift = float(shift)  # Convert int to float if needed

        if starttime is not None:
            if not isinstance(starttime, datetime):
                raise TypeError(f"starttime must be a datetime object, got {type(starttime)} instead.")
            self.starttime = starttime

        if endtime is not None:
            if not isinstance(endtime, datetime):
                raise TypeError(f"endtime must be a datetime object, got {type(endtime)} instead.")
            self.endtime = endtime

        # Recalculate duration when time is updated
        self.__post_init__()

    def copy(self):
        """Creates a deep copy of the Metadata object."""
        return Metadata(self.meas_number, self.meas_type, self.gender, self.pair_number, self.shift, self.starttime, self.endtime)

    def __repr__(self):
        return (f"Metadata(meas_number={self.meas_number!r}, meas_type={self.meas_type!r}, gender={self.gender!r}, pair_number={self.pair_number!r}, "
                f"shift={self.shift}, starttime={self.starttime}, "
                f"endtime={self.endtime}, duration_min={self.duration_min})")

#%%
class Meas:
    data: 'Data'
    metadata: Metadata
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, meas_number: int, meas_type: str, gender: str, pair_number: int, shift: float, starttime: datetime, endtime: datetime):
        self.data = Data(x_data, y_data)
        self.metadata = Metadata(meas_number, meas_type, gender, pair_number, shift, starttime, endtime)

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
        
    def copy(self):
        """Creates a deep copy of the Meas object."""
        return Meas(
            np.copy(self.data.x_data),
            np.copy(self.data.y_data),
            self.metadata.meas_number,
            self.metadata.meas_type,
            self.metadata.gender,
            self.metadata.pair_number,
            self.metadata.shift,
            self.metadata.starttime,
            self.metadata.endtime
        )

    def __repr__(self):
        return f"Meas(data={self.data!r}, metadata={self.metadata!r})"
    
    def __str__(self):
        return f"{self.metadata.meas_number}{self.metadata.meas_type}{self.metadata.gender}{self.metadata.pair_number}_{self.metadata.shift}"

#%%
def test_df(m1, m2):
    # Tworzenie DataFrame z referencjami do obiektów meas
    df = pd.DataFrame({
        'pair': [m1.pair, m2.pair],
        'meas_obj': [m1, m2]
    })
    
    # Aktualizacja x_data w obiekcie meas dla 'Pair1'
    df.loc[df['pair'] == 'Pair1', 'meas_obj'].values[0].update_data(np.array([100, 200, 300]), np.array([400, 500, 600]))
    
    # Sprawdzenie zmian
    print(df.loc[df['pair'] == 'Pair1', 'meas_obj'].values[0])

#%%




#%%
if __name__ == '__main__':
    # Zakładając, że masz już zdefiniowaną klasę Meas
    meas_instance = Meas(np.array([1, 2, 3]), np.array([4, 5, 6]), "example1", "pair1", 0.0, datetime.now(), datetime.now())
    
    # Głęboka kopia
    deep_copied_meas = copy.deepcopy(meas_instance)
    deep_copied_meas.update_metadata("deepcopy10", "pair10", 10.0, datetime.now(), datetime.now())
    
    df = pd.DataFrame({
        'pair': [meas_instance.metadata.pair, deep_copied_meas.metadata.pair],
        'meas_obj': [meas_instance, deep_copied_meas]
    })
    
    # Tworzenie głębokiej kopii DataFrame
    df_copy = copy.deepcopy(df)

    # Aktualizacja danych w kopii DataFrame
    # Sprawdź czy para 'pair10' istnieje w df_copy
    if 'pair10' in df_copy['pair'].values:
        print("Istnieje")
        df_copy.loc[df_copy['pair'] == 'pair10', 'meas_obj'].values[0].update(
            np.array([100, 200, 300]), 
            np.array([400, 500, 600]),
            "deepcopy100", "pair100", 100.0, datetime.now(), datetime.now()
        )

    
    
    