# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:19:39 2024

@author: Hubert Szewczyk
"""

from pathlib import Path
import logging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from utils.file_utils import create_directory


# Define directories
ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / 'logs'
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ANALYSIS_DATA_DIR = RESULTS_DIR / "analysis_data"

MIN_DURATION_RATIO = 0.5 #Minimum durration ratio for signals in intervals to be valid
SHIFT_MIN_MS = 1000    # Minimum shift in milliseconds
SHIFT_MAX_MS = 5000    # Maximum shift in milliseconds
SHIFT_STEP_MS = 1000   # Step size for each shift in milliseconds

# Create directories if they don't exist
for directory in [LOGS_DIR, DATA_DIR, RESULTS_DIR, PLOTS_DIR, ANALYSIS_DATA_DIR]:
    create_directory(directory)
    
LOGGING_LVL = 'INFO'  # Can be changed to 'DEBUG', 'ERROR', 'INFO', 'WARNING' , etc.
LOG_FILE = LOGS_DIR / 'app.log'

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# Map string levels to logging constants
log_level = getattr(logging, LOGGING_LVL.upper(), logging.WARNING)

# Set up logging
logging.basicConfig(
    level=log_level,  # Set the log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Define log format
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w')  # Log to file, overwrite each time
    ]
)

# Get the logger instance
logger = logging.getLogger(__name__)

#%%
COOPERATION_TIME_INTERVALS = {
#   (0, 8000): "z1_instr",#nie
    (8000, 28000): "z1_1_f",
    (28000, 48000): "z1_2_m",
    (48000, 68000): "z1_3_f",
    (68000, 88000): "z1_4_m",
    (88000, 108000): "z1_5_f",
    (108000, 128000): "z1_6_m",
    (8000, 128000): "z1",
#   (128000, 136000): "z1_odp_idle",#nie
#   (226000, 226000): "z1_odp",#nie
#   (246000, 286000): "pause",#nie
#   (286000, 294000): "z2_instr",#nie
    (294000, 314000): "z2_1_m",
    (314000, 334000): "z2_2_f",
    (334000, 354000): "z2_3_m",
    (354000, 374000): "z2_4_f",
    (374000, 394000): "z2_5_m",
    (394000, 414000): "z2_6_f",
    (294000, 414000): "z2",
#   (414000, 422000): "z2_odp_idle",#nie
#   (422000, 512000): "z2_odp",#nie
#   (512000, 547000): "baseline2_idle",#nie
    (547000, 787000): "baseline2",
}


BASELINE_TIME_INTERVALS = {
#   (0, 20000): "1_baseline1_idle",#nie
    (20000, 260000): "baseline1",
}


RELAXATION_TIME_INTERVALS = {
    (0, 237000): "baseline1",
    (320000, 740000): "z",
    (793000, 1033000): "baseline2"
}


# Define conditions with measurement type and regex pattern
CONDITIONS = [
    ('Relaxation', r'[12]r.*'),
    ('Baseline', r'[12]w.*\d+_1'),
    ('Cooperation', r'[12]w.*\d+_(?!1)')
]

intervals = {
    'BASELINE': BASELINE_TIME_INTERVALS,
    'COOPERATION': COOPERATION_TIME_INTERVALS,
    'RELAXATION': RELAXATION_TIME_INTERVALS
}

def get_time_intervals(condition: str) -> dict:
    """
    Returns the appropriate time intervals based on the measurement type.
    
    Args:
        condition (str): The measurement type, e.g., 'Baseline' or 'Cooperation'.
    
    Returns:
        dict: The corresponding dictionary of time intervals.
    """
    # Create the key name based on meas_number and condition
    key = f"{condition.upper()}"
    
    # Return the appropriate time intervals
    return intervals.get(key, None)

#%%
